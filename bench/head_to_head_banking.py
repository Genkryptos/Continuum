"""
bench/head_to_head_banking.py
=============================
**True** head-to-head: Continuum, the naive baselines, and the real Mem0 SDK
all scored on the *same sampled scenarios*, with the same ground truth.

Why this exists
---------------
``bench/mem0_compare.py`` runs Mem0 and prints Continuum's numbers beside it
from a hardcoded reference table. That is juxtaposition, not a comparison —
the two systems never saw the same scenario list. This harness fixes that:
it draws one sample, then runs every system over that identical sample.

Sampling
--------
The banking bi-temporal corpus is 500 questions (375 point-in-time + 125
backdated corrections). Running Mem0 over all 500 is impractical — Mem0 makes
a live LLM call *per stored fact*, so a full run is thousands of calls. So we
draw a **stratified sample** and deliberately **oversample the retroactive
stratum**, because that is the load-bearing claim and needs statistical power.

Because the strata are deliberately unbalanced relative to the corpus, a raw
blended score would misrepresent the corpus. We therefore report:

* **per-stratum rates** (the honest primary numbers), and
* a **corpus-weighted overall** = 0.75 x point-in-time + 0.25 x retroactive,
  reconstructing the true 375/125 corpus composition from the stratum rates.

Never quote the unweighted sample mean as a corpus score.

Run
---
::

    set -a && source .env && set +a
    python3.12 -m bench.head_to_head_banking --pit 20 --retro 20 --ss 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_CHROMA = Path(
    os.environ.get("MEM0_COMPARE_CHROMA")
    or (Path(tempfile.gettempdir()) / "head_to_head_chroma")
)
_LLM_MODEL = "openai/gpt-4o-mini"


# ── Mem0 plumbing (mirrors bench/mem0_compare.py) ────────────────────────────
def _build_memory():
    from mem0 import Memory
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("[mem0] set OPENROUTER_API_KEY (source .env)", file=sys.stderr)
        raise SystemExit(1)
    if _CHROMA.exists():
        shutil.rmtree(_CHROMA)
    cfg = {
        "llm": {"provider": "openai", "config": {
            "model": _LLM_MODEL, "api_key": key,
            "openai_base_url": "https://openrouter.ai/api/v1", "temperature": 0}},
        "embedder": {"provider": "huggingface", "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "chroma", "config": {
            "collection_name": "head_to_head", "path": str(_CHROMA)}},
    }
    return Memory.from_config(cfg)


def _answer(memories: list[str], question: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    mem_block = "\n".join(f"- {m}" for m in memories) or "(no memories)"
    prompt = (
        "Answer the question using ONLY these stored memories. Reply with the "
        "single specific value (a short phrase), or exactly UNKNOWN if the "
        "memories don't contain the answer. Pay attention to dates: the "
        "question may ask about a PAST point in time.\n\n"
        f"MEMORIES:\n{mem_block}\n\nQUESTION: {question}\n\nVALUE:"
    )
    r = client.chat.completions.create(
        model=_LLM_MODEL, temperature=0, max_tokens=40,
        messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip()


def _search_texts(mem, query: str, user_id: str) -> list[str]:
    res = mem.search(query, user_id=user_id, limit=20)
    rows = res.get("results", res) if isinstance(res, dict) else res
    return [r.get("memory", "") if isinstance(r, dict) else str(r) for r in (rows or [])]


def _match(answer: str, value: str) -> bool:
    return value.lower() in answer.lower()


# ── Sampling ─────────────────────────────────────────────────────────────────
def _sample_bitemporal(n_pit: int, n_retro: int, seed: int):
    from bench.bi_temporal_banking import _build_scenarios
    corpus = _build_scenarios(500)
    pit = [s for s in corpus if s.kind == "point_in_time"]
    retro = [s for s in corpus if s.kind == "retroactive_correction"]
    rng = random.Random(seed)
    return (rng.sample(pit, min(n_pit, len(pit)))
            + rng.sample(retro, min(n_retro, len(retro))),
            len(pit), len(retro))


def _sample_supersession(n: int, seed: int):
    from bench.supersession_banking import _build_scenarios
    corpus = _build_scenarios(50)
    rng = random.Random(seed)
    return rng.sample(corpus, min(n, len(corpus)))


# ── Bi-temporal: score every system on the SAME scenarios ────────────────────
def bitemporal_head_to_head(mem, scenarios) -> dict[str, Any]:
    from bench.bi_temporal import (
        _query_continuum_bitemporal,
        _query_naive_chronological,
        _query_naive_latest,
    )

    deterministic = {
        "continuum_bitemporal": _query_continuum_bitemporal,
        "naive_chronological": _query_naive_chronological,
        "naive_latest": _query_naive_latest,
    }
    tally: dict[str, dict[str, int]] = {
        name: {"pit_ok": 0, "pit_n": 0, "retro_ok": 0, "retro_n": 0}
        for name in [*deterministic, "mem0"]
    }
    samples: list[dict[str, Any]] = []

    for i, sc in enumerate(scenarios, 1):
        is_retro = sc.kind == "retroactive_correction"
        stratum = "retro" if is_retro else "pit"
        exp = sc.expected_value

        # --- deterministic systems (no LLM, no API) ---
        for name, fn in deterministic.items():
            ans = fn(list(sc.updates), sc.query_attribute, sc.as_of)
            ok = (ans is not None and exp is not None
                  and ans.lower() == exp.lower()) or (ans is None and exp is None)
            tally[name][f"{stratum}_n"] += 1
            tally[name][f"{stratum}_ok"] += bool(ok)

        # --- Mem0 (real SDK): ingest in learned-order, then retrieve + answer ---
        uid = f"bt_{uuid.uuid4().hex[:8]}"
        for u in sorted(sc.updates, key=lambda x: x.recorded_at):
            mem.add(u.text, user_id=uid)
        mems = _search_texts(mem, sc.query, uid)
        m_ans = _answer(mems, sc.query)
        m_ok = _match(m_ans, exp) if exp else ("unknown" in m_ans.lower())
        tally["mem0"][f"{stratum}_n"] += 1
        tally["mem0"][f"{stratum}_ok"] += bool(m_ok)

        if len(samples) < 6:
            samples.append({"kind": sc.kind, "query": sc.query, "expected": exp,
                            "mem0_answer": m_ans, "mem0_memories": mems[:6]})
        print(f"    bi-temporal {i}/{len(scenarios)} ({stratum})", flush=True)

    return {"tally": tally, "samples": samples}


# ── Supersession: score every system on the SAME scenarios ───────────────────
def supersession_head_to_head(mem, scenarios) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    from bench.supersession_correctness import (
        ContinuumSupersessionSystem,
        NaiveAppendSystem,
        _score,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    out: dict[str, dict[str, int]] = {}
    for name, cls in (("continuum_supersession", ContinuumSupersessionSystem),
                      ("naive_append", NaiveAppendSystem)):
        r = _score(cls(name=name, note="", embedder=embedder), list(scenarios))
        out[name] = {"ok": r.n_correct, "n": r.n_scenarios}

    ok = 0
    samples: list[dict[str, Any]] = []
    for i, sc in enumerate(scenarios, 1):
        uid = f"ss_{uuid.uuid4().hex[:8]}"
        for t in sc.turns:
            mem.add(t.text, user_id=uid)
        mems = _search_texts(mem, sc.query, uid)
        ans = _answer(mems, sc.query)
        hit = _match(ans, sc.current_answer) and not any(
            _match(ans, s) and not _match(ans, sc.current_answer)
            for s in sc.stale_answers)
        ok += bool(hit)
        if len(samples) < 6:
            samples.append({"query": sc.query, "expected": sc.current_answer,
                            "mem0_answer": ans, "mem0_memories": mems[:6]})
        print(f"    supersession {i}/{len(scenarios)}", flush=True)
    out["mem0"] = {"ok": ok, "n": len(scenarios)}
    return {"tally": out, "samples": samples}


# ── Reporting ────────────────────────────────────────────────────────────────
def _pct(ok: int, n: int) -> float:
    return round(100 * ok / n, 1) if n else 0.0


def _wilson(ok: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval (percent). Chosen over the normal approximation
    because these proportions sit at the 0 and 1 boundaries, where the normal
    interval is degenerate (it returns a zero-width interval at p=0 or p=1)."""
    if n == 0:
        return (0.0, 0.0)
    p = ok / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def _ci_str(ok: int, n: int) -> str:
    lo, hi = _wilson(ok, n)
    return f"[{lo:.0f}–{hi:.0f}]"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pit", type=int, default=20, help="point-in-time sample size")
    p.add_argument("--retro", type=int, default=20,
                   help="retroactive sample size (deliberately oversampled)")
    p.add_argument("--ss", type=int, default=25, help="supersession sample size")
    p.add_argument("--seed", type=int, default=20260804)
    a = p.parse_args(argv)

    bt_scen, corpus_pit, corpus_retro = _sample_bitemporal(a.pit, a.retro, a.seed)
    ss_scen = _sample_supersession(a.ss, a.seed)

    mem = _build_memory()
    print(f"  head-to-head on IDENTICAL scenarios — "
          f"bi-temporal n={len(bt_scen)} ({a.pit} pit + {a.retro} retro), "
          f"supersession n={len(ss_scen)}", flush=True)

    bt = bitemporal_head_to_head(mem, bt_scen)
    ss = supersession_head_to_head(mem, ss_scen)

    # Corpus weights reconstruct the true 375/125 composition from stratum rates.
    w_pit = corpus_pit / (corpus_pit + corpus_retro)
    w_retro = corpus_retro / (corpus_pit + corpus_retro)

    rows = []
    for name, t in bt["tally"].items():
        pit_p, retro_p = _pct(t["pit_ok"], t["pit_n"]), _pct(t["retro_ok"], t["retro_n"])
        rows.append({
            "system": name,
            "point_in_time": f"{t['pit_ok']}/{t['pit_n']}",
            "point_in_time_pct": pit_p,
            "point_in_time_ci95": list(_wilson(t["pit_ok"], t["pit_n"])),
            "retroactive": f"{t['retro_ok']}/{t['retro_n']}",
            "retroactive_pct": retro_p,
            "retroactive_ci95": list(_wilson(t["retro_ok"], t["retro_n"])),
            "corpus_weighted_pct": round(w_pit * pit_p + w_retro * retro_p, 1),
        })
    order = {"continuum_bitemporal": 0, "naive_chronological": 1,
             "naive_latest": 2, "mem0": 3}
    rows.sort(key=lambda r: order.get(r["system"], 9))

    print("\n" + "=" * 92)
    print("  HEAD-TO-HEAD — banking bi-temporal (same sampled scenarios, same ground truth)")
    print("=" * 92)
    print(f"  {'system':<24}{'point-in-time':>26}{'retroactive':>26}{'weighted':>12}")
    print("-" * 92)
    for r in rows:
        pit_cell = "{} {}%  {}".format(
            r["point_in_time"], r["point_in_time_pct"],
            "[{:.0f}-{:.0f}]".format(*r["point_in_time_ci95"]))
        retro_cell = "{} {}%  {}".format(
            r["retroactive"], r["retroactive_pct"],
            "[{:.0f}-{:.0f}]".format(*r["retroactive_ci95"]))
        weighted_cell = "{}%".format(r["corpus_weighted_pct"])
        print(f"  {r['system']:<24}{pit_cell:>26}{retro_cell:>26}{weighted_cell:>12}")
    print("  (bracketed = 95% Wilson score interval)")
    print("=" * 92)
    print(f"  weights: point-in-time {w_pit:.0%} · retroactive {w_retro:.0%} "
          f"(true corpus = {corpus_pit} pit / {corpus_retro} retro)")
    print("  retroactive stratum is OVERSAMPLED for power; never quote the raw sample mean.")

    print("\n" + "=" * 92)
    print("  HEAD-TO-HEAD — banking supersession (same sampled scenarios)")
    print("=" * 92)
    for name in ("continuum_supersession", "mem0", "naive_append"):
        t = ss["tally"][name]
        print(f"  {name:<24}{t['ok']}/{t['n']} = {_pct(t['ok'], t['n']):>5}%  "
              f"95% CI {_ci_str(t['ok'], t['n'])}")
    print("=" * 92)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"head_to_head_banking_{ts}.json"
    out.write_text(json.dumps({
        "benchmark": "head_to_head_banking",
        "timestamp": ts,
        "judge_model": _LLM_MODEL,
        "design": ("All systems scored on the SAME sampled scenarios. Retroactive "
                   "stratum deliberately oversampled for statistical power; "
                   "corpus_weighted_pct reconstructs the true 375/125 composition. "
                   "Do not quote the unweighted sample mean as a corpus score."),
        "sample": {"point_in_time": a.pit, "retroactive": a.retro,
                   "supersession": a.ss, "seed": a.seed,
                   "corpus_point_in_time": corpus_pit,
                   "corpus_retroactive": corpus_retro},
        "bi_temporal": rows,
        "supersession": [
            {"system": k, "correct": v["ok"], "n": v["n"], "pct": _pct(v["ok"], v["n"]),
             "ci95": list(_wilson(v["ok"], v["n"]))}
            for k, v in ss["tally"].items()
        ],
        "bi_temporal_samples": bt["samples"],
        "supersession_samples": ss["samples"],
    }, indent=2, default=str))
    print(f"\n  results: {out.relative_to(Path(__file__).resolve().parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
