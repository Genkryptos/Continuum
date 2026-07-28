"""
bench/mem0_compare.py
=====================
Run the **real Mem0 SDK** on Continuum's own supersession and bi-temporal
scenarios — the validation gate for the paper's central claim.

The existing `bench/bi_temporal.py` stubs Mem0 out ("framework not applicable").
That is an assumption, not a measurement. This harness makes it falsifiable:
it ingests each scenario's natural-language statements into a real, fully-local
Mem0 instance (HuggingFace embedder + Chroma vector store; only the LLM is
remote), then answers each query the way Mem0 is meant to be used — retrieve
memories, let an LLM answer over them — and scores against the same ground truth.

The prediction (Related Work / §3.2.1): Mem0 has no valid-/transaction-time axis
and resolves contradictions *destructively* (an UPDATE deletes the superseded
fact), so it should fail **point-in-time** queries (the historical value is gone)
even where it handles simple "latest value" supersession. This script measures
whether that holds.

    set -a && source .env && set +a
    python3.12 -m bench.mem0_compare --limit 3          # quick pipeline check
    python3.12 -m bench.mem0_compare                    # full 50 + 20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_CHROMA = Path("/private/tmp/claude-501/mem0_compare_chroma")
_JUDGE_MODEL = "openai/gpt-4o-mini"


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
            "model": _JUDGE_MODEL, "api_key": key,
            "openai_base_url": "https://openrouter.ai/api/v1", "temperature": 0}},
        "embedder": {"provider": "huggingface", "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "chroma", "config": {
            "collection_name": "mem0_compare", "path": str(_CHROMA)}},
    }
    return Memory.from_config(cfg)


def _answer(memories: list[str], question: str) -> str:
    """Answer `question` from Mem0's retrieved memories — the intended usage."""
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    mem_block = "\n".join(f"- {m}" for m in memories) or "(no memories)"
    prompt = (
        "Answer the question using ONLY these stored memories. Reply with the "
        "single specific value (a short phrase — a place, employer, status, etc.), "
        "or exactly UNKNOWN if the memories don't contain the answer. Pay attention "
        "to dates: the question may ask about a PAST point in time.\n\n"
        f"MEMORIES:\n{mem_block}\n\nQUESTION: {question}\n\nVALUE:"
    )
    r = client.chat.completions.create(
        model=_JUDGE_MODEL, temperature=0, max_tokens=40,
        messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip()


def _search_texts(mem, query: str, user_id: str) -> list[str]:
    res = mem.search(query, user_id=user_id, limit=20)
    rows = res.get("results", res) if isinstance(res, dict) else res
    return [r.get("memory", "") if isinstance(r, dict) else str(r) for r in (rows or [])]


def _match(answer: str, value: str) -> bool:
    return value.lower() in answer.lower()


# ── supersession ────────────────────────────────────────────────────────────
def run_supersession(mem, n: int, limit: int | None) -> dict[str, Any]:
    from bench.supersession_correctness import _build_scenarios
    scenarios = _build_scenarios(n)
    if limit:
        scenarios = scenarios[:limit]
    ok = 0
    samples = []
    for sc in scenarios:
        uid = f"ss_{uuid.uuid4().hex[:8]}"
        for t in sc.turns:
            mem.add(t.text, user_id=uid)
        mems = _search_texts(mem, sc.query, uid)
        ans = _answer(mems, sc.query)
        hit = _match(ans, sc.current_answer) and not any(
            _match(ans, s) and not _match(ans, sc.current_answer) for s in sc.stale_answers)
        ok += hit
        if len(samples) < 4:
            samples.append({"query": sc.query, "expected": sc.current_answer,
                            "mem0_answer": ans, "mem0_memories": mems})
    total = len(scenarios)
    return {"benchmark": "supersession", "n": total, "correct": ok,
            "pct": round(100 * ok / total, 1) if total else 0.0, "samples": samples}


# ── bi-temporal ─────────────────────────────────────────────────────────────
def run_bitemporal(mem, limit: int | None) -> dict[str, Any]:
    from bench.bi_temporal import _build_scenarios
    scenarios = _build_scenarios()
    if limit:
        scenarios = [s for s in scenarios if s.kind == "point_in_time"][:limit] + \
                    [s for s in scenarios if s.kind == "retroactive_correction"][:limit]
    pit_ok = pit_tot = retro_ok = retro_tot = 0
    samples = []
    for sc in scenarios:
        uid = f"bt_{uuid.uuid4().hex[:8]}"
        for u in sorted(sc.updates, key=lambda x: x.recorded_at):  # learned-order
            mem.add(u.text, user_id=uid)
        mems = _search_texts(mem, sc.query, uid)
        ans = _answer(mems, sc.query)
        exp = sc.expected_value
        correct = (_match(ans, exp) if exp else ("unknown" in ans.lower()))
        is_retro = sc.kind == "retroactive_correction"
        if is_retro:
            retro_tot += 1; retro_ok += correct
        else:
            pit_tot += 1; pit_ok += correct
        if len(samples) < 4:
            samples.append({"kind": sc.kind, "query": sc.query, "expected": exp,
                            "mem0_answer": ans, "mem0_memories": mems})
    tot = pit_tot + retro_tot
    return {"benchmark": "bi_temporal", "n": tot, "correct": pit_ok + retro_ok,
            "pct": round(100 * (pit_ok + retro_ok) / tot, 1) if tot else 0.0,
            "point_in_time": f"{pit_ok}/{pit_tot}",
            "retroactive": f"{retro_ok}/{retro_tot}", "samples": samples}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=int, default=50, help="supersession count")
    p.add_argument("--limit", type=int, default=None, help="cap per set (quick check)")
    a = p.parse_args(argv)

    mem = _build_memory()
    print(f"  running Mem0 (real SDK) on supersession + bi_temporal "
          f"(limit={a.limit}) …", flush=True)
    ss = run_supersession(mem, a.scenarios, a.limit)
    bt = run_bitemporal(mem, a.limit)

    print("\n" + "=" * 78)
    print("  Mem0 (real SDK, local) vs Continuum — same scenarios, same ground truth")
    print("=" * 78)
    print(f"  {'benchmark':<22}{'Mem0':>16}{'Continuum (known)':>26}")
    print("-" * 78)
    print(f"  {'supersession (recall)':<22}{ss['correct']}/{ss['n']} = {ss['pct']:>5}%"
          f"{'current() 100% · recall 20%':>26}")
    print(f"  {'bi_temporal overall':<22}{bt['correct']}/{bt['n']} = {bt['pct']:>5}%"
          f"{'100% (20/20)':>26}")
    print(f"  {'  point_in_time':<22}{bt['point_in_time']:>16}{'15/15 (100%)':>26}")
    print(f"  {'  retroactive':<22}{bt['retroactive']:>16}{'5/5 (100%)':>26}")
    print("=" * 78)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"mem0_compare_{ts}.json"
    out.write_text(json.dumps({"timestamp": ts, "judge_model": _JUDGE_MODEL,
                               "supersession": ss, "bi_temporal": bt}, indent=2, default=str))
    print(f"\n  results: {out.relative_to(Path(__file__).resolve().parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
