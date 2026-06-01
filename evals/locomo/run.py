"""
evals/locomo/run.py
===================
LOCOMO benchmark runner with a Continuum-vs-Mem0 head-to-head.

Same dataset, same answerer model, same LLM judge for both systems —
only the memory layer differs (see the two adapters). Writes
``results/locomo_v1/continuum.json`` and ``results/locomo_v1/mem0.json``
each with overall + per-category judged accuracy, p50/p95 latency, and
(continuum only) evidence recall.

Usage
-----
.. code-block:: bash

    # download the dataset once
    mkdir -p evals/locomo/data
    curl -L -o evals/locomo/data/locomo10.json \\
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

    # 50-question smoke, continuum side only
    OPENROUTER_API_KEY=… python3.12 -m evals.locomo.run \\
        --system continuum --provider openrouter --model openai/gpt-oss-120b \\
        --judge-provider groq --judge-model llama-3.3-70b-versatile \\
        --limit 50 --output results/locomo_smoke

    # full head-to-head
    OPENROUTER_API_KEY=… python3.12 -m evals.locomo.run \\
        --system both --provider openrouter --model openai/gpt-oss-120b \\
        --judge-provider groq --judge-model llama-3.3-70b-versatile \\
        --output results/locomo_v1

The CLI is a thin bridge onto :func:`run_locomo`, which tests drive
directly with fakes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.locomo.loader import LocomoQuestion, load_locomo

log = logging.getLogger(__name__)


# ── LLM construction (answerer + judge) ─────────────────────────────────────


def _build_llm(provider: str, model: str | None, rpm: int) -> Any:
    """Build an answerer/judge LLM by provider, reusing bootstrap clients."""
    from evals.longmemeval import bootstrap_ollama as boot

    if provider == "openrouter":
        return boot.OpenRouterLLM(model=model or boot.DEFAULT_OPENROUTER_MODEL, rpm=rpm)
    if provider == "groq":
        return boot.GroqLLM(model=model or boot.DEFAULT_GROQ_MODEL, rpm=rpm)
    if provider == "openai":
        return boot.OpenAILLM(model=model or boot.DEFAULT_OPENAI_MODEL, rpm=rpm)
    raise ValueError(f"unsupported provider for LOCOMO: {provider!r}")


# ── scoring / aggregation ───────────────────────────────────────────────────


def _recall(retrieved_dia_ids: list[str], evidence: list[str]) -> float | None:
    """Fraction of evidence dia_ids that were retrieved. None when no evidence."""
    if not evidence:
        return None
    got = set(retrieved_dia_ids)
    hit = sum(1 for e in evidence if e in got)
    return hit / len(evidence)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[min(len(s) - 1, round((len(s) - 1) * q))])


def _summarise(system: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category_label"]].append(r)

    def bucket(rs: list[dict]) -> dict[str, Any]:
        n = len(rs)
        correct = sum(1 for r in rs if r["correct"])
        lat = [r["latency_ms"] for r in rs if r.get("latency_ms") is not None]
        recs = [r["recall"] for r in rs if r.get("recall") is not None]
        return {
            "n": n,
            "judged_acc": correct / n if n else 0.0,
            "recall": (statistics.fmean(recs) if recs else None),
            "p50_ms": _percentile(lat, 0.50),
            "p95_ms": _percentile(lat, 0.95),
            "avg_llm_calls": (statistics.fmean([r.get("llm_calls", 0) for r in rs]) if rs else 0.0),
        }

    return {
        "system": system,
        "overall": bucket(rows),
        "per_category": {cat: bucket(rs) for cat, rs in sorted(by_cat.items())},
        "n_rows": len(rows),
    }


def _render(summary: dict[str, Any]) -> str:
    head = f"{'category':<16}{'n':>6}{'judged%':>10}{'recall%':>9}{'p50ms':>9}{'p95ms':>9}"
    lines = [f"LOCOMO — {summary['system']}", "=" * len(head), head, "-" * len(head)]

    def row(label: str, s: dict[str, Any]) -> str:
        rec = f"{s['recall'] * 100:>8.1f}" if s.get("recall") is not None else f"{'n/a':>8}"
        return (
            f"{label[:16]:<16}{s['n']:>6}{s['judged_acc'] * 100:>9.1f} "
            f"{rec} {s['p50_ms']:>8.0f} {s['p95_ms']:>8.0f}"
        )

    lines.append(row("OVERALL", summary["overall"]))
    for cat, s in summary["per_category"].items():
        lines.append(row(cat, s))
    return "\n".join(lines)


# ── per-system run ──────────────────────────────────────────────────────────


async def _run_system(
    system: str,
    samples: list[tuple[Any, list[LocomoQuestion]]],
    *,
    answerer: Any,
    judge: Any,
    limit: int | None,
    embedder: Any | None,
) -> list[dict[str, Any]]:
    from evals.locomo.continuum_adapter import ContinuumLocomoAnswerer
    from evals.locomo.mem0_adapter import Mem0LocomoAnswerer

    rows: list[dict[str, Any]] = []
    asked = 0
    for conversation, questions in samples:
        if not questions:
            continue
        if system == "continuum":
            ans: Any = ContinuumLocomoAnswerer(
                conversation,
                llm=answerer,
                embedder=embedder,
            )
        else:
            ans = Mem0LocomoAnswerer(conversation, llm=answerer)

        for q in questions:
            if limit is not None and asked >= limit:
                return rows
            asked += 1
            try:
                result = await ans.answer(q.question, category=q.category)
            except Exception as exc:
                log.exception("answer failed (%s) for %s", system, q.question[:60])
                result = {"answer": "", "latency_ms": None, "llm_calls": 0, "retrieved_dia_ids": []}
                result["error"] = repr(exc)
            answer = result.get("answer", "")
            try:
                correct = bool(answer) and await judge.is_correct(q.question, q.answer, answer)
            except Exception:
                log.exception("judge failed for %s", q.question[:60])
                correct = False
            rows.append(
                {
                    "sample_id": q.sample_id,
                    "question": q.question,
                    "expected": q.answer,
                    "answer": answer,
                    "category": q.category,
                    "category_label": q.category_label,
                    "correct": correct,
                    "latency_ms": result.get("latency_ms"),
                    "llm_calls": result.get("llm_calls", 0),
                    # dia_id recall only applies to systems that retrieve raw
                    # turns (continuum). Mem0 retrieves distilled facts with no
                    # dia_id mapping → recall is N/A (None), not 0.
                    "recall": (
                        _recall(result["retrieved_dia_ids"], q.evidence)
                        if "retrieved_dia_ids" in result
                        else None
                    ),
                    "error": result.get("error"),
                }
            )
            if asked % 25 == 0:
                ok = sum(1 for r in rows if r["correct"])
                log.info(
                    "%s: %d asked, %d correct (%.1f%%)", system, asked, ok, 100 * ok / len(rows)
                )
    return rows


async def run_locomo(
    *,
    dataset: Path,
    systems: list[str],
    answerer_factory,
    judge,
    output_dir: Path,
    limit: int | None = None,
    embedder: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Drive the LOCOMO benchmark for the requested systems. Returns
    ``{system: summary}``. Library-first; the CLI wraps this.
    """
    samples = load_locomo(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for system in systems:
        log.info("=== running system: %s ===", system)
        answerer = answerer_factory()
        rows = await _run_system(
            system,
            samples,
            answerer=answerer,
            judge=judge,
            limit=limit,
            embedder=embedder,
        )
        summary = _summarise(system, rows)
        summary["rows"] = rows
        out_path = output_dir / f"{system}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(_render(summary))
        print(f"  → {out_path}\n")
        summaries[system] = summary
    _print_delta(summaries)
    return summaries


def _print_delta(summaries: dict[str, dict[str, Any]]) -> None:
    if "continuum" in summaries and "mem0" in summaries:
        c = summaries["continuum"]["overall"]["judged_acc"] * 100
        m = summaries["mem0"]["overall"]["judged_acc"] * 100
        print("=" * 50)
        print(f"HEAD-TO-HEAD  continuum {c:.1f}%  vs  mem0 {m:.1f}%  (delta {c - m:+.1f}pp)")
        print("=" * 50)


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOCOMO benchmark — Continuum vs Mem0.")
    p.add_argument("--dataset", type=Path, default=Path("evals/locomo/data/locomo10.json"))
    p.add_argument("--system", choices=("continuum", "mem0", "both"), default="both")
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--model", default=None)
    p.add_argument("--rpm", type=int, default=60)
    p.add_argument("--judge-provider", default="groq")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-max-tokens", type=int, default=8)
    p.add_argument("--judge-rpm", type=int, default=28)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", type=Path, default=Path("results/locomo_v1"))
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    from evals.longmemeval.bootstrap_ollama import _Embedder
    from evals.longmemeval.judge import LLMJudgeScorer

    systems = ["continuum", "mem0"] if args.system == "both" else [args.system]
    judge_llm = _build_llm(args.judge_provider, args.judge_model, args.judge_rpm)
    judge = LLMJudgeScorer(llm=judge_llm, max_tokens=args.judge_max_tokens)
    embedder = _Embedder()

    def answerer_factory() -> Any:
        return _build_llm(args.provider, args.model, args.rpm)

    asyncio.run(
        run_locomo(
            dataset=args.dataset,
            systems=systems,
            answerer_factory=answerer_factory,
            judge=judge,
            output_dir=args.output,
            limit=args.limit,
            embedder=embedder,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())


__all__ = ["run_locomo"]
