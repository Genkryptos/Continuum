"""
evals/longmemeval/rescore_with_judge.py
========================================
Re-grade an existing LongMemEval result file with an LLM judge so we
can separate "model was actually wrong" from "substring scorer rejected
a correctly-paraphrased answer".

Usage
-----
.. code-block:: bash

    GROQ_API_KEY=... \\
    python -m evals.longmemeval.rescore_with_judge \\
        --input  results/optimizer_iteration_1.json \\
        --output results/optimizer_iteration_3_judged.json \\
        --judge-model llama-3.1-8b-instant \\
        --rpm 28

Pipeline (per row from the input results file)
----------------------------------------------
1. Look up the original question text via ``question_id`` in the
   LongMemEval dataset (results files don't carry the question).
2. Call :class:`LLMJudgeScorer.is_correct(question, expected, actual)`
   on the row's stored answer.
3. Recompute ``correct`` / ``failure_category`` and write a new results
   file alongside the old one.

The script doesn't re-run the underlying LLM — only the judge. That's
the whole point: cheap, fast, and isolates the scoring lens.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evals.longmemeval.baseline import (
    BaselineResults,
    RowResult,
    _classify_failure,
    _failure_breakdown,
)
from evals.longmemeval.bootstrap_ollama import GroqLLM
from evals.longmemeval.judge import LLMJudgeScorer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_questions(dataset_path: Path) -> dict[str, str]:
    """Return a ``question_id → question`` map from the LongMemEval JSON."""
    raw = json.loads(dataset_path.read_text())
    return {r["question_id"]: r["question"] for r in raw}


def _row_from_dict(d: dict[str, Any]) -> RowResult:
    """Rehydrate a RowResult, tolerating missing optional fields."""
    field_names = {f.name for f in dataclasses.fields(RowResult)}
    return RowResult(**{k: v for k, v in d.items() if k in field_names})


async def _rescore(
    *,
    input_path: Path,
    dataset_path: Path,
    judge: LLMJudgeScorer,
    limit: int | None,
) -> tuple[BaselineResults, dict[str, int]]:
    payload = json.loads(input_path.read_text())
    rows = [_row_from_dict(r) for r in payload["rows"]]
    if limit:
        rows = rows[:limit]
    questions = _load_questions(dataset_path)

    flipped = {"false_to_true": 0, "true_to_false": 0, "unchanged": 0}
    new_rows: list[RowResult] = []
    for idx, r in enumerate(rows):
        question = questions.get(r.question_id, "")
        if not question:
            log.warning("no question text for %s — keeping original verdict",
                        r.question_id)
            new_rows.append(r)
            continue
        try:
            verdict = await judge.is_correct(
                question, r.expected_answer, r.answer
            )
        except Exception:
            log.exception("judge call failed for %s", r.question_id)
            new_rows.append(r)
            continue
        # Track flips for reporting.
        if verdict == r.correct:
            flipped["unchanged"] += 1
        elif verdict and not r.correct:
            flipped["false_to_true"] += 1
        else:
            flipped["true_to_false"] += 1
        # Rebuild the row with the new verdict + recomputed failure class.
        new_category = (
            None if verdict
            else _classify_failure(
                answer=r.answer, error=r.error,
                retrieved=r.retrieved_session_ids,
                expected=r.expected_session_ids,
            )
        )
        new_rows.append(dataclasses.replace(
            r, correct=verdict, failure_category=new_category,
        ))
        if (idx + 1) % 25 == 0:
            log.info(
                "judged %d/%d  (flipped: +%d −%d unchanged=%d)",
                idx + 1, len(rows),
                flipped["false_to_true"], flipped["true_to_false"],
                flipped["unchanged"],
            )

    results = BaselineResults(
        dataset=payload.get("dataset", "longmemeval-s"),
        answerer=payload.get("answerer", "unknown"),
        rows=new_rows,
        started_at=payload.get("started_at", ""),
        finished_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    return results, flipped


def _print_summary(
    *,
    before: dict[str, Any],
    after: BaselineResults,
    flipped: dict[str, int],
) -> None:
    # Apples-to-apples comparison: substring vs LLM-judge ON THE SAME
    # rows that were re-graded. Computed from the input file's per-row
    # records (which carry the original verdict) intersected with the
    # judged set by question_id.
    judged_ids = {r.question_id for r in after.rows}
    judged_before_rows = [
        r for r in before["rows"] if r["question_id"] in judged_ids
    ]
    n = len(after.rows) or 1
    b_correct = sum(1 for r in judged_before_rows if r["correct"])
    b_acc = b_correct / n
    a_acc = after.accuracy

    full_before_acc = float(before["metrics"]["accuracy"])
    full_before_n = int(before["metrics"]["n_questions"])

    b_bd: dict[str, int] = {}
    for r in judged_before_rows:
        cat = r.get("failure_category")
        if cat:
            b_bd[cat] = b_bd.get(cat, 0) + 1
    a_bd = _failure_breakdown(after.rows)

    print()
    print("=" * 70)
    print("LLM-judge re-scoring  — before vs after")
    print("=" * 70)
    print(f"  rows judged:           {len(after.rows)} of {full_before_n} "
          f"({len(after.rows)/full_before_n:.0%})")
    print(f"  flipped False → True:  {flipped['false_to_true']}")
    print(f"  flipped True → False:  {flipped['true_to_false']}")
    print(f"  unchanged:             {flipped['unchanged']}")
    print()
    print("  apples-to-apples on the same judged rows:")
    print(f"    accuracy (substring): {b_correct}/{n}  {b_acc:.1%}")
    print(f"    accuracy (LLM judge): {sum(1 for r in after.rows if r.correct)}"
          f"/{n}  {a_acc:.1%}")
    print(f"    delta:                {(a_acc - b_acc)*100:+.1f} pp")
    print()
    print("  for reference (full benchmark substring acc): "
          f"{full_before_acc:.1%}  ({full_before_n} rows)")
    print()
    print("  failure breakdown (on judged rows):")
    print(f"    {'category':<18} {'before':>10} {'after':>10}")
    for cat in ("missing_fact", "wrong_retrieval", "llm_error", "other"):
        print(f"    {cat:<18} {b_bd.get(cat, 0):>10} {a_bd.get(cat, 0):>10}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-grade an existing LongMemEval results file with an LLM judge.",
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Path to the existing results JSON.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the LLM-judged results JSON.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "LongMemEval" / "data"
                                       / "longmemeval_s_cleaned.json",
        help="Path to the LongMemEval source JSON (for question lookup).",
    )
    p.add_argument(
        "--provider", default="groq",
        choices=["groq", "openai", "openrouter"],
        help=(
            "Which LLM provider hosts the judge. 'groq' needs "
            "GROQ_API_KEY; 'openai' needs OPENAI_API_KEY; 'openrouter' "
            "needs OPENROUTER_API_KEY. Pick the one whose key you've "
            "already got loaded."
        ),
    )
    p.add_argument(
        "--judge-model", default=None,
        help=(
            "Model used for judging. Defaults: 'llama-3.1-8b-instant' "
            "on groq, 'gpt-4o-mini' on openai, 'openai/gpt-oss-120b' on "
            "openrouter. Cheap is fine — the judge just compares strings "
            "semantically."
        ),
    )
    p.add_argument("--rpm", type=int, default=28,
                   help="Client-side RPM cap for judge calls.")
    p.add_argument(
        "--judge-max-tokens", type=int, default=8,
        help=(
            "Token cap on the judge's yes/no reply. 8 is plenty for a "
            "NON-reasoning model. REASONING judges (gpt-oss-120b, "
            "deepseek-r1, o-series) burn this thinking and emit no "
            "verdict — the parser then falls back to 'wrong' and marks "
            "EVERYTHING incorrect. Set 1024-2048 if you must judge with "
            "a reasoning model; better, judge with a non-reasoning one."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Re-grade only the first N rows (handy for validation).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.input.exists():
        raise SystemExit(f"input file not found: {args.input}")

    judge_model = args.judge_model or {
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-oss-120b",
        "groq": "llama-3.1-8b-instant",
    }.get(args.provider, "llama-3.1-8b-instant")
    log.info(
        "loading judge provider=%s model=%s @ %d RPM",
        args.provider, judge_model, args.rpm,
    )
    if args.provider == "openai":
        from evals.longmemeval.bootstrap_ollama import OpenAILLM
        llm = OpenAILLM(model=judge_model, rpm=args.rpm)
    elif args.provider == "openrouter":
        from evals.longmemeval.bootstrap_ollama import OpenRouterLLM
        llm = OpenRouterLLM(model=judge_model, rpm=args.rpm)
    else:
        llm = GroqLLM(model=judge_model, rpm=args.rpm)
    judge = LLMJudgeScorer(llm=llm, max_tokens=args.judge_max_tokens)

    log.info("re-scoring %s …", args.input)
    results, flipped = await _rescore(
        input_path=args.input,
        dataset_path=args.dataset,
        judge=judge,
        limit=args.limit,
    )

    before = json.loads(args.input.read_text())
    _print_summary(before=before, after=results, flipped=flipped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results.to_dict(), indent=2, default=str))
    log.info("wrote %s", args.output)

    await llm.aclose()
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    return asyncio.run(main_async(_parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "main_async"]
