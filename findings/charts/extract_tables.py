"""
findings/charts/extract_tables.py
=================================
Regenerate every table in ``findings/longmemeval_2026-05.md`` from the
raw JSON files in ``results/``. Prints to stdout; no side effects.

Usage:
    cd /Users/mayanksahu/Continuum
    python3 findings/charts/extract_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = (
    ROOT / "evals" / "longmemeval" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
)

#: The 6 full LongMemEval-S runs cited in the report. Order matches §2 + §5.
RUNS: list[tuple[str, str, str]] = [
    ("Scout-17B topk",         "results/baseline_2026-05-21.json",                         "Llama-4-Scout-17B"),
    ("gpt-4o-mini topk",       "results/gpt4omini_topk/baseline_2026-05-22.json",          "gpt-4o-mini"),
    ("gpt-4o-mini decompose",  "results/gpt4omini_decompose/baseline_2026-05-22.json",     "gpt-4o-mini"),
    ("70B topk",               "results/nvidia_70b_adaptive/baseline_2026-05-22.json",     "Llama-3.3-70B"),
    ("70B long-ctx",           "results/nvidia_longctx/baseline_2026-05-22.json",          "Llama-3.3-70B"),
    ("gpt-4o-mini scorer",     "results/optimizer_iter2_scorer/baseline_2026-05-22.json",  "gpt-4o-mini"),
]

CATEGORIES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def _load(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _category_map() -> dict[str, str]:
    return {
        e["question_id"]: e.get("question_type", "?")
        for e in json.loads(DATASET.read_text())
    }


def _primary_accuracy(m: dict[str, Any]) -> tuple[float, str]:
    """
    Pick the primary accuracy figure from a metrics block.

    ``judged_accuracy`` is the promoted metric whenever the run wired in
    the LLM judge; older result files only carry ``accuracy`` and the
    legacy substring number — fall back to those.
    """
    judged = m.get("judged_accuracy")
    if judged is not None:
        return float(judged), "judge"
    # Pre-judge runs: ``accuracy`` was the substring scorer; keep it as the
    # primary so historical tables still render, but tag the lens.
    return float(m.get("accuracy", 0.0)), "substring"


def _substring_accuracy(m: dict[str, Any]) -> float:
    """Substring lens — kept as a reference-only column."""
    return float(
        m.get("substring_accuracy", m.get("accuracy", 0.0))
    )


def headline_table() -> None:
    print("=" * 104)
    print("§2 — HEADLINE RESULTS (n=500 each)")
    print("    'judged_acc%' is the primary metric (LLM judge, gpt-4o-mini).")
    print("    'substr%' is the legacy substring scorer — REFERENCE ONLY.")
    print("=" * 104)
    print(
        f"{'run':<26}{'judged_acc%':>13}{'substr% (ref)':>16}{'rec%':>7}"
        f"{'p50ms':>9}{'p95ms':>9}{'avg_tok':>9}{'fail/total':>14}"
    )
    print("-" * 104)
    for name, path, _ in RUNS:
        d = _load(path)
        m = d["metrics"]
        primary, lens = _primary_accuracy(m)
        substring = _substring_accuracy(m)
        fail = sum(m["failure_breakdown"].values())
        # When the run didn't have a judge, mark the primary column so the
        # reader doesn't mistake substring numbers for judged ones.
        primary_cell = f"{primary*100:>11.1f}"
        if lens == "substring":
            primary_cell = f"{primary*100:>9.1f}*"
        print(
            f"{name:<26}"
            f"{primary_cell:>13} "
            f"{substring*100:>13.1f}  "
            f"{m['recall']*100:>6.1f} "
            f"{m['latency_p50_ms']:>8.0f} "
            f"{m['latency_p95_ms']:>8.0f} "
            f"{m['avg_tokens']:>8.0f} "
            f"{fail:>6}/{m['n_questions']}"
        )
    print()
    print("  * = substring scorer used as primary (judge not run for this row).")


def failure_table() -> None:
    print()
    print("=" * 96)
    print("§4 — FAILURE-MODE BREAKDOWN")
    print("=" * 96)
    print(f"{'run':<26}{'missing_fact':>15}{'wrong_retrieval':>18}{'llm_error':>12}{'other':>8}")
    print("-" * 96)
    for name, path, _ in RUNS:
        d = _load(path)
        fb = d["metrics"]["failure_breakdown"]
        print(
            f"{name:<26}"
            f"{fb.get('missing_fact', 0):>15}"
            f"{fb.get('wrong_retrieval', 0):>18}"
            f"{fb.get('llm_error', 0):>12}"
            f"{fb.get('other', 0):>8}"
        )


def _row_primary_correct(r: dict[str, Any]) -> bool:
    """
    Resolve the primary verdict for a single row.

    Prefers the judge verdict when present, otherwise falls back to the
    row-level ``correct`` flag (which itself mirrors the judge inside the
    runner — this fallback is only hit on legacy result files where the
    judge column was never written).
    """
    judge = r.get("judge_correct")
    if judge is not None:
        return bool(judge)
    return bool(r.get("correct", False))


def category_table() -> None:
    print()
    print("=" * 96)
    print("§5 — PER-CATEGORY ACCURACY (LLM judge primary; substring = reference only)")
    print("=" * 96)
    cats = _category_map()
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for name, path, _ in RUNS:
        d = _load(path)
        cat: dict[str, list[int]] = {}
        for r in d["rows"]:
            c = cats.get(r["question_id"], "?")
            bucket = cat.setdefault(c, [0, 0])
            bucket[0] += 1
            bucket[1] += 1 if _row_primary_correct(r) else 0
        table[name] = {c: (cat.get(c, [0, 0])[0], cat.get(c, [0, 0])[1]) for c in CATEGORIES}
    header = f"{'category':<28}" + "".join(f"{name[:15]:>16}" for name, _, _ in RUNS)
    print(header)
    print("-" * len(header))
    for c in CATEGORIES:
        # one of the runs (any) gives us n per category
        first = next(iter(table.values()))
        n = first[c][0]
        row = f"{c:<24}(n={n:>3})"
        for name, _, _ in RUNS:
            total, ok = table[name][c]
            row += f"{ok / total * 100:>15.1f}%"
        print(row)


def main() -> None:
    headline_table()
    failure_table()
    category_table()


if __name__ == "__main__":
    main()
