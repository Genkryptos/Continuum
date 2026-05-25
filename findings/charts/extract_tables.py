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


def headline_table() -> None:
    print("=" * 96)
    print("§2 — HEADLINE RESULTS (n=500 each)")
    print("=" * 96)
    print(
        f"{'run':<26}{'acc%':>7}{'rec%':>7}{'p50ms':>9}{'p95ms':>9}"
        f"{'avg_tok':>9}{'fail/total':>14}"
    )
    print("-" * 96)
    for name, path, _ in RUNS:
        d = _load(path)
        m = d["metrics"]
        fail = sum(m["failure_breakdown"].values())
        print(
            f"{name:<26}"
            f"{m['accuracy']*100:>6.1f} "
            f"{m['recall']*100:>6.1f} "
            f"{m['latency_p50_ms']:>8.0f} "
            f"{m['latency_p95_ms']:>8.0f} "
            f"{m['avg_tokens']:>8.0f} "
            f"{fail:>6}/{m['n_questions']}"
        )


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


def category_table() -> None:
    print()
    print("=" * 96)
    print("§5 — PER-CATEGORY ACCURACY (substring)")
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
            bucket[1] += 1 if r["correct"] else 0
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
