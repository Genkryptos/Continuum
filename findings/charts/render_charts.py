"""
findings/charts/render_charts.py
================================
Render the three figures referenced in
``findings/longmemeval_2026-05.md`` as PNGs.

Outputs (next to this script):

* ``fig1_accuracy_vs_recall.png`` — the 32% ceiling. Scatter showing
  that going from 77% recall to 100% recall does not move accuracy.
* ``fig2_per_category.png``       — per-category accuracy, grouped
  bars across the 6 full runs.
* ``fig3_failure_split.png``      — missing_fact vs wrong_retrieval
  across runs; visualises the 90% wrong_retrieval finding.

Usage:
    cd /Users/mayanksahu/Continuum
    python3 findings/charts/render_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Headless backend — works on CI without a display.
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATASET = (
    ROOT / "evals" / "longmemeval" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
)

RUNS: list[tuple[str, str]] = [
    ("Scout-17B topk",         "results/baseline_2026-05-21.json"),
    ("gpt-4o-mini topk",       "results/gpt4omini_topk/baseline_2026-05-22.json"),
    ("gpt-4o-mini decompose",  "results/gpt4omini_decompose/baseline_2026-05-22.json"),
    ("70B topk",               "results/nvidia_70b_adaptive/baseline_2026-05-22.json"),
    ("70B long-ctx",           "results/nvidia_longctx/baseline_2026-05-22.json"),
    ("gpt-4o-mini scorer",     "results/optimizer_iter2_scorer/baseline_2026-05-22.json"),
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
    return json.loads((ROOT / path).read_text())


def _category_map() -> dict[str, str]:
    return {
        e["question_id"]: e.get("question_type", "?")
        for e in json.loads(DATASET.read_text())
    }


def fig1_accuracy_vs_recall() -> Path:
    """Scatter of (recall, accuracy) across the 6 runs."""
    _fig, ax = plt.subplots(figsize=(8, 5))
    for name, path in RUNS:
        d = _load(path)
        m = d["metrics"]
        ax.scatter(m["recall"] * 100, m["accuracy"] * 100, s=110, alpha=0.85)
        ax.annotate(
            name, (m["recall"] * 100, m["accuracy"] * 100),
            xytext=(7, 4), textcoords="offset points", fontsize=9,
        )
    # Horizontal line at the 32 % ceiling.
    ax.axhline(32.2, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate(
        "32 % ceiling — same regardless of recall",
        xy=(85, 32.5), fontsize=8, color="grey",
    )
    ax.set_xlabel("Retrieval recall (%)")
    ax.set_ylabel("Answer accuracy — substring (%)")
    ax.set_title(
        "LongMemEval-S: accuracy is flat from 77 % → 100 % recall.\n"
        "The ceiling is not retrieval."
    )
    ax.set_xlim(60, 105)
    ax.set_ylim(20, 40)
    ax.grid(True, alpha=0.3)
    out = OUT / "fig1_accuracy_vs_recall.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def fig2_per_category() -> Path:
    """Grouped bars: accuracy per category, one group per run."""
    cats_map = _category_map()
    data: dict[str, list[float]] = {name: [] for name, _ in RUNS}
    for name, path in RUNS:
        d = _load(path)
        bucket: dict[str, list[int]] = {}
        for r in d["rows"]:
            c = cats_map.get(r["question_id"], "?")
            b = bucket.setdefault(c, [0, 0])
            b[0] += 1
            b[1] += 1 if r["correct"] else 0
        for c in CATEGORIES:
            total, ok = bucket.get(c, [0, 0])
            data[name].append(ok / total * 100 if total else 0)

    x = np.arange(len(CATEGORIES))
    width = 0.13
    _fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, _) in enumerate(RUNS):
        ax.bar(x + (i - 2.5) * width, data[name], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in CATEGORIES], fontsize=9)
    ax.set_ylabel("Accuracy (substring, %)")
    ax.set_title(
        "LongMemEval-S per-category accuracy across 6 full runs.\n"
        "Catastrophic on multi-session / temporal-reasoning; uniform 0 % on "
        "preference (scorer artifact)."
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    out = OUT / "fig2_per_category.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def fig3_failure_split() -> Path:
    """Stacked bar: missing_fact vs wrong_retrieval per run."""
    names = [n for n, _ in RUNS]
    miss = []
    wrong = []
    for _, path in RUNS:
        d = _load(path)
        fb = d["metrics"]["failure_breakdown"]
        miss.append(fb.get("missing_fact", 0))
        wrong.append(fb.get("wrong_retrieval", 0))
    _fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, miss, label="missing_fact (retrieval miss)", color="#e07b91")
    ax.bar(x, wrong, bottom=miss, label="wrong_retrieval (model failed despite right ctx)",
           color="#7095c2")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Failure count (n=500 each)")
    ax.set_title(
        "Failure mix: wrong_retrieval outnumbers missing_fact ~10× in every run.\n"
        "The model fails with the right context in front of it."
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate ratios.
    for i, (m, w) in enumerate(zip(miss, wrong, strict=True)):
        total = m + w
        if total:
            ax.annotate(
                f"{w / total * 100:.0f}% wrong_retrieval",
                (i, m + w + 5), ha="center", fontsize=7, color="#3f5878",
            )
    out = OUT / "fig3_failure_split.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def main() -> None:
    print("rendering figures…")
    for fn in (fig1_accuracy_vs_recall, fig2_per_category, fig3_failure_split):
        p = fn()
        print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
