"""
findings/charts/budget_curve.py
===============================
Render the cost/accuracy curve from a budget ablation
(``docs/LOW_BUDGET_PLAN.md``, Phase 1).

Reads one or more ``results/budget_<chars>/`` directories, each holding
``judged.json`` (accuracy — ``metrics.accuracy`` is the *judged* number,
since ``rescore_with_judge`` overwrites ``correct`` in place) and
``budget_report.json`` (the measured token axis). Prints:

* the aggregate curve — judged accuracy vs measured context tokens, with
  the delta from the highest-budget point, which is the claim under test;
* the same curve per ``question_type`` — the router's design input, and
  the only view that says *which* questions are paying for the budget;
* a validity column, because a budget that never bound is not a point on
  the curve at all.

Usage::

    python findings/charts/budget_curve.py results/budget_*/
    python findings/charts/budget_curve.py results/budget_*/ --csv curve.csv

The chart itself is optional (``--png``); the table is the deliverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

#: Categories in the order they appear in the paper's tables — strongest
#: to weakest on the v2.0 run, so the reader sees the saturated ones
#: (which should be cheap to serve) before the hard ones.
_CATEGORY_ORDER = [
    "single-session-assistant",
    "single-session-user",
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-preference",
]


class Point:
    """One budget point: config, accuracy, and the measured token axis."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.judged = _load(directory / "judged.json")
        self.report = _load(directory / "budget_report.json")

        if self.judged is None and self.report is None:
            raise FileNotFoundError(
                f"{directory}: neither judged.json nor budget_report.json"
            )

        cfg = (self.report or {}).get("config", {})
        self.max_chars: int = int(
            cfg.get("max_context_chars") or _chars_from_name(directory)
        )
        self.max_tokens: int = int(cfg.get("max_answer_prompt_tokens") or 0)

        summary = (self.report or {}).get("summary", {})
        self.avg_tokens: float = float(summary.get("avg_context_tokens", 0.0))
        self.p95_tokens: float = float(summary.get("context_tokens_p95", 0.0))
        self.chars_per_token: float = float(
            summary.get("measured_chars_per_token", 0.0)
        )
        self.pct_bound: float = float(summary.get("pct_rows_budget_bound", 0.0))
        self.tokens_measured: bool = bool(summary.get("tokens_measured", True))
        self.unpriced: list[str] = list(summary.get("unpriced_models") or [])

        metrics = (self.judged or self.report or {}).get("metrics", {})
        if self.judged is not None:
            # rescore_with_judge rewrites `correct`, so `accuracy` here is
            # the judged number. `judged_accuracy` stays None in this
            # pipeline — do not read it.
            self.accuracy: float | None = float(metrics.get("accuracy", 0.0))
            self.n: int = int(metrics.get("n_questions", 0))
            self.cost: float = float(metrics.get("total_cost_usd", 0.0))
            self.by_type: dict[str, Any] = dict(metrics.get("by_question_type", {}))
            self.judged_available = True
        else:
            s = (self.report or {}).get("summary", {})
            self.accuracy = float(s.get("accuracy", 0.0))
            self.n = int(s.get("n_questions", 0))
            self.cost = float(s.get("total_cost_usd", 0.0))
            self.by_type = dict((self.report or {}).get("by_question_type", {}))
            self.judged_available = False


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"WARNING: {path} is not valid JSON ({exc}) — skipping", file=sys.stderr)
        return None


def _chars_from_name(directory: Path) -> int:
    """Fall back to the trailing integer in ``budget_<chars>``."""
    tail = directory.name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


#: Tolerance for the "within target" comparison. A point exactly on the
#: 2pp line computes as -2.0000000000000018 in float and would otherwise
#: be reported as a break — the wrong call on the one point most likely
#: to land there.
_EPS = 1e-9


def _fmt_pp(delta: float) -> str:
    return f"{delta:+.1f}pp" if abs(delta) >= 0.05 else "  0.0pp"


def _holds(point: Point, anchor: Point, target_pp: float) -> bool:
    """True when ``point`` is within ``target_pp`` of the anchor AND real."""
    if point.accuracy is None or point.pct_bound < 1.0:
        return False
    return (point.accuracy - anchor.accuracy) * 100 >= -target_pp - _EPS


def render(points: list[Point], *, target_pp: float = 2.0) -> None:
    points.sort(key=lambda p: -p.max_chars)
    anchor = points[0]
    n = anchor.n

    print()
    print("=" * 84)
    print(f"  BUDGET ABLATION — cost/accuracy curve   (n={n} per point)")
    print("=" * 84)

    if not all(p.judged_available for p in points):
        missing = [p.dir.name for p in points if not p.judged_available]
        print(f"  NOTE: unjudged points fall back to substring accuracy: {missing}")
        print("        Substring reports 0% on single-session-preference — the")
        print("        curve is not readable until those points are judged.")
    if any(not p.tokens_measured for p in points):
        print("  WARNING: some points used chars/3.9 ESTIMATES, not tiktoken.")
    unpriced = sorted({m for p in points for m in p.unpriced})
    if unpriced:
        print(f"  WARNING: unpriced models (cost understated): {', '.join(unpriced)}")

    print()
    print(f"  {'budget':>8}  {'avg tok':>8}  {'p95 tok':>8}  {'acc':>7}  "
          f"{'vs anchor':>10}  {'bound':>7}  {'$':>7}  verdict")
    print("  " + "-" * 80)
    for p in points:
        delta = (p.accuracy - anchor.accuracy) * 100 if p.accuracy is not None else 0.0
        if p is anchor:
            verdict = "anchor"
        elif p.pct_bound < 1.0:
            # Not a real point: the retriever never offered enough context
            # to fill the budget, so the accuracy delta is run noise.
            verdict = "NOT BINDING"
        elif _holds(p, anchor, target_pp):
            verdict = f"within {target_pp:.0f}pp  <-- target"
        else:
            verdict = "breaks"
        print(
            f"  {p.max_chars:>8,}  {p.avg_tokens:>8,.0f}  {p.p95_tokens:>8,.0f}  "
            f"{(p.accuracy or 0) * 100:>6.1f}%  {_fmt_pp(delta):>10}  "
            f"{p.pct_bound:>6.0f}%  {p.cost:>7.3f}  {verdict}"
        )

    ratios = [p.chars_per_token for p in points if p.chars_per_token]
    if ratios:
        print()
        print(f"  measured chars/token: {min(ratios):.2f}-{max(ratios):.2f} "
              f"(the assumed 4.0 was never used)")

    # ── The headline ────────────────────────────────────────────────
    holding = [p for p in points[1:] if _holds(p, anchor, target_pp)]
    print()
    if holding and anchor.avg_tokens:
        best = min(holding, key=lambda p: p.avg_tokens)
        factor = anchor.avg_tokens / best.avg_tokens if best.avg_tokens else 0.0
        print(f"  HEADLINE: {(best.accuracy or 0) * 100:.1f}% at "
              f"{best.avg_tokens:,.0f} tokens vs {(anchor.accuracy or 0) * 100:.1f}% "
              f"at {anchor.avg_tokens:,.0f} — "
              f"{_fmt_pp((best.accuracy - anchor.accuracy) * 100).strip()} "
              f"at {factor:.1f}x fewer tokens.")
    else:
        print(f"  HEADLINE: no budget below the anchor holds within "
              f"{target_pp:.0f}pp. Report the curve and where it breaks — "
              f"that is still the finding.")

    _render_by_category(points, anchor)


def _render_by_category(points: list[Point], anchor: Point) -> None:
    """Per-category accuracy and context cost — the router's design input."""
    cats = [c for c in _CATEGORY_ORDER if c in anchor.by_type]
    cats += [c for c in sorted(anchor.by_type) if c not in _CATEGORY_ORDER]
    if not cats:
        return

    print()
    print("=" * 84)
    print("  PER CATEGORY — accuracy (delta vs anchor) and avg context tokens")
    print("=" * 84)
    header = f"  {'category':<27} {'n':>4}"
    for p in points:
        header += f"  {p.max_chars // 1000:>5}k"
    print(header + "   avg tok @anchor")
    print("  " + "-" * 80)

    for cat in cats:
        base = anchor.by_type.get(cat, {})
        row = f"  {cat:<27} {int(base.get('n_questions', 0)):>4}"
        base_acc = float(base.get("accuracy", 0.0)) * 100
        for p in points:
            cell = p.by_type.get(cat)
            if not cell:
                row += f"  {'--':>6}"
                continue
            acc = float(cell.get("accuracy", 0.0)) * 100
            row += f"  {acc:>5.0f}%" if p is anchor else f"  {acc - base_acc:>+5.0f}"
        row += f"   {float(base.get('avg_context_tokens', 0.0)):>10,.0f}"
        print(row)
    print()
    print("  Anchor column is absolute accuracy; the rest are pp deltas from it.")
    print("  Categories that stay flat as the budget falls are SMALL-class")
    print("  candidates for the Phase-2 router; the ones that break are LARGE.")


def write_csv(points: list[Point], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "max_context_chars", "avg_context_tokens", "p95_context_tokens",
            "accuracy", "n_questions", "pct_rows_budget_bound",
            "measured_chars_per_token", "total_cost_usd", "judged",
        ])
        for p in points:
            w.writerow([
                p.max_chars, round(p.avg_tokens, 1), round(p.p95_tokens, 1),
                round(p.accuracy or 0.0, 4), p.n, round(p.pct_bound, 1),
                p.chars_per_token, round(p.cost, 4), p.judged_available,
            ])
    print(f"\n  csv -> {path}")


def write_png(points: list[Point], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping --png)", file=sys.stderr)
        return

    pts = sorted(points, key=lambda p: p.avg_tokens)
    xs = [p.avg_tokens for p in pts]
    ys = [(p.accuracy or 0.0) * 100 for p in pts]
    anchor = max(points, key=lambda p: p.max_chars)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", color="#2b6cb0")
    ax.axhline((anchor.accuracy or 0.0) * 100, ls="--", lw=1,
               color="#718096", label="64k anchor")
    ax.axhline((anchor.accuracy or 0.0) * 100 - 2, ls=":", lw=1,
               color="#c53030", label="-2pp target")
    for p, x, y in zip(pts, xs, ys, strict=True):
        ax.annotate(f"{p.max_chars // 1000}k", (x, y),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("measured context tokens per question (log)")
    ax.set_ylabel("judged accuracy (%)")
    ax.set_title("LongMemEval-S — accuracy vs context budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  png -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="results/budget_*/ directories")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--target-pp", type=float, default=2.0,
                    help="Accuracy budget vs the anchor (default 2.0pp).")
    args = ap.parse_args(argv)

    points: list[Point] = []
    for d in args.dirs:
        if not d.is_dir():
            continue
        try:
            points.append(Point(d))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)
    if not points:
        print("no usable budget points found", file=sys.stderr)
        return 1

    render(points, target_pp=args.target_pp)
    if args.csv:
        write_csv(points, args.csv)
    if args.png:
        write_png(points, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
