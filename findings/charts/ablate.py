"""
findings/charts/ablate.py
=========================
WS-5 measurement harness — **A/B ablation across full runs.**

The v1 development hit a real trap (documented in
``findings/reasoning_loop_2026-06.md``): a change that helped the *failure*
set silently regressed the *success* set by −6pp, because only failures were
inspected. This tool makes that impossible to miss: it diffs two (or more)
LongMemEval result JSONs **per category on the full set** and flags any
category that regressed — especially the already-solved ones.

Every v1.1 lever (WS-1..WS-6) must show its before/after through this tool
before it lands. A lever that doesn't move the full-set number, or that
regresses a solved category beyond noise, does not ship.

Input
-----
Result JSONs written by ``evals.longmemeval.baseline`` (the
``{metrics, rows[]}`` shape). The first file is the **baseline**; each
subsequent file is a **variant** compared against it.

Usage
-----
    python3.12 findings/charts/ablate.py BASELINE.json VARIANT.json [VARIANT2.json ...]
    python3.12 findings/charts/ablate.py --regression-threshold 2.0 a.json b.json

Exit code is non-zero if the regression guard trips (a solved category, i.e.
baseline accuracy >= --solved-threshold, drops by more than
--regression-threshold pp) — so it can gate CI / a pre-merge check.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _row_correct(row: dict[str, Any]) -> bool | None:
    """Judged verdict if present, else the primary ``correct`` flag.

    Returns ``None`` only when the row carries no usable verdict at all
    (so it's excluded from the denominator rather than counted wrong).
    """
    jc = row.get("judge_correct")
    if jc is not None:
        return bool(jc)
    if "correct" in row:
        return bool(row["correct"])
    if "substring_correct" in row:
        return bool(row["substring_correct"])
    return None


def _category(row: dict[str, Any]) -> str:
    return str(row.get("question_type") or row.get("category") or "unknown")


class RunStats:
    """Per-category + overall accuracy for one result JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        payload = json.loads(path.read_text())
        rows: list[dict[str, Any]] = payload.get("rows", [])
        self.metrics: dict[str, Any] = payload.get("metrics", {})

        # category -> (n_scored, n_correct)
        self._cat: dict[str, list[int]] = {}
        self._n = 0
        self._correct = 0
        latencies: list[float] = []
        cost = 0.0
        for r in rows:
            verdict = _row_correct(r)
            cost += float(r.get("cost_usd", 0.0) or 0.0)
            lat = r.get("latency_ms")
            if isinstance(lat, (int, float)):
                latencies.append(float(lat))
            if verdict is None:
                continue
            cat = _category(r)
            bucket = self._cat.setdefault(cat, [0, 0])
            bucket[0] += 1
            bucket[1] += int(verdict)
            self._n += 1
            self._correct += int(verdict)

        self.total_cost = cost or float(self.metrics.get("total_cost_usd", 0.0) or 0.0)
        self.p50_latency = statistics.median(latencies) if latencies else 0.0

    def overall(self) -> float:
        return 100.0 * self._correct / self._n if self._n else 0.0

    def categories(self) -> dict[str, tuple[int, float]]:
        """category -> (n, accuracy%)."""
        return {c: (n, 100.0 * k / n if n else 0.0) for c, (n, k) in self._cat.items()}


def _fmt_delta(d: float) -> str:
    return f"{d:+.1f}"


def ablate(
    paths: list[Path],
    *,
    regression_threshold: float,
    solved_threshold: float,
) -> int:
    runs = [RunStats(p) for p in paths]
    base = runs[0]
    base_cats = base.categories()

    labels = [p.stem for p in paths]
    print("=" * 78)
    print("ABLATION — judged accuracy per category (baseline =", labels[0] + ")")
    print("=" * 78)

    all_cats = sorted({c for r in runs for c in r.categories()})
    header = f"{'category':<28}{'n':>5}  " + "".join(f"{lbl[:12]:>13}" for lbl in labels)
    if len(runs) == 2:
        header += f"{'Δpp':>8}"
    print(header)
    print("-" * len(header))

    regressions: list[tuple[str, float, float, float]] = []  # cat, n, base%, delta
    for cat in all_cats:
        n, base_acc = base_cats.get(cat, (0, 0.0))
        cells = ""
        for r in runs:
            rc = r.categories().get(cat)
            cells += f"{rc[1]:>12.1f}%" if rc else f"{'—':>13}"
        line = f"{cat:<28}{n:>5}  {cells}"
        if len(runs) == 2 and cat in base_cats and cat in runs[1].categories():
            delta = runs[1].categories()[cat][1] - base_acc
            line += f"{_fmt_delta(delta):>8}"
            if base_acc >= solved_threshold and delta < -regression_threshold:
                regressions.append((cat, n, base_acc, delta))
        print(line)

    print("-" * len(header))
    overall_cells = "".join(f"{r.overall():>12.1f}%" for r in runs)
    overall_line = f"{'OVERALL':<28}{base._n:>5}  {overall_cells}"
    if len(runs) == 2:
        overall_line += f"{_fmt_delta(runs[1].overall() - base.overall()):>8}"
    print(overall_line)

    # Cost + latency receipts
    print()
    for r, lbl in zip(runs, labels, strict=True):
        print(f"  {lbl:<24} cost=${r.total_cost:.4f}  p50={r.p50_latency:.0f}ms")

    # Regression guard
    print()
    if len(runs) != 2:
        print("(regression guard runs only for a 2-file A/B)")
        return 0
    if regressions:
        print("✗ REGRESSION GUARD TRIPPED — solved categories dropped:")
        for cat, n, base_acc, delta in regressions:
            print(f"    {cat} (n={n}): {base_acc:.1f}% → {base_acc + delta:.1f}%  ({_fmt_delta(delta)}pp)")
        print(
            f"\n  A category at ≥{solved_threshold:.0f}% fell by >{regression_threshold:.1f}pp. "
            "Investigate the success set before landing — this is the v1 −6pp mistake."
        )
        return 1
    print(
        f"✓ regression guard clear — no solved category (≥{solved_threshold:.0f}%) "
        f"dropped by >{regression_threshold:.1f}pp."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-category A/B ablation for LongMemEval runs.")
    ap.add_argument("paths", nargs="+", type=Path, help="result JSONs; first = baseline")
    ap.add_argument(
        "--regression-threshold",
        type=float,
        default=2.0,
        help="pp drop on a solved category that trips the guard (default 2.0)",
    )
    ap.add_argument(
        "--solved-threshold",
        type=float,
        default=90.0,
        help="baseline accuracy%% above which a category counts as 'solved' (default 90)",
    )
    args = ap.parse_args(argv)
    for p in args.paths:
        if not p.exists():
            ap.error(f"no such file: {p}")
    return ablate(
        args.paths,
        regression_threshold=args.regression_threshold,
        solved_threshold=args.solved_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
