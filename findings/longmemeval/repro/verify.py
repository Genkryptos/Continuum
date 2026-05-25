"""
findings/longmemeval/repro/verify.py
====================================
Compare a freshly-run baseline JSON against the expected numbers in
``expected_results.json``. Exits ``0`` if every metric is within the
declared tolerance, ``1`` otherwise.

Usage:
    python3 findings/longmemeval/repro/verify.py \\
        --run-name gpt4omini_topk \\
        --result results/repro/gpt4omini_topk/baseline_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXPECTED = Path(__file__).resolve().parent / "expected_results.json"


def _load_run_expectation(name: str) -> dict:
    cfg = json.loads(EXPECTED.read_text())
    for r in cfg["runs"]:
        if r["name"] == name:
            return {"expected": r["expected"], "tol": cfg["_meta"]["tolerance_pp"]}
    raise SystemExit(f"unknown run name: {name!r}")


def _latest(glob_pattern: str) -> Path:
    matches = sorted(glob.glob(glob_pattern), reverse=True)
    if not matches:
        raise SystemExit(f"no file matches {glob_pattern!r}")
    return Path(matches[0])


def verify(name: str, result_path: Path) -> int:
    cfg = _load_run_expectation(name)
    exp = cfg["expected"]
    tol = float(cfg["tol"])

    data = json.loads(result_path.read_text())
    m = data["metrics"]
    fb = m.get("failure_breakdown", {})

    actual = {
        "accuracy_pct":   m["accuracy"] * 100,
        "recall_pct":     m["recall"] * 100,
        "n_questions":    m["n_questions"],
        "missing_fact":   fb.get("missing_fact", 0),
        "wrong_retrieval": fb.get("wrong_retrieval", 0),
    }

    print()
    print("=" * 72)
    print(f"  REPRO VERIFY — {name}")
    print(f"  source: {result_path.relative_to(REPO) if result_path.is_relative_to(REPO) else result_path}")
    print(f"  tolerance: ±{tol}pp on percentages, ±10 on counts")
    print("=" * 72)
    print(f"  {'metric':<22}{'expected':>14}{'actual':>14}{'diff':>14}{'status':>10}")
    print("-" * 72)
    fail = 0
    for k, e in exp.items():
        a = actual[k]
        diff = a - e
        # Use percentage tolerance for accuracy/recall, absolute count
        # tolerance for the integer failure-bucket counts.
        if k.endswith("_pct"):
            within = abs(diff) <= tol
        elif k == "n_questions":
            within = a == e
        else:
            within = abs(diff) <= 10
        flag = "OK" if within else "FAIL"
        fail += 0 if within else 1
        print(
            f"  {k:<22}{e:>14}{a:>14.1f}{diff:>+14.1f}{flag:>10}"
            if isinstance(a, float)
            else f"  {k:<22}{e:>14}{a:>14}{diff:>+14}{flag:>10}"
        )
    print("=" * 72)
    if fail:
        print(f"  ❌ {fail} metric(s) outside tolerance")
        return 1
    print("  ✅ all metrics within tolerance")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True,
                   help="Name from expected_results.json (e.g. gpt4omini_topk).")
    p.add_argument("--result", required=True,
                   help="Path or glob to the freshly-run baseline_*.json.")
    args = p.parse_args(argv)
    path = _latest(args.result) if "*" in args.result else Path(args.result)
    return verify(args.run_name, path)


if __name__ == "__main__":
    sys.exit(main())
