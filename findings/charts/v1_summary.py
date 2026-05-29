"""
findings/charts/v1_summary.py
==============================
Render the headline numbers for a v1 full LongMemEval-S sweep run with
the iterative reasoner. Reads a baseline JSON produced by
``evals.longmemeval.bootstrap_ollama`` and prints:

* overall judged accuracy
* per-category (``question_type``) accuracy + n + abstain rate
* average LLM-call count (per row)
* p50 / p95 latency (ms)
* total cost (USD, from ``metrics.total_cost_usd`` when present,
  otherwise summed from ``row.cost_usd``)

Distinct from :mod:`findings.charts.iterative_smoke_report` in two
ways: (1) v1_summary surfaces dollar cost (the smoke report doesn't),
(2) v1_summary is one tight table tuned for the final-sweep report,
not the broader per-mode breakdown used during development.

Usage
-----
Produce the input by backgrounding the spec'd sweep on a box with
``OPENAI_API_KEY`` set — Task A's gate killed ``--abstain-threshold``
so the spec'd command must drop that flag::

    OPENAI_API_KEY=… python3.12 -m evals.longmemeval.bootstrap_ollama \\
        --reasoner iterative --max-llm-calls 6 --max-rounds 2 \\
        --use-ltm --ltm-backend in_memory --no-llm-promoter \\
        --retriever hybrid --judge llm --span-fallback \\
        --output results/v1_final

Then render::

    python3.12 findings/charts/v1_summary.py results/v1_final/baseline_<DATE>.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── parsing ────────────────────────────────────────────────────────────────


def _judged(row: dict[str, Any]) -> bool | None:
    v = row.get("judge_correct")
    if v is None:
        return bool(row.get("correct", False)) if "correct" in row else None
    return bool(v)


def _telem(row: dict[str, Any]) -> dict[str, Any]:
    t = row.get("telemetry")
    return dict(t) if isinstance(t, dict) else {}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, round((len(s) - 1) * q))
    return float(s[idx])


# ── aggregation ────────────────────────────────────────────────────────────


def _bucket_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Per-bucket numbers used by every row of the table."""
    if not rows:
        return {"n": 0, "judged_acc": 0.0, "abstain_rate": 0.0,
                "avg_llm_calls": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "total_cost_usd": 0.0}
    judged = [_judged(r) for r in rows]
    judged_known = [v for v in judged if v is not None]
    abstained = [bool(_telem(r).get("abstained")) for r in rows]
    calls = [int(_telem(r).get("llm_call_count") or 0) for r in rows]
    latencies = [
        float(r["latency_ms"]) for r in rows
        if isinstance(r.get("latency_ms"), (int, float))
    ]
    costs = [
        float(r.get("cost_usd") or 0.0) for r in rows
    ]
    return {
        "n": len(rows),
        "judged_acc": (
            sum(1 for v in judged_known if v) / len(judged_known)
            if judged_known else 0.0
        ),
        "abstain_rate": sum(abstained) / len(rows),
        "avg_llm_calls": statistics.fmean(calls) if calls else 0.0,
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "total_cost_usd": sum(costs),
    }


def _by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[str(r.get("question_type") or "unknown")].append(r)
    return dict(out)


# ── rendering ──────────────────────────────────────────────────────────────


def _render_table(overall: dict, per_cat: dict[str, dict]) -> str:
    head = (
        f"{'category':<28}"
        f"{'n':>6}"
        f"{'judged%':>10}"
        f"{'abst%':>8}"
        f"{'avg_LLM':>10}"
        f"{'p50ms':>9}"
        f"{'p95ms':>9}"
        f"{'cost_$':>10}"
    )
    sep = "-" * len(head)
    lines = ["v1 IterativeReasoner — full LongMemEval-S sweep",
             "=" * len(head), head, sep]
    # Overall first
    lines.append(_row("OVERALL", overall))
    # Per-category, sorted alphabetically for stable diffs across runs.
    for name in sorted(per_cat):
        lines.append(_row(name, per_cat[name]))
    return "\n".join(lines)


def _row(label: str, s: dict[str, float | int]) -> str:
    return (
        f"{label[:28]:<28}"
        f"{int(s['n']):>6}"
        f"{float(s['judged_acc'])*100:>9.1f} "
        f"{float(s['abstain_rate'])*100:>7.1f} "
        f"{float(s['avg_llm_calls']):>9.2f} "
        f"{float(s['p50_ms']):>8.0f} "
        f"{float(s['p95_ms']):>8.0f} "
        f"{float(s['total_cost_usd']):>9.2f}"
    )


# ── entry ──────────────────────────────────────────────────────────────────


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    metrics = payload.get("metrics") or {}
    per_cat = {k: _bucket_stats(v) for k, v in _by_category(rows).items()}
    overall = _bucket_stats(rows)
    # Prefer the run-level total_cost_usd if the rig already aggregated it
    # — that path includes any cost the rig measured outside of row.cost_usd.
    if isinstance(metrics.get("total_cost_usd"), (int, float)):
        overall["total_cost_usd"] = float(metrics["total_cost_usd"])
    # Surface judged_accuracy at the run level when present (newer
    # judge-inline runs); fall back to the row-derived figure.
    if isinstance(metrics.get("judged_accuracy"), (int, float)):
        overall["judged_acc"] = float(metrics["judged_accuracy"])
    return {
        "overall": overall,
        "per_question_type": per_cat,
        "n_rows": len(rows),
        "dataset": payload.get("dataset"),
        "answerer": payload.get("answerer"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render v1-final headline numbers from a baseline JSON.",
    )
    p.add_argument("input", type=Path,
                   help="Path to a baseline_*.json from bootstrap_ollama.")
    p.add_argument("--no-write", action="store_true",
                   help="Don't write the structured summary JSON next to the input.")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(args.input.read_text())
    except json.JSONDecodeError as exc:
        print(f"failed to parse {args.input}: {exc}", file=sys.stderr)
        return 2

    summary = _summarise(payload)

    if summary.get("dataset") or summary.get("answerer"):
        print(f"dataset: {summary.get('dataset')}  answerer: {summary.get('answerer')}")
    print(_render_table(summary["overall"], summary["per_question_type"]))

    if not args.no_write:
        # Use a "summary_" prefix so the output doesn't collide with the
        # input under ``baseline_*.json`` globs (the prior naming
        # ``<stem>.v1_summary.json`` did, breaking re-runs).
        out_path = args.input.with_name("summary_" + args.input.stem + ".json")
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nstructured summary → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
