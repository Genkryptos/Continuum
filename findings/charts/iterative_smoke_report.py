"""
findings/charts/iterative_smoke_report.py
==========================================
Read a LongMemEval baseline-results JSON produced by an iterative-reasoner
run and print the headline numbers the spec asks for:

* per-category judged accuracy + abstain rate + head-short-circuit rate
* overall mean LLM-call count + cap-hit rate
* p50 / p95 latency

Also writes a structured summary to ``<input>.iterative_summary.json``
next to the input so downstream tooling doesn't have to re-parse the
table.

Usage
-----
The input JSON is whatever :mod:`evals.longmemeval.bootstrap_ollama`
writes when ``--reasoner iterative`` is on. Producing it requires a
provider key + outbound HTTP; the agent sandbox can't run this, but
the user can. The canonical smoke is::

    OPENAI_API_KEY=… python3.12 -m evals.longmemeval.bootstrap_ollama \\
        --reasoner iterative --max-llm-calls 6 --max-rounds 2 \\
        --use-ltm --ltm-backend in_memory --no-llm-promoter \\
        --retriever hybrid --judge llm --span-fallback \\
        --question-types multi-session,temporal-reasoning \\
        --limit 100 \\
        --output results/iter_reasoner_smoke

Then::

    python3.12 findings/charts/iterative_smoke_report.py \\
        results/iter_reasoner_smoke/baseline_<DATE>.json

Add ``--by-mode`` to also break results down by the reasoner's
TaskMode classification recorded in ``row.telemetry.task_mode``.

Pure stdlib — no extra deps, no charts, just a tight text table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── parsing helpers ────────────────────────────────────────────────────────


def _judged(row: dict[str, Any]) -> bool | None:
    """The row's primary verdict under the judge, falling back to ``correct``.

    Older runs without an inline judge only populate ``correct``; newer
    runs add ``judge_correct``. We treat the judged verdict as the
    primary metric when present so the column label "judged accuracy"
    means what it says.
    """
    v = row.get("judge_correct")
    if v is None:
        return bool(row.get("correct", False)) if "correct" in row else None
    return bool(v)


def _telem(row: dict[str, Any]) -> dict[str, Any]:
    t = row.get("telemetry")
    return dict(t) if isinstance(t, dict) else {}


def _latencies(rows: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get("latency_ms")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, round((len(s) - 1) * q))
    return float(s[idx])


# ── grouped aggregation ────────────────────────────────────────────────────


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict]]:
    """
    Group rows by row[key] for top-level keys, OR by row.telemetry[key]
    when the top-level lookup is empty. This is how we get
    ``question_type`` (top-level) AND ``task_mode`` (telemetry) under
    one function.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket = r.get(key)
        if bucket is None:
            bucket = _telem(r).get(key)
        groups[str(bucket or "unknown")].append(r)
    return dict(groups)


def _summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Single-bucket aggregation. Safe on empty input."""
    if not rows:
        return {
            "n": 0, "judged_acc": 0.0, "abstain_rate": 0.0,
            "head_short_circuit_rate": 0.0, "avg_llm_calls": 0.0,
            "max_llm_calls": 0, "cap_hit_rate": 0.0,
            "p50_ms": 0.0, "p95_ms": 0.0,
        }
    judged = [_judged(r) for r in rows]
    judged_known = [v for v in judged if v is not None]
    abstained = [bool(_telem(r).get("abstained")) for r in rows]
    head_short = [bool(_telem(r).get("head_short_circuit")) for r in rows]
    calls = [int(_telem(r).get("llm_call_count") or 0) for r in rows]
    cap = max(calls) if calls else 0
    cap_hits = sum(1 for c in calls if c >= cap and cap > 0)
    lat = _latencies(rows)
    return {
        "n": len(rows),
        "judged_acc": (
            sum(1 for v in judged_known if v) / len(judged_known)
            if judged_known else 0.0
        ),
        "judged_n": len(judged_known),
        "abstain_rate": sum(abstained) / len(rows),
        "head_short_circuit_rate": sum(head_short) / len(rows),
        "avg_llm_calls": statistics.fmean(calls) if calls else 0.0,
        "max_llm_calls": cap,
        "cap_hit_rate": cap_hits / len(rows) if rows else 0.0,
        "p50_ms": _percentile(lat, 0.50),
        "p95_ms": _percentile(lat, 0.95),
    }


# ── rendering ──────────────────────────────────────────────────────────────


def _render_table(title: str, summaries: dict[str, dict[str, Any]]) -> str:
    head = (
        f"{'bucket':<28}"
        f"{'n':>6}"
        f"{'judged%':>10}"
        f"{'abst%':>8}"
        f"{'head%':>8}"
        f"{'avg_LLM':>9}"
        f"{'cap%':>7}"
        f"{'p50ms':>9}"
        f"{'p95ms':>9}"
    )
    sep = "-" * len(head)
    out = [title, "=" * len(head), head, sep]
    for bucket in sorted(summaries):
        s = summaries[bucket]
        out.append(
            f"{bucket[:28]:<28}"
            f"{s['n']:>6}"
            f"{s['judged_acc']*100:>9.1f} "
            f"{s['abstain_rate']*100:>7.1f} "
            f"{s['head_short_circuit_rate']*100:>7.1f} "
            f"{s['avg_llm_calls']:>8.2f} "
            f"{s['cap_hit_rate']*100:>6.1f} "
            f"{s['p50_ms']:>8.0f} "
            f"{s['p95_ms']:>8.0f}"
        )
    return "\n".join(out)


# ── entry ──────────────────────────────────────────────────────────────────


def _summarise(payload: dict[str, Any], *, by_mode: bool) -> dict[str, Any]:
    rows = payload.get("rows") or []
    overall = _summary_for(rows)
    per_cat = {k: _summary_for(v) for k, v in _group_by(rows, "question_type").items()}
    summary: dict[str, Any] = {
        "overall": overall,
        "per_question_type": per_cat,
    }
    if by_mode:
        per_mode = {k: _summary_for(v) for k, v in _group_by(rows, "task_mode").items()}
        summary["per_task_mode"] = per_mode
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render the iterative-reasoner smoke-run numbers from a baseline JSON.",
    )
    p.add_argument("input", type=Path,
                   help="Path to a baseline_*.json written by bootstrap_ollama.")
    p.add_argument("--by-mode", action="store_true",
                   help="Also break results down by the reasoner's TaskMode.")
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

    summary = _summarise(payload, by_mode=args.by_mode)

    print(_render_table(
        "Iterative-reasoner smoke — overall + per question_type",
        {"OVERALL": summary["overall"], **summary["per_question_type"]},
    ))
    if args.by_mode and summary.get("per_task_mode"):
        print()
        print(_render_table(
            "Per TaskMode (from row.telemetry.task_mode)",
            summary["per_task_mode"],
        ))

    if not args.no_write:
        out_path = args.input.with_name(args.input.stem + ".iterative_summary.json")
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nstructured summary → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
