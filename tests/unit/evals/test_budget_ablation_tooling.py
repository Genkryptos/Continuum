"""
tests/unit/evals/test_budget_ablation_tooling.py
================================================
Phase-1 tooling for the cost/accuracy ablation (``docs/LOW_BUDGET_PLAN.md``):
the frozen stratified sample and the curve renderer.

The properties that matter are about *comparability*, not formatting:

* the sample must be proportional to the real category mix, or the
  subset's aggregate accuracy can't be compared to the published
  full-500 number;
* it must be deterministic, because every budget point and every later
  phase re-runs against the identical rows;
* the renderer must refuse to treat a non-binding budget as a point on
  the curve — that is the failure mode most likely to produce a
  confident, wrong headline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from findings.charts.budget_curve import Point, _holds, render
from scripts.build_budget_sample import _allocate, build_sample

pytestmark = pytest.mark.unit


#: The real LongMemEval-S mix (n=500).
_SOURCE_MIX = {
    "knowledge-update": 78,
    "multi-session": 133,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "single-session-user": 70,
    "temporal-reasoning": 133,
}


def _dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qtype, n in _SOURCE_MIX.items():
        rows.extend(
            {"question_id": f"{qtype}-{i:03d}", "question_type": qtype}
            for i in range(n)
        )
    return rows


# ── stratified sample ──────────────────────────────────────────────────


def test_allocation_sums_to_exactly_the_requested_size() -> None:
    """
    Largest-remainder, not round(). A subset of 249 or 251 would quietly
    break the "half the dataset" framing the sample is chosen for.
    """
    for size in (50, 100, 250, 333, 499):
        alloc = _allocate(_SOURCE_MIX, sum(_SOURCE_MIX.values()), size)
        assert sum(alloc.values()) == size, f"size={size} allocated {sum(alloc.values())}"


def test_sample_preserves_the_category_mix() -> None:
    rows = _dataset()
    qids, breakdown = build_sample(rows, size=250, seed=0)

    assert len(qids) == 250
    assert len(set(qids)) == 250, "duplicate question ids in the sample"
    for category, n_src in _SOURCE_MIX.items():
        want = 100 * n_src / len(rows)
        got = 100 * breakdown[category] / len(qids)
        assert abs(got - want) < 0.5, f"{category}: {got:.1f}% vs {want:.1f}%"


def test_every_category_cell_is_readable() -> None:
    """n>=15 per category, or the per-category curve says nothing."""
    _, breakdown = build_sample(_dataset(), size=250, seed=0)
    assert min(breakdown.values()) >= 15


def test_sample_is_deterministic_across_calls() -> None:
    rows = _dataset()
    a, _ = build_sample(rows, size=250, seed=0)
    b, _ = build_sample(rows, size=250, seed=0)
    assert a == b


def test_different_seed_gives_a_different_sample() -> None:
    rows = _dataset()
    a, _ = build_sample(rows, size=250, seed=0)
    b, _ = build_sample(rows, size=250, seed=7)
    assert a != b, "seed had no effect — the sample is not actually sampled"


def test_sample_preserves_dataset_source_order() -> None:
    rows = _dataset()
    order = {r["question_id"]: i for i, r in enumerate(rows)}
    qids, _ = build_sample(rows, size=250, seed=0)
    positions = [order[q] for q in qids]
    assert positions == sorted(positions)


def test_committed_sample_matches_the_generator() -> None:
    """
    The committed sample is frozen: every budget point runs against it.
    If regenerating it here produces different ids, someone changed the
    generator after the ablation started and the points no longer compare.
    """
    path = Path(__file__).resolve().parents[3] / "samples" / "budget_strat_250.json"
    if not path.exists():
        pytest.skip("samples/budget_strat_250.json not built yet")
    payload = json.loads(path.read_text())
    dataset_path = Path(payload["created_from"])
    if not dataset_path.exists():
        pytest.skip("source dataset not available")

    rows = json.loads(dataset_path.read_text())
    regenerated, _ = build_sample(
        rows, size=payload["total"], seed=payload["seed"],
    )
    assert regenerated == payload["question_ids"]


# ── curve renderer ─────────────────────────────────────────────────────


def _point(tmp: Path, chars: int, acc: float, tokens: float, bound: float) -> Point:
    d = tmp / f"budget_{chars}"
    d.mkdir(parents=True, exist_ok=True)
    by_type = {
        "multi-session": {
            "n_questions": 67, "accuracy": acc, "recall": 0.98,
            "avg_context_tokens": tokens, "pct_budget_bound": bound,
        },
    }
    (d / "judged.json").write_text(json.dumps({
        "metrics": {
            "n_questions": 250, "accuracy": acc, "total_cost_usd": 0.5,
            "by_question_type": by_type,
        },
    }))
    (d / "budget_report.json").write_text(json.dumps({
        "config": {"max_context_chars": chars},
        "summary": {
            "n_questions": 250, "accuracy": acc,
            "avg_context_tokens": tokens, "context_tokens_p95": tokens * 1.4,
            "measured_chars_per_token": 4.1, "pct_rows_budget_bound": bound,
            "total_cost_usd": 0.5, "tokens_measured": True,
            "unpriced_models": [],
        },
        "by_question_type": by_type, "rows": [],
    }))
    return Point(d)


def test_point_exactly_on_the_target_line_counts_as_holding(tmp_path: Path) -> None:
    """
    76.4 - 74.4 computes as 2.0000000000000018 in float. Without a
    tolerance the one point most likely to land on the line gets called
    a break.
    """
    anchor = _point(tmp_path, 64000, 0.764, 15600, 100.0)
    edge = _point(tmp_path, 8000, 0.744, 1980, 74.0)
    assert _holds(edge, anchor, 2.0)


def test_point_just_past_the_line_does_not_hold(tmp_path: Path) -> None:
    anchor = _point(tmp_path, 64000, 0.764, 15600, 100.0)
    past = _point(tmp_path, 8000, 0.740, 1980, 74.0)
    assert not _holds(past, anchor, 2.0)


def test_non_binding_budget_never_counts_as_holding(tmp_path: Path) -> None:
    """
    A budget the retriever never filled is not a cheaper configuration —
    it is the same configuration with a bigger number on the flag. Its
    accuracy delta is run noise and must not become a headline.
    """
    anchor = _point(tmp_path, 64000, 0.764, 15600, 100.0)
    never_bound = _point(tmp_path, 32000, 0.764, 15600, 0.0)
    assert not _holds(never_bound, anchor, 2.0)


def test_render_reports_no_holding_point_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A negative result must still render — it is the finding either way."""
    points = [
        _point(tmp_path, 64000, 0.764, 15600, 100.0),
        _point(tmp_path, 8000, 0.500, 1980, 90.0),
    ]
    render(points, target_pp=2.0)
    out = capsys.readouterr().out
    assert "no budget below the anchor holds" in out
    assert "breaks" in out


def test_render_names_the_cheapest_holding_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    points = [
        _point(tmp_path, 64000, 0.764, 15600, 100.0),
        _point(tmp_path, 16000, 0.752, 3950, 92.0),
        _point(tmp_path, 8000, 0.751, 1980, 74.0),
    ]
    render(points, target_pp=2.0)
    out = capsys.readouterr().out
    assert "HEADLINE" in out
    # The cheapest holding point wins, not merely the first one seen.
    assert "1,980 tokens" in out
    assert "7.9x fewer tokens" in out
