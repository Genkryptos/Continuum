"""
tests/unit/evals/test_ablate.py
===============================
Unit tests for the WS-5 ablation harness (``findings/charts/ablate.py``):
per-category accuracy, A/B delta, and the regression guard that codifies
"never silently regress a solved category" (the v1 −6pp lesson).

Hermetic — synthetic result JSONs written to ``tmp_path``; no eval run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_ABLATE = Path(__file__).resolve().parents[3] / "findings" / "charts" / "ablate.py"


def _load_ablate() -> Any:
    spec = importlib.util.spec_from_file_location("ablate", _ABLATE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ablate = _load_ablate()


def _write_run(
    path: Path,
    cats: dict[str, tuple[int, int]],
    *,
    cost_per_row: float = 0.001,
    latency_ms: float = 100.0,
) -> Path:
    """Write a result JSON. ``cats`` maps category -> (n_total, n_correct)."""
    rows: list[dict[str, Any]] = []
    for cat, (n, k) in cats.items():
        for i in range(n):
            rows.append(
                {
                    "question_id": f"{cat}-{i}",
                    "question_type": cat,
                    "judge_correct": i < k,
                    "cost_usd": cost_per_row,
                    "latency_ms": latency_ms,
                }
            )
    path.write_text(json.dumps({"metrics": {}, "rows": rows}))
    return path


# ── per-category accuracy + overall ──────────────────────────────────────────


def test_runstats_per_category_and_overall(tmp_path: Path) -> None:
    p = _write_run(tmp_path / "a.json", {"solved": (10, 10), "temporal": (10, 4)})
    rs = ablate.RunStats(p)
    cats = rs.categories()
    assert cats["solved"] == (10, 100.0)
    assert cats["temporal"] == (10, 40.0)
    assert rs.overall() == pytest.approx(70.0)  # 14/20


def test_runstats_reads_cost_and_latency(tmp_path: Path) -> None:
    p = _write_run(tmp_path / "a.json", {"x": (5, 3)}, cost_per_row=0.002, latency_ms=250.0)
    rs = ablate.RunStats(p)
    assert rs.total_cost == pytest.approx(0.010)  # 5 × 0.002
    assert rs.p50_latency == pytest.approx(250.0)


def test_row_correct_falls_back_to_correct_then_substring() -> None:
    assert ablate._row_correct({"judge_correct": True}) is True
    assert ablate._row_correct({"judge_correct": None, "correct": True}) is True
    assert ablate._row_correct({"correct": False}) is False
    assert ablate._row_correct({"substring_correct": True}) is True
    # No verdict at all → excluded from the denominator.
    assert ablate._row_correct({"question_type": "x"}) is None


# ── regression guard ─────────────────────────────────────────────────────────


def test_guard_clear_on_improvement(tmp_path: Path, capsys: Any) -> None:
    base = _write_run(tmp_path / "base.json", {"solved": (10, 10), "temporal": (10, 4)})
    better = _write_run(tmp_path / "better.json", {"solved": (10, 10), "temporal": (10, 7)})
    rc = ablate.main([str(base), str(better)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "regression guard clear" in out
    assert "+30.0" in out  # temporal 40 → 70


def test_guard_trips_on_solved_category_regression(tmp_path: Path, capsys: Any) -> None:
    base = _write_run(tmp_path / "base.json", {"solved": (10, 10), "temporal": (10, 4)})
    # solved drops 100 → 80 (−20pp on a ≥90% category) while temporal improves
    worse = _write_run(tmp_path / "worse.json", {"solved": (10, 8), "temporal": (10, 9)})
    rc = ablate.main([str(base), str(worse)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "REGRESSION GUARD TRIPPED" in out
    assert "solved" in out


def test_small_regression_below_threshold_does_not_trip(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base.json", {"solved": (100, 100)})
    # 100 → 99 is only −1pp, under the default 2pp threshold
    near = _write_run(tmp_path / "near.json", {"solved": (100, 99)})
    assert ablate.main([str(base), str(near)]) == 0


def test_non_solved_category_regression_is_not_guarded(tmp_path: Path) -> None:
    # A weak category dropping is allowed (only *solved* cats are guarded);
    # the guard exists to protect wins, not to freeze every number.
    base = _write_run(tmp_path / "base.json", {"weak": (10, 5)})
    worse = _write_run(tmp_path / "worse.json", {"weak": (10, 2)})
    assert ablate.main([str(base), str(worse)]) == 0


def test_three_way_runs_skip_guard(tmp_path: Path, capsys: Any) -> None:
    a = _write_run(tmp_path / "a.json", {"x": (10, 5)})
    b = _write_run(tmp_path / "b.json", {"x": (10, 6)})
    c = _write_run(tmp_path / "c.json", {"x": (10, 7)})
    rc = ablate.main([str(a), str(b), str(c)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "regression guard runs only for a 2-file A/B" in out
