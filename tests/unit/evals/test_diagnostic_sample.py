"""
tests/unit/evals/test_diagnostic_sample.py
==========================================
Unit tests for :mod:`scripts.build_diagnostic_sample`.

Confirms:

* Composition is honoured when every bucket has enough rows.
* Same seed → same id list (determinism).
* Abstain detection works via ``_abs`` suffix AND expected-answer phrase.
* The script gracefully under-fills a bucket without crashing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_module():
    """Load scripts/build_diagnostic_sample.py directly (no package install)."""
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_build_diag_sample",
        repo / "scripts" / "build_diagnostic_sample.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # 20 of each main category — well above the per-bucket spec.
    for cat in (
        "single-session-user",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    ):
        for i in range(20):
            rows.append({
                "question_id": f"{cat}-{i:03d}",
                "question_type": cat,
                "answer": f"answer-{cat}-{i}",
            })
    # 12 abstain — some via _abs suffix, some via expected-answer phrase.
    for i in range(6):
        rows.append({
            "question_id": f"single-session-user-{i:03d}_abs",
            "question_type": "single-session-user",
            "answer": "user did not mention this",
        })
    for i in range(6):
        rows.append({
            "question_id": f"multi-session-extra-{i:03d}",
            "question_type": "multi-session",
            "answer": "Not enough information to answer.",
        })
    return rows


def test_build_sample_honors_composition():
    mod = _load_module()
    rows = _make_rows()
    ids, breakdown = mod.build_sample(rows, seed=42)
    expected = {
        "single-session-user":        10,
        "multi-session":              10,
        "temporal-reasoning":         10,
        "knowledge-update":           10,
        "single-session-preference":   5,
        "abstain":                     5,
    }
    for cat, want in expected.items():
        assert breakdown[cat] == want, f"bucket {cat}: want {want}, got {breakdown[cat]}"
    # 50 unique ids assuming no duplicates across buckets (abstain is
    # checked first so the bucket is exclusive).
    assert len(ids) == 50
    assert len(set(ids)) == len(ids)


def test_build_sample_is_deterministic():
    mod = _load_module()
    rows = _make_rows()
    a, _ = mod.build_sample(rows, seed=42)
    b, _ = mod.build_sample(rows, seed=42)
    assert a == b


def test_build_sample_changes_with_seed():
    mod = _load_module()
    rows = _make_rows()
    a, _ = mod.build_sample(rows, seed=42)
    b, _ = mod.build_sample(rows, seed=7)
    assert a != b  # extremely unlikely to coincide


def test_abstain_detection_via_abs_suffix():
    mod = _load_module()
    assert mod._is_abstain_row({
        "question_id": "x_abs",
        "question_type": "single-session-user",
        "answer": "the user said tea",
    })


def test_abstain_detection_via_phrase():
    mod = _load_module()
    assert mod._is_abstain_row({
        "question_id": "x",
        "question_type": "single-session-user",
        "answer": "Not enough information to answer this question.",
    })
    assert not mod._is_abstain_row({
        "question_id": "x",
        "question_type": "single-session-user",
        "answer": "The answer is blue.",
    })


def test_build_sample_handles_undersized_bucket():
    """When a bucket is short, warn but don't crash. Other buckets unaffected."""
    mod = _load_module()
    rows = _make_rows()
    # Drop temporal-reasoning down to 3 rows; bucket should under-fill to 3.
    pruned = [r for r in rows if not (
        r["question_type"] == "temporal-reasoning"
        and int(r["question_id"].rsplit("-", 1)[1]) >= 3
    )]
    ids, breakdown = mod.build_sample(pruned, seed=42)
    assert breakdown["temporal-reasoning"] == 3
    # The other buckets still get their full share.
    assert breakdown["single-session-user"] == 10
    assert breakdown["multi-session"] == 10
    assert len(ids) == 50 - (10 - 3)
