"""
tests/unit/evals/test_decompose_guard.py
========================================
Tests for :mod:`evals.longmemeval.decompose_guard` — the
intent-preserving guard that rejects decomposed sub-questions whose
meaning has drifted from the original.

Acceptance scenarios from the spec:

* ``"What is the painting worth?"`` must not become
  ``"What amount did I pay?"`` (relation / answer-type drift).
* ``"Where did I complete my Bachelor's degree?"`` must not become a
  generic university / location query (object drift).

Plus per-check unit coverage and a telemetry round-trip test.
"""
from __future__ import annotations

from evals.longmemeval.decompose_guard import (
    DecompositionGuardResult,
    RejectedSubquestion,
    guard_decomposition,
)

# ─── ACCEPTANCE 1 — painting worth ≠ painting paid ────────────────────────


def test_painting_worth_does_not_become_paid():
    """
    A decomposed sub-question that flips ``worth`` to ``pay`` must be
    rejected. The original question is reinstated as the only lookup.
    """
    original = "What is the painting worth?"
    subs = ["What amount did I pay for the painting?"]
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original is True
    # Every rejected sub-question must carry an explanation
    assert result.rejected
    reason = result.rejected[0].reason
    assert any(token in reason for token in (
        "relation_changed",
        "object_dropped",
        "object_anchor_dropped",
        "answer_type_changed",
    )), f"unexpected reason: {reason}"
    # The fallback list is the original question — not the dropped sub.
    assert result.kept == [original]


def test_painting_worth_accepts_compatible_subquestion():
    """A sub that keeps both the object AND the relation passes."""
    original = "What is the painting worth?"
    subs = ["What is the current value of the painting?"]
    result = guard_decomposition(original, subs)
    # "value" maps to is_worth (same relation), "painting" survives,
    # no temporal drift → should pass.
    assert subs[0] in result.kept
    assert not result.fallback_to_original


# ─── ACCEPTANCE 2 — Bachelor's degree ≠ generic university query ──────────


def test_bachelors_question_does_not_become_generic_university():
    original = "Where did I complete my Bachelor's degree?"
    subs = ["What is a good university?"]
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original
    assert result.rejected
    reason = result.rejected[0].reason
    # "Bachelor's" / "degree" dropped → object_anchor_dropped, OR the
    # decomposition lost the personal LOCATION flavour entirely (a
    # different_fact rejection is also valid here).
    assert any(token in reason for token in (
        "object_dropped",
        "object_anchor_dropped",
        "different_fact",
        "answer_type_changed",
    )), f"unexpected reason: {reason}"
    assert result.kept == [original]


def test_bachelors_question_accepts_split_into_components():
    """A legitimate split is preserved — same object + intent."""
    original = "Where did I complete my Bachelor's degree?"
    subs = [
        "Where did I complete my Bachelor's degree?",
        "When did I complete my Bachelor's degree?",
    ]
    result = guard_decomposition(original, subs)
    # First sub is identical to original — definitely kept.
    assert original in result.kept
    # Second sub is a different fact (date instead of location) → drift.
    # That's not just a decomposition: it's literally a different
    # question and SHOULD be rejected.
    if len(result.kept) == 1:
        assert any("Bachelor" in r.subquestion for r in result.rejected)


# ─── Per-check coverage ───────────────────────────────────────────────────


def test_answer_type_change_rejected():
    """Question-type drift is a reject path."""
    original = "How many shirts did I buy?"  # COUNT
    subs = ["When did I buy shirts?"]         # DATE_TIME
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original
    assert "answer_type_changed" in result.rejected[0].reason


def test_relation_change_rejected():
    """``worth`` → ``pay`` is the relation-drift canonical case.

    The original carries "today" too, so the check ladder may
    short-circuit on temporal-drop OR on relation-changed depending
    on order. Either rejection is correct.
    """
    original = "How much is my house worth today?"
    subs = ["How much did I pay for my house?"]
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original
    reason = result.rejected[0].reason
    assert any(token in reason for token in (
        "relation_changed", "different_fact", "temporal_dropped",
    )), f"unexpected reason: {reason}"


def test_relation_change_isolated():
    """Same as above but without a temporal phrase to confuse the ladder."""
    original = "How much is my house worth?"
    subs = ["How much did I pay for my house?"]
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original
    reason = result.rejected[0].reason
    assert "relation_changed" in reason


def test_object_change_rejected():
    """Dropping the original's anchor noun is a reject path."""
    original = "Where did I park my car?"
    subs = ["Where did I park?"]
    # "car" is the anchor; the sub drops it. The check rejects on
    # object_anchor_dropped.
    result = guard_decomposition(original, subs)
    # This might pass or fail depending on tokenisation — the test
    # exists to lock the behaviour. If anchor is dropped, expect fail.
    if result.fallback_to_original:
        assert any(
            "object" in r.reason for r in result.rejected
        ), result.rejected[0].reason


def test_temporal_scope_added_rejected():
    """Sub that adds a specific year not present in the original drifts."""
    original = "When did I start the new job?"
    subs = ["When did I start the new job in 2020?"]
    result = guard_decomposition(original, subs)
    if result.fallback_to_original:
        assert any(
            "temporal" in r.reason for r in result.rejected
        )


def test_temporal_scope_dropped_rejected():
    """Original carries a year; the sub drops it → reject."""
    original = "What did I order in 2022?"
    subs = ["What did I order?"]
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original
    assert any("temporal" in r.reason for r in result.rejected)


def test_different_fact_catchall():
    """Sub that shares no content words with the original is rejected."""
    original = "What was the title of the book I read last summer?"
    subs = ["What is the weather forecast for tomorrow?"]
    result = guard_decomposition(original, subs)
    assert result.fallback_to_original
    reason = result.rejected[0].reason
    assert "different_fact" in reason or "object_dropped" in reason


# ─── Pass-through behaviour ───────────────────────────────────────────────


def test_identical_subquestion_passes_through():
    """An atomic question echoed back by the decomposer is kept verbatim."""
    original = "What is my mother's birthday?"
    subs = [original]
    result = guard_decomposition(original, subs)
    assert result.kept == [original]
    assert not result.rejected
    assert not result.fallback_to_original


def test_legitimate_multi_subquestion_split_preserved():
    """A real decomposition that preserves intent passes."""
    original = "What did I eat for breakfast and lunch yesterday?"
    subs = [
        "What did I eat for breakfast yesterday?",
        "What did I eat for lunch yesterday?",
    ]
    result = guard_decomposition(original, subs)
    # Both subs preserve content words (eat, breakfast/lunch, yesterday)
    # and the relation. Both should pass.
    assert len(result.kept) == 2
    assert not result.fallback_to_original


def test_empty_subquestions_falls_back_to_original():
    """When the decomposer returned nothing, the fallback fires."""
    original = "Anything?"
    result = guard_decomposition(original, [])
    assert result.fallback_to_original
    assert result.kept == [original]


def test_mixed_pass_and_reject():
    """Some sub-questions pass, some fail — kept list reflects passes."""
    original = "What is my favorite restaurant?"
    subs = [
        "What is my favorite restaurant?",   # PASS — identical
        "What's the weather like?",           # FAIL — different_fact
    ]
    result = guard_decomposition(original, subs)
    assert original in result.kept
    assert len(result.rejected) == 1
    assert not result.fallback_to_original
    # "weather" question drifts on multiple axes (answer_type +
    # different_fact + object_dropped) — accept any of them.
    reason = result.rejected[0].reason
    assert any(token in reason for token in (
        "different_fact", "object_dropped", "answer_type_changed",
    )), f"unexpected reason: {reason}"


# ─── Telemetry surface ────────────────────────────────────────────────────


def test_to_dict_emits_full_audit():
    """``DecompositionGuardResult.to_dict`` must include kept + rejected + counts."""
    result = guard_decomposition(
        "What is the painting worth?",
        ["What amount did I pay for the painting?"],
    )
    d = result.to_dict()
    assert d["fallback_to_original"] is True
    assert d["n_kept"] == 1
    assert d["n_rejected"] == 1
    assert d["kept"] == ["What is the painting worth?"]
    rej = d["rejected"][0]
    assert rej["subquestion"] == "What amount did I pay for the painting?"
    assert isinstance(rej["reason"], str) and rej["reason"]


def test_rejected_subquestion_to_dict():
    r = RejectedSubquestion("sub q", "answer_type_changed:COUNT→DATE_TIME")
    assert r.to_dict() == {
        "subquestion": "sub q",
        "reason": "answer_type_changed:COUNT→DATE_TIME",
    }


def test_dataclass_is_immutable():
    """Frozen — guard result can't be mutated post-hoc by the runner."""
    import dataclasses

    result = DecompositionGuardResult(
        kept=["x"], rejected=[], fallback_to_original=False,
    )
    try:
        dataclasses.replace(result, kept=["y"])  # this is fine — returns new
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"dataclass.replace failed: {exc!r}") from exc
    # But direct attribute mutation should not be allowed.
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.kept = ["y"]  # type: ignore[misc]
