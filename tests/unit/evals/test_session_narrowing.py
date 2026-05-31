"""
tests/unit/evals/test_session_narrowing.py
==========================================
Tests for :mod:`evals.longmemeval.session_narrowing`.

Acceptance scenarios from the spec:

* When the expected session is rank 1 and ``partial_recall = 1.0``,
  unrelated ShareGPT / UltraChat sessions must not enter the final
  answer context.
* Recall is preserved on multi-session questions (the narrowing
  bypass fires).
* Narrowing is refused when the top session has no verified evidence
  or the verifier's confidence is too low — accuracy can't improve
  by silently dropping evidence.

Plus per-branch unit tests and a telemetry round-trip.
"""

from __future__ import annotations

from continuum.core.types import MemoryItem
from evals.longmemeval.candidates import Candidate
from evals.longmemeval.session_narrowing import (
    NarrowResult,
    narrow_to_top_session,
)


def _item(session_id: str, *, content: str = "", item_id: str = "") -> MemoryItem:
    return MemoryItem(
        id=item_id or f"item-{session_id}",
        content=content or f"content from {session_id}",
        session_id=session_id,
        metadata={"session_id": session_id, "role": "user"},
    )


def _verified(
    session_id: str,
    *,
    value: str = "verified value",
    confidence: float = 0.8,
    subject: str = "x",
    relation: str = "got_at",
    answer_type: str = "location",
    candidate_type: str = "location",
) -> Candidate:
    return Candidate(
        value=value,
        normalized_value=value.lower(),
        candidate_type=candidate_type,  # type: ignore[arg-type]
        answer_type=answer_type,
        subject=subject,
        relation=relation,
        object_=value,
        source_session_id=session_id,
        confidence=confidence,
        claim=f"claim from {session_id}",
        source_span=value,
    )


# ─── ACCEPTANCE — unrelated sessions are dropped when top is verified ─────


def test_unrelated_sessions_dropped_when_top_session_verified():
    """
    The retrieval bundle contains:
      rank 1: expected session (S_correct) — has a verified candidate
      rank 2: ShareGPT distractor (S_chat_1)
      rank 3: UltraChat distractor (S_chat_2)

    After narrowing, only items from S_correct survive. The dropped
    session ids are recorded for telemetry.
    """
    items = [
        _item("S_correct", content="user: I got my Bachelor's at UCLA."),
        _item("S_chat_1", content="user: Hi, how are you today?"),
        _item("S_chat_2", content="assistant: Sure, I can help with that."),
    ]
    verified = [
        _verified("S_correct", value="UCLA", confidence=0.9),
    ]
    result = narrow_to_top_session(items, verified)

    assert result.narrowed
    assert result.chosen_session_id == "S_correct"
    assert result.dropped_session_ids == ["S_chat_1", "S_chat_2"]
    assert len(result.items) == 1
    assert result.items[0].metadata["session_id"] == "S_correct"
    assert result.reason == "narrowed_to_top_session"
    assert result.verified_confidence == 0.9


def test_narrow_preserves_all_items_from_top_session():
    """When the top session contributed multiple items, all of them survive."""
    items = [
        _item("S1", content="rank 1 — first window", item_id="s1-a"),
        _item("S2", content="rank 2 — distractor", item_id="s2-a"),
        _item("S1", content="rank 3 — second S1 window", item_id="s1-b"),
    ]
    verified = [_verified("S1", confidence=0.8)]
    result = narrow_to_top_session(items, verified)

    assert result.narrowed
    assert result.chosen_session_id == "S1"
    assert [it.id for it in result.items] == ["s1-a", "s1-b"]


def test_narrow_keeps_rank_order_within_top_session():
    """Narrowing must not reshuffle items."""
    items = [
        _item("S1", item_id="s1-A"),
        _item("S1", item_id="s1-B"),
        _item("S2", item_id="s2-A"),
    ]
    verified = [_verified("S1", confidence=0.7)]
    result = narrow_to_top_session(items, verified)
    assert [it.id for it in result.items] == ["s1-A", "s1-B"]


# ─── Bypass paths — narrowing must not silently drop evidence ─────────────


def test_multi_session_hint_bypasses_narrowing():
    """Explicit multi-session questions keep all sessions."""
    items = [
        _item("S1"),
        _item("S2"),
        _item("S3"),
    ]
    verified = [_verified("S1", confidence=0.95)]  # high conf but irrelevant
    result = narrow_to_top_session(
        items,
        verified,
        multi_session_hint=True,
    )
    assert not result.narrowed
    assert result.reason == "multi_session_hint"
    assert len(result.items) == 3
    # Every session id survives.
    sids = {it.metadata["session_id"] for it in result.items}
    assert sids == {"S1", "S2", "S3"}


def test_top_session_unverified_keeps_all():
    """No verified candidate from S1 → don't narrow."""
    items = [_item("S1"), _item("S2")]
    verified = [_verified("S2", confidence=0.9)]  # verifier passed S2, not S1
    result = narrow_to_top_session(items, verified)
    assert not result.narrowed
    assert result.reason == "top_session_unverified"
    assert len(result.items) == 2


def test_top_session_low_confidence_keeps_all():
    """Verified at top, but below confidence floor → don't narrow."""
    items = [_item("S1"), _item("S2")]
    verified = [_verified("S1", confidence=0.3)]  # below default 0.5 floor
    result = narrow_to_top_session(items, verified, min_verified_confidence=0.5)
    assert not result.narrowed
    assert "top_session_confidence_below_threshold" in result.reason
    assert result.verified_confidence == 0.3
    assert len(result.items) == 2


def test_no_verified_candidates_keeps_all():
    """Verifier emptied the list entirely → narrowing is unsafe."""
    items = [_item("S1"), _item("S2")]
    result = narrow_to_top_session(items, [])
    assert not result.narrowed
    assert result.reason == "top_session_unverified"
    assert len(result.items) == 2


def test_single_session_only_skips_narrowing():
    """When everything's already from one session, there's nothing to drop."""
    items = [_item("S1", item_id="a"), _item("S1", item_id="b")]
    verified = [_verified("S1", confidence=0.95)]
    result = narrow_to_top_session(items, verified)
    assert not result.narrowed
    assert result.reason == "single_session_only"
    assert result.chosen_session_id == "S1"
    assert len(result.items) == 2


def test_empty_items_returns_no_op():
    result = narrow_to_top_session([], [])
    assert not result.narrowed
    assert result.reason == "no_items"
    assert result.items == []
    assert result.dropped_session_ids == []


# ─── Threshold + selection details ────────────────────────────────────────


def test_picks_highest_confidence_when_top_has_multiple_verified():
    """Two verified candidates from S1 — picks the higher confidence."""
    items = [_item("S1"), _item("S2")]
    verified = [
        _verified("S1", value="A", confidence=0.55),
        _verified("S1", value="B", confidence=0.85),
        _verified("S2", value="C", confidence=0.99),  # irrelevant
    ]
    result = narrow_to_top_session(items, verified)
    assert result.narrowed
    assert result.verified_confidence == 0.85


def test_threshold_is_inclusive():
    """A verified candidate exactly at the threshold passes."""
    items = [_item("S1"), _item("S2")]
    verified = [_verified("S1", confidence=0.5)]
    result = narrow_to_top_session(items, verified, min_verified_confidence=0.5)
    assert result.narrowed
    assert result.verified_confidence == 0.5


def test_higher_threshold_blocks_narrowing():
    items = [_item("S1"), _item("S2")]
    verified = [_verified("S1", confidence=0.6)]
    result = narrow_to_top_session(items, verified, min_verified_confidence=0.7)
    assert not result.narrowed
    assert "below_threshold" in result.reason


# ─── Edge cases ────────────────────────────────────────────────────────────


def test_items_without_session_id_grouped_together():
    """Items lacking a session_id share the empty-string bucket."""
    items = [
        MemoryItem(id="anon-1", content="orphan", session_id=None, metadata={}),
        _item("S1"),
    ]
    verified = [_verified("S1", confidence=0.9)]
    result = narrow_to_top_session(items, verified)
    # Top session is the empty-string bucket because anon-1 ranked first.
    assert result.chosen_session_id == ""
    assert not result.narrowed
    assert result.reason == "top_session_unverified"


def test_narrow_telemetry_records_dropped_sessions():
    """The dropped session ids are emitted as a list in the telemetry dict."""
    items = [_item("S1"), _item("S2"), _item("S3")]
    verified = [_verified("S1", confidence=0.95)]
    result = narrow_to_top_session(items, verified)
    d = result.to_dict()
    assert d["narrowed"] is True
    assert d["chosen_session_id"] == "S1"
    assert d["dropped_session_ids"] == ["S2", "S3"]
    assert d["verified_confidence"] == 0.95
    assert d["n_input_sessions"] == 3
    assert d["n_output_items"] == 1
    assert d["reason"] == "narrowed_to_top_session"


def test_narrow_result_is_immutable():
    """``NarrowResult`` is frozen so trace consumers can't mutate it."""
    import dataclasses

    import pytest

    items = [_item("S1")]
    verified = [_verified("S1", confidence=0.9)]
    result = narrow_to_top_session(items, verified)
    # Default-frozen dataclass — attribute writes raise
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.narrowed = False  # type: ignore[misc]


# ─── Full trace pattern (the spec acceptance) ──────────────────────────────


def test_realistic_trace_expected_at_rank1_with_full_recall():
    """
    Mirror of the production trace pattern: 5 retrieved items, top
    item is from the expected session, the other 4 are noisy chat
    distractors. ``partial_recall = 1.0`` means the expected session
    is present — narrowing must drop the distractors and keep ONLY
    the answer-bearing window.
    """
    items = [
        _item("expected", content="user: My favorite restaurant is Lucca."),
        _item("sharegpt_a", content="user: explain transformers"),
        _item("sharegpt_b", content="user: write me a poem about cats"),
        _item("ultrachat_a", content="assistant: Hello, how can I help?"),
        _item("ultrachat_b", content="user: what's the capital of France?"),
    ]
    verified = [
        _verified(
            "expected",
            value="Lucca",
            confidence=0.85,
            subject="favorite restaurant",
            relation="is",
            answer_type="entity",
            candidate_type="entity",
        ),
    ]
    result = narrow_to_top_session(items, verified)
    assert result.narrowed
    assert result.chosen_session_id == "expected"
    # All four distractor sessions are recorded as dropped.
    assert set(result.dropped_session_ids) == {
        "sharegpt_a",
        "sharegpt_b",
        "ultrachat_a",
        "ultrachat_b",
    }
    # The model now only sees one item — the answer-bearing window.
    assert len(result.items) == 1
    assert "Lucca" in result.items[0].content


def test_narrow_result_dataclass_shape():
    """Smoke test the construction of NarrowResult independently."""
    nr = NarrowResult(
        items=[],
        chosen_session_id="s1",
        dropped_session_ids=["s2"],
        narrowed=True,
        reason="narrowed_to_top_session",
        verified_confidence=0.9,
        n_input_sessions=2,
    )
    assert nr.narrowed
    assert nr.to_dict()["chosen_session_id"] == "s1"
