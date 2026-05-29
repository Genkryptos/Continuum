"""
tests/unit/evals/test_iterative_recall.py
=========================================
Regression test for the "recall is always 0 under --reasoner iterative"
bug.

Root cause: ``_IterativeReasoningAdapter`` delegated retrieval to an
internal shim and never set ``adapter.last_ctx``. The baseline runner
derives ``retrieved_session_ids`` (and therefore recall) from
``adapter.last_ctx.items[*].metadata["session_id"]`` — with ``last_ctx``
left at ``None`` it returned ``[]`` for every row.

The fix is :func:`_merge_retrieved_contexts`, which unions every
sub-question's retrieved bundle (deduping by item id) while preserving
each item's metadata so the session ids survive.
"""
from __future__ import annotations

import uuid

from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, TokenBudget
from evals.longmemeval.baseline import _extract_retrieved_session_ids
from evals.longmemeval.bootstrap_ollama import _merge_retrieved_contexts

import pytest

pytestmark = pytest.mark.unit


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000, stm_reserved=500, mtm_reserved=500,
        ltm_reserved=2000, response_reserved=500,
    )


def _item(session_id: str, content: str = "x") -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=content,
        tier=MemoryTier.LTM,
        metadata={"role": "user", "session_id": session_id},
    )


def _bundle(*session_ids: str) -> ContextBundle:
    items = [_item(s) for s in session_ids]
    return ContextBundle(
        items=items, messages=[], tokens_used=0, budget=_budget(),
    )


def test_merge_preserves_session_ids_across_subqs() -> None:
    """The union of two sub-question bundles exposes every session id."""
    contexts = [_bundle("answer_a", "noise_1"), _bundle("answer_b", "noise_2")]
    merged = _merge_retrieved_contexts(contexts, _budget())
    assert merged is not None
    sids = {it.metadata["session_id"] for it in merged.items}
    assert sids == {"answer_a", "noise_1", "answer_b", "noise_2"}


def test_merge_dedups_by_item_id() -> None:
    """A session surfacing under two sub-questions is counted once."""
    shared = _item("answer_a")
    b1 = ContextBundle(items=[shared], messages=[], tokens_used=0, budget=_budget())
    b2 = ContextBundle(items=[shared], messages=[], tokens_used=0, budget=_budget())
    merged = _merge_retrieved_contexts([b1, b2], _budget())
    assert merged is not None
    assert len(merged.items) == 1


def test_merge_empty_returns_none() -> None:
    assert _merge_retrieved_contexts([], _budget()) is None


def test_recall_extraction_sees_merged_session_ids() -> None:
    """
    End-to-end of the bug: a fake adapter whose ``last_ctx`` is the
    merged bundle yields the expected retrieved_session_ids — proving
    recall would now be non-zero.
    """
    merged = _merge_retrieved_contexts(
        [_bundle("answer_d61669c7", "distractor")], _budget(),
    )

    class _FakeAdapter:
        last_ctx = merged

    ids = _extract_retrieved_session_ids(_FakeAdapter())
    assert "answer_d61669c7" in ids
    assert "distractor" in ids


def test_recall_extraction_empty_when_last_ctx_none() -> None:
    """The pre-fix behaviour: no last_ctx → empty ids → recall 0."""
    class _FakeAdapter:
        last_ctx = None

    assert _extract_retrieved_session_ids(_FakeAdapter()) == []
