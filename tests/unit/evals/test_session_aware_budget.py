"""
Budget policy and presentation order for the session-aware retriever.

The 500-question run showed ``items_offered = items_admitted = 8.0`` and
``items_dropped = 0.0`` on *every* ``missing_aggregation`` failure: the
reader never had more than the fixed ``session_top_k × turns_per_session``
budget and discarded none of it. These tests pin the two levers that
follow from that — a global turn re-rank, and ``chrono_sort`` actually
taking effect on this code path — plus the ``max_items`` floor that keeps
the cosine arm of the hybrid from being starved relative to BM25.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from continuum.core.types import MemoryItem, Query, TokenBudget
from evals.longmemeval.bootstrap_ollama import (
    FlatHaystackStore,
    SessionAwareSemanticRetriever,
)


class _ScriptedEmbedder:
    """Maps text -> a vector whose cosine against the query is a fixed score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t == "QUERY":
                rows.append([1.0, 0.0])
            else:
                s = self.scores[t]
                rows.append([s, float(np.sqrt(max(0.0, 1.0 - s * s)))])
        return np.asarray(rows, dtype=np.float64)


# Three sessions, four turns each. Session A holds the three strongest
# turns overall — the shape a quota of 2-per-session cannot serve.
_SCORES = {
    "a0": 0.99,
    "a1": 0.95,
    "a2": 0.93,
    "a3": 0.10,
    "b0": 0.90,
    "b1": 0.20,
    "b2": 0.15,
    "b3": 0.05,
    "c0": 0.50,
    "c1": 0.12,
    "c2": 0.08,
    "c3": 0.02,
}


def _budget() -> TokenBudget:
    return TokenBudget(
        total=10_000,
        stm_reserved=10_000,
        mtm_reserved=0,
        ltm_reserved=0,
        response_reserved=0,
    )


def _store() -> FlatHaystackStore:
    store = FlatHaystackStore()
    for name in _SCORES:  # dict order == chronological haystack order
        store.items.append(
            MemoryItem(
                id=name,
                content=name,
                metadata={"session_id": name[0], "role": "user"},
            )
        )
    return store


def _bundle(**kwargs):
    r = SessionAwareSemanticRetriever(
        store=_store(),
        embedder=_ScriptedEmbedder(_SCORES),
        **kwargs,
    )
    r._refresh_index()
    return r, r._bundle_topk(Query(text="QUERY"), _budget())


def _contents(bundle) -> list[str]:
    return [i.content for i in bundle.items]


def test_quota_is_the_default_and_unchanged() -> None:
    """The pinned arm: two turns from each of the top two sessions."""
    _r, b = _bundle(session_top_k=2, turns_per_session=2, max_items=16, top_k=4)
    assert _contents(b) == ["a0", "a1", "b0", "b1"]


def test_quota_caps_evidence_that_clusters_in_one_session() -> None:
    """``a2`` outranks ``b1`` globally but the quota cannot admit it."""
    _r, b = _bundle(session_top_k=2, turns_per_session=2, max_items=16, top_k=4)
    assert "a2" not in _contents(b)
    assert _SCORES["a2"] > _SCORES["b1"]


def test_global_rerank_admits_the_strongest_turns_overall() -> None:
    """Same two sessions, same budget of 4 — but ranked across them."""
    _r, b = _bundle(
        session_top_k=2,
        turns_per_session=2,
        max_items=4,
        top_k=4,
        turn_rerank="global",
    )
    assert _contents(b) == ["a0", "a1", "a2", "b0"]


def test_global_rerank_still_respects_session_top_k() -> None:
    """Session C is outside the top two, so none of its turns appear."""
    _r, b = _bundle(
        session_top_k=2,
        turns_per_session=2,
        max_items=12,
        top_k=12,
        turn_rerank="global",
    )
    assert not any(c.startswith("c") for c in _contents(b))


@pytest.mark.parametrize("rerank", ["quota", "global"])
def test_chrono_sort_restores_timeline_order(rerank: str) -> None:
    """
    Regression: this override replaced the parent's whole selection step,
    so ``--chrono-sort`` was a silent no-op under
    ``--session-aware-retrieval`` — the pinned configuration.
    """
    _r, plain = _bundle(
        session_top_k=2,
        turns_per_session=2,
        max_items=4,
        top_k=4,
        turn_rerank=rerank,
    )
    _r2, chrono = _bundle(
        session_top_k=2,
        turns_per_session=2,
        max_items=4,
        top_k=4,
        turn_rerank=rerank,
        chrono_sort=True,
    )
    order = list(_SCORES)
    picked = _contents(chrono)
    assert sorted(picked) == sorted(_contents(plain))  # same selection
    assert picked == sorted(picked, key=order.index)  # chronological


def test_max_items_never_falls_below_top_k() -> None:
    """
    RRF fusion and the reader both truncate to ``top_k``. A ``max_items``
    under it supplies the cosine arm with fewer candidates than BM25 gets,
    which quietly shrinks the budget instead of widening it.
    """
    r, _b = _bundle(
        session_top_k=4,
        turns_per_session=2,
        max_items=4,
        top_k=12,
    )
    assert r.max_items == 12


def test_debug_info_records_the_arm() -> None:
    _r, b = _bundle(
        session_top_k=2,
        turns_per_session=2,
        max_items=4,
        top_k=4,
        turn_rerank="global",
        chrono_sort=True,
    )
    assert b.debug_info["turn_rerank"] == "global"
    assert b.debug_info["chrono_sort"] is True
    assert b.debug_info["max_items"] == 4


def test_retrieve_entrypoint_agrees_with_bundle() -> None:
    r = SessionAwareSemanticRetriever(
        store=_store(),
        embedder=_ScriptedEmbedder(_SCORES),
        session_top_k=2,
        turns_per_session=2,
        max_items=4,
        top_k=4,
        turn_rerank="global",
        mode="topk",
    )
    got = asyncio.run(r.retrieve(Query(text="QUERY"), _budget()))
    assert [i.content for i in got.items] == ["a0", "a1", "a2", "b0"]
