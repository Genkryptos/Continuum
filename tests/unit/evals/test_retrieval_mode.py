"""
tests/unit/evals/test_retrieval_mode.py
=======================================
Unit tests for the adaptive retrieval mode in
``evals.longmemeval.bootstrap_ollama.STMSemanticRetriever``.

The retriever picks between two modes:
  * top-K cosine retrieval (small slice of a huge haystack)
  * long-context (the whole haystack, when it fits the window)

These tests use a fake embedder so they run offline and instantly —
no sentence-transformers download, no network.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from continuum.core.types import MemoryItem, MemoryTier, Query, TokenBudget
from evals.longmemeval.bootstrap_ollama import (
    FlatHaystackStore,
    STMSemanticRetriever,
)


class _FakeEmbedder:
    """Deterministic embedder — hashes text to a fixed 8-dim vector."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=8).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            rows.append(v)
        return np.stack(rows, axis=0)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000,
        stm_reserved=500,
        mtm_reserved=500,
        ltm_reserved=2000,
        response_reserved=500,
    )


async def _store_with(n: int, *, words_each: int = 5) -> FlatHaystackStore:
    store = FlatHaystackStore()
    for i in range(n):
        await store.append(
            MemoryItem(
                id=str(uuid.uuid4()),
                content=" ".join([f"word{i}"] * words_each),
                tier=MemoryTier.STM,
                metadata={"role": "user", "session_id": f"s{i}"},
            )
        )
    return store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        STMSemanticRetriever(
            store=FlatHaystackStore(),
            embedder=_FakeEmbedder(),
            mode="banana",
        )


# ---------------------------------------------------------------------------
# Mode resolution logic (_resolve_mode)
# ---------------------------------------------------------------------------


def test_topk_mode_always_resolves_topk() -> None:
    r = STMSemanticRetriever(
        store=FlatHaystackStore(),
        embedder=_FakeEmbedder(),
        mode="topk",
        max_context_tokens=100,
    )
    assert r._resolve_mode(total_tokens=10) == "topk"
    assert r._resolve_mode(total_tokens=10_000) == "topk"


def test_auto_mode_picks_long_when_it_fits() -> None:
    r = STMSemanticRetriever(
        store=FlatHaystackStore(),
        embedder=_FakeEmbedder(),
        mode="auto",
        max_context_tokens=1000,
    )
    assert r._resolve_mode(total_tokens=500) == "long"  # fits
    assert r._resolve_mode(total_tokens=5000) == "topk"  # doesn't fit


def test_long_mode_falls_back_to_topk_when_too_big() -> None:
    r = STMSemanticRetriever(
        store=FlatHaystackStore(),
        embedder=_FakeEmbedder(),
        mode="long",
        max_context_tokens=1000,
    )
    assert r._resolve_mode(total_tokens=900) == "long"  # fits
    assert r._resolve_mode(total_tokens=2000) == "topk"  # can't fit


# ---------------------------------------------------------------------------
# retrieve() behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topk_returns_at_most_top_k_items() -> None:
    store = await _store_with(30)
    r = STMSemanticRetriever(
        store=store,
        embedder=_FakeEmbedder(),
        top_k=4,
        mode="topk",
    )
    ctx = await r.retrieve(Query(text="word3"), _budget())
    assert len(ctx.items) == 4
    assert ctx.debug_info["retrieval_mode"] == "topk"


@pytest.mark.asyncio
async def test_long_mode_returns_all_items() -> None:
    store = await _store_with(30)
    r = STMSemanticRetriever(
        store=store,
        embedder=_FakeEmbedder(),
        top_k=4,
        mode="long",
        max_context_tokens=100_000,
    )
    ctx = await r.retrieve(Query(text="word3"), _budget())
    # All 30 items returned, not just top-4.
    assert len(ctx.items) == 30
    assert ctx.debug_info["retrieval_mode"] == "long"
    assert ctx.debug_info["long_context_items"] == 30
    assert ctx.debug_info["long_context_tokens"] > 0


@pytest.mark.asyncio
async def test_long_mode_preserves_chronological_order() -> None:
    store = await _store_with(10)
    r = STMSemanticRetriever(
        store=store,
        embedder=_FakeEmbedder(),
        mode="long",
        max_context_tokens=100_000,
    )
    ctx = await r.retrieve(Query(text="anything"), _budget())
    # Order matches insertion order (oldest first).
    assert [it.content.split()[0] for it in ctx.items] == [f"word{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_long_mode_falls_back_to_topk_over_budget() -> None:
    # 30 items × 5 words ≈ 150 tokens total; cap at 20 forces fallback.
    store = await _store_with(30, words_each=5)
    r = STMSemanticRetriever(
        store=store,
        embedder=_FakeEmbedder(),
        top_k=4,
        mode="long",
        max_context_tokens=20,
    )
    ctx = await r.retrieve(Query(text="word3"), _budget())
    assert len(ctx.items) == 4
    assert ctx.debug_info["retrieval_mode"] == "topk"


@pytest.mark.asyncio
async def test_auto_mode_long_for_small_haystack() -> None:
    store = await _store_with(8, words_each=3)
    r = STMSemanticRetriever(
        store=store,
        embedder=_FakeEmbedder(),
        top_k=4,
        mode="auto",
        max_context_tokens=100_000,
    )
    ctx = await r.retrieve(Query(text="word1"), _budget())
    assert ctx.debug_info["retrieval_mode"] == "long"
    assert len(ctx.items) == 8


@pytest.mark.asyncio
async def test_empty_store_returns_empty_bundle() -> None:
    r = STMSemanticRetriever(
        store=FlatHaystackStore(),
        embedder=_FakeEmbedder(),
        mode="auto",
    )
    ctx = await r.retrieve(Query(text="x"), _budget())
    assert ctx.items == []
    assert ctx.debug_info["retrieval_mode"] == "empty"
