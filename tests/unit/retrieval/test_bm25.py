"""
tests/unit/retrieval/test_bm25.py
=================================
Unit tests for :mod:`continuum.retrieval.bm25` — the pure
:class:`BM25Index` and the :class:`BM25Retriever` protocol shim.

Every test is hermetic — no fixtures touch disk, network, or any model
download. Items are tiny inline `MemoryItem` instances so the test runs
in milliseconds.
"""

from __future__ import annotations

import uuid

import pytest

from continuum.core.types import (
    MemoryItem,
    MemoryTier,
    Query,
    TokenBudget,
)
from continuum.retrieval.bm25 import BM25Index, BM25Retriever, tokenize

pytestmark = pytest.mark.unit


# ── helpers ────────────────────────────────────────────────────────────────


def _mi(content: str, *, id: str | None = None) -> MemoryItem:
    return MemoryItem(
        id=id or str(uuid.uuid4()),
        content=content,
        tier=MemoryTier.LTM,
    )


def _budget() -> TokenBudget:
    return TokenBudget(
        total=4000,
        stm_reserved=0,
        mtm_reserved=0,
        ltm_reserved=2000,
        response_reserved=100,
    )


# ── tokenize ───────────────────────────────────────────────────────────────


def test_tokenize_lowercases_and_splits_on_punct() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("Roscioli's pasta — Roman, Italian, sublime.") == [
        "roscioli",
        "s",
        "pasta",
        "roman",
        "italian",
        "sublime",
    ]


def test_tokenize_empty_inputs() -> None:
    assert tokenize("") == []
    assert tokenize("   \t\n") == []


def test_tokenize_keeps_digits() -> None:
    # Number-bearing queries (LongMemEval count questions) need to
    # surface ``5 eggs`` style facts — both tokens must survive.
    assert tokenize("Use 5 eggs and 2 tomatoes") == [
        "use",
        "5",
        "eggs",
        "and",
        "2",
        "tomatoes",
    ]


def test_tokenize_unicode_letters() -> None:
    # Accents must survive — LongMemEval rows include "café", "São Paulo", etc.
    assert tokenize("Café São Paulo") == ["café", "são", "paulo"]


# ── BM25Index ──────────────────────────────────────────────────────────────


def test_empty_corpus_returns_empty() -> None:
    """An empty corpus must short-circuit — rank_bm25 itself raises on []."""
    idx = BM25Index([])
    assert idx.is_empty is True
    assert idx.query("anything", k=10) == []


def test_empty_query_returns_empty() -> None:
    idx = BM25Index([_mi("Boston is the capital of Massachusetts.")])
    assert idx.query("", k=5) == []
    assert idx.query("   \n", k=5) == []


def test_exact_match_outranks_partial_overlap() -> None:
    """Rare-token matches dominate. The doc that mentions every query
    term should outrank the one that only shares 'eggs'."""
    items = [
        _mi("Use 5 eggs for the carbonara."),
        _mi("Eggs are nutritious."),
        _mi("Pasta is great with marinara sauce."),  # no overlap
    ]
    idx = BM25Index(items)
    hits = idx.query("how many eggs for carbonara", k=3)
    assert [si.item.content for si in hits[:1]] == [items[0].content]
    # The unrelated doc should be ranked last (or absent).
    assert items[2].content not in [si.item.content for si in hits[:1]]


def test_topk_cap_respected() -> None:
    items = [_mi(f"doc {i} term") for i in range(5)]
    idx = BM25Index(items)
    hits = idx.query("term", k=3)
    assert len(hits) == 3


def test_query_with_no_overlap_returns_zero_relevance() -> None:
    """All-misses → all zero relevance via the min-max guard."""
    items = [_mi("apples"), _mi("oranges"), _mi("bananas")]
    idx = BM25Index(items)
    hits = idx.query("xyzzy", k=3)
    # BM25 may still return some ordering, but relevance must be 0.0
    # because every raw score is identical (zero).
    assert hits  # docs returned in corpus order
    assert all(si.scores.relevance == 0.0 for si in hits)


def test_score_breakdown_normalized_to_unit_interval() -> None:
    """relevance ∈ [0, 1] for every hit, even with high raw BM25 scores."""
    items = [_mi("Boston Boston Boston")] + [_mi("filler") for _ in range(20)]
    idx = BM25Index(items)
    hits = idx.query("Boston", k=10)
    for si in hits:
        assert 0.0 <= si.scores.relevance <= 1.0


def test_stable_ordering_on_ties() -> None:
    """Identical content → tie-broken by corpus order (stable)."""
    items = [_mi(f"same text {i}", id=str(i)) for i in range(3)]
    # All three docs have one token in common with the query.
    idx = BM25Index(items)
    hits = idx.query("same", k=3)
    # Corpus order is preserved among ties.
    assert [si.item.content for si in hits] == [it.content for it in items]


# ── BM25Retriever (RetrieverProtocol shim) ─────────────────────────────────


@pytest.mark.asyncio
async def test_retriever_returns_empty_bundle_on_empty_corpus() -> None:
    async def provider(_q: Query) -> list[MemoryItem]:
        return []

    retr = BM25Retriever(corpus_provider=provider, name="bm25_test")
    bundle = await retr.retrieve(Query(text="anything"), _budget())
    assert bundle.items == []
    assert bundle.tokens_used == 0
    assert bundle.debug_info["retriever"] == "bm25_test"
    assert bundle.debug_info["corpus_size"] == 0


@pytest.mark.asyncio
async def test_retriever_ranks_top_k_from_corpus() -> None:
    items = [
        _mi("Boston is the capital of Massachusetts."),
        _mi("New York is in New York."),
        _mi("Pasta is a kind of food."),
    ]

    def provider(_q: Query) -> list[MemoryItem]:  # sync provider also OK
        return items

    retr = BM25Retriever(corpus_provider=provider, top_k=2)
    bundle = await retr.retrieve(Query(text="capital of Massachusetts"), _budget())
    assert len(bundle.items) == 2
    assert bundle.items[0].content == items[0].content


@pytest.mark.asyncio
async def test_retriever_swallows_provider_exception() -> None:
    async def provider(_q: Query) -> list[MemoryItem]:
        raise RuntimeError("db down")

    retr = BM25Retriever(corpus_provider=provider, name="bm25_test")
    bundle = await retr.retrieve(Query(text="x"), _budget())
    assert bundle.items == []
    assert "error" in bundle.debug_info


@pytest.mark.asyncio
async def test_retriever_uses_query_top_k_when_not_overridden() -> None:
    items = [_mi(f"doc {i} term") for i in range(10)]
    retr = BM25Retriever(corpus_provider=lambda _q: items)
    q = Query(text="term", top_k=4)
    bundle = await retr.retrieve(q, _budget())
    assert len(bundle.items) == 4
