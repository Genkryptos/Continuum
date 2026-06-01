"""
tests/unit/retrieval/test_rrf.py
================================
Unit tests for :mod:`continuum.retrieval.rrf`.

The child retrievers are :class:`_StaticRetriever` instances that
return a pre-baked list of items so the test can assert the exact
fused order — no embedder, no index, no LLM in the loop.
"""

from __future__ import annotations

import uuid

import pytest

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    Query,
    TokenBudget,
)
from continuum.retrieval.rrf import (
    DEFAULT_K,
    HybridRetriever,
    ReciprocalRankFusion,
)

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


class _StaticRetriever:
    """Returns a pre-baked list of items as the ranked bundle."""

    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items

    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle:
        return ContextBundle(
            items=list(self._items),
            messages=[],
            tokens_used=0,
            budget=budget,
        )


class _BoomRetriever:
    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle:
        raise RuntimeError("boom")


# ── basic merge ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rrf_no_retrievers_returns_empty_bundle() -> None:
    rrf = ReciprocalRankFusion([])
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    assert bundle.items == []
    assert bundle.debug_info["rrf"] == "no retrievers"


@pytest.mark.asyncio
async def test_rrf_single_retriever_passthrough_order() -> None:
    a, b, c = _mi("A"), _mi("B"), _mi("C")
    rrf = ReciprocalRankFusion([_StaticRetriever([a, b, c])], top_k=10)
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    assert [it.id for it in bundle.items] == [a.id, b.id, c.id]


@pytest.mark.asyncio
async def test_rrf_score_formula_matches_specification() -> None:
    """An item that's rank 1 in both retrievers should beat one at rank 1
    in only one, even when the latter is at rank 2 in the other.

    Concrete scores with k=60:
      shared  → 1/61 + 1/61 = 0.03278…
      only_in_A → 1/61 + 0  = 0.01639…    (so shared > only_in_A)
      ranked2_in_B → 1/61 + 1/62 = 0.03252… < shared
    """
    shared = _mi("shared", id="shared")
    only_a = _mi("only-A", id="oA")
    only_b = _mi("only-B", id="oB")
    r_a = _StaticRetriever([shared, only_a])
    r_b = _StaticRetriever([shared, only_b])
    rrf = ReciprocalRankFusion([r_a, r_b], k=60, top_k=10)
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    assert bundle.items[0].id == "shared"
    # The two singletons tie on RRF score, but first_seen ordering
    # places only_a first because it appeared in retriever A.
    assert [it.id for it in bundle.items[1:]] == ["oA", "oB"]
    # Scores match the formula.
    expected_shared = 1 / 61 + 1 / 61
    expected_singleton = 1 / 62
    assert bundle.debug_info["rrf_scores"]["shared"] == pytest.approx(
        expected_shared,
        abs=1e-6,
    )
    assert bundle.debug_info["rrf_scores"]["oA"] == pytest.approx(
        expected_singleton,
        abs=1e-6,
    )


@pytest.mark.asyncio
async def test_rrf_tie_break_is_deterministic() -> None:
    """Two items that collected the same RRF mass tie-break by
    first-seen order so output is deterministic across runs."""
    x = _mi("x", id="x")
    y = _mi("y", id="y")
    r_a = _StaticRetriever([x, y])  # x rank 1, y rank 2
    r_b = _StaticRetriever([y, x])  # y rank 1, x rank 2
    rrf = ReciprocalRankFusion([r_a, r_b], k=60, top_k=10)
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    # Both have score 1/61 + 1/62. first_seen(x) == 0 (in r_a),
    # first_seen(y) == 1 — so x comes first.
    assert [it.id for it in bundle.items] == ["x", "y"]


@pytest.mark.asyncio
async def test_rrf_top_k_cap_applied_to_merged_list() -> None:
    items = [_mi(f"d{i}", id=f"d{i}") for i in range(5)]
    r_a = _StaticRetriever(items)
    r_b = _StaticRetriever(list(reversed(items)))  # disagree completely
    rrf = ReciprocalRankFusion([r_a, r_b], k=60, top_k=3)
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    assert len(bundle.items) == 3


@pytest.mark.asyncio
async def test_rrf_top_k_falls_back_to_query_top_k() -> None:
    items = [_mi(f"d{i}", id=f"d{i}") for i in range(5)]
    rrf = ReciprocalRankFusion([_StaticRetriever(items)], k=60)
    bundle = await rrf.retrieve(Query(text="x", top_k=2), _budget())
    assert len(bundle.items) == 2


# ── failure handling ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rrf_boom_retriever_contributes_nothing() -> None:
    a = _mi("A", id="A")
    rrf = ReciprocalRankFusion(
        [_StaticRetriever([a]), _BoomRetriever()],
        k=60,
        top_k=10,
    )
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    assert [it.id for it in bundle.items] == ["A"]


@pytest.mark.asyncio
async def test_rrf_all_boom_returns_empty_bundle() -> None:
    rrf = ReciprocalRankFusion([_BoomRetriever(), _BoomRetriever()])
    bundle = await rrf.retrieve(Query(text="x"), _budget())
    assert bundle.items == []


# ── factory ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_retriever_factory_uses_rrf_default_k() -> None:
    a = _mi("A", id="A")
    cosine = _StaticRetriever([a])
    bm25 = _StaticRetriever([a])
    hybrid = HybridRetriever(cosine, bm25)
    assert isinstance(hybrid, ReciprocalRankFusion)
    bundle = await hybrid.retrieve(Query(text="x"), _budget())
    assert bundle.debug_info["rrf_k"] == DEFAULT_K
    assert bundle.items[0].id == "A"


def test_constructor_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be > 0"):
        ReciprocalRankFusion([], k=0)
