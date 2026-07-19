"""
tests/unit/test_retriever.py
============================
Unit tests for ``continuum.retrieval.retriever.Retriever`` — the full
pipeline, driven by injected fakes (no DB / LLM / model / network).
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest

from continuum.core.config import RetrieverConfig
from continuum.core.protocols import RetrieverProtocol
from continuum.core.types import (
    Edge,
    MemoryItem,
    MemoryTier,
    Query,
    ScoreBreakdown,
    ScoredItem,
    SummaryBlock,
    TokenBudget,
)
from continuum.retrieval import Retriever

pytestmark = pytest.mark.unit

BUDGET = TokenBudget(
    total=8000,
    stm_reserved=1000,
    mtm_reserved=2000,
    ltm_reserved=2000,
    response_reserved=1000,
)


# ---------------------------------------------------------------------------
# Builders / fakes
# ---------------------------------------------------------------------------


def _mi(
    content: str, *, kind: str | None = None, role: str | None = None, mid: uuid.UUID | None = None
) -> MemoryItem:
    meta: dict[str, Any] = {}
    if kind:
        meta["kind"] = kind
    if role:
        meta["role"] = role
    return MemoryItem(
        id=str(mid or uuid.uuid4()),
        content=content,
        tier=MemoryTier.LTM,
        metadata=meta,
    )


def _si(content: str, composite: float, **kw: Any) -> ScoredItem:
    item = _mi(content, **kw)
    return ScoredItem(
        item=item,
        scores=ScoreBreakdown(
            relevance=composite, importance=0.0, recency=0.0, confidence=0.0, composite=composite
        ),
    )


class FakeLTM:
    def __init__(
        self,
        hits: list[ScoredItem] | None = None,
        neighbors: dict[str, list[Edge]] | None = None,
        *,
        raises: bool = False,
    ) -> None:
        self._hits = hits or []
        self._nbrs = neighbors or {}
        self.raises = raises
        self.search_queries: list[str] = []

    async def search_hybrid(self, q: Any, k: int) -> list[ScoredItem]:
        if self.raises:
            raise RuntimeError("ltm down")
        self.search_queries.append(q.text)
        return self._hits[:k]

    async def neighbors(self, node_id: Any, hops: int = 1) -> list[Edge]:
        return self._nbrs.get(str(node_id), [])


class FakeSTM:
    def __init__(self, items: list[MemoryItem], *, raises: bool = False) -> None:
        self._items = items
        self.raises = raises
        self.n: int | None = None

    async def get_recent(self, sid: str, n: int) -> list[MemoryItem]:
        if self.raises:
            raise RuntimeError("stm down")
        self.n = n
        return self._items[-n:]


class FakeMTM:
    def __init__(self, blocks: list[SummaryBlock], *, raises: bool = False) -> None:
        self._blocks = blocks
        self.raises = raises
        self.budget_seen: int | None = None

    async def recent(self, token_budget: int, *, session_id: Any = None) -> list[SummaryBlock]:
        if self.raises:
            raise RuntimeError("mtm down")
        self.budget_seen = token_budget
        return self._blocks


class FakeScorer:
    """breakdown().composite = configured map[content] (else 0)."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def breakdown(self, item: Any, query: Any, now: Any) -> ScoreBreakdown:
        c = self._scores.get(item.content, 0.0)
        return ScoreBreakdown(relevance=c, importance=0.0, recency=0.0, confidence=0.0, composite=c)

    def score(self, item: Any, query: Any, now: Any) -> float:
        return self._scores.get(item.content, 0.0)


class FakeReranker:
    """Re-orders by a configured per-content rank score (desc)."""

    def __init__(self, order: dict[str, float]) -> None:
        self._order = order
        self.called_with: str | None = None

    async def rerank(self, query: str, items: list[ScoredItem]) -> list[ScoredItem]:
        self.called_with = query
        return sorted(
            items,
            key=lambda s: self._order.get(s.item.content, 0.0),
            reverse=True,
        )


def _q(text: str = "what is X?", session: str | None = "s1") -> Query:
    return Query(text=text, session_id=session)


# ---------------------------------------------------------------------------
# Protocol / smoke
# ---------------------------------------------------------------------------


def test_satisfies_retriever_protocol() -> None:
    assert isinstance(Retriever(), RetrieverProtocol)


class TestRetrievalRelevanceSurvivesScoring:
    """Regression: the stores return hits WITHOUT their embedding vector, so the
    real Scorer computed ``cosine(query.embedding, None) == 0`` as the relevance
    for every item. Ranking then collapsed to importance/recency — the SAME
    results for every query. The hybrid (RRF) relevance must survive scoring."""

    @staticmethod
    def _hit(content: str, rrf: float, importance: float) -> ScoredItem:
        item = MemoryItem(
            content=content,
            tier=MemoryTier.LTM,
            importance=importance,
            embedding=None,  # what search_hybrid actually returns
        )
        return ScoredItem(
            item=item,
            scores=ScoreBreakdown(
                relevance=rrf, importance=0.0, recency=0.0, confidence=0.0, composite=rrf
            ),
        )

    async def test_best_match_wins_over_high_importance_distractors(self) -> None:
        # The true match has the top RRF score but the LOWEST importance. Under
        # the old behaviour importance won and it ranked last.
        ltm = FakeLTM(
            [
                self._hit("the answer", 0.0164, 0.0),
                self._hit("distractor A", 0.0161, 1.0),
                self._hit("distractor B", 0.0159, 1.0),
            ]
        )
        r = Retriever(RetrieverConfig(), ltm=ltm, stm=FakeSTM([]))  # real Scorer
        bundle = await r.retrieve(_q(), BUDGET)
        assert bundle.items[0].content == "the answer"

    async def test_custom_scorer_without_weights_keeps_its_own_order(self) -> None:
        # An injected scorer need not expose config.weights; its ordering wins.
        ltm = FakeLTM([self._hit("low rrf", 0.001, 0.0), self._hit("high rrf", 0.9, 0.0)])
        r = Retriever(
            RetrieverConfig(),
            ltm=ltm,
            stm=FakeSTM([]),
            scorer=FakeScorer({"low rrf": 1.0, "high rrf": 0.0}),
        )
        bundle = await r.retrieve(_q(), BUDGET)
        assert bundle.items[0].content == "low rrf"  # FakeScorer's order, not RRF


class TestFullPipeline:
    async def test_runs_and_returns_contextbundle(self) -> None:
        ltm = FakeLTM([_si("fact one", 0.9), _si("fact two", 0.5)])
        stm = FakeSTM([_mi("hello", role="user"), _mi("hi", role="assistant")])
        mtm = FakeMTM([SummaryBlock(text="project summary", id=uuid.uuid4())])
        r = Retriever(
            RetrieverConfig(),
            ltm=ltm,
            stm=stm,
            mtm=mtm,
            scorer=FakeScorer({"fact one": 0.9, "fact two": 0.5}),
        )

        bundle = await r.retrieve(_q(), BUDGET)

        assert bundle.budget is BUDGET
        assert {"stm", "mtm", "ltm"} == set(bundle.tier_breakdown)
        assert bundle.tier_breakdown["ltm"] > 0
        assert bundle.tier_breakdown["mtm"] > 0
        assert bundle.tier_breakdown["stm"] > 0
        # items = LTM facts → MTM summaries → STM turns (that order).
        assert bundle.items[0].content == "fact one"
        assert "project summary" in [i.content for i in bundle.items]
        assert bundle.items[-1].content == "hi"
        assert len(bundle.messages) == len(bundle.items)

    async def test_token_tracking_exact(self) -> None:
        # whitespace counter: word counts are deterministic.
        ltm = FakeLTM([_si("alpha beta gamma", 0.9)])  # 3
        stm = FakeSTM([_mi("one two", role="user")])  # 2
        mtm = FakeMTM([SummaryBlock(text="x y z w", id=uuid.uuid4())])  # 4
        r = Retriever(
            RetrieverConfig(),
            ltm=ltm,
            stm=stm,
            mtm=mtm,
            scorer=FakeScorer({"alpha beta gamma": 0.9}),
        )
        b = await r.retrieve(_q(), BUDGET)
        assert b.tier_breakdown == {"ltm": 3, "mtm": 4, "stm": 2}
        assert b.tokens_used == 9
        assert b.tokens_remaining == BUDGET.total - 9

    async def test_mtm_called_with_budget(self) -> None:
        mtm = FakeMTM([])
        r = Retriever(RetrieverConfig(), mtm=mtm)
        await r.retrieve(_q(), BUDGET)
        assert mtm.budget_seen == BUDGET.mtm_reserved

    async def test_stm_uses_config_turns(self) -> None:
        stm = FakeSTM([_mi(f"t{i}") for i in range(20)])
        r = Retriever(RetrieverConfig(stm_turns=4), stm=stm)
        await r.retrieve(_q(), BUDGET)
        assert stm.n == 4


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------


class TestGraphExpansion:
    async def test_adds_neighbours_of_entity_hits(self) -> None:
        ent = _si("Acme Corp", 0.9, kind="entity")
        nbr = _mi("Acme HQ is in NYC", mid=uuid.uuid4())
        ltm = FakeLTM(
            hits=[ent],
            neighbors={
                ent.item.id: [
                    Edge(source_id=uuid.UUID(ent.item.id), target_id=uuid.UUID(nbr.id), target=nbr)
                ]
            },
        )
        r = Retriever(
            RetrieverConfig(),
            ltm=ltm,
            scorer=FakeScorer({"Acme Corp": 0.9, "Acme HQ is in NYC": 0.7}),
        )
        b = await r.retrieve(_q(), BUDGET)
        contents = [i.content for i in b.items]
        assert "Acme HQ is in NYC" in contents  # neighbour pulled in
        assert b.debug_info["graph_added"] == 1
        nbr_item = next(i for i in b.items if i.content == "Acme HQ is in NYC")
        assert nbr_item.metadata["via_graph"] == ent.item.id

    async def test_non_entity_not_expanded(self) -> None:
        fact = _si("just a fact", 0.9, kind="fact")
        ltm = FakeLTM(
            hits=[fact],
            neighbors={
                fact.item.id: [
                    Edge(
                        source_id=uuid.uuid4(),
                        target_id=uuid.uuid4(),
                        target=_mi("should not appear"),
                    )
                ]
            },
        )
        r = Retriever(RetrieverConfig(), ltm=ltm, scorer=FakeScorer({"just a fact": 0.9}))
        b = await r.retrieve(_q(), BUDGET)
        assert b.debug_info["graph_added"] == 0
        assert "should not appear" not in [i.content for i in b.items]

    async def test_neighbour_dedup(self) -> None:
        shared = uuid.uuid4()
        ent = _si("Ent", 0.9, kind="entity", mid=shared)
        # neighbour has the SAME id as an existing hit → must not duplicate.
        dup = _mi("Ent", mid=shared)
        ltm = FakeLTM(
            hits=[ent],
            neighbors={str(shared): [Edge(source_id=shared, target_id=shared, target=dup)]},
        )
        r = Retriever(RetrieverConfig(), ltm=ltm, scorer=FakeScorer({"Ent": 0.9}))
        b = await r.retrieve(_q(), BUDGET)
        assert b.debug_info["graph_added"] == 0
        assert [i.content for i in b.items].count("Ent") == 1

    async def test_graph_disabled_when_n_zero(self) -> None:
        ent = _si("E", 0.9, kind="entity")
        ltm = FakeLTM(
            hits=[ent],
            neighbors={
                ent.item.id: [
                    Edge(source_id=uuid.uuid4(), target_id=uuid.uuid4(), target=_mi("nbr"))
                ]
            },
        )
        r = Retriever(RetrieverConfig(graph_expand_n=0), ltm=ltm, scorer=FakeScorer({"E": 0.9}))
        b = await r.retrieve(_q(), BUDGET)
        assert b.debug_info["graph_added"] == 0


# ---------------------------------------------------------------------------
# Scoring + reranking (NDCG)
# ---------------------------------------------------------------------------


def _ndcg(order: list[str], rel: dict[str, float], k: int) -> float:
    def dcg(seq: list[str]) -> float:
        return sum(rel.get(c, 0.0) / math.log2(i + 2) for i, c in enumerate(seq[:k]))

    ideal = sorted(order, key=lambda c: rel.get(c, 0.0), reverse=True)
    idcg = dcg(ideal)
    return dcg(order) / idcg if idcg else 0.0


class TestScoringRerank:
    async def test_sorted_by_score_then_reranked(self) -> None:
        # Scorer order: A(0.4) < B(0.6) < C(0.5)  →  scored: B, C, A
        # True relevance: A best. Reranker fixes it → A first.
        hits = [_si("A", 0.4), _si("B", 0.6), _si("C", 0.5)]
        ltm = FakeLTM(hits=hits)
        scorer = FakeScorer({"A": 0.4, "B": 0.6, "C": 0.5})
        rer = FakeReranker({"A": 1.0, "C": 0.5, "B": 0.1})  # A is truly best
        r = Retriever(RetrieverConfig(), ltm=ltm, scorer=scorer, reranker=rer)

        b = await r.retrieve(_q(), BUDGET)
        order = [i.content for i in b.items]

        assert rer.called_with == "what is X?"
        assert order[0] == "A"  # rerank moved A to top
        rel = {"A": 3.0, "C": 1.0, "B": 0.0}  # ground truth
        scored_order = ["B", "C", "A"]  # pre-rerank
        assert _ndcg(order, rel, 3) >= _ndcg(scored_order, rel, 3)
        assert _ndcg(order, rel, 3) == pytest.approx(1.0)

    async def test_ltm_top_k_limits_final(self) -> None:
        hits = [_si(f"f{i}", 1.0 - i * 0.01) for i in range(30)]
        r = Retriever(
            RetrieverConfig(ltm_top_k=5),
            ltm=FakeLTM(hits=hits),
            scorer=FakeScorer({f"f{i}": 1.0 - i * 0.01 for i in range(30)}),
        )
        b = await r.retrieve(_q(), BUDGET)
        # 5 LTM (capped) + 0 mtm + 0 stm
        assert b.tier_breakdown["ltm"] > 0
        assert len(b.items) == 5

    async def test_real_scorer_integration(self) -> None:
        # No fake scorer → default continuum.scoring.Scorer runs end to end.
        a = _si("aligned", 0.0)
        a.item.embedding = [1.0, 0.0]
        b_ = _si("orthogonal", 0.0)
        b_.item.embedding = [0.0, 1.0]
        ltm = FakeLTM(hits=[b_, a])
        r = Retriever(RetrieverConfig(), ltm=ltm)
        q = Query(text="q", embedding=[1.0, 0.0])
        bundle = await r.retrieve(q, BUDGET)
        assert bundle.items[0].content == "aligned"  # higher cosine wins


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------


class TestHyDE:
    async def test_off_by_default(self) -> None:
        called = {"n": 0}

        async def hyde(_q: Query) -> str:
            called["n"] += 1
            return "rewritten"

        ltm = FakeLTM([_si("x", 0.5)])
        r = Retriever(RetrieverConfig(), ltm=ltm, hyde_fn=hyde, scorer=FakeScorer({"x": 0.5}))
        await r.retrieve(_q("original"), BUDGET)
        assert called["n"] == 0
        assert ltm.search_queries == ["original"]

    async def test_enabled_rewrites_search_query(self) -> None:
        async def hyde(_q: Query) -> str:
            return "hypothetical answer doc"

        ltm = FakeLTM([_si("x", 0.5)])
        r = Retriever(
            RetrieverConfig(hyde_enabled=True), ltm=ltm, hyde_fn=hyde, scorer=FakeScorer({"x": 0.5})
        )
        b = await r.retrieve(_q("original"), BUDGET)
        assert ltm.search_queries == ["hypothetical answer doc"]
        assert b.debug_info["hyde"] is True


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGraceful:
    async def test_no_tiers_returns_empty_bundle(self) -> None:
        b = await Retriever(RetrieverConfig()).retrieve(_q(), BUDGET)
        assert b.items == [] and b.messages == []
        assert b.tokens_used == 0

    async def test_ltm_failure_still_returns_stm_mtm(self) -> None:
        ltm = FakeLTM(raises=True)
        stm = FakeSTM([_mi("recent", role="user")])
        r = Retriever(RetrieverConfig(), ltm=ltm, stm=stm)
        b = await r.retrieve(_q(), BUDGET)
        assert b.debug_info["ltm_hybrid"] == "failed"
        assert "recent" in [i.content for i in b.items]  # STM survived

    async def test_reranker_failure_keeps_scored_order(self) -> None:
        class BadReranker:
            async def rerank(self, q: str, items: list[Any]) -> list[Any]:
                raise RuntimeError("model oom")

        ltm = FakeLTM([_si("hi", 0.9), _si("lo", 0.1)])
        r = Retriever(
            RetrieverConfig(),
            ltm=ltm,
            reranker=BadReranker(),
            scorer=FakeScorer({"hi": 0.9, "lo": 0.1}),
        )
        b = await r.retrieve(_q(), BUDGET)
        assert b.debug_info["reranked"] == "failed"
        assert b.items[0].content == "hi"  # scored order kept

    async def test_stm_mtm_failures_isolated(self) -> None:
        ltm = FakeLTM([_si("fact", 0.9)])
        r = Retriever(
            RetrieverConfig(),
            ltm=ltm,
            stm=FakeSTM([], raises=True),
            mtm=FakeMTM([], raises=True),
            scorer=FakeScorer({"fact": 0.9}),
        )
        b = await r.retrieve(_q(), BUDGET)
        assert [i.content for i in b.items] == ["fact"]  # LTM only, no crash
        assert b.tier_breakdown == {"stm": 0, "mtm": 0, "ltm": 1}
