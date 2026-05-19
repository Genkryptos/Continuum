"""
tests/unit/test_lexical_hybrid.py
=================================
Unit tests for ``continuum.stores.postgres.retrieval``.

* ``reciprocal_rank_fusion`` is pure → tested directly (no DB).
* ``PostgresRetriever`` is driven by a SQL-dispatching ``MockConnection``
  injected via ``conn_factory`` — no psycopg3, no PostgreSQL.

The headline behavioural test (`hybrid recall > dense-only recall`) uses a
relevant-set the dense channel misses but the lexical channel finds.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from continuum.core.types import ScoredItem
from continuum.stores.postgres.retrieval import (
    FusionResult,
    PostgresRetriever,
    reciprocal_rank_fusion,
)

pytestmark = pytest.mark.unit

# Stable ids for deterministic assertions.
U1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
U2 = uuid.UUID("00000000-0000-0000-0000-000000000002")
U3 = uuid.UUID("00000000-0000-0000-0000-000000000003")
U4 = uuid.UUID("00000000-0000-0000-0000-000000000004")
U5 = uuid.UUID("00000000-0000-0000-0000-000000000005")


# ---------------------------------------------------------------------------
# Pure RRF
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_formula_uses_k_60(self) -> None:
        # U3 appears in both lists → score = 1/(60+1) + 1/(60+1).
        fused = reciprocal_rank_fusion([U3], [U3])
        assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 61)

    def test_doc_in_both_lists_outranks_doc_in_one(self) -> None:
        dense = [U1, U2, U3]
        sparse = [U3, U4]
        fused = reciprocal_rank_fusion(dense, sparse)
        # U3 is the only doc in BOTH → must rank first.
        assert fused[0].id == U3
        assert fused[0].dense_rank == 3
        assert fused[0].sparse_rank == 1
        assert fused[0].rrf_score == pytest.approx(1 / 63 + 1 / 61)

    def test_single_list_contribution_only(self) -> None:
        fused = {f.id: f for f in reciprocal_rank_fusion([U1], [U2])}
        assert fused[U1].rrf_score == pytest.approx(1 / 61)
        assert fused[U1].sparse_rank is None
        assert fused[U2].rrf_score == pytest.approx(1 / 61)
        assert fused[U2].dense_rank is None

    def test_top_k_truncation(self) -> None:
        fused = reciprocal_rank_fusion([U1, U2, U3], [U4, U5], top_k=2)
        assert len(fused) == 2

    def test_custom_rrf_k(self) -> None:
        fused = reciprocal_rank_fusion([U1], [], rrf_k=10)
        assert fused[0].rrf_score == pytest.approx(1 / 11)

    def test_duplicates_keep_best_rank(self) -> None:
        # U1 appears twice; only its first (rank 1) occurrence counts.
        fused = reciprocal_rank_fusion([U1, U2, U1], [])
        f = {x.id: x for x in fused}
        assert f[U1].dense_rank == 1
        assert f[U2].dense_rank == 2

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_sorted_descending_and_stable(self) -> None:
        fused = reciprocal_rank_fusion([U1, U2, U3], [U3, U2, U1])
        scores = [f.rrf_score for f in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Mock DB layer
# ---------------------------------------------------------------------------


class MockCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class MockConnection:
    """Dispatches by SQL fingerprint to canned dense / sparse / item rows."""

    def __init__(
        self,
        *,
        dense: list[tuple[uuid.UUID, float]] | None = None,
        sparse: list[tuple[uuid.UUID, float]] | None = None,
        items: dict[uuid.UUID, str] | None = None,
    ) -> None:
        self.dense = dense or []
        self.sparse = sparse or []
        self.items = items or {}
        self.executed: list[tuple[str, Any]] = []

    async def execute(self, sql: str, params: Any = None) -> MockCursor:
        self.executed.append((sql, params))

        if sql.startswith("SET pg_trgm.similarity_threshold"):
            return MockCursor()

        if "id = ANY(" in sql:  # hydration
            wanted = set(params["ids"])
            return MockCursor(
                [
                    {
                        "id": i,
                        "text": self.items[i],
                        "kind": "fact",
                        "importance": 0.7,
                        "confidence": 0.9,
                        "created_at": None,
                        "tags": {},
                    }
                    for i in self.items
                    if i in wanted
                ]
            )

        if "embedding <=>" in sql:  # dense KNN
            return MockCursor([{"id": i, "sim": s} for i, s in self.dense])

        if 'similarity("text"' in sql:  # lexical
            return MockCursor([{"id": i, "sim": s} for i, s in self.sparse])

        return MockCursor()


def _retriever(conn: MockConnection) -> PostgresRetriever:
    @asynccontextmanager
    async def _factory() -> AsyncIterator[MockConnection]:
        yield conn

    return PostgresRetriever(conn_factory=_factory)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_dsn_or_factory(self) -> None:
        with pytest.raises(ValueError, match="dsn or conn_factory"):
            PostgresRetriever()

    def test_defaults(self) -> None:
        r = PostgresRetriever(dsn="postgresql://x/y")
        assert r.rrf_k == 60
        assert r.trgm_threshold == 0.25


# ---------------------------------------------------------------------------
# Lexical search
# ---------------------------------------------------------------------------


class TestLexicalSearch:
    async def test_exact_match_found(self) -> None:
        conn = MockConnection(sparse=[(U1, 1.0), (U2, 0.4)])
        r = _retriever(conn)
        out = await r.lexical_search("connection pooling", k=10)
        assert out == [(U1, 1.0), (U2, 0.4)]

    async def test_fuzzy_typo_match_found(self) -> None:
        # The DB (pg_trgm) still ranks the intended row highly for a typo;
        # we simulate that ranked result and assert the function surfaces it.
        conn = MockConnection(sparse=[(U1, 0.55)])
        r = _retriever(conn)
        out = await r.lexical_search("conection poling", k=5)  # 2 typos
        assert out == [(U1, 0.55)]

    async def test_sql_filters_live_rows_and_uses_trgm(self) -> None:
        conn = MockConnection(sparse=[(U1, 0.9)])
        r = _retriever(conn)
        await r.lexical_search("hello", k=3)

        sqls = [s for s, _ in conn.executed]
        # threshold GUC set first
        assert any(s.startswith("SET pg_trgm.similarity_threshold = 0.25") for s in sqls)
        lex = next(s for s in sqls if 'similarity("text"' in s)
        assert "invalidated_at IS NULL" in lex
        assert '"text" %% %(q)s' in lex  # GIN trigram prefilter
        assert "ORDER  BY sim DESC" in lex
        assert "LIMIT  %(k)s" in lex

    async def test_empty_query_short_circuits(self) -> None:
        conn = MockConnection()
        r = _retriever(conn)
        assert await r.lexical_search("   ", k=5) == []
        assert conn.executed == []  # no DB round-trip


# ---------------------------------------------------------------------------
# Dense search
# ---------------------------------------------------------------------------


class TestDenseSearch:
    async def test_returns_pairs_and_uses_halfvec(self) -> None:
        conn = MockConnection(dense=[(U1, 0.95), (U2, 0.80)])
        r = _retriever(conn)
        out = await r.dense_search([0.1, 0.2, 0.3], k=10)
        assert out == [(U1, 0.95), (U2, 0.80)]

        dsql = next(s for s, _ in conn.executed if "embedding <=>" in s)
        assert "%(q)s::halfvec" in dsql
        assert "invalidated_at IS NULL" in dsql
        assert "embedding IS NOT NULL" in dsql

    async def test_empty_embedding_short_circuits(self) -> None:
        conn = MockConnection()
        r = _retriever(conn)
        assert await r.dense_search([], k=5) == []
        assert conn.executed == []


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    async def test_returns_scored_items_with_rrf_metadata(self) -> None:
        conn = MockConnection(
            dense=[(U1, 0.9), (U2, 0.7)],
            sparse=[(U2, 0.8), (U3, 0.6)],
            items={U1: "alpha", U2: "beta", U3: "gamma"},
        )
        r = _retriever(conn)
        results = await r.hybrid_search("q", [0.1, 0.2], k=3)

        assert all(isinstance(s, ScoredItem) for s in results)
        # U2 is in BOTH channels → highest RRF → first.
        assert results[0].item.content == "beta"
        meta = results[0].item.metadata["retrieval"]
        assert meta["dense_rank"] == 2
        assert meta["sparse_rank"] == 1
        assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
        # Sorted by RRF (composite) descending.
        scores = [s.score for s in results]
        assert scores == sorted(scores, reverse=True)

    async def test_runs_both_channels(self) -> None:
        conn = MockConnection(dense=[(U1, 0.9)], sparse=[(U2, 0.5)], items={U1: "a", U2: "b"})
        r = _retriever(conn)
        await r.hybrid_search("q", [0.1], k=2)
        sqls = [s for s, _ in conn.executed]
        assert any("embedding <=>" in s for s in sqls)  # dense ran
        assert any('similarity("text"' in s for s in sqls)  # lexical ran

    async def test_vanished_row_is_skipped(self) -> None:
        # U3 ranked by lexical but missing from items (invalidated between
        # ranking and hydration) → must be dropped, not crash.
        conn = MockConnection(
            dense=[(U1, 0.9)],
            sparse=[(U3, 0.9)],
            items={U1: "alpha"},  # U3 absent
        )
        r = _retriever(conn)
        results = await r.hybrid_search("q", [0.1], k=5)
        assert [s.item.content for s in results] == ["alpha"]

    async def test_hybrid_recall_beats_dense_only(self) -> None:
        """
        Relevant set = {U1, U5}. The dense channel never returns U5;
        the lexical channel does. Dense-only recall@2 = 0.5; hybrid
        recall@2 = 1.0.
        """
        relevant = {U1, U5}

        dense = [(U1, 0.91), (U2, 0.88), (U3, 0.80), (U4, 0.75)]  # no U5
        sparse = [(U5, 0.70), (U1, 0.40)]

        # Dense-only top-2:
        dense_only_top2 = {i for i, _ in dense[:2]}  # {U1, U2}
        dense_recall = len(dense_only_top2 & relevant) / len(relevant)
        assert dense_recall == 0.5

        conn = MockConnection(
            dense=dense,
            sparse=sparse,
            items={U1: "u1", U2: "u2", U3: "u3", U4: "u4", U5: "u5"},
        )
        r = _retriever(conn)
        hybrid = await r.hybrid_search("q", [0.1, 0.2], k=2)
        hybrid_top2 = {s.item.id for s in hybrid}

        hybrid_recall = len({uuid.UUID(i) for i in hybrid_top2} & relevant) / len(relevant)

        assert hybrid_recall == 1.0
        assert hybrid_recall > dense_recall


# ---------------------------------------------------------------------------
# FusionResult dataclass
# ---------------------------------------------------------------------------


def test_fusion_result_fields() -> None:
    fr = FusionResult(id=U1, rrf_score=0.5, dense_rank=1, sparse_rank=None)
    assert fr.id == U1
    assert fr.rrf_score == 0.5
    assert fr.dense_rank == 1
    assert fr.sparse_rank is None
