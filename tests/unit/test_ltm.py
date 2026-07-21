"""
tests/unit/test_ltm.py
======================
Unit tests for ``continuum.stores.postgres.ltm.PostgresLTM``, driven by a
SQL-dispatching ``MockConnection`` (no psycopg3 / psycopg_pool / Postgres).

Covers: upsert (insert + by-id update), partial update, bi-temporal
invalidate (no DELETE), single-SQL dense⊕sparse⊕RRF search_hybrid, and the
recursive-CTE neighbors walk.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from continuum.core.protocols import LTMProtocol
from continuum.core.types import (
    BiTemporalRange,
    Edge,
    MemoryItem,
    MemoryTier,
    PrunePolicy,
    Query,
    ScoredItem,
)
from continuum.stores.postgres.ltm import PostgresLTM

pytestmark = pytest.mark.unit

N1 = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
N2 = uuid.UUID("00000000-0000-0000-0000-0000000000a2")
E1 = uuid.UUID("00000000-0000-0000-0000-0000000000e1")


# ---------------------------------------------------------------------------
# Mock DB
# ---------------------------------------------------------------------------


class MockCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class MockConnection:
    def __init__(
        self,
        *,
        search_rows: list[dict[str, Any]] | None = None,
        neighbor_rows: list[dict[str, Any]] | None = None,
        prune_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.prepared: list[bool] = []
        self._search = search_rows or []
        self._neighbors = neighbor_rows or []
        self._prune = prune_rows or []

    async def execute(
        self, sql: str, params: Any = None, *, prepare: bool = False, **_: Any
    ) -> MockCursor:
        self.executed.append((sql, params))
        self.prepared.append(prepare)
        if sql.startswith("SET pg_trgm.similarity_threshold"):
            return MockCursor()
        if "WITH RECURSIVE walk" in sql:
            return MockCursor(self._neighbors)
        if "ORDER  BY rrf DESC" in sql:
            return MockCursor(self._search)
        if "ORDER BY COALESCE(last_access" in sql:
            return MockCursor(self._prune)
        return MockCursor()  # INSERT / UPDATE


def _ltm(conn: MockConnection, **kw: Any) -> PostgresLTM:
    @asynccontextmanager
    async def _factory() -> AsyncIterator[MockConnection]:
        yield conn

    return PostgresLTM(conn_factory=_factory, **kw)


def _search_row(nid: uuid.UUID, text: str, rrf: float) -> dict[str, Any]:
    return {
        "id": str(nid),
        "text": text,
        "kind": "fact",
        "importance": 0.7,
        "confidence": 0.9,
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        "tags": {"topic": "x"},
        "rrf": rrf,
    }


# ---------------------------------------------------------------------------
# Construction / protocol
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_a_source(self) -> None:
        with pytest.raises(ValueError, match="dsn, pool, or conn_factory"):
            PostgresLTM()

    def test_embedding_type_validated(self) -> None:
        with pytest.raises(ValueError, match="bare identifier"):
            PostgresLTM(dsn="postgresql://x/y", embedding_type="halfvec;DROP")

    def test_satisfies_ltm_protocol(self) -> None:
        assert isinstance(_ltm(MockConnection()), LTMProtocol)


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    async def test_insert_new_returns_uuid_and_sets_ltm(self) -> None:
        conn = MockConnection()
        ltm = _ltm(conn)
        item = MemoryItem(
            id=str(N1), content="a durable fact", tier=MemoryTier.MTM, metadata={"kind": "fact"}
        )

        rid = await ltm.upsert(item)

        assert rid == N1
        assert item.tier == MemoryTier.LTM
        sql, params = conn.executed[0]
        assert "INSERT INTO memory_nodes" in sql
        assert "'LTM'" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql
        assert "updated_at  = now()" in sql
        assert params["id"] == str(N1)
        assert params["kind"] == "fact"

    async def test_update_existing_uses_same_id(self) -> None:
        conn = MockConnection()
        ltm = _ltm(conn)
        item = MemoryItem(id=str(N1), content="v1")
        assert await ltm.upsert(item) == N1
        item.content = "v2"
        assert await ltm.upsert(item) == N1  # same id → ON CONFLICT path
        assert all("ON CONFLICT (id) DO UPDATE" in s for s, _ in conn.executed)

    async def test_embedding_cast_vs_null(self) -> None:
        conn = MockConnection()
        ltm = _ltm(conn, embedding_type="halfvec")
        await ltm.upsert(MemoryItem(content="vec", embedding=[0.1, 0.2]))
        sql, params = conn.executed[0]
        assert "::halfvec" in sql and params["embedding"] == "[0.1,0.2]"

        conn2 = MockConnection()
        await _ltm(conn2).upsert(MemoryItem(content="no vec"))
        sql2, params2 = conn2.executed[0]
        assert "NULL" in sql2 and "embedding" not in params2

    async def test_valid_range_persisted(self) -> None:
        conn = MockConnection()
        vf = datetime(2025, 3, 1, tzinfo=UTC)
        await _ltm(conn).upsert(
            MemoryItem(content="bitemporal", valid_range=BiTemporalRange(valid_from=vf))
        )
        _, params = conn.executed[0]
        assert params["valid_from"] == vf
        assert params["valid_to"] is None

    async def test_non_uuid_id_gets_fresh_uuid(self) -> None:
        conn = MockConnection()
        item = MemoryItem(id="not-a-uuid", content="x")
        rid = await _ltm(conn).upsert(item)
        assert isinstance(rid, uuid.UUID)
        assert item.id == str(rid)


# ---------------------------------------------------------------------------
# update (partial)
# ---------------------------------------------------------------------------


class TestUpdate:
    async def test_only_allowlisted_fields_and_touches_updated_at(self) -> None:
        conn = MockConnection()
        await _ltm(conn).update(N1, {"importance": 0.95, "bogus": 1, "content": "new text"})
        sql, params = conn.executed[0]
        assert sql.startswith("UPDATE memory_nodes SET ")
        assert "importance = %(v_importance)s" in sql
        assert '"text" = %(v_content)s' in sql
        assert "updated_at = now()" in sql
        assert "WHERE id = %(id)s AND invalidated_at IS NULL" in sql
        assert "bogus" not in sql  # non-updatable ignored
        assert params["v_importance"] == 0.95
        assert params["v_content"] == "new text"

    async def test_embedding_and_tags_casts(self) -> None:
        conn = MockConnection()
        await _ltm(conn).update(N1, {"embedding": [1.0, 2.0], "tags": {"a": 1}})
        sql, params = conn.executed[0]
        assert "embedding = %(v_embedding)s::halfvec" in sql
        assert "tags = %(v_tags)s::jsonb" in sql
        assert params["v_embedding"] == "[1.0,2.0]"
        assert json.loads(params["v_tags"]) == {"a": 1}

    async def test_no_updatable_fields_is_noop(self) -> None:
        conn = MockConnection()
        await _ltm(conn).update(N1, {"unknown": 1})
        assert conn.executed == []  # no DB round-trip


# ---------------------------------------------------------------------------
# invalidate (bi-temporal — never deletes)
# ---------------------------------------------------------------------------


class TestInvalidate:
    async def test_sets_timestamp_no_delete(self) -> None:
        conn = MockConnection()
        when = datetime(2025, 6, 1, tzinfo=UTC)
        await _ltm(conn).invalidate(N1, when)
        sql, params = conn.executed[0]
        assert "UPDATE memory_nodes SET invalidated_at = %(at)s" in sql
        assert "DELETE" not in sql.upper()
        assert "invalidated_at IS NULL" in sql  # idempotent guard
        assert params == {"id": str(N1), "at": when}

    async def test_default_at_is_now(self) -> None:
        conn = MockConnection()
        await _ltm(conn).invalidate(str(N1))
        _, params = conn.executed[0]
        assert isinstance(params["at"], datetime)


# ---------------------------------------------------------------------------
# search_hybrid (single SQL: dense ⊕ sparse ⊕ RRF)
# ---------------------------------------------------------------------------


class TestSearchHybrid:
    async def test_both_channels_rrf_and_scoreditems(self) -> None:
        rows = [_search_row(N1, "best", 0.033), _search_row(N2, "next", 0.016)]
        conn = MockConnection(search_rows=rows)
        ltm = _ltm(conn)
        q = Query(text="db pooling", embedding=[0.1, 0.2], tiers=[MemoryTier.LTM])

        out = await ltm.search_hybrid(q, k=5)

        assert [type(s) for s in out] == [ScoredItem, ScoredItem]
        assert out[0].item.content == "best"
        assert out[0].score == pytest.approx(0.033)  # composite == rrf
        assert out[0].item.metadata["retrieval"]["rrf_score"] == pytest.approx(0.033)
        # Inspect the SQL the store built.
        hybrid_sql = next(s for s, _ in conn.executed if "ORDER  BY rrf DESC" in s)
        assert "WITH dense AS" in hybrid_sql
        assert "sparse AS" in hybrid_sql
        assert "1.0/(60 + d.rk)" in hybrid_sql
        assert "1.0/(60 + s.rk)" in hybrid_sql
        assert "n.invalidated_at IS NULL" in hybrid_sql
        assert "n.layer = ANY(%(layers)s)" in hybrid_sql
        # pg_trgm threshold set before the sparse query; query prepared.
        assert any(s.startswith("SET pg_trgm.similarity_threshold") for s, _ in conn.executed)
        assert conn.prepared[-1] is True

    async def test_dense_only_when_no_text(self) -> None:
        conn = MockConnection(search_rows=[_search_row(N1, "x", 0.01)])
        await _ltm(conn).search_hybrid(Query(text="", embedding=[0.1]), k=3)
        sql = next(s for s, _ in conn.executed if "ORDER  BY rrf DESC" in s)
        assert "dense AS" in sql
        assert "sparse AS" not in sql
        assert "+ 0)" in sql or "0 +" in sql  # sparse term collapses to 0

    async def test_sparse_only_when_no_embedding(self) -> None:
        conn = MockConnection(search_rows=[_search_row(N1, "x", 0.01)])
        await _ltm(conn).search_hybrid(Query(text="hello"), k=3)
        sql = next(s for s, _ in conn.executed if "ORDER  BY rrf DESC" in s)
        assert "sparse AS" in sql
        assert "dense AS" not in sql

    async def test_no_channels_returns_empty_no_db(self) -> None:
        conn = MockConnection()
        out = await _ltm(conn).search_hybrid(Query(text="   "), k=3)
        assert out == []
        assert conn.executed == []

    async def test_metadata_filter_and_as_of(self) -> None:
        conn = MockConnection(search_rows=[])
        as_of = datetime(2025, 1, 1, tzinfo=UTC)
        q = Query(text="hi", embedding=[0.1], metadata_filter={"lang": "py"}, as_of=as_of)
        await _ltm(conn).search_hybrid(q, k=2)
        sql, params = next((s, p) for s, p in conn.executed if "ORDER  BY rrf DESC" in s)
        assert "n.tags @> %(filt)s::jsonb" in sql
        assert "valid_from" in sql and "valid_to" in sql
        assert json.loads(params["filt"]) == {"lang": "py"}
        assert params["as_of"] == as_of

    async def test_layers_mapping(self) -> None:
        conn = MockConnection(search_rows=[])
        await _ltm(conn).search_hybrid(Query(text="x", tiers=[MemoryTier.LTM, MemoryTier.MTM]), k=2)
        _, params = next((s, p) for s, p in conn.executed if "ORDER  BY rrf DESC" in s)
        assert params["layers"] == ["LTM", "MTM"]  # .name, not .value


# ---------------------------------------------------------------------------
# neighbors (recursive CTE)
# ---------------------------------------------------------------------------


class TestNeighbors:
    def _edge_row(self, depth: int = 1) -> dict[str, Any]:
        return {
            "edge_id": str(E1),
            "source_id": str(N1),
            "target_id": str(N2),
            "predicate": "SUPPORTS",
            "weight": 0.8,
            "depth": depth,
            "text": "neighbour node",
            "kind": "fact",
            "importance": 0.6,
            "confidence": 0.9,
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "tags": {},
        }

    async def test_returns_edges_with_hydrated_targets(self) -> None:
        conn = MockConnection(neighbor_rows=[self._edge_row()])
        edges = await _ltm(conn).neighbors(N1, hops=2)

        assert len(edges) == 1
        e = edges[0]
        assert isinstance(e, Edge)
        assert e.source_id == N1 and e.target_id == N2
        assert e.predicate == "SUPPORTS" and e.weight == 0.8 and e.depth == 1
        assert e.target is not None and e.target.content == "neighbour node"

        sql, params = conn.executed[0]
        assert "WITH RECURSIVE walk" in sql
        assert "e.invalidated_at IS NULL" in sql  # edges bi-temporal
        assert "tn.invalidated_at IS NULL" in sql  # target nodes bi-temporal
        assert "NOT e.target_id = ANY(w.path)" in sql  # cycle guard
        assert "LIMIT  %(cap)s" in sql
        assert params["maxd"] == 2
        assert conn.prepared[0] is True

    async def test_depth_kwarg_overrides_hops(self) -> None:
        # ContinuumSession._incremental_index calls neighbors(id, depth=1).
        conn = MockConnection(neighbor_rows=[])
        await _ltm(conn).neighbors(str(N1), hops=9, depth=1)
        _, params = conn.executed[0]
        assert params["maxd"] == 1

    async def test_empty_graph(self) -> None:
        conn = MockConnection(neighbor_rows=[])
        assert await _ltm(conn).neighbors(N1) == []


# ---------------------------------------------------------------------------
# prune (forgetting)
# ---------------------------------------------------------------------------


def _prune_row(nid: uuid.UUID, text: str) -> dict[str, Any]:
    return {"id": str(nid), "text": text}


class TestPrune:
    @staticmethod
    def _policy(**kw: Any) -> PrunePolicy:
        kw.setdefault("unused_for", timedelta(days=90))
        return PrunePolicy(**kw)

    async def test_dry_run_reports_but_never_writes(self) -> None:
        # The safety property: a dry run must be readable proof of what WOULD
        # happen, not a preview that already happened.
        conn = MockConnection(prune_rows=[_prune_row(N1, "old fact"), _prune_row(N2, "older")])
        report = await _ltm(conn).prune(self._policy(), dry_run=True)

        assert report.matched == 2
        assert report.pruned == 0 and report.dry_run is True
        assert report.samples == ("old fact", "older")
        assert not any(sql.lstrip().startswith("UPDATE") for sql, _ in conn.executed)

    async def test_apply_invalidates_the_matched_rows(self) -> None:
        conn = MockConnection(prune_rows=[_prune_row(N1, "old fact")])
        report = await _ltm(conn).prune(self._policy(), dry_run=False)

        assert report.matched == 1 and report.pruned == 1 and report.dry_run is False
        update_sql, update_params = conn.executed[-1]
        assert update_sql.lstrip().startswith("UPDATE")
        assert "SET invalidated_at" in update_sql
        assert "DELETE" not in update_sql  # bi-temporal: retire, never destroy
        assert update_params["ids"] == [N1]
        # Re-pruning an already-retired row must be a no-op, as with invalidate().
        assert "invalidated_at IS NULL" in update_sql

    async def test_scoped_to_this_namespace(self) -> None:
        # Forgetting must never reach across tenants — the same bug class as the
        # cross-namespace recall leak.
        conn = MockConnection(prune_rows=[])
        await _ltm(conn, namespace="alice").prune(self._policy())
        sql, params = conn.executed[0]
        assert "namespace = %(ns)s" in sql and params["ns"] == "alice"

    async def test_only_superseded_facts_by_default(self) -> None:
        conn = MockConnection(prune_rows=[])
        await _ltm(conn).prune(self._policy())
        sql, params = conn.executed[0]
        assert "valid_to IS NOT NULL" in sql  # the fact is no longer true
        assert "valid_to <= %(at)s" in sql
        assert params["at"] is not None

    async def test_superseded_only_false_widens_the_net(self) -> None:
        conn = MockConnection(prune_rows=[])
        await _ltm(conn).prune(self._policy(superseded_only=False))
        sql, _ = conn.executed[0]
        assert "valid_to IS NOT NULL" not in sql

    async def test_unused_cutoff_falls_back_to_created_at(self) -> None:
        # A row nobody ever read has last_access NULL; it must still age out.
        now = datetime(2026, 7, 1, tzinfo=UTC)
        conn = MockConnection(prune_rows=[])
        await _ltm(conn).prune(self._policy(unused_for=timedelta(days=30)), now=now)
        sql, params = conn.executed[0]
        assert "COALESCE(last_access, created_at) <= %(cutoff)s" in sql
        assert params["cutoff"] == datetime(2026, 6, 1, tzinfo=UTC)

    async def test_coldest_first_within_the_limit(self) -> None:
        conn = MockConnection(prune_rows=[])
        await _ltm(conn).prune(self._policy(limit=25))
        sql, params = conn.executed[0]
        assert "ORDER BY COALESCE(last_access, created_at) ASC" in sql
        assert "LIMIT %(lim)s" in sql and params["lim"] == 25

    async def test_optional_ceilings_are_omitted_when_unset(self) -> None:
        conn = MockConnection(prune_rows=[])
        await _ltm(conn).prune(self._policy())
        sql, params = conn.executed[0]
        assert "importance" not in sql and "access_count" not in sql
        assert "max_imp" not in params and "max_acc" not in params

    async def test_ceilings_are_bound_when_set(self) -> None:
        conn = MockConnection(prune_rows=[])
        await _ltm(conn).prune(self._policy(max_importance=0.2, max_access_count=0))
        sql, params = conn.executed[0]
        assert "COALESCE(importance, 0) <= %(max_imp)s" in sql
        assert "COALESCE(access_count, 0) <= %(max_acc)s" in sql
        assert params["max_imp"] == 0.2 and params["max_acc"] == 0

    async def test_nothing_matched_writes_nothing(self) -> None:
        conn = MockConnection(prune_rows=[])
        report = await _ltm(conn).prune(self._policy(), dry_run=False)
        assert report.matched == 0 and report.pruned == 0
        assert len(conn.executed) == 1  # the SELECT only

    def test_policy_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError, match="unused_for"):
            PrunePolicy(unused_for=timedelta(days=-1))
        with pytest.raises(ValueError, match="limit"):
            PrunePolicy(unused_for=timedelta(days=1), limit=0)
