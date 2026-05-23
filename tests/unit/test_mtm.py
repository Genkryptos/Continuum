"""
tests/unit/test_mtm.py
======================
Unit tests for ``continuum.stores.postgres.mtm.PostgresMTM`` and the
``SummaryBlock`` type, driven by a SQL-dispatching ``MockConnection``
(no psycopg3, no PostgreSQL, no tiktoken required).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from continuum.core.protocols import MTMProtocol
from continuum.core.types import MemoryItem, MemoryTier, ProcessingState, SummaryBlock
from continuum.stores.postgres.mtm import PostgresMTM, count_tokens

pytestmark = pytest.mark.unit

# Deterministic token counter for tests: 1 token per whitespace word.
WORDS = lambda s: len(s.split())  # noqa: E731


def _ts(sec: int) -> datetime:
    return datetime(2024, 1, 1, 12, 0, sec, tzinfo=UTC)


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
    """Records executes; serves canned rows for SELECT/scan; pages the scan."""

    def __init__(
        self,
        *,
        recent_rows: list[dict[str, Any]] | None = None,
        scan_pages: list[list[dict[str, Any]]] | None = None,
        dedup_row: dict[str, Any] | None = None,
    ) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._recent_rows = recent_rows or []
        # Each call to the scan SQL pops one page; empty list ⇒ no more.
        self._scan_pages = list(scan_pages or [])
        self._dedup_row = dedup_row

    async def execute(self, sql: str, params: Any = None) -> MockCursor:
        self.executed.append((sql, params))
        if "INSERT INTO" in sql:
            return MockCursor()
        if "LEFT" in sql and "JOIN" in sql:  # scan_unprocessed
            page = self._scan_pages.pop(0) if self._scan_pages else []
            return MockCursor(page)
        if "embedding <=>" in sql and "LIMIT  1" in sql:  # dedupe probe
            return MockCursor([self._dedup_row] if self._dedup_row else [])
        if "ORDER  BY created_at DESC" in sql:  # recent
            return MockCursor(self._recent_rows)
        return MockCursor()


def _mtm(conn: MockConnection, **kw: Any) -> PostgresMTM:
    @asynccontextmanager
    async def _factory() -> AsyncIterator[MockConnection]:
        yield conn

    kw.setdefault("token_counter", WORDS)
    return PostgresMTM(conn_factory=_factory, **kw)


def _node_row(
    *,
    idx: int,
    text: str,
    tokens: int,
    session: str | None = "s1",
    user: str | None = None,
) -> dict[str, Any]:
    tags: dict[str, Any] = {"tokens": tokens}
    if session is not None:
        tags["session_id"] = session
    if user is not None:
        tags["user_id"] = user
    return {
        "id": str(uuid.UUID(int=idx)),
        "text": text,
        "embedding": None,
        "created_at": _ts(idx),
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# SummaryBlock dataclass + conversions
# ---------------------------------------------------------------------------


class TestSummaryBlock:
    def test_defaults(self) -> None:
        b = SummaryBlock(text="hello world")
        assert isinstance(b.id, uuid.UUID)
        assert b.tokens == 0
        assert b.processed is False
        assert b.embedding is None

    def test_to_memory_item(self) -> None:
        bid = uuid.uuid4()
        b = SummaryBlock(
            text="abc",
            tokens=3,
            id=bid,
            session_id="s9",
            user_id="u7",
            processed=True,
        )
        mi = b.to_memory_item()
        assert mi.id == str(bid)
        assert mi.content == "abc"
        assert mi.tier == MemoryTier.MTM
        assert mi.session_id == "s9"
        assert mi.user_id == "u7"
        assert mi.processing_state == ProcessingState.PROCESSED
        assert mi.metadata["tokens"] == 3

    def test_from_memory_item_round_trip(self) -> None:
        mi = MemoryItem(
            content="round trip text",
            tier=MemoryTier.MTM,
            session_id="s1",
            user_id="u1",
            agent_id="a1",
            metadata={"tokens": 4, "k": "v"},
            processing_state=ProcessingState.UNPROCESSED,
        )
        b = SummaryBlock.from_memory_item(mi)
        assert b.text == "round trip text"
        assert b.tokens == 4
        assert b.session_id == "s1"
        assert b.user_id == "u1"
        assert b.agent_id == "a1"
        assert b.processed is False
        assert b.metadata["k"] == "v"
        assert str(b.id) == mi.id  # uuid preserved

    def test_from_memory_item_non_uuid_id_gets_fresh(self) -> None:
        mi = MemoryItem(id="not-a-uuid", content="x")
        b = SummaryBlock.from_memory_item(mi, tokens=7)
        assert isinstance(b.id, uuid.UUID)
        assert b.tokens == 7


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_is_zero(self) -> None:
        assert count_tokens("") == 0

    def test_nonempty_is_positive(self) -> None:
        assert count_tokens("the quick brown fox") > 0

    def test_longer_text_more_tokens(self) -> None:
        assert count_tokens("a " * 100) > count_tokens("a")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_dsn_or_factory(self) -> None:
        with pytest.raises(ValueError, match="dsn or conn_factory"):
            PostgresMTM()

    def test_embedding_type_validated(self) -> None:
        with pytest.raises(ValueError, match="bare identifier"):
            PostgresMTM(dsn="postgresql://x/y", embedding_type="halfvec; DROP")

    def test_satisfies_mtm_protocol(self) -> None:
        mtm = _mtm(MockConnection())
        assert isinstance(mtm, MTMProtocol)  # runtime_checkable, name-based


# ---------------------------------------------------------------------------
# add_summary
# ---------------------------------------------------------------------------


class TestAddSummary:
    async def test_inserts_mtm_row_and_returns_uuid(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        bid = uuid.uuid4()
        block = SummaryBlock(text="alpha beta", id=bid, session_id="s1", user_id="u1")

        returned = await mtm.add_summary(block)

        assert returned == bid
        sql, params = conn.executed[0]
        assert "INSERT INTO memory_nodes" in sql
        assert "'MTM'" in sql
        assert params["id"] == str(bid)
        tags = json.loads(params["tags"])
        assert tags["tokens"] == 2  # "alpha beta" → 2 words
        assert tags["session_id"] == "s1"
        assert tags["user_id"] == "u1"

    async def test_uses_existing_token_count(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        await mtm.add_summary(SummaryBlock(text="ignored words here", tokens=99))
        _, params = conn.executed[0]
        assert json.loads(params["tags"])["tokens"] == 99

    async def test_accepts_memory_item_from_flush_to(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        item = MemoryItem(
            content="from stm flush",
            tier=MemoryTier.STM,
            session_id="s2",
            user_id="u2",
        )
        returned = await mtm.add_summary(item)  # flush_to passes MemoryItem
        assert isinstance(returned, uuid.UUID)
        sql, params = conn.executed[0]
        assert "'MTM'" in sql
        assert json.loads(params["tags"])["session_id"] == "s2"
        assert json.loads(params["tags"])["user_id"] == "u2"

    async def test_embedding_cast_when_present(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn, embedding_type="halfvec")
        await mtm.add_summary(SummaryBlock(text="vec", embedding=[0.1, 0.2, 0.3]))
        sql, params = conn.executed[0]
        assert "::halfvec" in sql
        assert params["embedding"] == "[0.1,0.2,0.3]"

    async def test_embedding_null_when_absent(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        await mtm.add_summary(SummaryBlock(text="no vec"))
        sql, params = conn.executed[0]
        assert "NULL" in sql
        assert "embedding" not in params


class TestDedupeOnWrite:
    def test_threshold_validated(self) -> None:
        with pytest.raises(ValueError, match="dedupe_threshold"):
            PostgresMTM(dsn="postgresql://x/y", dedupe_threshold=1.5)

    async def test_duplicate_skips_insert_returns_existing(self) -> None:
        existing = uuid.uuid4()
        conn = MockConnection(dedup_row={"id": str(existing), "sim": 0.95})
        mtm = _mtm(conn, embedding_type="vector", dedupe_threshold=0.92)

        returned = await mtm.add_summary(SummaryBlock(text="near dup", embedding=[0.1, 0.2, 0.3]))

        assert returned == existing  # existing id, not a new one
        sqls = [s for s, _ in conn.executed]
        assert any("embedding <=>" in s for s in sqls)  # dedupe probe ran
        assert not any("INSERT INTO" in s for s in sqls)  # insert skipped

    async def test_below_threshold_inserts_new(self) -> None:
        conn = MockConnection(
            dedup_row={"id": str(uuid.uuid4()), "sim": 0.80}  # < 0.92
        )
        mtm = _mtm(conn, embedding_type="vector", dedupe_threshold=0.92)
        block = SummaryBlock(text="distinct", embedding=[0.9, 0.0, 0.0])

        returned = await mtm.add_summary(block)

        assert returned == block.id  # brand-new row
        assert any("INSERT INTO" in s for s, _ in conn.executed)

    async def test_no_embedding_skips_dedupe_probe(self) -> None:
        conn = MockConnection(dedup_row={"id": str(uuid.uuid4()), "sim": 1.0})
        mtm = _mtm(conn, dedupe_threshold=0.92)
        await mtm.add_summary(SummaryBlock(text="no embedding"))
        # No embedding → dedupe probe must be skipped, insert proceeds.
        assert not any("embedding <=>" in s for s, _ in conn.executed)
        assert any("INSERT INTO" in s for s, _ in conn.executed)


# ---------------------------------------------------------------------------
# recent (token budget)
# ---------------------------------------------------------------------------


class TestRecent:
    async def test_returns_blocks_within_budget(self) -> None:
        rows = [
            _node_row(idx=3, text="c", tokens=4),
            _node_row(idx=2, text="b", tokens=4),
            _node_row(idx=1, text="a", tokens=4),
        ]
        mtm = _mtm(MockConnection(recent_rows=rows))
        out = await mtm.recent(token_budget=10)
        # 4 + 4 = 8 ≤ 10; third (would be 12) excluded.
        assert [b.text for b in out] == ["c", "b"]
        assert all(isinstance(b, SummaryBlock) for b in out)

    async def test_single_oversized_newest_still_returned(self) -> None:
        rows = [_node_row(idx=1, text="huge", tokens=999)]
        mtm = _mtm(MockConnection(recent_rows=rows))
        out = await mtm.recent(token_budget=10)
        assert len(out) == 1
        assert out[0].text == "huge"

    async def test_zero_budget_returns_empty(self) -> None:
        conn = MockConnection(recent_rows=[_node_row(idx=1, text="x", tokens=1)])
        mtm = _mtm(conn)
        assert await mtm.recent(token_budget=0) == []
        assert conn.executed == []  # no query issued

    async def test_session_filter_in_sql(self) -> None:
        conn = MockConnection(recent_rows=[])
        mtm = _mtm(conn)
        await mtm.recent(token_budget=100, session_id="proj-x")
        sql, params = conn.executed[0]
        assert "tags->>'session_id' = %(sid)s" in sql
        assert params["sid"] == "proj-x"
        assert "ORDER  BY created_at DESC" in sql

    async def test_user_filter_and_excluded_session_in_sql(self) -> None:
        conn = MockConnection(recent_rows=[])
        mtm = _mtm(conn)
        await mtm.recent(
            token_budget=100,
            user_id="user-1",
            exclude_session_id="chat-current",
        )
        sql, params = conn.executed[0]
        assert "tags->>'user_id' = %(uid)s" in sql
        assert "tags->>'session_id' <> %(exclude_sid)s" in sql
        assert params["uid"] == "user-1"
        assert params["exclude_sid"] == "chat-current"


# ---------------------------------------------------------------------------
# scan_unprocessed
# ---------------------------------------------------------------------------


class TestScanUnprocessed:
    async def test_yields_only_unprocessed_blocks(self) -> None:
        page = [
            _node_row(idx=1, text="u1", tokens=2),
            _node_row(idx=2, text="u2", tokens=2),
        ]
        conn = MockConnection(scan_pages=[page])  # single short page → stop
        mtm = _mtm(conn)

        seen = [b async for b in mtm.scan_unprocessed()]

        assert [b.text for b in seen] == ["u1", "u2"]
        assert all(b.processed is False for b in seen)
        sql, _ = conn.executed[0]
        assert "LEFT" in sql and "JOIN memory_promotions" in sql
        assert "mp.id IS NULL" in sql  # the "no promotions row" predicate

    async def test_keyset_pagination_advances(self) -> None:
        full = [_node_row(idx=i, text=f"n{i}", tokens=1) for i in range(1, 4)]
        tail = [_node_row(idx=9, text="last", tokens=1)]
        conn = MockConnection(scan_pages=[full, tail])
        mtm = _mtm(conn)

        seen = [b async for b in mtm.scan_unprocessed(batch_size=3)]

        assert [b.text for b in seen] == ["n1", "n2", "n3", "last"]
        # 2nd page query must carry a non-null keyset cursor.
        _, page2_params = conn.executed[1]
        assert page2_params["after_ts"] is not None
        assert page2_params["after_id"] is not None

    async def test_agent_filter(self) -> None:
        conn = MockConnection(scan_pages=[[]])
        mtm = _mtm(conn)
        _ = [b async for b in mtm.scan_unprocessed(agent_id="agent-7")]
        sql, params = conn.executed[0]
        assert "mn.tags->>'agent_id' = %(agent)s" in sql
        assert params["agent"] == "agent-7"

    async def test_empty_store_yields_nothing(self) -> None:
        conn = MockConnection(scan_pages=[[]])
        mtm = _mtm(conn)
        assert [b async for b in mtm.scan_unprocessed()] == []


# ---------------------------------------------------------------------------
# mark_processed
# ---------------------------------------------------------------------------


class TestMarkProcessed:
    async def test_inserts_promotion_markers(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        ids = [uuid.uuid4(), uuid.uuid4()]

        result = await mtm.mark_processed(ids)

        assert result is None
        sql, params = conn.executed[0]
        assert "INSERT INTO memory_promotions" in sql
        assert "NOT EXISTS" in sql  # idempotent guard
        assert params["ids"] == [str(i) for i in ids]

    async def test_empty_ids_is_noop(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        assert await mtm.mark_processed([]) is None
        assert conn.executed == []  # no DB round-trip

    async def test_accepts_str_ids(self) -> None:
        conn = MockConnection()
        mtm = _mtm(conn)
        await mtm.mark_processed(["11111111-1111-1111-1111-111111111111"])
        _, params = conn.executed[0]
        assert params["ids"] == ["11111111-1111-1111-1111-111111111111"]


# ---------------------------------------------------------------------------
# Lifecycle: scan → mark → scan excludes
# ---------------------------------------------------------------------------


class TestWorkQueueLifecycle:
    async def test_mark_then_rescan_excludes(self) -> None:
        """
        Simulates the Promoter loop: first scan sees a block; after
        mark_processed the (mock) promotions row exists, so the second
        scan's page is empty.
        """
        block_row = _node_row(idx=1, text="candidate", tokens=2)
        # page 1 returns the block; after mark_processed, page 2 is empty.
        conn = MockConnection(scan_pages=[[block_row], []])
        mtm = _mtm(conn)

        first = [b async for b in mtm.scan_unprocessed()]
        assert [b.text for b in first] == ["candidate"]

        await mtm.mark_processed([b.id for b in first])

        second = [b async for b in mtm.scan_unprocessed()]
        assert second == []  # excluded after marking
