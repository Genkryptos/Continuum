"""
tests/unit/test_postgres_stm.py
================================
Unit tests for ``PostgresSTM`` using injected mock connections.

No real PostgreSQL instance is required — all DB I/O is replaced with an
in-process ``MockConnection`` / ``MockCursor`` pair injected via the
``conn_factory`` constructor parameter.

Design
------
``PostgresSTM._connect()`` calls ``self._conn_factory()`` and uses it as an
async context-manager that yields a connection.  Our ``make_conn_factory()``
helper returns an ``asynccontextmanager`` wrapping a ``MockConnection``.

The ``MockConnection`` records every ``execute(sql, params)`` call so tests
can assert on what SQL was sent and with what parameters.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from continuum.core.protocols import STMProtocol
from continuum.core.types import MemoryItem, MemoryTier, ProcessingState
from continuum.stores.stm.postgres_stm import PostgresSTM

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers — mock DB layer
# ---------------------------------------------------------------------------


class MockCursor:
    """Fake psycopg3 cursor that returns canned rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class MockConnection:
    """
    Fake psycopg3 async connection.

    Stores every ``execute`` call in ``self.calls`` for later assertion.
    The ``cursor_rows`` mapping lets callers specify per-SQL return rows
    (keyed by a substring of the SQL string for easy matching).
    """

    def __init__(
        self,
        default_rows: list[dict[str, Any]] | None = None,
        cursor_rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._default_rows = default_rows or []
        self._cursor_rows = cursor_rows or {}

    async def execute(self, sql: str, params: Any = None) -> MockCursor:
        self.calls.append((sql, params))
        # Pick rows by matching a substring of the SQL
        for key, rows in self._cursor_rows.items():
            if key in sql:
                return MockCursor(rows)
        return MockCursor(self._default_rows)


def make_conn_factory(
    conn: MockConnection,
) -> Any:
    """
    Return a ``conn_factory`` callable compatible with ``PostgresSTM``.

    The factory is an async function returning an async context manager that
    yields *conn*.  ``PostgresSTM._connect()`` calls it as::

        async with self._conn_factory() as conn:
            ...
    """

    @asynccontextmanager
    async def _factory() -> AsyncIterator[MockConnection]:
        yield conn

    return _factory


def _make_stm(
    conn: MockConnection,
    max_tokens: int = 1_000,
    reserved: int = 100,
) -> PostgresSTM:
    """Build a PostgresSTM with *conn* injected and schema bootstrap skipped."""
    stm = PostgresSTM(
        conn_factory=make_conn_factory(conn),
        max_tokens=max_tokens,
        reserved_for_response=reserved,
    )
    stm._initialized = True  # skip ensure_schema in unit tests
    return stm


def _make_item(
    content: str = "hello world",
    session_id: str = "sess-1",
    tokens: int = 2,
    importance: float = 0.5,
) -> MemoryItem:
    return MemoryItem(
        content=content,
        session_id=session_id,
        importance=importance,
        metadata={"role": "user", "tokens": tokens},
    )


def _row(
    content: str = "hello world",
    session_id: str = "sess-1",
    tokens: int = 2,
    importance: float = 0.5,
    idx: int = 0,
) -> dict[str, Any]:
    """Build a fake stm_messages row dict."""
    return {
        "id": f"row-{idx}",
        "session_id": session_id,
        "agent_id": None,
        "user_id": None,
        "role": "user",
        "content": content,
        "tokens": tokens,
        "importance": importance,
        "confidence": 1.0,
        "created_at": datetime(2024, 1, 1, 12, 0, idx, tzinfo=UTC),
        "metadata": json.dumps({"role": "user", "tokens": tokens}),
    }


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestPostgresSTMConstruction:
    def test_requires_at_least_one_source(self) -> None:
        with pytest.raises(ValueError, match="Provide at least one"):
            PostgresSTM()

    def test_accepts_dsn(self) -> None:
        stm = PostgresSTM(dsn="postgresql://localhost/test")
        assert stm._dsn == "postgresql://localhost/test"

    def test_conn_factory_takes_precedence(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        assert stm._conn_factory is not None

    def test_default_token_settings(self) -> None:
        conn = MockConnection()
        stm = PostgresSTM(conn_factory=make_conn_factory(conn))
        assert stm.max_tokens == 4096
        assert stm.reserved_for_response == 512


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


class TestAppend:
    async def test_append_executes_insert(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        item = _make_item()

        await stm.append(item)

        assert len(conn.calls) == 1
        sql, params = conn.calls[0]
        assert "INSERT INTO stm_messages" in sql
        assert "ON CONFLICT" in sql
        assert params["content"] == "hello world"
        assert params["session_id"] == "sess-1"

    async def test_append_sets_stm_tier(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        item = _make_item()
        item.tier = MemoryTier.LTM  # force non-STM tier before append
        assert item.tier == MemoryTier.LTM  # sanity: mutation took effect

        await stm.append(item)

        assert item.tier == MemoryTier.STM  # append must stamp STM tier

    async def test_append_uses_metadata_tokens_if_present(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        item = _make_item(tokens=42)

        await stm.append(item)

        _, params = conn.calls[0]
        assert params["tokens"] == 42

    async def test_append_computes_tokens_if_absent(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        item = MemoryItem(content="one two three", session_id="s", metadata={"role": "user"})
        # "tokens" not in metadata → fallback to whitespace split → 3

        await stm.append(item)

        _, params = conn.calls[0]
        assert params["tokens"] == 3

    async def test_append_upsert_on_duplicate_id(self) -> None:
        """Two appends with the same id trigger ON CONFLICT path."""
        conn = MockConnection()
        stm = _make_stm(conn)
        item = _make_item()

        await stm.append(item)
        await stm.append(item)  # same id

        assert len(conn.calls) == 2
        for sql, _ in conn.calls:
            assert "ON CONFLICT (id) DO UPDATE" in sql


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------


class TestWindow:
    async def test_window_returns_items_within_budget(self) -> None:
        # budget = 1000 - 100 = 900; rows total 700 tokens → all fit
        rows = [_row(tokens=300, idx=0), _row(tokens=400, idx=1)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn, max_tokens=1_000, reserved=100)

        result = await stm.window("sess-1")

        assert len(result) == 2

    async def test_window_respects_token_budget(self) -> None:
        # budget = 500; rows are fetched newest-first (400, 300); 400 fits, 400+300=700 > 500
        rows = [_row(tokens=400, idx=1), _row(tokens=300, idx=0)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn, max_tokens=600, reserved=100)  # budget=500

        result = await stm.window("sess-1")

        assert len(result) == 1  # only the 400-token row fits
        assert result[0].metadata["tokens"] == 400

    async def test_window_returns_oldest_first(self) -> None:
        # Mock simulates DB returning newest-first: idx 3, 2, 1
        # window() reverses → result is oldest-first: msg-1, msg-2, msg-3
        rows = [_row(content=f"msg-{i}", tokens=10, idx=i) for i in range(3, 0, -1)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)

        result = await stm.window("sess-1")

        assert result[0].content == "msg-1"  # oldest (smallest idx) → first
        assert result[-1].content == "msg-3"  # newest (largest idx) → last

    async def test_window_explicit_max_tokens_overrides_default(self) -> None:
        rows = [_row(tokens=200, idx=0), _row(tokens=200, idx=1)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn, max_tokens=10_000, reserved=0)

        # Explicit budget of 250 → only one 200-token row fits
        result = await stm.window("sess-1", max_tokens=250)

        assert len(result) == 1

    async def test_window_empty_session(self) -> None:
        conn = MockConnection(default_rows=[])
        stm = _make_stm(conn)

        result = await stm.window("no-such-session")

        assert list(result) == []


# ---------------------------------------------------------------------------
# get_recent
# ---------------------------------------------------------------------------


class TestGetRecent:
    async def test_get_recent_returns_n_items(self) -> None:
        rows = [_row(content=f"m{i}", tokens=5, idx=i) for i in range(5, 0, -1)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)

        result = await stm.get_recent("sess-1", n=5)

        assert len(result) == 5

    async def test_get_recent_passes_limit_to_sql(self) -> None:
        conn = MockConnection(default_rows=[])
        stm = _make_stm(conn)

        await stm.get_recent("sess-1", n=7)

        sql, params = conn.calls[0]
        assert "LIMIT" in sql
        assert params["n"] == 7

    async def test_get_recent_oldest_first(self) -> None:
        # Mock: DB returns newest-first ["newest" (idx=3), "oldest" (idx=1)]
        # get_recent reverses → result is oldest-first ["oldest", "newest"]
        rows = [_row(content="newest", idx=3), _row(content="oldest", idx=1)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)

        result = await stm.get_recent("sess-1", n=2)

        assert result[0].content == "oldest"  # oldest item first after reversal
        assert result[1].content == "newest"  # newest item last

    async def test_get_recent_empty_session(self) -> None:
        conn = MockConnection(default_rows=[])
        stm = _make_stm(conn)

        result = await stm.get_recent("empty", n=10)

        assert list(result) == []


# ---------------------------------------------------------------------------
# flush_to
# ---------------------------------------------------------------------------


class MockMTM:
    """Minimal MTMProtocol stand-in that records add_summary calls."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.received: list[MemoryItem] = []
        self._fail_on = fail_on

    async def add_summary(self, item: MemoryItem) -> None:
        if self._fail_on and item.id == self._fail_on:
            raise RuntimeError(f"simulated failure on {item.id}")
        self.received.append(item)


class TestFlushTo:
    async def test_flush_to_full_success_returns_count(self) -> None:
        rows = [_row(content=f"m{i}", idx=i) for i in range(3)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)
        mtm = MockMTM()

        count = await stm.flush_to("sess-1", mtm)

        assert count == 3
        assert len(mtm.received) == 3

    async def test_flush_to_deletes_transferred_ids(self) -> None:
        rows = [_row(idx=0), _row(idx=1)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)
        mtm = MockMTM()

        await stm.flush_to("sess-1", mtm)

        # Last call must be the DELETE
        delete_calls = [c for c in conn.calls if "DELETE" in c[0] and "ANY" in c[0]]
        assert len(delete_calls) == 1
        _, params = delete_calls[0]
        assert set(params["ids"]) == {"row-0", "row-1"}

    async def test_flush_to_partial_failure_deletes_only_succeeded(self) -> None:
        rows = [_row(idx=0), _row(idx=1), _row(idx=2)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)
        mtm = MockMTM(fail_on="row-1")  # row-1 will raise

        count = await stm.flush_to("sess-1", mtm)

        assert count == 1  # only row-0 was transferred
        delete_calls = [c for c in conn.calls if "DELETE" in c[0] and "ANY" in c[0]]
        assert len(delete_calls) == 1
        _, params = delete_calls[0]
        assert params["ids"] == ["row-0"]

    async def test_flush_to_empty_session_returns_zero(self) -> None:
        conn = MockConnection(default_rows=[])
        stm = _make_stm(conn)
        mtm = MockMTM()

        count = await stm.flush_to("empty", mtm)

        assert count == 0
        assert mtm.received == []

    async def test_flush_to_marks_items_unprocessed(self) -> None:
        rows = [_row(idx=0)]
        conn = MockConnection(default_rows=rows)
        stm = _make_stm(conn)
        received: list[MemoryItem] = []

        class CaptureMTM:
            async def add_summary(self, item: MemoryItem) -> None:
                received.append(item)

        await stm.flush_to("sess-1", CaptureMTM())  # type: ignore[arg-type]

        assert received[0].processing_state == ProcessingState.UNPROCESSED


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    async def test_clear_executes_delete(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)

        await stm.clear("sess-1")

        assert len(conn.calls) == 1
        sql, params = conn.calls[0]
        assert "DELETE FROM stm_messages" in sql
        assert params["session_id"] == "sess-1"

    async def test_clear_different_sessions_independent(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)

        await stm.clear("sess-A")
        await stm.clear("sess-B")

        sessions = [p["session_id"] for _, p in conn.calls]
        assert sessions == ["sess-A", "sess-B"]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_stats_required_keys(self) -> None:
        stats_row = [{"message_count": 5, "total_tokens": 250}]
        conn = MockConnection(default_rows=stats_row)
        stm = _make_stm(conn, max_tokens=1_000, reserved=100)

        result = await stm.stats("sess-1")

        assert set(result.keys()) == {
            "session_id",
            "message_count",
            "total_tokens",
            "max_tokens",
            "utilization",
            "budget_remaining",
        }

    async def test_stats_correct_values(self) -> None:
        stats_row = [{"message_count": 4, "total_tokens": 180}]
        conn = MockConnection(default_rows=stats_row)
        stm = _make_stm(conn, max_tokens=1_000, reserved=100)  # budget = 900

        result = await stm.stats("sess-1")

        assert result["message_count"] == 4
        assert result["total_tokens"] == 180
        assert result["max_tokens"] == 1_000
        assert result["budget_remaining"] == 720  # 900 - 180
        assert abs(result["utilization"] - 180 / 900) < 1e-9

    async def test_stats_zero_state(self) -> None:
        stats_row = [{"message_count": 0, "total_tokens": 0}]
        conn = MockConnection(default_rows=stats_row)
        stm = _make_stm(conn, max_tokens=1_000, reserved=100)

        result = await stm.stats("empty-session")

        assert result["message_count"] == 0
        assert result["total_tokens"] == 0
        assert result["utilization"] == 0.0
        assert result["budget_remaining"] == 900

    async def test_stats_session_id_in_result(self) -> None:
        stats_row = [{"message_count": 1, "total_tokens": 10}]
        conn = MockConnection(default_rows=stats_row)
        stm = _make_stm(conn)

        result = await stm.stats("my-session")

        assert result["session_id"] == "my-session"


# ---------------------------------------------------------------------------
# _row_to_item
# ---------------------------------------------------------------------------


class TestRowToItem:
    def test_string_metadata_parsed(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        row = _row()
        row["metadata"] = json.dumps({"role": "assistant", "tokens": 7})

        item = stm._row_to_item(row)

        assert item.metadata["role"] == "assistant"
        assert item.metadata["tokens"] == 7

    def test_dict_metadata_passed_through(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        row = _row()
        row["metadata"] = {"role": "system", "tokens": 3}  # already a dict

        item = stm._row_to_item(row)

        assert item.metadata["role"] == "system"

    def test_tier_always_stm(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)

        item = stm._row_to_item(_row())

        assert item.tier == MemoryTier.STM

    def test_datetime_preserved(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        ts = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        row = _row()
        row["created_at"] = ts

        item = stm._row_to_item(row)

        assert item.created_at == ts

    def test_iso_string_created_at_parsed(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        row = _row()
        row["created_at"] = "2024-06-15T10:30:00+00:00"

        item = stm._row_to_item(row)

        assert item.created_at.year == 2024
        assert item.created_at.month == 6

    def test_missing_metadata_defaults_gracefully(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        row = _row()
        row["metadata"] = None  # NULL from DB

        item = stm._row_to_item(row)

        assert isinstance(item.metadata, dict)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_postgres_stm_satisfies_stm_protocol(self) -> None:
        conn = MockConnection()
        stm = _make_stm(conn)
        # STMProtocol is @runtime_checkable
        assert isinstance(stm, STMProtocol)

    def test_all_protocol_methods_present(self) -> None:
        required = {"append", "window", "get_recent", "flush_to", "clear", "stats"}
        conn = MockConnection()
        stm = _make_stm(conn)
        missing = required - set(dir(stm))
        assert missing == set(), f"Missing methods: {missing}"


# ---------------------------------------------------------------------------
# ensure_schema (schema bootstrap)
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    async def test_ensure_schema_executes_ddl(self) -> None:
        conn = MockConnection()
        stm = PostgresSTM(
            conn_factory=make_conn_factory(conn),
            max_tokens=1_000,
        )
        # _initialized starts False → ensure_schema will run

        await stm.ensure_schema()

        ddl_sqls = [sql for sql, _ in conn.calls]
        assert any("CREATE TABLE IF NOT EXISTS stm_messages" in s for s in ddl_sqls)
        assert any("CREATE INDEX IF NOT EXISTS" in s for s in ddl_sqls)

    async def test_ensure_schema_idempotent(self) -> None:
        conn = MockConnection()
        stm = PostgresSTM(
            conn_factory=make_conn_factory(conn),
            max_tokens=1_000,
        )

        await stm.ensure_schema()
        call_count_after_first = len(conn.calls)

        await stm.ensure_schema()  # second call should be a no-op

        assert len(conn.calls) == call_count_after_first  # no additional DDL
