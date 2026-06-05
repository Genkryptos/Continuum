"""
tests/unit/db/test_clear.py
==========================
Hermetic tests for ``continuum.db.clear`` — discovery (preserves
``schema_migrations``) and the truncate flow via an injected fake connection.
No real Postgres.
"""

from __future__ import annotations

import io

import pytest

from continuum.db.clear import clear_all, discover_data_tables

pytestmark = pytest.mark.unit


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, tables: list[str], counts: dict[str, int] | None = None) -> None:
        self._tables = tables
        self._counts = counts or {}
        self.executed: list[str] = []
        self.committed = False

    def execute(self, sql: str) -> _FakeCursor:
        self.executed.append(sql)
        s = sql.strip().upper()
        if "PG_TABLES" in s:
            return _FakeCursor([(t,) for t in self._tables])
        if s.startswith("SELECT COUNT(*)"):
            tbl = sql.split('"')[1] if '"' in sql else ""
            return _FakeCursor([(self._counts.get(tbl, 0),)])
        return _FakeCursor([])

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _patch(monkeypatch: pytest.MonkeyPatch, conn: _FakeConn) -> None:
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)


def test_discover_excludes_schema_migrations() -> None:
    conn = _FakeConn(["stm_messages", "memory_nodes", "schema_migrations"])
    assert discover_data_tables(conn) == ["stm_messages", "memory_nodes"]


def test_clear_truncates_all_data_tables_with_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(
        ["stm_messages", "memory_nodes", "schema_migrations"],
        counts={"stm_messages": 3, "memory_nodes": 2},
    )
    _patch(monkeypatch, conn)
    buf = io.StringIO()
    n = clear_all("postgresql://x", yes=True, out=buf)
    assert n == 2
    truncate = [s for s in conn.executed if s.strip().upper().startswith("TRUNCATE")]
    assert len(truncate) == 1
    assert '"stm_messages"' in truncate[0] and '"memory_nodes"' in truncate[0]
    assert "schema_migrations" not in truncate[0]  # preserved
    assert "RESTART IDENTITY CASCADE" in truncate[0]
    assert conn.committed is True
    assert "cleared 2 table(s)" in buf.getvalue()


def test_clear_refuses_without_yes_when_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(["stm_messages"], counts={"stm_messages": 5})
    _patch(monkeypatch, conn)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    buf = io.StringIO()
    n = clear_all("postgresql://x", yes=False, out=buf)
    assert n == 0
    assert not any(s.strip().upper().startswith("TRUNCATE") for s in conn.executed)
    assert conn.committed is False
    assert "re-run with --yes" in buf.getvalue()


def test_clear_noop_when_no_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(["schema_migrations"])  # only the preserved table
    _patch(monkeypatch, conn)
    buf = io.StringIO()
    n = clear_all("postgresql://x", yes=True, out=buf)
    assert n == 0
    assert "nothing to clear" in buf.getvalue()
