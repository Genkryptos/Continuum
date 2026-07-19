"""
tests/unit/db/test_migrate.py
=============================
Hermetic tests for ``continuum.db.migrate`` — the migration runner. The pure
logic (discovery, version parsing, ordering, pending-set) is tested directly;
``apply_migrations`` is exercised in --dry-run mode through an injected fake
connection so no real Postgres is needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from continuum.db.migrate import (
    Migration,
    apply_migrations,
    discover_migrations,
    pending,
)

pytestmark = pytest.mark.unit


def _write(d: Path, name: str, body: str = "SELECT 1;") -> None:
    (d / name).write_text(body)


# ── discovery + ordering ──────────────────────────────────────────────────


def test_discover_sorts_by_version_and_parses_prefix(tmp_path: Path) -> None:
    # Intentionally out of order on disk; numeric prefix drives the order.
    _write(tmp_path, "010_late.sql")
    _write(tmp_path, "002_pgvector_upgrade.sql")
    _write(tmp_path, "001_ltm_schema.sql")
    _write(tmp_path, "README.md", "not a migration")  # ignored
    _write(tmp_path, "notes.txt", "ignored")
    got = discover_migrations(tmp_path)
    assert [m.version for m in got] == ["001", "002", "010"]
    assert [m.name for m in got] == [
        "001_ltm_schema.sql",
        "002_pgvector_upgrade.sql",
        "010_late.sql",
    ]


def test_discover_empty_dir(tmp_path: Path) -> None:
    assert discover_migrations(tmp_path) == []


def test_non_numeric_prefixed_sql_is_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "baseline.sql")  # no leading digits
    _write(tmp_path, "001_ok.sql")
    assert [m.version for m in discover_migrations(tmp_path)] == ["001"]


# ── pending ────────────────────────────────────────────────────────────────


def test_pending_filters_applied_versions() -> None:
    all_m = [
        Migration("001", Path("001_a.sql")),
        Migration("002", Path("002_b.sql")),
        Migration("003", Path("003_c.sql")),
    ]
    assert [m.version for m in pending(all_m, {"001", "002"})] == ["003"]
    assert pending(all_m, {"001", "002", "003"}) == []
    assert [m.version for m in pending(all_m, set())] == ["001", "002", "003"]


# ── real migrations directory sanity ───────────────────────────────────────


def test_repo_migrations_are_discoverable() -> None:
    # The shipped migrations/ should yield the known 001..004 set, in order.
    found = discover_migrations(Path(__file__).resolve().parents[3] / "migrations")
    versions = [m.version for m in found]
    assert versions == sorted(versions)  # ordered
    assert {"001", "002", "003", "004"} <= set(versions)


# ── apply_migrations: dry-run + up-to-date (fake connection) ───────────────


class _FakeCursor:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[str]]:
        return self._rows


class _FakeConn:
    """Minimal psycopg-like connection: records executed SQL, returns applied rows."""

    def __init__(self, applied: list[str]) -> None:
        self._applied = applied
        self.executed: list[str] = []

    def execute(self, sql: str) -> _FakeCursor:
        self.executed.append(sql)
        if "schema_migrations" in sql and sql.strip().upper().startswith("SELECT"):
            return _FakeCursor([(v,) for v in self._applied])
        return _FakeCursor([])

    def commit(self) -> None:  # pragma: no cover - not hit in dry-run
        pass

    def rollback(self) -> None:  # pragma: no cover
        pass

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _patch_connect(monkeypatch: pytest.MonkeyPatch, conn: _FakeConn) -> None:
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)


def test_apply_dry_run_lists_pending_without_executing_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "001_a.sql", "CREATE TABLE a();")
    _write(tmp_path, "002_b.sql", "CREATE TABLE b();")
    conn = _FakeConn(applied=["001"])  # 001 already applied → only 002 pending
    _patch_connect(monkeypatch, conn)

    buf = io.StringIO()
    n = apply_migrations("postgresql://x", directory=tmp_path, dry_run=True, out=buf)

    assert n == 0
    out = buf.getvalue()
    assert "002_b.sql" in out and "001_a.sql" not in out
    assert "dry run" in out
    # dry-run must not run the CREATE TABLE statements
    assert not any("CREATE TABLE" in s for s in conn.executed)


def test_apply_reports_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "001_a.sql")
    conn = _FakeConn(applied=["001"])
    _patch_connect(monkeypatch, conn)

    buf = io.StringIO()
    n = apply_migrations("postgresql://x", directory=tmp_path, out=buf)
    assert n == 0
    assert "up to date" in buf.getvalue()


def test_apply_runs_pending_and_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "001_a.sql", "CREATE TABLE a();")
    conn = _FakeConn(applied=[])  # nothing applied → 001 runs
    _patch_connect(monkeypatch, conn)

    buf = io.StringIO()
    n = apply_migrations("postgresql://x", directory=tmp_path, out=buf)
    assert n == 1
    assert any("CREATE TABLE a()" in s for s in conn.executed)
    assert "applied 1 migration" in buf.getvalue()
