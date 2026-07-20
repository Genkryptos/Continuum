"""
tests/e2e/test_mcp_end_to_end.py
================================
Drives the real ``continuum-mcp`` binary over stdio against a real, migrated
Postgres. Sparse-only (no embedder download), so it runs in CI on the pgvector
service container.

Each test pins a bug this cycle produced — the ones no unit test caught because
they lived between correct layers. Assertions are against ground truth (the row
in the database, the exact answer), never "it returned something".
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp", reason="e2e needs the [mcp] extra")

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ── the headline promise: memory survives across sessions ─────────────────────


def test_memory_persists_across_sessions(e2e_dsn: str) -> None:
    from tests.e2e.conftest import MCPClient

    writer = MCPClient(e2e_dsn)
    err, ack = writer.tool("remember", text="The deploy target is staging first.")
    assert err is False
    assert ack == "stored"  # durable backend → terse ack, not the in-memory warning
    writer.close()

    reader = MCPClient(e2e_dsn)  # brand-new process, shares only the database
    try:
        err, hits = reader.tool("recall", query="where do we deploy first", k=3)
        assert err is False
        assert any("staging" in h["content"].lower() for h in hits)
    finally:
        reader.close()


# ── attribute-keyed current(): exact SQL, no embedder, honours valid time ─────


def test_current_is_exact_and_bitemporal(e2e_dsn: str) -> None:
    from tests.e2e.conftest import MCPClient

    c = MCPClient(e2e_dsn)
    try:
        c.tool(
            "remember", text="I live in Boston.", occurred_at="2026-01-10", attribute="residence"
        )
        c.tool(
            "remember",
            text="I moved to New York City.",
            occurred_at="2026-06-15",
            attribute="residence",
        )

        _, now = c.tool("current", subject="user", attribute="residence")
        assert "New York" in now  # latest wins

        _, past = c.tool("current", subject="user", attribute="residence", as_of="2026-03-01")
        assert "Boston" in past  # bi-temporal: before the move

        _, missing = c.tool("current", subject="user", attribute="employer")
        assert missing == "not found"  # authoritative, does not invent a value
    finally:
        c.close()


# ── the durability lie: a write that never persists must not report success ───


def test_in_memory_backend_admits_it_is_not_durable() -> None:
    from tests.e2e.conftest import MCPClient

    # No DSN → in-memory fallback. The ack must say so, not lie "stored" — that
    # silent lie was invisible to every benchmark this cycle.
    c = MCPClient(dsn=None)
    try:
        err, ack = c.tool("remember", text="x")
        assert err is False
        assert "NOT durable" in ack
    finally:
        c.close()


# ── adversarial inputs that used to crash or misbehave ────────────────────────


def test_negative_and_zero_k_return_nothing(e2e_dsn: str) -> None:
    from tests.e2e.conftest import MCPClient

    c = MCPClient(e2e_dsn)
    try:
        # Need enough rows that the bug is observable: items[:-5] on 1 item is
        # empty either way. With 8, the un-guarded slice returned 3.
        for i in range(8):
            c.tool("remember", text=f"searchable fact number {i}")
        for k in (-5, 0):
            err, hits = c.tool("recall", query="searchable fact", k=k)
            assert err is False and hits == []  # slice bug: k=-5 used to return rows
        err, hits = c.tool("recall", query="searchable fact", k=3)
        assert err is False and len(hits) >= 1
    finally:
        c.close()


def test_explicit_as_of_does_not_crash(e2e_dsn: str) -> None:
    from tests.e2e.conftest import MCPClient

    # The MCP layer parses as_of to a NAIVE datetime; the store returns AWARE
    # ones. Comparing them raised TypeError -> a failed tool call.
    c = MCPClient(e2e_dsn)
    try:
        c.tool(
            "remember", text="I live in Bhilai.", occurred_at="2026-01-01", attribute="residence"
        )
        for as_of in ("0001-01-01", "2027-01-01", "not-a-date"):
            err, _ = c.tool("current", subject="user", attribute="residence", as_of=as_of)
            assert err is False
    finally:
        c.close()


def test_sql_injection_is_inert_and_table_survives(e2e_dsn: str) -> None:
    from tests.e2e.conftest import MCPClient

    c = MCPClient(e2e_dsn)
    try:
        err, ack = c.tool("remember", text="'; DROP TABLE memory_nodes; --")
        assert err is False and ack == "stored"
        # The table still exists and the payload is stored as inert text.
        err, hits = c.tool("recall", query="DROP TABLE", k=5)
        assert err is False
        assert any("DROP TABLE" in h["content"] for h in hits)
    finally:
        c.close()


# ── compound statements are split before storage ─────────────────────────────


def test_compound_statement_is_split(e2e_dsn: str) -> None:
    from tests.e2e.conftest import MCPClient

    c = MCPClient(e2e_dsn)
    try:
        # This exact sentence embedded between its topics and matched neither.
        c.tool("remember", text="We're going with Postgres 16, pgvector needs 0.8 or newer.")
        err, hits = c.tool("recall", query="Postgres", k=8)
        assert err is False
        contents = [h["content"] for h in hits]
        # split → a short "Postgres 16" fact exists on its own, not only the compound
        assert any("Postgres 16" in c and "pgvector" not in c for c in contents)
    finally:
        c.close()
