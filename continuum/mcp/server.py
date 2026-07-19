"""
continuum.mcp.server
====================
An MCP server exposing Continuum's memory as tools, so any MCP client (Claude
Code, Cursor, …) gets persistent memory with zero glue:

    recall(query, k)             — retrieve relevant memories
    remember(text, occurred_at)  — store a fact/turn
    current(subject, attribute)  — the current value after supersession
    timeline(entity, since, until) — bi-temporal history, oldest→newest

Run it:  ``continuum-mcp``  (stdio transport). Backing store is
``Memory.in_memory()`` by default; set ``CONTINUUM_DB_DSN`` (or configure
``continuum.yaml``) and it will use Postgres via a full session.

The tool *logic* lives in plain ``_recall``/``_remember``/… functions (unit-
testable without the MCP runtime); the FastMCP tools are thin wrappers.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from continuum.memory import Memory

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

__all__ = ["build_server", "main"]

log = logging.getLogger(__name__)


# ── tool logic (plain, testable) ──────────────────────────────────────────────


def _item_dict(item: Any) -> dict[str, Any]:
    """JSON-safe view of a MemoryItem for tool output."""
    created = getattr(item, "created_at", None)
    return {
        "content": getattr(item, "content", "") or "",
        "id": str(getattr(item, "id", "")),
        "session_id": getattr(item, "session_id", None),
        "created_at": created.isoformat() if created else None,
    }


async def _recall(mem: Memory, query: str, k: int = 8) -> list[dict[str, Any]]:
    return [_item_dict(h) for h in await mem.recall(query, k=k)]


async def _remember(mem: Memory, text: str, occurred_at: str | None = None) -> str:
    when: datetime | None = None
    if occurred_at:
        try:
            when = datetime.fromisoformat(occurred_at)
        except ValueError:
            when = None
    await mem.add(text, occurred_at=when)
    return "stored"


async def _current(mem: Memory, subject: str, attribute: str) -> str:
    return (await mem.current(subject, attribute)) or "not found"


async def _timeline(
    mem: Memory,
    entity: str,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    def _parse(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    items = await mem.timeline(entity, since=_parse(since), until=_parse(until))
    return [_item_dict(h) for h in items]


# ── server assembly ───────────────────────────────────────────────────────────


def _default_memory() -> Memory:
    """Postgres-backed (durable, dense recall) when a DSN is configured, else
    in-memory (ephemeral, recency recall).

    A DSN in ``CONTINUUM_DB_DSN`` / ``DATABASE_URL`` selects the production
    stack via :meth:`Memory.from_postgres`. The local bge-m3 embedder is
    attached by default (dense/semantic recall) — set ``CONTINUUM_MCP_EMBEDDINGS=0``
    for a sparse-only, no-download setup. If the backend can't be built (no
    migrated DB, missing extra), we log and fall back to in-memory so the server
    still starts. Inject your own store with ``build_server(memory=...)``."""
    dsn = os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL")
    if dsn:
        embeddings = os.environ.get("CONTINUUM_MCP_EMBEDDINGS", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        try:
            return Memory.from_postgres(dsn, embeddings=embeddings)
        except Exception:
            log.exception("continuum-mcp: Postgres backend unavailable — using in-memory")
    return Memory.in_memory()


def build_server(
    memory: Memory | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> FastMCP:
    """Build the Continuum MCP server. Inject *memory* for tests / custom stores.

    *host* / *port* set the bind address for the HTTP transports
    (``streamable-http`` / ``sse``); they are ignored under stdio.
    """
    from mcp.server.fastmcp import FastMCP

    mem = memory or _default_memory()
    settings: dict[str, Any] = {}
    if host is not None:
        settings["host"] = host
    if port is not None:
        settings["port"] = port
    server = FastMCP("continuum", **settings)

    @server.tool()
    async def recall(query: str, k: int = 8) -> list[dict[str, Any]]:
        """Retrieve up to k memories relevant to the query, best-first."""
        return await _recall(mem, query, k)

    @server.tool()
    async def remember(text: str, occurred_at: str | None = None) -> str:
        """Store a fact or turn in memory. occurred_at is an optional ISO date."""
        return await _remember(mem, text, occurred_at)

    @server.tool()
    async def current(subject: str, attribute: str) -> str:
        """The current value for an attribute after supersession (e.g. residence)."""
        return await _current(mem, subject, attribute)

    @server.tool()
    async def timeline(
        entity: str,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Bi-temporal history for an entity, oldest→newest (ISO date bounds)."""
        return await _timeline(mem, entity, since, until)

    return server


def main(argv: list[str] | None = None) -> None:
    """Console entry point (``continuum-mcp``).

    Default transport is **stdio**: an MCP client (Claude Code, Cursor, …) spawns
    this process and talks over stdin/stdout — you do not pre-start it.

    Pass ``--http`` to instead run a standalone **Streamable-HTTP** server that a
    client connects to by URL — a genuinely always-on server::

        continuum-mcp --http --host 127.0.0.1 --port 8000
        claude mcp add continuum --transport http http://127.0.0.1:8000/mcp

    ``--sse`` selects the legacy SSE transport (endpoint ``/sse``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="continuum-mcp",
        description="Continuum MCP server — memory as recall/remember/current/timeline tools.",
    )
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--http",
        action="store_true",
        help="run a standalone Streamable-HTTP server (endpoint /mcp) instead of stdio",
    )
    transport.add_argument(
        "--sse",
        action="store_true",
        help="run a standalone legacy SSE server (endpoint /sse) instead of stdio",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind host for --http/--sse (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="bind port for --http/--sse (default: 8000)"
    )
    args = parser.parse_args(argv)

    if args.http or args.sse:
        server = build_server(host=args.host, port=args.port)
        server.run(transport="sse" if args.sse else "streamable-http")
    else:
        build_server().run()


if __name__ == "__main__":
    main()
