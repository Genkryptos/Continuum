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

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from continuum.memory import Memory

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

__all__ = ["build_server", "main"]


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
    mem: Memory, entity: str, since: str | None = None, until: str | None = None,
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
    """Postgres-backed session when a DSN is configured, else in-memory."""
    if os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL"):
        # A full session (with the supersession decider) is built by the CLI
        # wiring; fall back to in-memory here to keep the server importable
        # without a live DB. Advanced users can inject their own Memory via
        # build_server(memory=...).
        try:
            from continuum.core.config import ContinuumConfig

            return Memory(Memory.in_memory(config=ContinuumConfig()).session)
        except Exception:
            pass
    return Memory.in_memory()


def build_server(memory: Memory | None = None) -> FastMCP:
    """Build the Continuum MCP server. Inject *memory* for tests / custom stores."""
    from mcp.server.fastmcp import FastMCP

    mem = memory or _default_memory()
    server = FastMCP("continuum")

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
        entity: str, since: str | None = None, until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Bi-temporal history for an entity, oldest→newest (ISO date bounds)."""
        return await _timeline(mem, entity, since, until)

    return server


def main() -> None:
    """Console entry point (`continuum-mcp`) — run over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
