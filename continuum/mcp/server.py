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

import asyncio
import contextlib
import logging
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

from continuum.memory import Memory

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

__all__ = ["BackendUnavailableError", "build_server", "main"]

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


def _parse_when(raw: str) -> datetime | None:
    """ISO date/datetime, or a bare year. ``None`` when it cannot be trusted.

    Deliberately narrow. ``03/15/2026`` is March 15th to an American and
    invalid to most of the world, and guessing wrong writes a false claim about
    *when something was true* — worse than recording no date at all. A bare year
    is unambiguous, so it is accepted as January 1st.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    if len(text) == 4 and text.isdigit():
        try:
            return datetime(int(text), 1, 1)
        except ValueError:
            return None
    return None


async def _remember(
    mem: Memory,
    text: str,
    occurred_at: str | None = None,
    attribute: str | None = None,
) -> str:
    # Say what actually happened. `Memory.add` no-ops on empty text, so a bare
    # "stored" here would report a write that never occurred — the same lie the
    # durability message below exists to prevent.
    if not text or not text.strip():
        return "nothing to store (empty text) - no memory was written"

    # Refuse credential-shaped content outright. This tool is called by the
    # MODEL, usually while summarising something the user pasted, so "they asked
    # for it" is not a safe assumption — and a key written here lands in a
    # database that gets backed up and recalled into future prompts. Refusing
    # loudly beats storing quietly: the caller can say what it skipped.
    from continuum.promotion.capture import looks_like_secret

    if looks_like_secret(text):
        return (
            "REFUSED - this looks like a credential (API key, token, password, "
            "card or ID number) and was NOT stored. Memory is long-lived and gets "
            "recalled into later prompts; put secrets in a secret manager instead."
        )

    when: datetime | None = None
    date_ignored = False
    if occurred_at:
        when = _parse_when(occurred_at)
        date_ignored = when is None
    # split=True and auto_attribute=True by default: callers hand us whole spoken
    # sentences. Splitting a compound one keeps each part embeddable; inferring
    # the attribute (when not given) lets `current` answer it exactly. Both are
    # conservative and no-op unless confident.
    await mem.add(text, occurred_at=when, attribute=attribute, auto_attribute=True, split=True)
    # Report DURABILITY, not just success. A bare "stored" is indistinguishable
    # between a real write and one into an ephemeral store that vanishes at
    # exit — the caller believes it has memory and only finds out much later.
    # This is the common misconfiguration: a DSN in the MCP registration is only
    # picked up by NEW client sessions, so a session started earlier keeps
    # writing to the in-memory fallback while reporting success.
    # A date we could not parse is dropped — say so. Silently discarding it
    # leaves the caller believing the fact is anchored in time, and valid time
    # is what `current` and `as_of` reason over.
    note = (
        f" (note: occurred_at {occurred_at!r} was not understood and NO date was "
        "recorded - use ISO, e.g. 2026-03-15 or 2026-03-15T09:30)"
        if date_ignored
        else ""
    )
    if getattr(mem, "is_durable", True):
        return "stored" + note
    return (
        "stored IN-MEMORY ONLY - this is NOT durable and will be lost when the "
        "server exits. Set CONTINUUM_DB_DSN (and restart the client session) for "
        "a persistent store." + note
    )


async def _current(mem: Memory, subject: str, attribute: str, as_of: str | None = None) -> str:
    when: datetime | None = None
    if as_of:
        try:
            when = datetime.fromisoformat(as_of)
        except ValueError:
            when = None
    return (await mem.current(subject, attribute, as_of=when)) or "not found"


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


# ── backend availability / auto-start ─────────────────────────────────────────


class BackendUnavailableError(RuntimeError):
    """The configured memory backend could not be reached.

    Raised instead of quietly using an in-memory store: when a DSN is configured,
    answering from a phantom store that disappears at exit is worse than an error,
    because the caller believes it has durable memory.
    """


def _configured_dsn() -> str | None:
    return os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL")


async def _dsn_reachable(dsn: str, timeout: float = 2.0) -> bool:
    """Is the DSN's **database** actually usable?

    Deliberately not just a port check. A port probe reports success whenever
    *anything* is listening — including a different PostgreSQL instance that
    grabbed the port and does not have this database at all (easy to hit when
    two versions are installed). That false positive is worse than a plain
    outage, because the caller proceeds against the wrong store. So connect for
    real when psycopg is importable, and fall back to TCP only when it isn't.
    """
    try:
        import psycopg
    except ImportError:
        return await _port_open(dsn, timeout)
    try:
        conn = await asyncio.wait_for(
            psycopg.AsyncConnection.connect(dsn, connect_timeout=int(timeout) or 1), timeout
        )
    except Exception:
        return False
    with contextlib.suppress(Exception):
        await conn.close()
    return True


async def _port_open(dsn: str, timeout: float = 2.0) -> bool:
    """TCP-only liveness probe — a weaker fallback when psycopg is unavailable."""
    from urllib.parse import urlsplit

    parts = urlsplit(dsn)
    host, port = parts.hostname or "127.0.0.1", parts.port or 5432
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (TimeoutError, OSError):
        return False
    writer.close()
    # Bounded: wait_closed() can block indefinitely against a peer that accepts
    # but never completes the close handshake — which would hang the very probe
    # meant to keep the server responsive.
    with contextlib.suppress(Exception):
        await asyncio.wait_for(writer.wait_closed(), timeout)
    return True


async def _autostart(dsn: str, *, timeout: float = 60.0) -> bool:
    """Run ``CONTINUUM_MCP_AUTOSTART`` and wait for *dsn* to accept connections.

    Opt-in only: with the variable unset we never execute anything. It exists so
    an MCP client (Claude Code, Cursor, …) can bring a stopped database up on the
    first memory call instead of the session simply failing, e.g.::

        CONTINUUM_MCP_AUTOSTART="brew services start postgresql@16"
    """
    cmd = (os.environ.get("CONTINUUM_MCP_AUTOSTART") or "").strip()
    if not cmd:
        return False
    log.warning("continuum-mcp: backend unreachable — running CONTINUUM_MCP_AUTOSTART: %s", cmd)
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except Exception:
        log.exception("continuum-mcp: autostart command failed")
        return False

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await _dsn_reachable(dsn):
            log.warning("continuum-mcp: backend is up")
            return True
        await asyncio.sleep(0.5)
    log.error("continuum-mcp: backend still unreachable %.0fs after autostart", timeout)
    return False


# ── observability ─────────────────────────────────────────────────────────────


def _configure_logging() -> None:
    """Send logs to **stderr** (stdout is the stdio protocol channel — logging
    there would corrupt it). Level from ``CONTINUUM_MCP_LOG_LEVEL`` (default
    WARNING, so the server is quiet unless asked; set INFO to trace every tool
    call). The server is otherwise a black box, which is exactly what hid this
    project's worst bugs — a silent fallback, a write that never persisted."""
    level = (os.environ.get("CONTINUUM_MCP_LOG_LEVEL") or "WARNING").upper()
    root = logging.getLogger("continuum")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("continuum-mcp %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        # Don't also propagate to the root logger — FastMCP installs its own
        # handler there, which would print every continuum line a second time.
        root.propagate = False
    root.setLevel(getattr(logging, level, logging.WARNING))


@contextlib.contextmanager
def _observe(tool: str, **fields: Any) -> Any:
    """Time a tool call and log one INFO line: name, inputs, result, duration.

    Yields a callable to record the result (``obs("hits=%d", n)``). An exception
    is logged at ERROR with its timing, then re-raised — a failing tool is no
    longer invisible.
    """
    import time

    start = time.perf_counter()
    detail: list[str] = []

    def _record(fmt: str, *args: Any) -> None:
        detail.append(fmt % args if args else fmt)

    ctx = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    try:
        yield _record
    except Exception:
        ms = (time.perf_counter() - start) * 1000
        log.exception("tool=%s %s FAILED [%.0fms]", tool, ctx, ms)
        raise
    else:
        ms = (time.perf_counter() - start) * 1000
        log.info("tool=%s %s %s [%.0fms]", tool, ctx, " ".join(detail), ms)


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
        namespace = (os.environ.get("CONTINUUM_MCP_NAMESPACE") or "default").strip() or "default"
        try:
            return Memory.from_postgres(
                dsn,
                embeddings=embeddings,
                namespace=namespace,
                # STM is keyed by session_id, not namespace — bind it to the
                # namespace too, or a second tenant's "recent turns" (which
                # `recall` also reads) would leak. LTM scoping alone is not enough.
                session_id=namespace,
                **_supersession_kwargs(),
            )
        except Exception:
            log.exception("continuum-mcp: Postgres backend unavailable — using in-memory")
    return Memory.in_memory()


def _supersession_kwargs() -> dict[str, Any]:
    """Decider wiring for :meth:`Memory.from_postgres`, or ``{}`` to leave it off.

    Enabled by ``CONTINUUM_MCP_SUPERSESSION=1``. Off by default and refused
    without an API key: the decider asks an LLM to adjudicate contradictions, and
    with no LLM it returns NOOP for the ambiguous band — which, honoured, would
    silently discard the new fact.

    With it on, "pricing is 9 dollars" followed by "switched to 12 dollars"
    retires the first instead of leaving both to compete in recall.
    """
    if (os.environ.get("CONTINUUM_MCP_SUPERSESSION") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {}
    from continuum.promotion.openrouter import (
        build_openrouter_completion_fn,
        resolve_openrouter_key,
    )

    key = resolve_openrouter_key()
    if not key:
        log.warning(
            "continuum-mcp: CONTINUUM_MCP_SUPERSESSION is set but no OPENROUTER_API_KEY "
            "was found — supersession stays OFF (enabling it without an LLM would drop facts)"
        )
        return {}
    model = (os.environ.get("CONTINUUM_MCP_SUPERSESSION_MODEL") or "openai/gpt-4o-mini").strip()
    log.warning("continuum-mcp: supersession ENABLED via %s", model)
    return {
        "supersession_completion_fn": build_openrouter_completion_fn(key),
        "supersession_model": model,
    }


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

    _configure_logging()
    mem = memory or _default_memory()
    settings: dict[str, Any] = {}
    if host is not None:
        settings["host"] = host
    if port is not None:
        settings["port"] = port
    server = FastMCP("continuum", **settings)

    # The backing session must be started before use — that is when the Postgres
    # connection pool opens and the background workers spin up. (In-memory needs
    # nothing, which is why this was easy to miss.) Start it lazily on the first
    # tool call, once, so the process still spawns instantly for stdio clients.
    started = False
    start_lock = asyncio.Lock()

    async def _ready() -> Memory:
        nonlocal started
        if started:
            return mem
        async with start_lock:
            if started:
                return mem
            # Check the DSN is reachable BEFORE opening the pool.
            # psycopg_pool.open() does not raise on a dead DSN — it opens
            # optimistically and the failure surfaces later inside a pool worker,
            # tearing down that request's task group so the client gets *no
            # reply at all*. Probing first turns that into a clean, reportable
            # error (and gives autostart somewhere to hook in).
            dsn = _configured_dsn()
            if dsn and not await _dsn_reachable(dsn) and not await _autostart(dsn):
                raise BackendUnavailableError(
                    f"Continuum memory backend is unreachable at {dsn}. Start it, or set "
                    f"CONTINUUM_MCP_AUTOSTART to a command that does "
                    f"(e.g. 'brew services start postgresql@16')."
                )
            try:
                await mem.start()
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Continuum memory backend failed to start{f' ({dsn})' if dsn else ''}: {exc}"
                ) from exc
            started = True
        return mem

    @server.tool()
    async def recall(query: str, k: int = 8) -> list[dict[str, Any]]:
        """Retrieve up to k memories relevant to the query, best-first."""
        with _observe("recall", query=query, k=k) as obs:
            hits = await _recall(await _ready(), query, k)
            obs("hits=%d", len(hits))
            return hits

    @server.tool()
    async def remember(
        text: str, occurred_at: str | None = None, attribute: str | None = None
    ) -> str:
        """Store a fact or turn in memory.

        occurred_at is an optional ISO date for when the fact became true.
        attribute names what the fact is ABOUT ("residence", "employer", …) —
        tag it and `current` can answer that attribute exactly.
        """
        with _observe("remember", chars=len(text), attribute=attribute) as obs:
            ack = await _remember(await _ready(), text, occurred_at, attribute)
            obs("durable=%s", not ack.startswith("stored IN-MEMORY"))
            return ack

    @server.tool()
    async def current(subject: str, attribute: str, as_of: str | None = None) -> str:
        """The current value for an attribute after supersession (e.g. residence).

        as_of is an optional ISO date to ask what was current back then.
        """
        with _observe("current", attribute=attribute, as_of=as_of) as obs:
            value = await _current(await _ready(), subject, attribute, as_of)
            obs("found=%s", value != "not found")
            return value

    @server.tool()
    async def timeline(
        entity: str,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Bi-temporal history for an entity, oldest→newest (ISO date bounds)."""
        with _observe("timeline", entity=entity) as obs:
            items = await _timeline(await _ready(), entity, since, until)
            obs("items=%d", len(items))
            return items

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
