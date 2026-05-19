"""
continuum/stores/stm/postgres_stm.py
======================================
``PostgresSTM`` — a ``psycopg3`` async implementation of ``STMProtocol``
that persists conversation messages in a lightweight PostgreSQL table.

Why PostgresSTM instead of InMemorySTM?
----------------------------------------
* **Multi-process / multi-pod agents** — processes share the same Postgres
  cluster so STM survives restarts and is visible across instances.
* **Durable dev sessions** — messages survive the agent process crashing.
* **Audit trail** — raw SQL queries let you inspect what the agent saw.

Schema
------
All sessions are stored in a single ``stm_messages`` table (see
``_DDL_CREATE_TABLE``).  Call ``await stm.ensure_schema()`` once at startup,
or use ``async with PostgresSTM(...) as stm:`` which calls it automatically.

Connection strategy
-------------------
``PostgresSTM`` is designed for **dependency injection** of the connection
factory, which makes unit-testing trivial:

    # Production — raw DSN (one connection per operation, suitable for low volume)
    stm = PostgresSTM(dsn="postgresql://user:pw@host/db")

    # Production — connection pool (recommended for high-throughput agents)
    from psycopg_pool import AsyncConnectionPool
    pool = AsyncConnectionPool("postgresql://...", min_size=2)
    stm  = PostgresSTM(pool=pool)

    # Tests — inject a mock factory (no real DB needed)
    stm = PostgresSTM(conn_factory=my_mock_factory)

``conn_factory`` must be an ``async`` callable that returns an object
usable as an async context manager yielding a ``psycopg.AsyncConnection``.

Token accounting
----------------
``PostgresSTM`` stores a pre-computed ``tokens`` column.  The tokenizer
callable (defaults to whitespace split) is called at ``append`` time if
``item.metadata["tokens"]`` is absent.  Always inject a real tokenizer
(e.g. tiktoken) in production to keep token counts accurate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from continuum.core.protocols import MTMProtocol
from continuum.core.types import MemoryItem, MemoryTier, ProcessingState

log = logging.getLogger(__name__)


# psycopg3 is imported lazily inside _connect() so that the module can be
# imported (and conn_factory-based unit tests can run) without psycopg3
# installed.  Only the DSN / pool code paths actually need the real driver.
def _require_psycopg() -> tuple[Any, Any]:
    """Import psycopg3 or raise a clear error pointing to the install docs."""
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg, dict_row
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "psycopg3 is required for PostgresSTM.\n"
            "Install it with:  pip install 'psycopg[binary]>=3.1'"
        ) from exc


# ---------------------------------------------------------------------------
# Schema DDL — idempotent, safe to run on every startup
# ---------------------------------------------------------------------------

_DDL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS stm_messages (
    id          TEXT        NOT NULL,
    session_id  TEXT        NOT NULL,
    agent_id    TEXT,
    user_id     TEXT,
    role        TEXT        NOT NULL DEFAULT 'user',
    content     TEXT        NOT NULL,
    tokens      INTEGER     NOT NULL DEFAULT 0,
    importance  REAL        NOT NULL DEFAULT 0.5,
    confidence  REAL        NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata    TEXT        NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);
"""

_DDL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_stm_session_created
    ON stm_messages (session_id, created_at ASC);
"""

# ---------------------------------------------------------------------------
# PostgresSTM
# ---------------------------------------------------------------------------

#: Type alias for the injectable connection factory
ConnFactory = Callable[[], Any]  # async context manager → AsyncConnection


class PostgresSTM:
    """
    psycopg3 async ``STMProtocol`` storing messages in PostgreSQL.

    Parameters
    ----------
    dsn:
        PostgreSQL DSN (``postgresql://user:pw@host/db``).
        Ignored when *pool* or *conn_factory* is provided.
    pool:
        ``psycopg_pool.AsyncConnectionPool`` instance.
        Each operation acquires / releases a pooled connection via
        ``pool.connection()``.
    conn_factory:
        Advanced override — an async callable that returns an async context
        manager yielding an ``AsyncConnection``.  Takes precedence over
        *dsn* and *pool*.  Primarily for unit tests.
    max_tokens:
        Total token capacity per session.
        ``window()`` respects this limit; ``append()`` does not enforce it
        (the engine trusts callers to check utilisation and compress/evict).
    reserved_for_response:
        Tokens withheld from the usable window for the model's reply.
        Effective budget = ``max_tokens - reserved_for_response``.
    tokenizer:
        Maps ``content`` → token count.  Defaults to whitespace split.
        **Replace with tiktoken in production.**
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Any | None = None,
        conn_factory: ConnFactory | None = None,
        max_tokens: int = 4096,
        reserved_for_response: int = 512,
        tokenizer: Callable[[str], int] | None = None,
    ) -> None:
        if conn_factory is None and pool is None and dsn is None:
            raise ValueError("Provide at least one of: dsn, pool, or conn_factory.")
        self._dsn = dsn
        self._pool = pool
        self._conn_factory: ConnFactory | None = conn_factory
        self.max_tokens = max_tokens
        self.reserved_for_response = reserved_for_response
        self._tokenizer: Callable[[str], int] = tokenizer or (lambda s: len(s.split()))
        self._init_lock = asyncio.Lock()
        self._initialized = False

    # ── Connection context manager ─────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[Any]:
        """
        Yield a live ``psycopg.AsyncConnection`` configured with ``dict_row``.

        Priority: conn_factory > pool > dsn.
        """
        if self._conn_factory is not None:
            async with self._conn_factory() as conn:
                yield conn
            return

        if self._pool is not None:
            psycopg, dict_row = _require_psycopg()
            async with self._pool.connection() as conn:
                # Ensure dict_row on pooled connections
                conn.row_factory = dict_row
                yield conn
            return

        # Raw DSN — open and close per operation (fine for low-volume use)
        psycopg, dict_row = _require_psycopg()
        conn = await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True, row_factory=dict_row
        )
        try:
            yield conn
        finally:
            await conn.close()

    # ── Schema bootstrap ───────────────────────────────────────────────────

    async def ensure_schema(self) -> None:
        """
        Create ``stm_messages`` and its index if they do not exist.

        Thread-safe via an ``asyncio.Lock``; safe to call multiple times —
        all DDL uses ``IF NOT EXISTS``.
        """
        async with self._init_lock:
            if self._initialized:
                return
            async with self._connect() as conn:
                await conn.execute(_DDL_CREATE_TABLE)
                await conn.execute(_DDL_CREATE_INDEX)
            self._initialized = True
            log.debug("stm_messages schema ensured")

    async def _ensure(self) -> None:
        """Lazily call ``ensure_schema`` before the first operation."""
        if not self._initialized:
            await self.ensure_schema()

    # ── Async context manager ──────────────────────────────────────────────

    async def __aenter__(self) -> PostgresSTM:
        await self.ensure_schema()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass  # connections are closed after each operation / by the pool

    # ── STMProtocol — append ───────────────────────────────────────────────

    async def append(self, item: MemoryItem) -> None:
        """
        Persist *item* as a row in ``stm_messages``.

        Sets ``item.tier = MemoryTier.STM``.  Token count is taken from
        ``item.metadata["tokens"]`` if present; otherwise computed by the
        configured tokenizer and stored back in the metadata.
        """
        await self._ensure()
        item.tier = MemoryTier.STM

        tokens: int = int(item.metadata.get("tokens") or self._tokenizer(item.content))
        role: str = str(item.metadata.get("role", "user"))

        # Store enriched metadata (include computed tokens for round-trips)
        meta: dict[str, Any] = dict(item.metadata)
        meta["tokens"] = tokens
        meta["role"] = role

        sql = """
            INSERT INTO stm_messages
                (id, session_id, agent_id, user_id, role, content,
                 tokens, importance, confidence, created_at, metadata)
            VALUES
                (%(id)s, %(session_id)s, %(agent_id)s, %(user_id)s,
                 %(role)s, %(content)s, %(tokens)s, %(importance)s,
                 %(confidence)s, %(created_at)s, %(metadata)s)
            ON CONFLICT (id) DO UPDATE
                SET content    = EXCLUDED.content,
                    tokens     = EXCLUDED.tokens,
                    importance = EXCLUDED.importance,
                    metadata   = EXCLUDED.metadata
        """
        async with self._connect() as conn:
            await conn.execute(
                sql,
                {
                    "id": item.id,
                    "session_id": item.session_id or "unknown",
                    "agent_id": item.agent_id,
                    "user_id": item.user_id,
                    "role": role,
                    "content": item.content,
                    "tokens": tokens,
                    "importance": item.importance,
                    "confidence": item.confidence,
                    "created_at": item.created_at,
                    "metadata": json.dumps(meta),
                },
            )
        log.debug(
            "STM append session=%s id=%s role=%s tokens=%d",
            item.session_id,
            item.id,
            role,
            tokens,
        )

    # ── STMProtocol — window ───────────────────────────────────────────────

    async def window(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> Sequence[MemoryItem]:
        """
        Return messages for *session_id* that fit within the token budget.

        Fetches newest-first and accumulates until the budget is exhausted,
        then reverses to yield oldest-first (chat-prompt order).

        Parameters
        ----------
        max_tokens:
            Explicit budget cap.  When ``None``, uses
            ``self.max_tokens - self.reserved_for_response``.
        """
        await self._ensure()
        budget = (
            max_tokens
            if max_tokens is not None
            else max(self.max_tokens - self.reserved_for_response, 0)
        )

        sql = """
            SELECT id, session_id, agent_id, user_id, role, content,
                   tokens, importance, confidence, created_at, metadata
            FROM   stm_messages
            WHERE  session_id = %(session_id)s
            ORDER  BY created_at DESC
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql, {"session_id": session_id})
            rows: list[dict[str, Any]] = await cur.fetchall()

        items: list[MemoryItem] = []
        used = 0
        for row in rows:
            t = int(row["tokens"])
            if used + t > budget:
                break
            items.append(self._row_to_item(row))
            used += t

        items.reverse()  # oldest-first
        return items

    # ── STMProtocol — get_recent ───────────────────────────────────────────

    async def get_recent(
        self,
        session_id: str,
        n: int = 10,
    ) -> Sequence[MemoryItem]:
        """
        Return the *n* most-recent messages, oldest-first, ignoring token budget.
        """
        await self._ensure()
        sql = """
            SELECT id, session_id, agent_id, user_id, role, content,
                   tokens, importance, confidence, created_at, metadata
            FROM   stm_messages
            WHERE  session_id = %(session_id)s
            ORDER  BY created_at DESC
            LIMIT  %(n)s
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql, {"session_id": session_id, "n": n})
            rows: list[dict[str, Any]] = await cur.fetchall()

        items = [self._row_to_item(row) for row in rows]
        items.reverse()  # oldest-first
        return items

    # ── STMProtocol — flush_to ─────────────────────────────────────────────

    async def flush_to(
        self,
        session_id: str,
        target: MTMProtocol,
    ) -> int:
        """
        Transfer all messages for *session_id* to MTM *target*, then delete.

        The delete is issued only for items that were successfully handed
        to ``target.add_summary`` — partial failures result in partial
        deletes so no messages are lost.
        """
        await self._ensure()
        # Fetch without a token cap so we get everything
        sql_select = """
            SELECT id, session_id, agent_id, user_id, role, content,
                   tokens, importance, confidence, created_at, metadata
            FROM   stm_messages
            WHERE  session_id = %(session_id)s
            ORDER  BY created_at ASC
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql_select, {"session_id": session_id})
            rows: list[dict[str, Any]] = await cur.fetchall()

        if not rows:
            return 0

        items = [self._row_to_item(row) for row in rows]
        transferred_ids: list[str] = []

        for item in items:
            item.processing_state = ProcessingState.UNPROCESSED
            try:
                await target.add_summary(item)
                transferred_ids.append(item.id)
            except Exception:
                log.exception(
                    "flush_to: failed to transfer %s — aborting after %d/%d items",
                    item.id,
                    len(transferred_ids),
                    len(items),
                )
                break

        if transferred_ids:
            await self._delete_ids(transferred_ids)

        log.info(
            "flush_to session=%s: transferred %d/%d",
            session_id,
            len(transferred_ids),
            len(items),
        )
        return len(transferred_ids)

    # ── STMProtocol — clear ────────────────────────────────────────────────

    async def clear(self, session_id: str) -> None:
        """Delete all STM rows for *session_id*."""
        await self._ensure()
        async with self._connect() as conn:
            await conn.execute(
                "DELETE FROM stm_messages WHERE session_id = %(session_id)s",
                {"session_id": session_id},
            )
        log.debug("STM clear session=%s", session_id)

    # ── STMProtocol — stats ────────────────────────────────────────────────

    async def stats(self, session_id: str) -> dict[str, Any]:
        """
        Return utilisation statistics for *session_id*.

        Keys: ``session_id``, ``message_count``, ``total_tokens``,
        ``max_tokens``, ``utilization``, ``budget_remaining``.
        """
        await self._ensure()
        sql = """
            SELECT COUNT(*)                    AS message_count,
                   COALESCE(SUM(tokens), 0)    AS total_tokens
            FROM   stm_messages
            WHERE  session_id = %(session_id)s
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql, {"session_id": session_id})
            row: dict[str, Any] | None = await cur.fetchone()

        eff_budget = max(self.max_tokens - self.reserved_for_response, 0)
        total = int(row["total_tokens"]) if row else 0
        count = int(row["message_count"]) if row else 0
        return {
            "session_id": session_id,
            "message_count": count,
            "total_tokens": total,
            "max_tokens": self.max_tokens,
            "utilization": total / eff_budget if eff_budget else 0.0,
            "budget_remaining": max(eff_budget - total, 0),
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _row_to_item(self, row: dict[str, Any]) -> MemoryItem:
        """Convert a ``stm_messages`` row dict to a ``MemoryItem``."""
        raw_meta = row.get("metadata") or "{}"
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta)
        meta.setdefault("role", row.get("role", "user"))

        raw_ts = row.get("created_at")
        if isinstance(raw_ts, datetime):
            created_at = raw_ts
        elif raw_ts is not None:
            created_at = datetime.fromisoformat(str(raw_ts))
        else:
            created_at = datetime.now(UTC)

        return MemoryItem(
            id=str(row["id"]),
            content=str(row["content"]),
            tier=MemoryTier.STM,
            importance=float(row.get("importance", 0.5)),
            confidence=float(row.get("confidence", 1.0)),
            created_at=created_at,
            session_id=str(row["session_id"]),
            agent_id=row.get("agent_id"),
            user_id=row.get("user_id"),
            metadata=meta,
        )

    async def _delete_ids(self, ids: list[str]) -> None:
        """Hard-delete a list of message ids from the table."""
        if not ids:
            return
        # psycopg3 handles Python lists as SQL arrays for ANY(...)
        sql = "DELETE FROM stm_messages WHERE id = ANY(%(ids)s)"
        async with self._connect() as conn:
            await conn.execute(sql, {"ids": ids})
