"""
continuum/stores/postgres/mtm.py
================================
``PostgresMTM`` — mid-term memory persisted in ``memory_nodes`` with
``layer = 'MTM'``, implementing the project's ``MTMProtocol``.

Method surface
--------------
The codebase's runtime ``MTMProtocol`` is structural / ``@runtime_checkable``
(membership is checked by *method name*).  ``PostgresMTM`` exposes the four
required names while shaping each one around the ergonomic
:class:`~continuum.core.types.SummaryBlock` (and ``UUID`` ids) requested for
the MTM surface:

* ``async add_summary(block) -> UUID``
      Accepts a :class:`SummaryBlock` *or* a :class:`MemoryItem` (so
      ``STM.flush_to``, which hands over ``MemoryItem`` objects, keeps
      working unchanged).  Inserts a ``layer='MTM'`` row.
* ``async recent(token_budget) -> list[SummaryBlock]``
      Newest-first, trimmed to a tiktoken budget.
* ``async scan_unprocessed() -> AsyncIterator[SummaryBlock]``
      Streams, oldest-first, every MTM block with **no** ``memory_promotions``
      row — the Promoter's work queue.
* ``async mark_processed(ids) -> None``
      Records a ``memory_promotions`` marker so the block leaves the queue.

Schema reuse (no new migration)
-------------------------------
``memory_nodes`` (migration 001) has no ``tokens`` / ``session_id`` columns,
so those live in the existing ``tags`` JSONB:
``tags = {"tokens": 42, "session_id": "...", "agent_id": "..."}``.
"Processed" is *derived*, not a column: a block is processed once a
``memory_promotions`` row references it (``target_id = memory_nodes.id``) —
exactly the relationship migration 001 already models.

Token counting
--------------
``tiktoken`` (``cl100k_base``) is imported lazily and cached.  If tiktoken is
unavailable a coarse ``len(text)//4`` fallback is used so the module works
everywhere; inject ``token_counter=`` to override.

Connections
-----------
psycopg3 is imported lazily; pass ``conn_factory=`` to unit-test without
psycopg/PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from continuum.core.types import MemoryItem, SummaryBlock
from continuum.db.pgvector_upgrade import to_halfvec_literal

log = logging.getLogger(__name__)

DEFAULT_TABLE = "memory_nodes"
DEFAULT_PROMOTIONS = "memory_promotions"
_MAX_RECENT_SCAN = 500  # hard cap on rows pulled before budget-trimming


# ============================================================================
# Token counting (tiktoken cl100k_base, lazy + cached)
# ============================================================================


@lru_cache(maxsize=1)
def _cl100k() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - exercised via fallback
        log.debug("tiktoken unavailable — using len//4 token estimate")
        return None


def count_tokens(text: str) -> int:
    """Token count via tiktoken ``cl100k_base``; coarse fallback if absent."""
    if not text:
        return 0
    enc = _cl100k()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


# ============================================================================
# PostgresMTM
# ============================================================================


class PostgresMTM:
    """
    Mid-term memory backed by ``memory_nodes`` (``layer='MTM'``).

    Parameters
    ----------
    dsn:
        ``postgresql://…`` DSN. Ignored when *conn_factory* is given.
    conn_factory:
        Async callable → async context manager yielding a connection
        (tests / pools).
    table / promotions_table:
        Override object names (default ``memory_nodes`` /
        ``memory_promotions``).
    embedding_type:
        SQL type the ``embedding`` column was migrated to — ``'halfvec'``
        (post-migration 002, default) or ``'vector'``. Validated to be a
        bare identifier before interpolation.
    token_counter:
        ``(text) -> int``. Defaults to :func:`count_tokens`.
    dedupe_threshold:
        When set (``0 < t ≤ 1``), :meth:`add_summary` first cosine-searches
        existing live MTM blocks; if the nearest is ``≥ t`` similar the new
        block is **not** inserted and the existing block's id is returned
        (dedupe-on-write). Requires the incoming block to carry an embedding;
        ``None`` (default) disables dedupe.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        conn_factory: Any | None = None,
        table: str = DEFAULT_TABLE,
        promotions_table: str = DEFAULT_PROMOTIONS,
        embedding_type: str = "halfvec",
        token_counter: Callable[[str], int] | None = None,
        dedupe_threshold: float | None = None,
    ) -> None:
        if conn_factory is None and dsn is None:
            raise ValueError("Provide either dsn or conn_factory.")
        if not embedding_type.isalpha():
            raise ValueError(f"embedding_type must be a bare identifier, got {embedding_type!r}")
        if dedupe_threshold is not None and not 0.0 < dedupe_threshold <= 1.0:
            raise ValueError(f"dedupe_threshold must be in (0, 1], got {dedupe_threshold!r}")
        self._dsn = dsn
        self._conn_factory = conn_factory
        self._table = table
        self._promotions = promotions_table
        self._embedding_type = embedding_type
        self._count = token_counter or count_tokens
        self._dedupe_threshold = dedupe_threshold

    # ── connection ──────────────────────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[Any]:
        if self._conn_factory is not None:
            async with self._conn_factory() as conn:
                yield conn
            return
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - via conn_factory in tests
            raise ImportError(
                "psycopg3 is required for PostgresMTM.\n"
                "Install it with:  pip install 'psycopg[binary]>=3.1'"
            ) from exc
        # __init__ guarantees (conn_factory or dsn); this branch is only
        # reached when conn_factory is None, so dsn must be set.
        assert self._dsn is not None, "PostgresMTM requires dsn or conn_factory"
        conn = await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True, row_factory=dict_row
        )
        try:
            yield conn
        finally:
            await conn.close()

    # ── add_summary ─────────────────────────────────────────────────────────

    async def add_summary(self, block: SummaryBlock | MemoryItem) -> uuid.UUID:
        """
        Persist *block* as a ``layer='MTM'`` row and return its UUID.

        Accepts a :class:`SummaryBlock` or a :class:`MemoryItem` (the latter
        is what ``STM.flush_to`` hands over). ``tokens`` is computed with the
        configured counter when the incoming block reports 0.
        """
        if isinstance(block, MemoryItem):
            block = SummaryBlock.from_memory_item(block)

        tokens = block.tokens or self._count(block.text)
        block.tokens = tokens

        tags: dict[str, Any] = dict(block.metadata)
        tags["tokens"] = tokens
        if block.session_id is not None:
            tags["session_id"] = block.session_id
        if block.agent_id is not None:
            tags["agent_id"] = block.agent_id
        if block.user_id is not None:
            tags["user_id"] = block.user_id

        params: dict[str, Any] = {
            "id": str(block.id),
            "text": block.text,
            "created_at": block.created_at,
            "tags": json.dumps(tags),
        }
        if block.embedding is not None:
            emb_sql = f"%(embedding)s::{self._embedding_type}"
            params["embedding"] = to_halfvec_literal(block.embedding)
        else:
            emb_sql = "NULL"

        sql = f"""
            INSERT INTO {self._table}
                (id, layer, "text", embedding, created_at, tags)
            VALUES
                (%(id)s, 'MTM', %(text)s, {emb_sql},
                 %(created_at)s, %(tags)s::jsonb)
            ON CONFLICT (id) DO UPDATE
                SET "text"  = EXCLUDED."text",
                    tags    = EXCLUDED.tags,
                    updated_at = now()
        """
        async with self._connect() as conn:
            if self._dedupe_threshold is not None and block.embedding is not None:
                dup = await self._find_duplicate(conn, params["embedding"])
                if dup is not None:
                    log.debug(
                        "MTM add_summary deduped id=%s → existing %s",
                        block.id,
                        dup,
                    )
                    return dup
            await conn.execute(sql, params)
        log.debug("MTM add_summary id=%s tokens=%d", block.id, tokens)
        return block.id

    # ── recent ──────────────────────────────────────────────────────────────

    async def recent(
        self,
        token_budget: int,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        exclude_session_id: str | None = None,
    ) -> list[SummaryBlock]:
        """
        Newest-first MTM blocks whose cumulative tokens fit *token_budget*.

        Rows are fetched ``created_at DESC``; blocks are accumulated until
        the next one would exceed the budget (a single oversized newest
        block is still returned so the caller always gets context).
        """
        if token_budget <= 0:
            return []

        where = ["layer = 'MTM'", "invalidated_at IS NULL"]
        params: dict[str, Any] = {"cap": _MAX_RECENT_SCAN}
        if session_id is not None:
            where.append("tags->>'session_id' = %(sid)s")
            params["sid"] = session_id
        if user_id is not None:
            where.append("tags->>'user_id' = %(uid)s")
            params["uid"] = user_id
        if exclude_session_id is not None:
            where.append("tags->>'session_id' <> %(exclude_sid)s")
            params["exclude_sid"] = exclude_session_id

        sql = f"""
            SELECT id, "text", embedding, created_at, tags
            FROM   {self._table}
            WHERE  {" AND ".join(where)}
            ORDER  BY created_at DESC
            LIMIT  %(cap)s
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()

        out: list[SummaryBlock] = []
        used = 0
        for row in rows:
            blk = self._row_to_block(row, processed=False)
            if out and used + blk.tokens > token_budget:
                break
            out.append(blk)
            used += blk.tokens
            if used >= token_budget:
                break
        return out

    # ── scan_unprocessed ────────────────────────────────────────────────────

    async def scan_unprocessed(
        self,
        *,
        agent_id: str | None = None,
        batch_size: int = 100,
    ) -> AsyncIterator[SummaryBlock]:
        """
        Stream every MTM block with **no** ``memory_promotions`` row.

        Yields oldest-first (``created_at ASC, id ASC``) using keyset
        pagination so the Promoter can process an unbounded backlog in
        bounded ``batch_size`` chunks without holding a long transaction.
        """
        agent_clause = ""
        if agent_id is not None:
            agent_clause = "AND mn.tags->>'agent_id' = %(agent)s"

        sql = f"""
            SELECT mn.id, mn."text", mn.embedding, mn.created_at, mn.tags
            FROM   {self._table} mn
            LEFT   JOIN {self._promotions} mp ON mp.target_id = mn.id
            WHERE  mn.layer = 'MTM'
              AND  mn.invalidated_at IS NULL
              AND  mp.id IS NULL
              {agent_clause}
              AND  (
                    %(after_ts)s::timestamptz IS NULL
                 OR (mn.created_at, mn.id) >
                    (%(after_ts)s::timestamptz, %(after_id)s::uuid)
              )
            ORDER  BY mn.created_at ASC, mn.id ASC
            LIMIT  %(batch)s
        """
        after_ts: datetime | None = None
        after_id: str | None = None
        while True:
            params: dict[str, Any] = {
                "after_ts": after_ts,
                "after_id": after_id,
                "batch": batch_size,
            }
            if agent_id is not None:
                params["agent"] = agent_id
            async with self._connect() as conn:
                cur = await conn.execute(sql, params)
                rows = await cur.fetchall()

            if not rows:
                return
            for row in rows:
                blk = self._row_to_block(row, processed=False)
                after_ts = blk.created_at
                after_id = str(blk.id)
                yield blk
            if len(rows) < batch_size:
                return

    # ── mark_processed ──────────────────────────────────────────────────────

    async def mark_processed(self, ids: Sequence[uuid.UUID | str]) -> None:
        """
        Record a ``memory_promotions`` marker for each id so future
        ``scan_unprocessed`` calls skip it.

        Idempotent: a block that already has a promotions row is left alone
        (``WHERE NOT EXISTS``), so re-marking is safe.
        """
        if not ids:
            return
        str_ids = [str(i) for i in ids]
        sql = f"""
            INSERT INTO {self._promotions} (target_id, op, candidate_text)
            SELECT mn.id, 'NOOP', NULL
            FROM   {self._table} mn
            WHERE  mn.id = ANY(%(ids)s)
              AND  mn.layer = 'MTM'
              AND  NOT EXISTS (
                    SELECT 1 FROM {self._promotions} mp
                    WHERE mp.target_id = mn.id
              )
        """
        async with self._connect() as conn:
            await conn.execute(sql, {"ids": str_ids})
        log.debug("MTM mark_processed %d ids", len(str_ids))

    # ── helpers ─────────────────────────────────────────────────────────────

    async def _find_duplicate(self, conn: Any, embedding_literal: str) -> uuid.UUID | None:
        """
        Return the id of the nearest live MTM block if its cosine similarity
        to *embedding_literal* is ``≥ self._dedupe_threshold``, else ``None``.

        ``<=>`` is pgvector cosine *distance*; similarity = ``1 - distance``.
        """
        sql = f"""
            SELECT id,
                   1 - (embedding <=> %(q)s::{self._embedding_type}) AS sim
            FROM   {self._table}
            WHERE  layer = 'MTM'
              AND  invalidated_at IS NULL
              AND  embedding IS NOT NULL
            ORDER  BY embedding <=> %(q)s::{self._embedding_type}
            LIMIT  1
        """
        cur = await conn.execute(sql, {"q": embedding_literal})
        row = await cur.fetchone()
        if not row or row.get("sim") is None:
            return None
        assert self._dedupe_threshold is not None
        if float(row["sim"]) >= self._dedupe_threshold:
            return uuid.UUID(str(row["id"]))
        return None

    def _row_to_block(self, row: dict[str, Any], *, processed: bool) -> SummaryBlock:
        raw_tags = row.get("tags") or {}
        tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else json.loads(raw_tags)

        raw_ts = row.get("created_at")
        if isinstance(raw_ts, datetime):
            created_at = raw_ts
        elif raw_ts is not None:
            created_at = datetime.fromisoformat(str(raw_ts))
        else:
            created_at = datetime.now(UTC)

        try:
            block_id = uuid.UUID(str(row["id"]))
        except (ValueError, KeyError):
            block_id = uuid.uuid4()

        tokens = int(tags.get("tokens") or 0)
        if tokens == 0:
            tokens = self._count(str(row.get("text") or ""))

        emb = row.get("embedding")
        embedding = list(emb) if isinstance(emb, (list, tuple)) else None

        return SummaryBlock(
            text=str(row.get("text") or ""),
            tokens=tokens,
            id=block_id,
            embedding=embedding,
            session_id=tags.get("session_id"),
            agent_id=tags.get("agent_id"),
            user_id=tags.get("user_id"),
            created_at=created_at,
            processed=processed,
            metadata={
                k: v for k, v in tags.items()
                if k not in ("tokens", "session_id", "agent_id", "user_id")
            },
        )


__all__ = ["PostgresMTM", "count_tokens"]
