"""
continuum/stores/postgres/ltm.py
================================
``PostgresLTM`` — bi-temporal long-term memory over ``memory_nodes`` /
``memory_edges`` (migrations 001 + 002 + 003), implementing the project's
``LTMProtocol``.

Bi-temporal contract
--------------------
Rows are **never deleted**. ``invalidate`` closes a fact's transaction-time
window (``invalidated_at``); every query filters ``invalidated_at IS NULL``
so historical versions stay queryable but never leak into "current" reads.

Method surface vs. the runtime Protocol
---------------------------------------
``LTMProtocol`` is ``@runtime_checkable`` (membership = method *name*).
``PostgresLTM`` exposes the four+one required names while shaping them to the
ergonomic ``UUID`` / :class:`~continuum.core.types.Edge` API requested for
the LTM surface:

* ``upsert(item)            -> UUID``     (protocol annotates ``-> str``)
* ``update(id, patch)       -> None``     (protocol annotates ``-> MemoryItem``)
* ``invalidate(id, at)      -> None``
* ``search_hybrid(q, k)     -> list[ScoredItem]``  (single SQL: dense⊕sparse⊕RRF)
* ``neighbors(node_id, hops)-> list[Edge]``  (recursive CTE; also accepts the
  protocol's ``depth=`` kwarg so ``ContinuumSession._incremental_index`` —
  which calls ``neighbors(id, depth=1)`` — keeps working unchanged)

Connection pooling
------------------
``psycopg_pool.AsyncConnectionPool`` is used for the DSN path (lazily opened,
reused, closed by :meth:`aclose`). A pre-built pool can be injected. Unit
tests inject ``conn_factory`` and need neither psycopg3 nor psycopg_pool.

Prepared statements
-------------------
The two hot read paths (:meth:`search_hybrid`, :meth:`neighbors`) execute
with ``prepare=True`` so psycopg3 server-prepares them on first use and
re-binds the plan thereafter.

Query-field mapping
-------------------
``Query`` has no ``layers`` / ``filters`` attributes; this maps the spec's
intent onto the real fields:
* "Query.layers filter" → ``query.tiers`` (``MemoryTier.name`` → ``layer``,
  e.g. ``MemoryTier.LTM`` → ``'LTM'`` — that's what the stores write).
* "Query.filters (JSONB tags)" → ``query.metadata_filter`` (``tags @> …``).
* ``query.as_of`` is honoured as a bi-temporal point-in-time when set.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from continuum.core.types import (
    BiTemporalRange,
    Edge,
    MemoryItem,
    MemoryTier,
    PrunePolicy,
    PruneReport,
    Query,
    ScoreBreakdown,
    ScoredItem,
)
from continuum.db.pgvector_upgrade import to_halfvec_literal

log = logging.getLogger(__name__)

DEFAULT_TABLE = "memory_nodes"
DEFAULT_EDGES = "memory_edges"
DEFAULT_RRF_K = 60
DEFAULT_TRGM_THRESHOLD = 0.25


#: Session ``hnsw.ef_search`` for the dense channel. pgvector's default of 40 is
#: tuned for speed and loses real memories: measured on 3,020 clustered personal
#: facts, only 13/20 needles reached the candidate pool at 40, versus 20/20 at
#: 400 — for **no** latency benefit (p50 0.9ms -> 0.5ms, p95 12.2ms -> 0.7ms; a
#: fuller graph walk terminates more predictably than a stuck one). Against a
#: ~100ms recall dominated by the embedder this is free. Lower it only if a very
#: large store makes the index scan itself expensive.
def _default_ef_search() -> int:
    """``CONTINUUM_HNSW_EF_SEARCH``, else 400.

    Environment-readable so the value can be swept without editing code — the
    first attempt to compare settings monkey-patched this module and silently
    measured the same number four times, because a signature default binds when
    the function is defined, not when it is called.
    """
    import os

    raw = os.environ.get("CONTINUUM_HNSW_EF_SEARCH", "")
    try:
        return max(1, min(1000, int(raw))) if raw.strip() else 400
    except ValueError:
        return 400


#: pgvector caps ``hnsw.ef_search`` at 1000; anything higher is rejected outright.
MAX_HNSW_EF_SEARCH = 1000
DEFAULT_HNSW_EF_SEARCH = 400
_NEIGHBOR_CAP = 500

# Columns a partial ``update(patch)`` is allowed to touch. The keys are the
# caller-facing names; the values are the *fixed* SQL column names — so even
# though we build SQL dynamically, the column list is a hard allowlist and
# never interpolates caller input.
_UPDATABLE: dict[str, str] = {
    "content": '"text"',
    "text": '"text"',
    "kind": "kind",
    "confidence": "confidence",
    "importance": "importance",
    "tags": "tags",
    "embedding": "embedding",
    "valid_from": "valid_from",
    "valid_to": "valid_to",
}
_RESERVED_TAG_KEYS = {"kind", "role", "tokens", "source_ids"}


def _require_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg, dict_row
    except ImportError as exc:  # pragma: no cover - via conn_factory in tests
        raise ImportError(
            "psycopg3 is required for PostgresLTM.\n"
            "Install it with:  pip install 'psycopg[binary,pool]>=3.1'"
        ) from exc


def _require_pool() -> Any:
    try:
        from psycopg_pool import AsyncConnectionPool

        return AsyncConnectionPool
    except ImportError as exc:  # pragma: no cover - via conn_factory in tests
        raise ImportError(
            "psycopg_pool is required for PostgresLTM's DSN pooling.\n"
            "Install it with:  pip install 'psycopg[binary,pool]>=3.1'"
        ) from exc


def _as_uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _is_live_text_conflict(exc: BaseException) -> bool:
    """True when a write lost the race to migration 007's uniqueness constraint.

    Matched on the index name rather than the SQLSTATE alone, so an unrelated
    unique violation (a genuine bug) still surfaces instead of being quietly
    swallowed as a duplicate.
    """
    return "memory_nodes_live_text_uniq" in str(exc)


class PostgresLTM:
    """
    Bi-temporal LTM store over ``memory_nodes`` / ``memory_edges``.

    Parameters
    ----------
    dsn:
        ``postgresql://…``. A pooled connection manager is created lazily
        from it. Ignored when *pool* or *conn_factory* is given.
    pool:
        Pre-built ``psycopg_pool.AsyncConnectionPool`` (not owned — caller
        closes it).
    conn_factory:
        Async callable → async context manager yielding a connection
        (tests). Highest priority.
    embedding_type:
        ``'halfvec'`` (post-migration 002, default) or ``'vector'``.
        Validated to be a bare identifier before interpolation.
    rrf_k:
        RRF constant (default 60, per Cormack et al. 2009).
    trgm_threshold:
        Session ``pg_trgm.similarity_threshold`` for the sparse channel.
    hnsw_ef_search:
        Session ``hnsw.ef_search`` for the dense channel — how hard pgvector
        looks before giving up. Its default (40) silently drops good matches
        once the planner starts using the index; see
        :data:`DEFAULT_HNSW_EF_SEARCH`.
    pool_min_size / pool_max_size:
        Sizing for the lazily-created DSN pool.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Any | None = None,
        conn_factory: Any | None = None,
        table: str = DEFAULT_TABLE,
        edges_table: str = DEFAULT_EDGES,
        embedding_type: str = "halfvec",
        rrf_k: int = DEFAULT_RRF_K,
        trgm_threshold: float = DEFAULT_TRGM_THRESHOLD,
        hnsw_ef_search: int | None = None,
        namespace: str = "default",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        if conn_factory is None and pool is None and dsn is None:
            raise ValueError("Provide one of: dsn, pool, or conn_factory.")
        if not embedding_type.isalpha():
            raise ValueError(f"embedding_type must be a bare identifier, got {embedding_type!r}")
        # Tenant scope. Every read filters on it and every write stamps it, so
        # one database can hold isolated stores. Migration 005 defaults existing
        # rows and this field to 'default', so an unscoped caller is unchanged.
        self._namespace = namespace
        self._dsn = dsn
        self._pool = pool
        self._owns_pool = False
        self._conn_factory = conn_factory
        self._table = table
        self._edges = edges_table
        self._embedding_type = embedding_type
        self._rrf_k = rrf_k
        self._trgm_threshold = trgm_threshold
        self._hnsw_ef_search = min(
            MAX_HNSW_EF_SEARCH,
            max(1, int(hnsw_ef_search) if hnsw_ef_search is not None else _default_ef_search()),
        )
        self._pool_min = pool_min_size
        self._pool_max = pool_max_size

    # ── connection / lifecycle ──────────────────────────────────────────────

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            pool_cls = _require_pool()
            self._pool = pool_cls(
                self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                open=False,
            )
            self._owns_pool = True
        # psycopg_pool.AsyncConnectionPool.open() is idempotent — calling it
        # on an already-open pool is a documented no-op, so no flag tracking.
        await self._pool.open()
        return self._pool

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[Any]:
        if self._conn_factory is not None:
            async with self._conn_factory() as conn:
                yield conn
            return
        _, dict_row = _require_psycopg()
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            conn.row_factory = dict_row
            yield conn

    async def aclose(self) -> None:
        """Close the pool if this instance created it."""
        if self._owns_pool and self._pool is not None:
            try:
                await self._pool.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("error closing LTM pool", exc_info=True)
            self._pool = None

    async def __aenter__(self) -> PostgresLTM:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    @staticmethod
    async def _exec(conn: Any, sql: str, params: Any, *, prepare: bool = False) -> Any:
        """psycopg3 ``conn.execute`` (server-prepared for hot reads)."""
        return await conn.execute(sql, params, prepare=prepare)

    # ── upsert ──────────────────────────────────────────────────────────────

    async def upsert(self, item: MemoryItem) -> uuid.UUID:
        """
        Insert *item* as a ``layer='LTM'`` row, or update the row with the
        same id (``ON CONFLICT (id) DO UPDATE``). Sets ``updated_at`` on both
        paths; ``created_at`` is preserved on update.

        EXPLAIN ANALYZE (single-row upsert, PK present)::

            Insert on memory_nodes  (cost=0.00..0.01 rows=0)
              Conflict Resolution: UPDATE
              Conflict Arbiter Indexes: memory_nodes_pkey
              ->  Result
            Execution Time: ~0.3 ms
        """
        item.tier = MemoryTier.LTM
        try:
            node_id = _as_uuid(item.id)
        except (ValueError, AttributeError):
            node_id = uuid.uuid4()
        item.id = str(node_id)

        meta = dict(item.metadata)
        kind = meta.get("kind")
        source_ids = meta.get("source_ids")
        tags = {k: v for k, v in meta.items() if k not in _RESERVED_TAG_KEYS}

        vf = item.valid_range.valid_from if item.valid_range else None
        vt = item.valid_range.valid_to if item.valid_range else None

        params: dict[str, Any] = {
            "id": str(node_id),
            "text": item.content,
            "kind": kind,
            "confidence": item.confidence,
            "importance": item.importance,
            "valid_from": vf,
            "valid_to": vt,
            "source_ids": list(source_ids) if source_ids else None,
            "tags": json.dumps(tags),
            "namespace": self._namespace,
        }
        if item.embedding is not None:
            emb_sql = f"%(embedding)s::{self._embedding_type}"
            params["embedding"] = to_halfvec_literal(item.embedding)
        else:
            emb_sql = "NULL"

        sql = f"""
            INSERT INTO {self._table}
                (id, layer, "text", kind, confidence, importance,
                 embedding, source_ids, valid_from, valid_to, tags,
                 namespace, created_at, updated_at)
            VALUES
                (%(id)s, 'LTM', %(text)s, %(kind)s, %(confidence)s,
                 %(importance)s, {emb_sql}, %(source_ids)s,
                 %(valid_from)s, %(valid_to)s, %(tags)s::jsonb,
                 %(namespace)s, now(), now())
            ON CONFLICT (id) DO UPDATE SET
                "text"      = EXCLUDED."text",
                kind        = EXCLUDED.kind,
                confidence  = EXCLUDED.confidence,
                importance  = EXCLUDED.importance,
                embedding   = EXCLUDED.embedding,
                source_ids  = EXCLUDED.source_ids,
                valid_from  = EXCLUDED.valid_from,
                valid_to    = EXCLUDED.valid_to,
                tags        = EXCLUDED.tags,
                updated_at  = now()
        """
        try:
            async with self._connect() as conn:
                await self._exec(conn, sql, params)
        except Exception as exc:
            # Migration 007 allows one live row per (namespace, text). Losing
            # that race means a concurrent writer stored this exact fact a
            # moment ago — which is the outcome we wanted, so reinforce theirs
            # instead of failing the caller's write. Any other error re-raises.
            if not _is_live_text_conflict(exc):
                raise
            existing = await self.touch_duplicate(
                item.content or "", valid_from=vf, attribute=(item.metadata or {}).get("attribute")
            )
            log.debug("LTM upsert lost the duplicate race — reinforced %s", existing)
            return existing or node_id
        log.debug("LTM upsert id=%s kind=%s", node_id, kind)
        return node_id

    # ── update (partial) ────────────────────────────────────────────────────

    async def update(self, id: uuid.UUID | str, patch: dict[str, Any]) -> None:
        """
        Apply a partial update — only keys in *patch* that map to a known
        column are written; ``updated_at`` is always bumped.

        Unknown keys are ignored (logged at debug). The column list comes
        from the fixed ``_UPDATABLE`` allowlist, so dynamic SQL never
        interpolates caller-controlled identifiers.
        """
        sets: list[str] = []
        params: dict[str, Any] = {"id": str(_as_uuid(id))}
        for key, value in patch.items():
            col = _UPDATABLE.get(key)
            if col is None:
                log.debug("LTM update: ignoring non-updatable field %r", key)
                continue
            pname = f"v_{key}"
            if col == "embedding" and value is not None:
                sets.append(f"embedding = %({pname})s::{self._embedding_type}")
                params[pname] = to_halfvec_literal(value)
            elif col == "tags":
                sets.append(f"tags = %({pname})s::jsonb")
                params[pname] = json.dumps(value)
            else:
                sets.append(f"{col} = %({pname})s")
                params[pname] = value

        if not sets:
            log.debug("LTM update: no updatable fields in patch — no-op")
            return

        sql = (
            f"UPDATE {self._table} "
            f"SET {', '.join(sets)}, updated_at = now() "
            f"WHERE id = %(id)s AND invalidated_at IS NULL"
        )
        async with self._connect() as conn:
            await self._exec(conn, sql, params)
        log.debug("LTM update id=%s fields=%s", params["id"], list(patch))

    # ── invalidate (bi-temporal close, no delete) ───────────────────────────

    async def invalidate(self, id: uuid.UUID | str, at: datetime | None = None) -> None:
        """
        Close *id*'s transaction-time window by setting ``invalidated_at``.

        The row is **kept** (bi-temporal history). Re-invalidating an already
        -invalidated row is a no-op (``AND invalidated_at IS NULL``).
        """
        when = at or datetime.now(UTC)
        sql = (
            f"UPDATE {self._table} "
            f"SET invalidated_at = %(at)s "
            f"WHERE id = %(id)s AND invalidated_at IS NULL"
        )
        async with self._connect() as conn:
            await self._exec(conn, sql, {"id": str(_as_uuid(id)), "at": when})
        log.debug("LTM invalidate id=%s at=%s", id, when)

    # ── duplicate suppression (reinforce, don't re-insert) ──────────────────

    async def touch_duplicate(
        self,
        text: str,
        *,
        valid_from: datetime | None = None,
        attribute: str | None = None,
    ) -> uuid.UUID | None:
        """Reinforce an existing identical memory; return its id, else ``None``.

        People restate things — across sessions, and automatic capture will see
        the same sentence again. Inserting a fresh row each time costs a row and
        an embedding (~77ms) forever, for a fact already known. Retrieval dedups
        at read, so the copies were invisible but never free.

        Restating is *evidence*, not noise, so the existing row is bumped
        (``access_count``, ``last_access``) rather than left alone — the scorer
        already reads reinforcement as mild recency strength. Anything the
        original lacked and this telling supplies (a valid time, an attribute)
        is filled in; nothing already recorded is overwritten, because the first
        telling is not less true than the second.

        Lookup and update are one statement, so they cannot interleave once a
        row exists. This is **not** a uniqueness guarantee: writers racing to
        store a brand-new fact all find nothing and all insert. Sequential
        writes — one user, one hook per prompt — collapse to a single row;
        genuinely simultaneous ones can leave a few copies, which the read path
        dedups anyway.

        Closing it properly needs a partial unique index on
        ``(namespace, md5(text)) WHERE invalidated_at IS NULL`` — deliberately
        NOT shipped: that index cannot be built on any store that already holds
        duplicates, which is every store written before this method existed, so
        the migration would hard-fail on upgrade. Retiring the existing copies
        first would be a silent data change to fix a storage-only defect. If you
        want it on a store you know is clean, the index above plus catching the
        unique violation here is the whole implementation.

        Exact text only — near-duplicates are the supersession decider's job,
        not a string comparison's.
        """
        sql = f"""
            UPDATE {self._table} AS m
            SET    access_count = m.access_count + 1,
                   last_access  = now(),
                   valid_from   = COALESCE(m.valid_from, %(valid_from)s::timestamptz),
                   tags         = CASE
                       WHEN %(attribute)s::text IS NOT NULL AND m.tags->>'attribute' IS NULL
                       THEN jsonb_set(COALESCE(m.tags, '{{}}'::jsonb), '{{attribute}}',
                                      to_jsonb(%(attribute)s::text))
                       ELSE m.tags END,
                   updated_at   = now()
            WHERE  m.id = (
                       SELECT id FROM {self._table}
                       WHERE  namespace = %(ns)s
                         AND  "text" = %(text)s
                         AND  invalidated_at IS NULL
                       ORDER  BY created_at DESC
                       LIMIT  1
                   )
            RETURNING m.id
        """
        params = {
            "ns": self._namespace,
            "text": text,
            "valid_from": valid_from,
            "attribute": attribute,
        }
        async with self._connect() as conn:
            cur = await self._exec(conn, sql, params)
            row = await cur.fetchone()
        if not row:
            return None
        log.debug("LTM duplicate reinforced ns=%s", self._namespace)
        return _as_uuid(row["id"])

    # ── embedding backfill (rows written by a sparse-only writer) ───────────

    async def rows_missing_embeddings(self, limit: int = 500) -> list[tuple[uuid.UUID, str]]:
        """Live rows in this namespace that have no vector yet, oldest first.

        The per-prompt capture hook is deliberately sparse — a fresh process
        cannot load a 2.3GB model every turn — so everything it writes lands
        with a NULL embedding and is invisible to dense search *forever*. The
        memories are there; semantic recall simply cannot see them. This is the
        read side of fixing that.
        """
        sql = f"""
            SELECT id, "text" FROM {self._table}
            WHERE  namespace = %(ns)s
              AND  invalidated_at IS NULL
              AND  embedding IS NULL
              AND  "text" <> ''
            ORDER  BY created_at ASC
            LIMIT  %(lim)s
        """
        async with self._connect() as conn:
            cur = await self._exec(conn, sql, {"ns": self._namespace, "lim": max(1, limit)})
            rows = await cur.fetchall()
        return [(_as_uuid(r["id"]), str(r["text"])) for r in rows]

    async def set_embeddings(self, vectors: dict[uuid.UUID, list[float]]) -> int:
        """Attach vectors to existing rows. Only fills NULLs — never overwrites."""
        if not vectors:
            return 0
        sql = (
            f"UPDATE {self._table} SET embedding = %(vec)s::{self._embedding_type} "
            f"WHERE id = %(id)s AND embedding IS NULL"
        )
        n = 0
        async with self._connect() as conn:
            for node_id, vec in vectors.items():
                await self._exec(conn, sql, {"id": str(node_id), "vec": to_halfvec_literal(vec)})
                n += 1
        log.info("LTM backfilled %d embedding(s) ns=%s", n, self._namespace)
        return n

    # ── prune (bulk invalidate by policy — still no DELETE) ─────────────────

    async def prune(
        self,
        policy: PrunePolicy,
        *,
        dry_run: bool = True,
        now: datetime | None = None,
    ) -> PruneReport:
        """Retire every row matching *policy*, within this store's namespace.

        Forgetting is the same bi-temporal close as :meth:`invalidate`, applied
        in bulk: rows are **not** deleted, they stop being visible. Every read
        filters ``invalidated_at IS NULL``, so a pruned fact leaves `recall`,
        `current` *and* `timeline` together — there is no half-forgotten state
        where history disagrees with the current answer.

        ``dry_run=True`` (the default) selects and reports without writing. A
        destructive maintenance operation should have to be asked for twice.
        """
        at = now or datetime.now(UTC)
        cutoff = at - policy.unused_for
        where = [
            "namespace = %(ns)s",
            "invalidated_at IS NULL",
            "COALESCE(last_access, created_at) <= %(cutoff)s",
        ]
        params: dict[str, Any] = {
            "ns": self._namespace,
            "cutoff": cutoff,
            "lim": policy.limit,
        }
        if policy.superseded_only:
            # A closed valid-time window in the past = the fact is no longer true.
            where.append("valid_to IS NOT NULL AND valid_to <= %(at)s")
            params["at"] = at
        if policy.contains is not None:
            # ILIKE with the pattern BOUND, never interpolated — this string
            # comes from a caller who may be pasting back a memory's own text.
            where.append('"text" ILIKE %(contains)s')
            params["contains"] = f"%{policy.contains}%"
        if policy.max_importance is not None:
            where.append("COALESCE(importance, 0) <= %(max_imp)s")
            params["max_imp"] = policy.max_importance
        if policy.max_access_count is not None:
            where.append("COALESCE(access_count, 0) <= %(max_acc)s")
            params["max_acc"] = policy.max_access_count

        # Oldest-touched first, so a `limit`-capped run forgets the coldest rows
        # rather than an arbitrary slice.
        select = (
            f'SELECT id, "text" FROM {self._table} '
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY COALESCE(last_access, created_at) ASC "
            f"LIMIT %(lim)s"
        )
        async with self._connect() as conn:
            cur = await self._exec(conn, select, params)
            rows = list(await cur.fetchall())
            matched = len(rows)
            samples = tuple(str(r["text"])[:80] for r in rows[:5])
            if dry_run or not rows:
                log.info("LTM prune ns=%s matched=%d dry_run=%s", self._namespace, matched, dry_run)
                return PruneReport(matched=matched, pruned=0, dry_run=dry_run, samples=samples)

            update = (
                f"UPDATE {self._table} SET invalidated_at = %(at)s "
                f"WHERE id = ANY(%(ids)s) AND invalidated_at IS NULL"
            )
            ids = [_as_uuid(r["id"]) for r in rows]
            await self._exec(conn, update, {"at": at, "ids": ids})
        log.info("LTM prune ns=%s matched=%d pruned=%d", self._namespace, matched, matched)
        return PruneReport(matched=matched, pruned=matched, dry_run=False, samples=samples)

    # ── hybrid search (single SQL: dense ⊕ sparse ⊕ RRF) ────────────────────

    async def search_hybrid(self, q: Query, k: int = 10) -> list[ScoredItem]:
        """
        Dense (halfvec ``<=>``) + sparse (pg_trgm ``similarity``) fused with
        Reciprocal Rank Fusion **in one SQL statement**, current rows only.

        Both channels are over-fetched to ``max(3k, 20)`` candidates so a
        document ranked modestly by one retriever but well by the other can
        still surface. Channels with no input (no ``q.embedding`` / empty
        ``q.text``) are omitted; if neither is available, returns ``[]``.

        EXPLAIN ANALYZE (both channels, ~10⁵ live rows)::

            Limit
              ->  Sort  (Sort Key: (rrf) DESC)
                    ->  Hash Left Join (dense)  ->  Hash Left Join (sparse)
                          ->  Bitmap Heap Scan on memory_nodes n
                                Recheck: (invalidated_at IS NULL)
              dense  CTE: Index Scan using memory_nodes_embedding_hnsw_idx
              sparse CTE: Bitmap Index Scan on memory_nodes_text_trgm_live_idx
            Execution Time: ~7 ms
        """
        layers = self._layers(q)
        cand = max(3 * k, 20)
        params: dict[str, Any] = {"k": k, "cand": cand, "layers": layers}

        # Optional bi-temporal point-in-time + JSONB tag filter, shared by
        # both channels and the outer query.
        extra = ["n.invalidated_at IS NULL", "n.namespace = %(ns)s"]
        cte_extra = ["invalidated_at IS NULL", "namespace = %(ns)s"]
        params["ns"] = self._namespace
        if q.metadata_filter:
            extra.append("n.tags @> %(filt)s::jsonb")
            cte_extra.append("tags @> %(filt)s::jsonb")
            params["filt"] = json.dumps(q.metadata_filter)
        if q.as_of is not None:
            params["as_of"] = q.as_of
            extra.append(
                "(n.valid_from IS NULL OR n.valid_from <= %(as_of)s) "
                "AND (n.valid_to IS NULL OR n.valid_to > %(as_of)s)"
            )
            cte_extra.append(
                "(valid_from IS NULL OR valid_from <= %(as_of)s) "
                "AND (valid_to IS NULL OR valid_to > %(as_of)s)"
            )
        cte_where = " AND ".join(["layer = ANY(%(layers)s)", *cte_extra])

        ctes: list[str] = []
        join_clause = ""
        present: list[str] = []

        if q.embedding:
            params["q"] = to_halfvec_literal(q.embedding)
            et = self._embedding_type
            ctes.append(
                f"""dense AS (
                    SELECT id,
                           row_number() OVER (
                               ORDER BY embedding <=> %(q)s::{et}
                           ) AS rk
                    FROM   {self._table}
                    WHERE  {cte_where} AND embedding IS NOT NULL
                    ORDER  BY embedding <=> %(q)s::{et}
                    LIMIT  %(cand)s
                )"""
            )
            join_clause += " LEFT JOIN dense d ON d.id = n.id"
            present.append("d.id IS NOT NULL")

        if q.text and q.text.strip():
            params["t"] = q.text
            ctes.append(
                f"""sparse AS (
                    SELECT id,
                           row_number() OVER (
                               ORDER BY similarity("text", %(t)s) DESC
                           ) AS rk
                    FROM   {self._table}
                    WHERE  {cte_where} AND "text" %% %(t)s
                    ORDER  BY similarity("text", %(t)s) DESC
                    LIMIT  %(cand)s
                )"""
            )
            join_clause += " LEFT JOIN sparse s ON s.id = n.id"
            present.append("s.id IS NOT NULL")

        if not ctes:
            return []

        d_term = (
            f"COALESCE(1.0/({self._rrf_k} + d.rk), 0)" if "d.id IS NOT NULL" in present else "0"
        )
        s_term = (
            f"COALESCE(1.0/({self._rrf_k} + s.rk), 0)" if "s.id IS NOT NULL" in present else "0"
        )

        sql = f"""
            WITH {", ".join(ctes)}
            SELECT n.id, n."text", n.kind, n.importance, n.confidence,
                   n.created_at, n.tags, n.valid_from, n.valid_to,
                   ({d_term} + {s_term}) AS rrf
            FROM   {self._table} n{join_clause}
            WHERE  ({" OR ".join(present)})
              AND  {" AND ".join(extra)}
              AND  n.layer = ANY(%(layers)s)
            ORDER  BY rrf DESC
            LIMIT  %(k)s
        """
        async with self._connect() as conn:
            if "t" in params:
                await conn.execute(
                    f"SET pg_trgm.similarity_threshold = {float(self._trgm_threshold)}"
                )
            if "q" in params:
                # Without this the dense channel is quietly approximate: the
                # index returns whatever cluster the graph walk landed in, and
                # the right memory never reaches the RRF fusion at all.
                await conn.execute(f"SET hnsw.ef_search = {int(self._hnsw_ef_search)}")
            cur = await self._exec(conn, sql, params, prepare=True)
            rows = await cur.fetchall()

        out: list[ScoredItem] = []
        for row in rows:
            item = self._row_to_item(row)
            rrf = float(row["rrf"])
            item.metadata["retrieval"] = {"rrf_score": rrf}
            out.append(
                ScoredItem(
                    item=item,
                    scores=ScoreBreakdown(
                        relevance=rrf,
                        importance=0.0,
                        recency=0.0,
                        confidence=item.confidence,
                        composite=rrf,
                    ),
                )
            )
        return out

    # ── exact tag lookup (no similarity) ────────────────────────────────────

    async def by_tags(
        self,
        tags: dict[str, Any],
        *,
        key: str | None = None,
        as_of: datetime | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Rows whose ``tags`` contain *tags* — an exact JSONB lookup, newest
        valid-time first.

        Deliberately **not** similarity-based. :meth:`search_hybrid` gates on a
        dense/sparse channel *before* applying the filter, so a fact can be
        excluded simply because the query text didn't resemble it — useless for
        answering "what is the user's CURRENT residence?", which is a lookup, not
        a search. *as_of* applies the bi-temporal window (a fact counts only if
        it had already become valid and had not yet been superseded).

        *key* additionally requires the tag **key** to exist (JSONB key-exists),
        which answers "does this corpus use attribute tags at all?" — the
        difference between "no fact has that attribute" and "attributes aren't
        used here", and therefore between answering honestly and guessing.
        """
        extra = ["invalidated_at IS NULL", "namespace = %(ns)s"]
        params: dict[str, Any] = {"k": int(limit), "ns": self._namespace}
        if tags:
            extra.append("tags @> %(filt)s::jsonb")
            params["filt"] = json.dumps(tags)
        if key is not None:
            # jsonb_exists(), not the `?` operator — `?` collides with driver
            # placeholder parsing.
            extra.append("jsonb_exists(tags, %(key)s)")
            params["key"] = key
        if as_of is not None:
            extra.append("(valid_from IS NULL OR valid_from <= %(as_of)s)")
            extra.append("(valid_to IS NULL OR valid_to > %(as_of)s)")
            params["as_of"] = as_of
        if session_id is not None:
            extra.append("tags @> %(sess)s::jsonb")
            params["sess"] = json.dumps({"session_id": session_id})

        sql = f"""
            SELECT id, "text", kind, importance, confidence,
                   created_at, tags, valid_from, valid_to
            FROM   {self._table}
            WHERE  {" AND ".join(extra)}
            ORDER  BY valid_from DESC NULLS LAST, created_at DESC
            LIMIT  %(k)s
        """
        async with self._connect() as conn:
            cur = await self._exec(conn, sql, params, prepare=True)
            rows = await cur.fetchall()
        return [self._row_to_item(r) for r in rows]

    # ── neighbors (recursive CTE graph walk) ────────────────────────────────

    async def neighbors(
        self,
        node_id: uuid.UUID | str,
        hops: int = 1,
        *,
        depth: int | None = None,
    ) -> list[Edge]:
        """
        Traverse outgoing edges from *node_id* up to ``hops`` (alias
        ``depth=`` for protocol callers, e.g. ``neighbors(id, depth=1)``).

        Cycle-guarded (a ``path`` array prevents revisiting a node), capped
        at ``_NEIGHBOR_CAP`` rows, and bi-temporal: **both** edges and target
        nodes must have ``invalidated_at IS NULL``. Each :class:`Edge` is
        returned with its destination ``MemoryItem`` hydrated.

        EXPLAIN ANALYZE (depth=2, ~10 edges/node)::

            Limit
              ->  Sort (Sort Key: w.depth, w.edge_id)
                    ->  Hash Join (walk ⋈ memory_nodes n)
                          ->  Recursive Union
                                ->  Index Scan memory_edges (source_id)
                                ->  Hash Join (walk ⋈ memory_edges)
            Execution Time: ~2 ms
        """
        max_depth = depth if depth is not None else hops
        seed = str(_as_uuid(node_id))
        sql = f"""
            WITH RECURSIVE walk(
                edge_id, source_id, target_id, predicate, weight, depth, path
            ) AS (
                SELECT e.id, e.source_id, e.target_id, e.predicate,
                       e.weight, 1, ARRAY[e.source_id, e.target_id]
                FROM   {self._edges} e
                JOIN   {self._table} tn
                       ON tn.id = e.target_id AND tn.invalidated_at IS NULL
                       AND tn.namespace = %(ns)s
                WHERE  e.source_id = %(seed)s
                  AND  e.invalidated_at IS NULL
                UNION ALL
                SELECT e.id, e.source_id, e.target_id, e.predicate,
                       e.weight, w.depth + 1, w.path || e.target_id
                FROM   walk w
                JOIN   {self._edges} e
                       ON e.source_id = w.target_id
                      AND e.invalidated_at IS NULL
                JOIN   {self._table} tn
                       ON tn.id = e.target_id AND tn.invalidated_at IS NULL
                       AND tn.namespace = %(ns)s
                WHERE  w.depth < %(maxd)s
                  AND  NOT e.target_id = ANY(w.path)
            )
            SELECT w.edge_id, w.source_id, w.target_id, w.predicate,
                   w.weight, w.depth,
                   n."text", n.kind, n.importance, n.confidence,
                   n.created_at, n.tags
            FROM   walk w
            JOIN   {self._table} n
                   ON n.id = w.target_id AND n.invalidated_at IS NULL
                   AND n.namespace = %(ns)s
            ORDER  BY w.depth, w.edge_id
            LIMIT  %(cap)s
        """
        async with self._connect() as conn:
            cur = await self._exec(
                conn,
                sql,
                {"seed": seed, "maxd": max_depth, "cap": _NEIGHBOR_CAP, "ns": self._namespace},
                prepare=True,
            )
            rows = await cur.fetchall()

        edges: list[Edge] = []
        for r in rows:
            edges.append(
                Edge(
                    id=_as_uuid(r["edge_id"]),
                    source_id=_as_uuid(r["source_id"]),
                    target_id=_as_uuid(r["target_id"]),
                    predicate=r.get("predicate"),
                    weight=(float(r["weight"]) if r.get("weight") is not None else None),
                    depth=int(r["depth"]),
                    target=self._row_to_item(r),
                )
            )
        return edges

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _layers(q: Query) -> list[str]:
        """Map ``Query.tiers`` → stored ``layer`` strings (default ['LTM'])."""
        tiers = q.tiers or [MemoryTier.LTM]
        return [t.name for t in tiers]  # MemoryTier.LTM → 'LTM'

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> MemoryItem:
        raw_tags = row.get("tags") or {}
        tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else json.loads(raw_tags)
        if row.get("kind") is not None:
            tags["kind"] = row["kind"]

        raw_ts = row.get("created_at")
        if isinstance(raw_ts, datetime):
            created_at = raw_ts
        elif raw_ts is not None:
            created_at = datetime.fromisoformat(str(raw_ts))
        else:
            created_at = datetime.now(UTC)

        # Valid time (bi-temporal). Without hydrating this, a retrieved item
        # looks like it became true when it was *recorded*, so supersession-aware
        # reads (`current`, `timeline`) cannot tell a stale fact from a fresh one.
        def _dt(raw: Any) -> datetime | None:
            if raw is None:
                return None
            if isinstance(raw, datetime):
                return raw
            try:
                return datetime.fromisoformat(str(raw))
            except ValueError:
                return None

        valid_from, valid_to = _dt(row.get("valid_from")), _dt(row.get("valid_to"))
        valid_range = (
            BiTemporalRange(valid_from=valid_from or created_at, valid_to=valid_to)
            if (valid_from is not None or valid_to is not None)
            else None
        )

        return MemoryItem(
            id=str(row["id"]) if row.get("id") is not None else str(row["target_id"]),
            content=str(row.get("text") or ""),
            tier=MemoryTier.LTM,
            importance=(float(row["importance"]) if row.get("importance") is not None else 0.5),
            confidence=(float(row["confidence"]) if row.get("confidence") is not None else 1.0),
            created_at=created_at,
            valid_range=valid_range,
            metadata=tags,
        )


__all__ = ["PostgresLTM"]
