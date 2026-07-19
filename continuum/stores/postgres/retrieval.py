"""
continuum/stores/postgres/retrieval.py
======================================
Hybrid retrieval over ``memory_nodes``: a **dense** semantic channel
(pgvector / ``halfvec``, migration 002) fused with a **sparse** lexical
channel (``pg_trgm``, migration 003) via **Reciprocal Rank Fusion**.

Why hybrid
----------
Dense vectors capture meaning but miss exact tokens (identifiers, error
codes, rare proper nouns, typo'd-but-literal queries). Trigram lexical
search nails those but ignores synonyms/paraphrase. Fusing the two
consistently beats either channel alone on recall — see Cormack, Clarke &
Büttcher, *"Reciprocal Rank Fusion Outperforms Condorcet and Individual
Rank Learning Methods"* (SIGIR 2009), which also motivates the constant
``k = 60`` used below.

RRF
---
For a document *d* ranked at (1-based) position ``r`` by a retriever, that
retriever contributes ``1 / (k + r)`` to *d*'s score; the final score is the
sum over the retrievers that ranked it (a retriever that did not return *d*
contributes nothing). With two channels::

    score(d) = 1/(60 + dense_rank)  +  1/(60 + sparse_rank)

Rank fusion is deliberately score-agnostic: it never compares a cosine
similarity to a trigram similarity (different, non-commensurable scales) —
only their *ordinal positions*. That robustness is the whole point.

Connections
-----------
psycopg3 is imported lazily; ``conn_factory`` can be injected for tests
(no psycopg/PostgreSQL needed). ``hybrid_search`` runs the two channels
concurrently, each acquiring its **own** connection — in production back
``conn_factory`` with a pool (or use ``dsn`` mode, which connects per call)
so the two queries don't contend on a single connection.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from continuum.core.types import (
    BiTemporalRange,
    MemoryItem,
    MemoryTier,
    ScoreBreakdown,
    ScoredItem,
)
from continuum.db.pgvector_upgrade import to_halfvec_literal

log = logging.getLogger(__name__)

DEFAULT_TABLE = "memory_nodes"
DEFAULT_RRF_K = 60
DEFAULT_TRGM_THRESHOLD = 0.25


# ============================================================================
# Pure RRF (no DB — fully unit-testable)
# ============================================================================


@dataclass
class FusionResult:
    """One fused candidate: its RRF score and the component ranks (1-based)."""

    id: uuid.UUID
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


def reciprocal_rank_fusion(
    dense_ids: Sequence[uuid.UUID],
    sparse_ids: Sequence[uuid.UUID],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[FusionResult]:
    """
    Fuse two ranked id lists with Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_ids / sparse_ids:
        Ids in rank order (index 0 == best) from each retriever. Duplicates
        within a list are ignored after their first (best) occurrence.
    rrf_k:
        The RRF constant (Cormack et al. use 60). Larger ``rrf_k`` flattens
        the contribution curve (positions matter less); smaller sharpens it.
    top_k:
        Truncate the fused output to this many results. ``None`` = return all.

    Returns
    -------
    list[FusionResult]
        Sorted by ``rrf_score`` desc. Ties broken deterministically by the
        better (smaller) available rank, then by id, so output is stable.
    """
    dense_rank: dict[uuid.UUID, int] = {}
    for pos, did in enumerate(dense_ids, start=1):
        dense_rank.setdefault(did, pos)
    sparse_rank: dict[uuid.UUID, int] = {}
    for pos, sid in enumerate(sparse_ids, start=1):
        sparse_rank.setdefault(sid, pos)

    fused: list[FusionResult] = []
    for cid in dense_rank.keys() | sparse_rank.keys():
        dr = dense_rank.get(cid)
        sr = sparse_rank.get(cid)
        score = 0.0
        if dr is not None:
            score += 1.0 / (rrf_k + dr)
        if sr is not None:
            score += 1.0 / (rrf_k + sr)
        fused.append(FusionResult(id=cid, rrf_score=score, dense_rank=dr, sparse_rank=sr))

    def _sort_key(fr: FusionResult) -> tuple[float, int, str]:
        best_rank = min(r for r in (fr.dense_rank, fr.sparse_rank) if r is not None)
        return (-fr.rrf_score, best_rank, str(fr.id))

    fused.sort(key=_sort_key)
    return fused if top_k is None else fused[:top_k]


# ============================================================================
# psycopg3 (lazy) + connection plumbing
# ============================================================================


def _require_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg, dict_row
    except ImportError as exc:  # pragma: no cover - exercised via conn_factory
        raise ImportError(
            "psycopg3 is required for PostgresRetriever.\n"
            "Install it with:  pip install 'psycopg[binary]>=3.1'"
        ) from exc


def _as_uuid(value: Any) -> uuid.UUID:
    """Coerce a DB id (UUID instance or str) to ``uuid.UUID``."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# ============================================================================
# PostgresRetriever
# ============================================================================


def _as_datetime(raw: Any) -> datetime | None:
    """Coerce a DB timestamp to datetime; None when absent/unparseable."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


@dataclass
class PostgresRetriever:
    """
    Dense + lexical + hybrid retrieval against ``memory_nodes``.

    Parameters
    ----------
    dsn:
        ``postgresql://…`` DSN. Ignored when *conn_factory* is given.
    conn_factory:
        Async callable returning an async context manager that yields a
        connection. Primarily for tests; in production prefer a pooled
        factory so concurrent channels don't share one connection.
    table:
        Target table (default ``memory_nodes``).
    rrf_k:
        RRF constant (default 60, per Cormack et al.).
    trgm_threshold:
        Session ``pg_trgm.similarity_threshold`` applied before lexical
        queries so genuine typo matches survive the ``%`` prefilter.
    """

    dsn: str | None = None
    conn_factory: Any | None = None
    table: str = DEFAULT_TABLE
    rrf_k: int = DEFAULT_RRF_K
    trgm_threshold: float = DEFAULT_TRGM_THRESHOLD

    def __post_init__(self) -> None:
        if self.conn_factory is None and self.dsn is None:
            raise ValueError("Provide either dsn or conn_factory.")

    # ── connection ──────────────────────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[Any]:
        if self.conn_factory is not None:
            async with self.conn_factory() as conn:
                yield conn
            return
        psycopg, dict_row = _require_psycopg()
        conn = await psycopg.AsyncConnection.connect(
            self.dsn, autocommit=True, row_factory=dict_row
        )
        try:
            yield conn
        finally:
            await conn.close()

    # ── lexical (sparse / pg_trgm) ──────────────────────────────────────────

    async def lexical_search(self, query_text: str, k: int) -> list[tuple[uuid.UUID, float]]:
        """
        Trigram lexical search over live rows.

        Uses ``similarity()`` for scoring and the ``%`` operator for the
        GIN-accelerated candidate prefilter, restricted to
        ``invalidated_at IS NULL``. Returns ``(id, similarity)`` pairs
        ordered by similarity descending, best first.

        ``query_text`` empty/whitespace → ``[]`` (trigrams of "" match
        nothing useful; skip the round-trip).
        """
        if not query_text or not query_text.strip():
            return []

        # `%(q)s` is the bind param; the literal pg_trgm `%` operator must be
        # doubled to survive psycopg's pyformat parsing.
        sql = f"""
            SELECT id, similarity("text", %(q)s) AS sim
            FROM   {self.table}
            WHERE  invalidated_at IS NULL
              AND  "text" %% %(q)s
            ORDER  BY sim DESC, id
            LIMIT  %(k)s
        """
        async with self._connect() as conn:
            # Session GUC (float-validated → safe to interpolate; SET can't
            # be parameterised). autocommit conn → session scope, not LOCAL.
            await conn.execute(f"SET pg_trgm.similarity_threshold = {float(self.trgm_threshold)}")
            cur = await conn.execute(sql, {"q": query_text, "k": k})
            rows = await cur.fetchall()

        return [(_as_uuid(r["id"]), float(r["sim"])) for r in rows]

    # ── dense (semantic / halfvec) ──────────────────────────────────────────

    async def dense_search(
        self, query_embedding: Sequence[float], k: int
    ) -> list[tuple[uuid.UUID, float]]:
        """
        Cosine KNN over the ``halfvec`` embedding column (live rows only).

        Returns ``(id, cosine_similarity)`` pairs, best first, where
        similarity = ``1 - (embedding <=> q)``.
        """
        if not query_embedding:
            return []
        q_lit = to_halfvec_literal(query_embedding)
        sql = f"""
            SELECT id,
                   1 - (embedding <=> %(q)s::halfvec) AS sim
            FROM   {self.table}
            WHERE  invalidated_at IS NULL
              AND  embedding IS NOT NULL
            ORDER  BY embedding <=> %(q)s::halfvec
            LIMIT  %(k)s
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql, {"q": q_lit, "k": k})
            rows = await cur.fetchall()
        return [(_as_uuid(r["id"]), float(r["sim"])) for r in rows]

    # ── hybrid (RRF fusion) ─────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        k: int,
    ) -> list[ScoredItem]:
        """
        Run dense + lexical concurrently and fuse with RRF (k=60).

        Over-fetches each channel (``3*k``, min 20) so a document ranked
        modestly by one retriever but well by the other can still surface —
        this is precisely where hybrid out-recalls dense-only. Returns the
        top-``k`` :class:`ScoredItem` ordered by RRF score; each item's
        ``metadata['retrieval']`` records the component ranks and fused score.
        """
        fetch_n = max(3 * k, 20)

        dense_pairs, sparse_pairs = await asyncio.gather(
            self.dense_search(query_embedding, fetch_n),
            self.lexical_search(query_text, fetch_n),
        )

        fused = reciprocal_rank_fusion(
            [i for i, _ in dense_pairs],
            [i for i, _ in sparse_pairs],
            rrf_k=self.rrf_k,
            top_k=k,
        )
        if not fused:
            return []

        dense_sim = dict(dense_pairs)
        sparse_sim = dict(sparse_pairs)
        items = await self._load_items([f.id for f in fused])

        scored: list[ScoredItem] = []
        for fr in fused:
            item = items.get(fr.id)
            if item is None:
                # Row vanished (invalidated) between ranking and hydration.
                continue
            item.metadata["retrieval"] = {
                "rrf_score": fr.rrf_score,
                "dense_rank": fr.dense_rank,
                "sparse_rank": fr.sparse_rank,
                "dense_similarity": dense_sim.get(fr.id),
                "sparse_similarity": sparse_sim.get(fr.id),
            }
            # RRF is the ranking signal → expose it as the composite so
            # ScoredItem.score orders correctly. relevance mirrors it;
            # confidence carries through from the stored fact.
            scored.append(
                ScoredItem(
                    item=item,
                    scores=ScoreBreakdown(
                        relevance=fr.rrf_score,
                        importance=0.0,
                        recency=0.0,
                        confidence=item.confidence,
                        composite=fr.rrf_score,
                    ),
                )
            )
        return scored

    # ── hydration ───────────────────────────────────────────────────────────

    async def _load_items(self, ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, MemoryItem]:
        """Fetch live ``memory_nodes`` rows for *ids* → ``{id: MemoryItem}``."""
        if not ids:
            return {}
        sql = f"""
            SELECT id, "text", kind, importance, confidence,
                   created_at, tags, valid_from, valid_to
            FROM   {self.table}
            WHERE  id = ANY(%(ids)s)
              AND  invalidated_at IS NULL
        """
        async with self._connect() as conn:
            cur = await conn.execute(sql, {"ids": list(ids)})
            rows = await cur.fetchall()
        return {_as_uuid(r["id"]): self._row_to_item(r) for r in rows}

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> MemoryItem:
        raw_ts = row.get("created_at")
        if isinstance(raw_ts, datetime):
            created_at = raw_ts
        elif raw_ts is not None:
            created_at = datetime.fromisoformat(str(raw_ts))
        else:
            created_at = datetime.now(UTC)

        tags = row.get("tags")
        metadata: dict[str, Any] = dict(tags) if isinstance(tags, dict) else {}
        if row.get("kind") is not None:
            metadata["kind"] = row["kind"]

        # Valid time (bi-temporal). Without hydrating this, every retrieved item
        # looks like it became true when it was *recorded*, so supersession-aware
        # reads (`current`, `timeline`) cannot tell a stale fact from a fresh one.
        valid_range = None
        vf, vt = _as_datetime(row.get("valid_from")), _as_datetime(row.get("valid_to"))
        if vf is not None or vt is not None:
            valid_range = BiTemporalRange(valid_from=vf or created_at, valid_to=vt)

        return MemoryItem(
            id=str(row["id"]),
            content=str(row.get("text") or ""),
            tier=MemoryTier.LTM,
            importance=float(row["importance"]) if row.get("importance") is not None else 0.5,
            confidence=float(row["confidence"]) if row.get("confidence") is not None else 1.0,
            created_at=created_at,
            valid_range=valid_range,
            metadata=metadata,
        )


__all__ = [
    "PostgresRetriever",
    "reciprocal_rank_fusion",
    "FusionResult",
]
