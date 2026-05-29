"""
continuum/stores/in_memory/ltm.py
==================================
Process-local :class:`continuum.core.protocols.LTMProtocol` implementation.

Mirrors :class:`continuum.stores.postgres.ltm.PostgresLTM` on the
public async surface (``upsert / update / invalidate / search_hybrid
/ neighbors / aclose``) so callers can swap backends without
touching downstream code. Supersession is modelled identically:
:meth:`invalidate` stamps ``invalidated_at`` on the row and every
read path filters ``invalidated_at is None`` — Postgres has the same
guarantee enforced inside its hybrid-search SQL.

Why this exists
---------------
The eval rig (:mod:`evals.longmemeval.continuum_ltm_store`) needs a
real LTM tier to drive the Mem0 promoter end-to-end without
provisioning Postgres. The Postgres LTM is the canonical
implementation; this module is a faithful reduction of that
behaviour to in-process dicts. Vector similarity and BM25 are
implemented via the same helpers shipped for the retriever (cosine
on numpy arrays, :class:`continuum.retrieval.bm25.BM25Index` for
sparse) and fused with
:class:`continuum.retrieval.rrf.ReciprocalRankFusion`'s rank-only
math — verbatim Postgres semantics modulo the storage layer.

Scope (deliberate, documented)
------------------------------
* No graph: :meth:`neighbors` returns ``[]``. Graph expansion isn't
  a knowledge-update lever and the eval rig doesn't read it.
* No bi-temporal point-in-time queries against ``valid_range``: the
  rig doesn't ask for them. We honour ``invalidated_at`` because
  *that* is supersession; bi-temporal point-in-time can be added if a
  caller needs it (it would slot into :meth:`search_hybrid` as a
  filter, no schema changes required).
"""
from __future__ import annotations

import logging
import math
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from continuum.core.types import (
    BiTemporalRange,
    MemoryItem,
    MemoryTier,
    Query,
    ScoreBreakdown,
    ScoredItem,
)
from continuum.retrieval.bm25 import BM25Index

log = logging.getLogger(__name__)


class InMemoryLTM:
    """In-process LTM with bi-temporal supersession via ``invalidated_at``."""

    def __init__(self) -> None:
        # ``id -> MemoryItem`` (live + invalidated). The historical
        # row is kept so the audit log can still address it.
        self._rows: dict[uuid.UUID, MemoryItem] = {}
        self._invalidated_at: dict[uuid.UUID, datetime] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        # No connections to close; conform to the protocol's signature.
        return None

    # ── writes ─────────────────────────────────────────────────────────────

    async def upsert(self, item: MemoryItem) -> uuid.UUID:
        """
        Insert or update by id. New items get an id assigned when
        ``item.id`` is empty or unparseable. Returns the persisted id.
        """
        node_id = _to_uuid(item.id) or uuid.uuid4()
        # Always stamp the LTM tier; callers sometimes pass STM items
        # in for convenience.
        stored = MemoryItem(
            id=str(node_id),
            content=item.content,
            tier=MemoryTier.LTM,
            importance=float(item.importance),
            confidence=float(item.confidence),
            created_at=item.created_at,
            session_id=item.session_id,
            agent_id=item.agent_id,
            user_id=item.user_id,
            embedding=list(item.embedding) if item.embedding is not None else None,
            valid_range=(
                item.valid_range
                if item.valid_range is not None
                else BiTemporalRange(valid_from=item.created_at)
            ),
            related_ids=list(item.related_ids),
            # MemoryItem.metadata isn't a declared field on every
            # build — fetch defensively.
            **_optional_metadata(item),
        )
        self._rows[node_id] = stored
        # Re-upserting a row clears any prior invalidation (mirrors
        # Postgres ``UPDATE … SET invalidated_at = NULL`` semantics).
        self._invalidated_at.pop(node_id, None)
        return node_id

    async def update(self, id: uuid.UUID | str, patch: dict[str, Any]) -> None:  # noqa: A002
        node_id = _to_uuid(id)
        if node_id is None or node_id not in self._rows:
            return
        row = self._rows[node_id]
        # Apply patch onto a fresh MemoryItem so dataclass invariants
        # stay intact. Only known fields are honoured; unknown keys
        # are ignored (Postgres silently does the same via JSONB merge).
        merged = MemoryItem(
            id=row.id,
            content=str(patch.get("content", row.content)),
            tier=row.tier,
            importance=float(patch.get("importance", row.importance)),
            confidence=float(patch.get("confidence", row.confidence)),
            created_at=row.created_at,
            session_id=row.session_id,
            agent_id=row.agent_id,
            user_id=row.user_id,
            embedding=patch.get("embedding", row.embedding),
            valid_range=row.valid_range,
            related_ids=list(row.related_ids),
            **_optional_metadata(row, override=patch.get("metadata")),
        )
        self._rows[node_id] = merged

    async def invalidate(
        self, id: uuid.UUID | str, at: datetime | None = None  # noqa: A002
    ) -> None:
        """
        Bi-temporal supersession: stamp ``invalidated_at`` so every
        downstream read path skips this row. Mirrors
        :meth:`continuum.stores.postgres.ltm.PostgresLTM.invalidate`.
        """
        node_id = _to_uuid(id)
        if node_id is None or node_id not in self._rows:
            return
        self._invalidated_at[node_id] = at or datetime.now(UTC)

    # ── reads ──────────────────────────────────────────────────────────────

    async def search_hybrid(
        self, q: Query, k: int = 10
    ) -> list[ScoredItem]:
        """
        Dense (cosine over ``embedding``) + sparse (BM25 over
        ``content``) fused by Reciprocal Rank Fusion, filtered to
        live (non-invalidated) rows. Same rank-fusion shape as the
        Postgres branch (see ``stores/postgres/ltm.py:_search_hybrid``
        — same ``1/(rrf_k + rank)`` accumulation).
        """
        live = [row for nid, row in self._rows.items() if nid not in self._invalidated_at]
        if not live or k <= 0:
            return []

        # Sparse channel — always runnable.
        sparse_index = BM25Index(live)
        sparse_hits = sparse_index.query(q.text or "", k=max(k * 3, 20))
        sparse_rank: dict[str, int] = {
            si.item.id: rank for rank, si in enumerate(sparse_hits, start=1)
        }

        # Dense channel — only when the caller embedded the query.
        dense_rank: dict[str, int] = {}
        if q.embedding is not None:
            scored = []
            qv = list(q.embedding)
            for row in live:
                if row.embedding is None:
                    continue
                scored.append((row.id, _cosine(qv, row.embedding)))
            scored.sort(key=lambda t: -t[1])
            for rank, (rid, _) in enumerate(scored[: max(k * 3, 20)], start=1):
                dense_rank[rid] = rank

        # RRF accumulation. k=60 mirrors Postgres' default.
        rrf_k = 60
        scores: dict[str, float] = {}
        for rid, rank in sparse_rank.items():
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
        for rid, rank in dense_rank.items():
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]

        by_id = {row.id: row for row in live}
        out: list[ScoredItem] = []
        for rid, score in ranked:
            row = by_id.get(rid)
            if row is None:
                continue
            sb = ScoreBreakdown.compute(
                relevance=min(1.0, score),
                importance=float(row.importance),
                recency=0.0,
                confidence=float(row.confidence),
            )
            out.append(ScoredItem(item=row, scores=sb))
        return out

    async def neighbors(
        self,
        node_id: uuid.UUID | str,
        hops: int = 1,
        *,
        depth: int | None = None,
    ) -> list[Any]:
        """No graph in v1 — see module docstring."""
        return []

    async def iter_live(self) -> AsyncIterator[MemoryItem]:
        """
        Stream all non-invalidated rows in insertion order. Used by
        :class:`evals.longmemeval.continuum_ltm_store.ContinuumLTMHaystackStore`
        to push the current LTM view back onto ``store.items`` for
        the rig's retrievers. Not part of the canonical LTMProtocol.
        """
        for nid, row in self._rows.items():
            if nid in self._invalidated_at:
                continue
            yield row

    # ── observability ──────────────────────────────────────────────────────

    @property
    def live_count(self) -> int:
        return sum(1 for nid in self._rows if nid not in self._invalidated_at)

    @property
    def total_count(self) -> int:
        return len(self._rows)


# ── helpers ────────────────────────────────────────────────────────────────


def _to_uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _optional_metadata(
    item: MemoryItem, *, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    MemoryItem's metadata attribute name varies slightly across
    builds; handle the common ``metadata`` field gracefully and
    ignore when absent.
    """
    md = getattr(item, "metadata", None)
    if not isinstance(md, dict):
        md = {}
    if override is not None:
        md = {**md, **override}
    # Pass through only when supported on the dataclass.
    return {"metadata": md} if hasattr(item, "metadata") else {}


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine. Returns 0.0 on degenerate vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


__all__ = ["InMemoryLTM"]
