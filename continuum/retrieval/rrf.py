"""
continuum/retrieval/rrf.py
==========================
Reciprocal Rank Fusion — combine N retrievers into a single ranked list
without needing comparable raw scores. Each retriever's hit at rank
``r`` contributes ``1 / (k + r)`` to the item's fused score; the merged
list is sorted by total contribution.

The reference (Cormack 2009) uses ``k = 60``. The constant matters less
than the rank-based shape: a top-1 from one retriever (≈ 1/61) is
worth more than a top-3 from another (≈ 1/63), and a top-1 from BOTH
is worth roughly double a top-1 from either alone — exactly the
property we want when fusing cosine and BM25 over the same corpus.

Composition pattern
-------------------
The child retrievers must conform to :class:`RetrieverProtocol`
(``async retrieve(query, budget) -> ContextBundle``). The fusion runs
them concurrently with :func:`asyncio.gather` (best-effort: any
retriever that raises contributes zero ranks rather than aborting the
whole bundle), then walks each :class:`ContextBundle.items` list in
the order it was returned and treats that as the rank order.

``HybridRetriever`` is a thin alias for the common "cosine + BM25"
case — see :func:`HybridRetriever`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from continuum.core.types import ContextBundle, Query, TokenBudget

log = logging.getLogger(__name__)

#: Cormack 2009 default. Trade-off: larger k smooths score differences
#: between top ranks (helpful when child retrievers disagree); smaller
#: k makes the top-1 dominate. 60 is a sane default; tune via the
#: constructor argument once you have a smoke to compare.
DEFAULT_K: int = 60


@runtime_checkable
class _Retriever(Protocol):
    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle: ...


class ReciprocalRankFusion:
    """
    Run multiple retrievers concurrently and merge their results by
    Reciprocal Rank Fusion.

    Parameters
    ----------
    retrievers:
        Child retrievers. Order does not affect the fused result —
        only ranks do.
    k:
        RRF smoothing constant. Defaults to :data:`DEFAULT_K` (60).
    top_k:
        How many items to surface from the merged ranking. Defaults to
        ``query.top_k`` at call time.

    Identity
    --------
    Items are deduplicated by ``MemoryItem.id``. When the same id
    appears in multiple child bundles, its scores accumulate and the
    representative copy is the first one seen — the child's
    metadata (e.g. ``via_graph`` tags from the LTM graph-expansion
    step) is preserved.

    Failure semantics
    -----------------
    Any retriever that raises during ``retrieve`` is logged and
    skipped — it contributes zero ranks. An exception in *every*
    retriever yields an empty bundle, not an exception. Matches the
    contract of :class:`continuum.retrieval.Retriever`.
    """

    def __init__(
        self,
        retrievers: Sequence[_Retriever],
        k: int = DEFAULT_K,
        *,
        top_k: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("RRF constant k must be > 0")
        self._retrievers = list(retrievers)
        self._k = k
        self._top_k = top_k

    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle:
        if not self._retrievers:
            return ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                debug_info={"rrf": "no retrievers"},
            )

        bundles = await asyncio.gather(
            *(self._safe_retrieve(r, query, budget) for r in self._retrievers),
            return_exceptions=False,
        )

        # ── RRF accumulation ─────────────────────────────────────────────
        scores: dict[str, float] = {}
        item_by_id: dict[str, Any] = {}
        per_retriever: list[dict[str, int]] = []  # debug: id → rank
        for bundle in bundles:
            ranks: dict[str, int] = {}
            for rank, item in enumerate(bundle.items, start=1):
                key = str(item.id)
                ranks[key] = rank
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._k + rank)
                item_by_id.setdefault(key, item)
            per_retriever.append(ranks)

        # ── sort + cap ──────────────────────────────────────────────────
        # Tie-break by first-appearance order so output is deterministic
        # when two items collected the same RRF mass (most common case:
        # both retrievers ranked them identically).
        first_seen: dict[str, int] = {}
        for ranks in per_retriever:
            for key in ranks:
                first_seen.setdefault(key, len(first_seen))
        merged_keys = sorted(
            scores.keys(),
            key=lambda key: (-scores[key], first_seen.get(key, 1 << 30)),
        )
        top_k = self._top_k if self._top_k is not None else query.top_k
        merged_keys = merged_keys[:top_k]

        merged_items = [item_by_id[key] for key in merged_keys]
        messages = [{"role": "system", "content": item.content} for item in merged_items]
        debug = {
            "rrf_k": self._k,
            "rrf_scores": {key: round(scores[key], 6) for key in merged_keys},
            "rrf_retriever_count": len(self._retrievers),
            "rrf_per_retriever_sizes": [len(r) for r in per_retriever],
        }
        return ContextBundle(
            items=merged_items,
            messages=messages,
            tokens_used=sum(len(it.content.split()) for it in merged_items),
            budget=budget,
            debug_info=debug,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    async def _safe_retrieve(
        self,
        retriever: _Retriever,
        query: Query,
        budget: TokenBudget,
    ) -> ContextBundle:
        try:
            return await retriever.retrieve(query, budget)
        except Exception:
            log.exception("child retriever failed in RRF — contributing 0 hits")
            return ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
            )


def HybridRetriever(  # noqa: N802 — factory-as-class convention
    cosine: _Retriever,
    bm25: _Retriever,
    *,
    k: int = DEFAULT_K,
    top_k: int | None = None,
) -> ReciprocalRankFusion:
    """
    Factory for the common cosine + BM25 fusion.

    A thin wrapper around :class:`ReciprocalRankFusion` named so the
    intent is obvious at call sites (``HybridRetriever(cosine, bm25)``
    rather than the more generic ``ReciprocalRankFusion([...])``).
    """
    return ReciprocalRankFusion([cosine, bm25], k=k, top_k=top_k)


__all__ = ["DEFAULT_K", "HybridRetriever", "ReciprocalRankFusion"]
