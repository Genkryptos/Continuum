"""
continuum/optimizer/strategies/semantic_dedupe.py
==================================================
``SemanticDedupe`` — drop near-duplicate LTM rows from the context bundle.

Algorithm
---------
1. Filter ``ctx.items`` to ``MemoryTier.LTM`` rows that carry an
   ``embedding``. Items missing an embedding are untouched (we can't
   compare them safely).
2. If fewer than ``skip_if_fewer_than`` candidates remain → no-op.
3. Build the normalised embedding matrix and compute the cosine
   similarity matrix in one ``E @ E.T`` numpy call (O(n²·d) but
   vectorised — fine up to a few thousand items per bundle).
4. Walk the upper triangle; for each pair with cosine ≥ ``threshold``,
   mark the **lower-scored** item for removal. The default scorer is
   ``importance + 0.1·confidence`` (importance dominant, confidence
   breaks ties); callers may inject any ``score_key: MemoryItem → float``.
5. Rebuild the bundle preserving the original prompt order.

Why a custom matrix instead of sklearn? The framework already pulls in
numpy; sklearn would add ~80 MB to the install footprint for a one-line
operation. Numerical equivalence (rtol < 1e-6) is asserted in the unit
tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from continuum.optimizer.base import BaseOptimizer, estimate_tokens_text
from continuum.optimizer.protocol import Cost

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SemanticDedupeConfig(BaseModel):
    """Configuration for :class:`SemanticDedupe`."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine-similarity threshold above which two LTM items are "
            "treated as duplicates. 0.92 is a conservative default that "
            "catches paraphrases without merging genuinely distinct facts."
        ),
    )
    skip_if_fewer_than: int = Field(
        default=10,
        ge=2,
        description=(
            "Skip the strategy entirely if the LTM tier has fewer than "
            "this many embedded items — the O(n²) cost isn't worth it "
            "and dedupe yield is negligible on tiny bundles."
        ),
    )


#: Default scorer when none is injected. Importance dominates;
#: confidence breaks ties.
def _default_score(item: MemoryItem) -> float:
    return float(item.importance) + 0.1 * float(item.confidence)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SemanticDedupe(BaseOptimizer):
    """
    Remove near-duplicate LTM rows.

    Parameters
    ----------
    threshold:
        Cosine threshold for "duplicate". Default 0.92.
    skip_if_fewer_than:
        Bail out when LTM has fewer than this many embedded items.
        Default 10.
    score_key:
        Optional ``(MemoryItem) -> float`` used to choose which of two
        duplicates to keep. Higher score wins; ties fall back to
        ``created_at`` (newer wins). Default scorer is
        ``importance + 0.1·confidence``.
    config:
        Bundle both scalars at once; explicit kwargs override.
    """

    def __init__(
        self,
        *,
        threshold: float | None = None,
        skip_if_fewer_than: int | None = None,
        score_key: Callable[[MemoryItem], float] | None = None,
        config: SemanticDedupeConfig | None = None,
    ) -> None:
        base = config or SemanticDedupeConfig()
        self.threshold: float = threshold if threshold is not None else base.threshold
        self.skip_if_fewer_than: int = (
            skip_if_fewer_than if skip_if_fewer_than is not None else base.skip_if_fewer_than
        )
        self.score_key: Callable[[MemoryItem], float] = score_key or _default_score

    # ── public API ──────────────────────────────────────────────────────────

    async def apply(self, ctx: ContextBundle, budget: TokenBudget) -> ContextBundle:
        ltm_items = [
            it for it in ctx.items if it.tier == MemoryTier.LTM and it.embedding is not None
        ]
        if len(ltm_items) < self.skip_if_fewer_than:
            return ctx

        removed_ids = self._find_duplicates_to_remove(ltm_items)
        if not removed_ids:
            return ctx

        new_items = [it for it in ctx.items if it.id not in removed_ids]
        new_messages = _rebuild_messages(ctx, new_items)
        new_breakdown = _tier_breakdown(new_items)

        return replace(
            ctx,
            items=new_items,
            messages=new_messages,
            tier_breakdown=new_breakdown,
            tokens_used=sum(estimate_tokens_text(i.content) for i in new_items),
            debug_info={
                **ctx.debug_info,
                "semantic_dedupe": {
                    "candidates": len(ltm_items),
                    "removed": len(removed_ids),
                    "threshold": self.threshold,
                },
            },
        )

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """
        Project tokens after dedup. Latency is dominated by the matrix
        multiply (~0.5 ms per 100 items at d=768) — capped at 50 ms upper
        bound for typical bundles.
        """
        ltm_items = [
            it for it in ctx.items if it.tier == MemoryTier.LTM and it.embedding is not None
        ]
        current = sum(estimate_tokens_text(it.content) for it in ctx.items)
        if len(ltm_items) < self.skip_if_fewer_than:
            return Cost(tokens=current, latency_ms=0.0)
        # Coarse projection: assume 5 % duplicate rate by default.
        avg_tokens = sum(estimate_tokens_text(it.content) for it in ltm_items) / max(
            1, len(ltm_items)
        )
        projected = current - int(avg_tokens * len(ltm_items) * 0.05)
        return Cost(
            tokens=max(0, projected),
            latency_ms=min(50.0, 0.005 * len(ltm_items)),
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _find_duplicates_to_remove(self, ltm_items: list[MemoryItem]) -> set[str]:
        """
        Build the cosine similarity matrix and pick lower-scored items in
        every pair ≥ ``threshold`` for removal.

        Transitive duplicates (A≈B≈C) are handled correctly: we walk
        every (i, j) pair, so all three pairings contribute and only the
        single best-scored row survives.
        """
        matrix = _embedding_matrix(ltm_items)
        if matrix is None:
            return set()

        sim = cosine_similarity_matrix(matrix)
        # Mask the diagonal and lower triangle — we only care about the
        # strict upper triangle (i < j) to count each pair once.
        n = len(ltm_items)
        to_remove: set[str] = set()
        scores = [self.score_key(it) for it in ltm_items]
        # Tie-breaker: created_at, fall back to id for full determinism.
        for i in range(n):
            if ltm_items[i].id in to_remove:
                continue
            for j in range(i + 1, n):
                if ltm_items[j].id in to_remove:
                    continue
                if sim[i, j] < self.threshold:
                    continue
                loser = _pick_loser(ltm_items[i], ltm_items[j], scores[i], scores[j])
                to_remove.add(loser.id)
        return to_remove


# ---------------------------------------------------------------------------
# Helpers (module-level so they're independently testable)
# ---------------------------------------------------------------------------


def _embedding_matrix(items: list[MemoryItem]) -> np.ndarray[Any, Any] | None:
    """
    Stack item embeddings into a (n, d) numpy array. Returns ``None`` if
    embeddings have inconsistent dimensions (defensive: shouldn't happen
    when a single embedder produced them, but if it does we'd rather
    skip than crash).
    """
    try:
        rows = [np.asarray(it.embedding, dtype=np.float32) for it in items]
    except Exception:
        log.exception("embedding conversion failed; skipping dedup")
        return None
    if not rows:
        return None
    dim = rows[0].shape[0]
    if any(r.shape != (dim,) for r in rows):
        log.warning("inconsistent embedding dimensions across LTM items; skipping semantic dedup")
        return None
    return np.stack(rows, axis=0)


def cosine_similarity_matrix(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Return the pairwise cosine-similarity matrix of *matrix* rows.

    Equivalent to ``sklearn.metrics.pairwise.cosine_similarity`` for the
    typical dense embedding case; verified numerically in the tests.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against zero-norm rows — they'd give NaNs after division.
    safe = np.where(norms == 0, 1.0, norms)
    normalised = matrix / safe
    # numpy can emit a spurious "divide by zero in matmul" warning on
    # float32 matrices that contain very small (sub-normal) values even
    # though the result is finite. Suppress it locally so callers under
    # `filterwarnings = error` don't blow up.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sim = normalised @ normalised.T
    # Mask zero-norm rows back to 0 so they never count as duplicates.
    zero_mask = (norms == 0).reshape(-1)
    if zero_mask.any():
        sim[zero_mask, :] = 0.0
        sim[:, zero_mask] = 0.0
    return np.asarray(sim)


def _pick_loser(a: MemoryItem, b: MemoryItem, score_a: float, score_b: float) -> MemoryItem:
    """Lower score loses; ties broken by older created_at, then id."""
    if score_a != score_b:
        return b if score_a > score_b else a
    ts_a = getattr(a, "created_at", None)
    ts_b = getattr(b, "created_at", None)
    if ts_a is not None and ts_b is not None and ts_a != ts_b:
        return a if ts_a < ts_b else b
    # Final deterministic tiebreak.
    return a if str(a.id) < str(b.id) else b


def _rebuild_messages(ctx: ContextBundle, items: list[MemoryItem]) -> list[dict[str, str]]:
    """
    Drop the messages whose paired item was removed; keep extras
    (messages with no item — e.g. injected system prompts) intact.
    """
    item_ids = {it.id for it in items}
    out: list[dict[str, str]] = []
    n_items_orig = len(ctx.items)
    for idx, msg in enumerate(ctx.messages):
        if idx >= n_items_orig:
            out.append(msg)
            continue
        if ctx.items[idx].id in item_ids:
            out.append(msg)
    return out


def _tier_breakdown(items: list[MemoryItem]) -> dict[str, int]:
    counts: dict[str, int] = {"stm": 0, "mtm": 0, "ltm": 0}
    for it in items:
        counts[it.tier.value] = counts.get(it.tier.value, 0) + estimate_tokens_text(it.content)
    return counts


__all__ = [
    "SemanticDedupe",
    "SemanticDedupeConfig",
    "cosine_similarity_matrix",
]
