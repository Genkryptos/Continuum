"""
continuum/optimizer/strategies/score_prune.py
==============================================
``ScoreAwareBudgetPrune`` — last-resort lossy strategy.

Run after every other optimisation in the chain. If the bundle still
exceeds budget, drop the cheapest-to-lose items until it fits.

Constraints (configurable)
--------------------------
* **STM is preserved** by default — recent conversation turns are the
  user's lived context; dropping them silently is the worst kind of
  context loss.
* **MTM keeps at least ``preserve_mtm_count`` blocks** — usually the
  freshest project summaries.
* **LTM is ordered by score ascending** and trimmed front-first.
* **MTM beyond the preserved count** is ordered by ``created_at``
  ascending (oldest first) and trimmed after LTM is exhausted.

If STM removal *is* enabled (``preserve_stm=False``), STM items are
removed last — oldest first — as a final escape hatch.

Edge case: ``too_large_stm_strategy``
-------------------------------------
A single STM turn larger than the entire budget is a pathological
situation a token-pruner cannot fix without help. Three policies are
supported:

* ``"keep"``  (default): leave the bundle alone — over-budget but
                 lossless. The caller decides whether to refuse the
                 request or call a summariser.
* ``"truncate"``: hard-truncate the offending turn(s) to the remaining
                 budget. Lossy but predictable.
* ``"summarise"``: delegate to an injected ``summarizer.summarize(text)``
                 collaborator. Falls back to ``truncate`` on failure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

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


class ScorePruneConfig(BaseModel):
    """Configuration for :class:`ScoreAwareBudgetPrune`."""

    model_config = ConfigDict(extra="forbid")

    preserve_stm: bool = Field(
        default=True,
        description=(
            "When True, STM is never touched. The recent conversation "
            "is the user's lived context — drop it only with explicit "
            "consent."
        ),
    )
    preserve_mtm_count: int = Field(
        default=2,
        ge=0,
        description=(
            "Lower bound on MTM rows that survive pruning. The most "
            "recent (by ``created_at``) are kept."
        ),
    )
    too_large_stm_strategy: Literal["keep", "truncate", "summarise"] = Field(
        default="keep",
        description=(
            "Behaviour when a single STM row alone exceeds the budget. "
            "'keep' stays over-budget but lossless; 'truncate' cuts; "
            "'summarise' delegates to the injected summarizer."
        ),
    )


def _default_score(item: MemoryItem) -> float:
    """importance + 0.1·confidence — same default as SemanticDedupe."""
    return float(item.importance) + 0.1 * float(item.confidence)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class ScoreAwareBudgetPrune(BaseOptimizer):
    """
    Drop the lowest-value items until the bundle fits the budget.

    Pruning order (lowest priority first → highest):
      1. LTM rows, ascending by ``score_key``.
      2. MTM rows beyond ``preserve_mtm_count``, oldest first.
      3. (only if ``preserve_stm=False``) STM rows, oldest first.

    Parameters
    ----------
    preserve_stm:
        Default True. See class docstring.
    preserve_mtm_count:
        Default 2.
    too_large_stm_strategy:
        Edge case for STM rows individually larger than the budget.
    score_key:
        Override the per-item value function. Default
        ``importance + 0.1·confidence``.
    summarizer:
        Required only when ``too_large_stm_strategy='summarise'``.
        ``async summarize(text: str) -> str``.
    config:
        Bundle all scalars at once; explicit kwargs override.
    """

    def __init__(
        self,
        *,
        preserve_stm: bool | None = None,
        preserve_mtm_count: int | None = None,
        too_large_stm_strategy: (Literal["keep", "truncate", "summarise"] | None) = None,
        score_key: Callable[[MemoryItem], float] | None = None,
        summarizer: Any | None = None,
        config: ScorePruneConfig | None = None,
    ) -> None:
        base = config or ScorePruneConfig()
        self.preserve_stm: bool = preserve_stm if preserve_stm is not None else base.preserve_stm
        self.preserve_mtm_count: int = (
            preserve_mtm_count if preserve_mtm_count is not None else base.preserve_mtm_count
        )
        self.too_large_stm_strategy: Literal["keep", "truncate", "summarise"] = (
            too_large_stm_strategy
            if too_large_stm_strategy is not None
            else base.too_large_stm_strategy
        )
        self.score_key: Callable[[MemoryItem], float] = score_key or _default_score
        self.summarizer = summarizer

    # ── public API ──────────────────────────────────────────────────────────

    async def apply(self, ctx: ContextBundle, budget: TokenBudget) -> ContextBundle:
        limit = budget.context_available
        current = sum(estimate_tokens_text(it.content) for it in ctx.items)
        if current <= limit:
            return ctx

        # Per-item token cost is computed once; we use it both for the
        # greedy budget arithmetic and the final breakdown.
        token_cost: dict[str, int] = {it.id: estimate_tokens_text(it.content) for it in ctx.items}

        removable_ids = self._pruning_order(ctx.items)
        survivors_set = {it.id for it in ctx.items}
        removed: list[str] = []
        for item_id in removable_ids:
            if current <= limit:
                break
            survivors_set.discard(item_id)
            removed.append(item_id)
            current -= token_cost[item_id]

        new_items = [it for it in ctx.items if it.id in survivors_set]

        # If we're still over budget the only remaining lever is STM —
        # either truncate, summarise, or accept the over-budget bundle.
        if current > limit:
            new_items, current = await self._handle_remaining_overage(new_items, current, limit)

        new_messages = _rebuild_messages(ctx, new_items)
        return replace(
            ctx,
            items=new_items,
            messages=new_messages,
            tier_breakdown=_tier_breakdown(new_items),
            tokens_used=current,
            debug_info={
                **ctx.debug_info,
                "score_prune": {
                    "removed": len(removed),
                    "before_tokens": sum(token_cost.values()),
                    "after_tokens": current,
                    "limit": limit,
                    "fit": current <= limit,
                },
            },
        )

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """Pure-arithmetic strategy; latency ≈ 0."""
        current = sum(estimate_tokens_text(it.content) for it in ctx.items)
        return Cost(tokens=current, latency_ms=0.0)

    # ── internals ───────────────────────────────────────────────────────────

    def _pruning_order(self, items: list[MemoryItem]) -> list[str]:
        """
        Return the **ordered** list of item ids eligible for removal.

        First-to-last in this list = first-to-removed. The full list is
        the worst-case sweep; we stop as soon as the budget is met.
        """
        ltm = [it for it in items if it.tier == MemoryTier.LTM]
        mtm = [it for it in items if it.tier == MemoryTier.MTM]
        stm = [it for it in items if it.tier == MemoryTier.STM]

        # LTM: ascending by score, then by created_at (older loses on ties).
        ltm_sorted = sorted(ltm, key=lambda it: (self.score_key(it), _ts_key(it)))

        # MTM: keep the freshest ``preserve_mtm_count``; everything older
        # is eligible, oldest-first.
        mtm_sorted = sorted(mtm, key=_ts_key)
        if self.preserve_mtm_count > 0:
            mtm_removable = mtm_sorted[: max(0, len(mtm_sorted) - self.preserve_mtm_count)]
        else:
            mtm_removable = mtm_sorted

        # STM: only if explicitly opted in.
        stm_removable: list[MemoryItem] = [] if self.preserve_stm else sorted(stm, key=_ts_key)

        return [it.id for it in (*ltm_sorted, *mtm_removable, *stm_removable)]

    async def _handle_remaining_overage(
        self,
        items: list[MemoryItem],
        current: int,
        limit: int,
    ) -> tuple[list[MemoryItem], int]:
        """
        Apply ``too_large_stm_strategy`` when the survivors still bust
        the budget. Returns the (possibly modified) item list + new
        token count.
        """
        if self.too_large_stm_strategy == "keep":
            return items, current

        # Find STM rows that are individually huge; modify in place
        # (functionally, via dataclass.replace).
        new_items: list[MemoryItem] = []
        for it in items:
            tokens = estimate_tokens_text(it.content)
            if it.tier == MemoryTier.STM and tokens + (current - tokens) > limit:
                # This row alone is too big.
                target_tokens = max(0, limit - (current - tokens))
                shrunk = await self._shrink(it, target_tokens)
                shrunk_tokens = estimate_tokens_text(shrunk.content)
                current = current - tokens + shrunk_tokens
                new_items.append(shrunk)
            else:
                new_items.append(it)
        return new_items, current

    async def _shrink(self, item: MemoryItem, target_tokens: int) -> MemoryItem:
        """Apply truncate or summarise according to strategy."""
        if target_tokens <= 0:
            return replace(item, content="")
        if self.too_large_stm_strategy == "summarise" and self.summarizer is not None:
            try:
                summary = await self.summarizer.summarize(item.content)
                return replace(item, content=str(summary))
            except Exception:
                log.exception("ScoreAwareBudgetPrune summariser failed; truncating")
        return replace(item, content=_truncate(item.content, target_tokens))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_key(item: MemoryItem) -> Any:
    ts = getattr(item, "created_at", None)
    return (ts is None, ts)


def _truncate(text: str, target_tokens: int) -> str:
    """Char-level truncation — ~4 chars/token for cl100k_base."""
    if target_tokens <= 0 or not text:
        return ""
    if estimate_tokens_text(text) <= target_tokens:
        return text
    return text[: target_tokens * 4].rstrip() + "…"


def _rebuild_messages(ctx: ContextBundle, items: list[MemoryItem]) -> list[dict[str, str]]:
    survivors = {it.id for it in items}
    out: list[dict[str, str]] = []
    n_items_orig = len(ctx.items)
    for idx, msg in enumerate(ctx.messages):
        if idx >= n_items_orig:
            out.append(msg)
            continue
        orig = ctx.items[idx]
        if orig.id in survivors:
            # Pull the (possibly modified) content from the survivors.
            new_content = next(
                (it.content for it in items if it.id == orig.id),
                msg["content"],
            )
            out.append({"role": msg["role"], "content": new_content})
    return out


def _tier_breakdown(items: list[MemoryItem]) -> dict[str, int]:
    counts: dict[str, int] = {"stm": 0, "mtm": 0, "ltm": 0}
    for it in items:
        counts[it.tier.value] = counts.get(it.tier.value, 0) + estimate_tokens_text(it.content)
    return counts


__all__ = ["ScoreAwareBudgetPrune", "ScorePruneConfig"]
