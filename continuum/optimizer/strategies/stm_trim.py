"""
continuum/optimizer/strategies/stm_trim.py
==========================================
``StmTrim`` — the cheapest optimisation strategy.

Keep the most recent ``keep_last`` STM turns verbatim (so the model sees
the live conversation in full); fold everything older into a single
summary item and shift it into MTM. The bundle's prompt order is
preserved (LTM → MTM → STM) and tier counts are updated accordingly.

Summarisation methods
---------------------
``concat`` (default, free, no LLM): join older turns with newlines and
            truncate to ``max_summary_chars``. Cheap, deterministic,
            tokenizer-friendly.
``llm``   : delegate to an injected ``summarizer.summarize(items) -> str``.
            Higher quality, requires a network call; failures fall back
            to ``concat`` so the chain never breaks the retrieval path.

Ordering rule
-------------
"Last N" is by :attr:`MemoryItem.created_at` ascending — newer items have
later timestamps. Items without a timestamp keep their list order; we use
a stable sort so they sit relative to each other where the Retriever put
them.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Literal
from uuid import uuid4

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


class StmTrimConfig(BaseModel):
    """Configuration for :class:`StmTrim`."""

    model_config = ConfigDict(extra="forbid")

    keep_last: int = Field(
        default=10,
        ge=0,
        description="How many recent STM turns to keep verbatim.",
    )
    summarization_method: Literal["concat", "llm"] = Field(
        default="concat",
        description=(
            "How to compress older STM turns. 'concat' joins+truncates "
            "(free, deterministic). 'llm' calls the injected summarizer."
        ),
    )
    max_summary_chars: int = Field(
        default=2000,
        ge=64,
        description=(
            "Character cap on the produced summary text (concat method) — "
            "keeps the resulting MTM row from blowing the budget itself."
        ),
    )


class StmTrim(BaseOptimizer):
    """
    Keep last N STM turns; fold older turns into a single MTM summary.

    Parameters
    ----------
    keep_last:
        Number of most-recent STM turns to keep verbatim. Default 10.
    summarization_method:
        ``"concat"`` (no LLM) or ``"llm"`` (delegate to *summarizer*).
    summarizer:
        Required only when ``summarization_method='llm'``. Must expose
        ``async summarize(items: list[MemoryItem]) -> str``. If the LLM
        call raises, ``StmTrim`` falls back to concat so the chain stays
        robust.
    max_summary_chars:
        Truncation cap for the produced summary (concat path).
    config:
        Alternative way to pass all parameters at once — when supplied,
        individual kwargs override its fields.
    """

    def __init__(
        self,
        *,
        keep_last: int | None = None,
        summarization_method: Literal["concat", "llm"] | None = None,
        summarizer: Any | None = None,
        max_summary_chars: int | None = None,
        config: StmTrimConfig | None = None,
    ) -> None:
        base = config or StmTrimConfig()
        self.keep_last: int = keep_last if keep_last is not None else base.keep_last
        self.summarization_method: Literal["concat", "llm"] = (
            summarization_method if summarization_method is not None else base.summarization_method
        )
        self.max_summary_chars: int = (
            max_summary_chars if max_summary_chars is not None else base.max_summary_chars
        )
        self.summarizer = summarizer

    # ── public API ──────────────────────────────────────────────────────────

    async def apply(self, ctx: ContextBundle, budget: TokenBudget) -> ContextBundle:
        stm_items, other_items = _partition_by_tier(ctx.items, MemoryTier.STM)
        if len(stm_items) <= self.keep_last:
            return ctx  # nothing to trim

        # Newer items at the end; "last N" = the tail of created_at-sorted.
        ordered = sorted(stm_items, key=_created_at_key)
        cutoff = len(ordered) - self.keep_last
        older = ordered[:cutoff]
        recent = ordered[cutoff:]

        try:
            summary_text = await self._summarise(older)
        except Exception:
            # The summarizer is the only place this method can raise; the
            # OptimizerChain would swallow it, but we'd rather degrade
            # cleanly than skip the trim altogether.
            log.exception("StmTrim summariser failed — falling back to concat")
            summary_text = _concat_summary(older, self.max_summary_chars)

        summary_item = _build_summary_item(summary_text, older)

        # Reassemble: keep non-STM order intact, prepend the new MTM
        # summary in front of any existing MTM rows so the most recently
        # compacted block appears closest to the model's eye.
        new_items = _splice(other_items, summary_item, recent)
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
                "stm_trim": {
                    "kept": len(recent),
                    "summarised": len(older),
                    "method": self.summarization_method,
                    "summary_chars": len(summary_text),
                },
            },
        )

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """
        Cheap projection: assume every trimmed turn collapses to ~80
        characters of summary text (worst case). Latency = 0 for concat,
        coarse 300 ms upper bound for LLM.
        """
        stm_items = [it for it in ctx.items if it.tier == MemoryTier.STM]
        if len(stm_items) <= self.keep_last:
            return Cost(
                tokens=sum(estimate_tokens_text(it.content) for it in ctx.items),
                latency_ms=0.0,
            )
        kept_tokens = sum(estimate_tokens_text(it.content) for it in stm_items[-self.keep_last :])
        non_stm_tokens = sum(
            estimate_tokens_text(it.content) for it in ctx.items if it.tier != MemoryTier.STM
        )
        # ~20 tokens per dropped turn is a generous upper bound for the
        # summary's contribution.
        summary_tokens = min(
            self.max_summary_chars // 4,
            20 * (len(stm_items) - self.keep_last),
        )
        return Cost(
            tokens=non_stm_tokens + summary_tokens + kept_tokens,
            latency_ms=300.0 if self.summarization_method == "llm" else 0.0,
        )

    # ── internals ───────────────────────────────────────────────────────────

    async def _summarise(self, older: list[MemoryItem]) -> str:
        if self.summarization_method == "llm" and self.summarizer is not None:
            return str(await self.summarizer.summarize(older))
        return _concat_summary(older, self.max_summary_chars)


# ---------------------------------------------------------------------------
# Helpers (module-level so they're testable and free of self-state)
# ---------------------------------------------------------------------------


def _partition_by_tier(
    items: list[MemoryItem], tier: MemoryTier
) -> tuple[list[MemoryItem], list[MemoryItem]]:
    """Split *items* into (matching tier, rest), preserving order."""
    a: list[MemoryItem] = []
    b: list[MemoryItem] = []
    for it in items:
        (a if it.tier == tier else b).append(it)
    return a, b


def _created_at_key(item: MemoryItem) -> Any:
    """Sort key that pushes items without timestamps to the front."""
    ts = getattr(item, "created_at", None)
    return (ts is None, ts)


def _concat_summary(items: list[MemoryItem], max_chars: int) -> str:
    """Free-of-charge summary: join texts, truncate to *max_chars*."""
    if not items:
        return ""
    joined = " | ".join(it.content.strip() for it in items if it.content)
    if len(joined) <= max_chars:
        return joined
    # Truncate on a word boundary when possible.
    cut = joined[: max_chars - 1]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut + "…"


def _build_summary_item(text: str, older: list[MemoryItem]) -> MemoryItem:
    """Wrap *text* in a fresh MTM-tier item, carrying useful provenance."""
    return MemoryItem(
        id=str(uuid4()),
        content=text,
        tier=MemoryTier.MTM,
        importance=0.5,
        confidence=1.0,
        metadata={
            "kind": "stm_trim_summary",
            "compacted_from": [str(it.id) for it in older],
            "compacted_count": len(older),
        },
    )


def _splice(
    other_items: list[MemoryItem],
    summary_item: MemoryItem,
    recent: list[MemoryItem],
) -> list[MemoryItem]:
    """
    Rebuild the items list keeping the prompt order LTM → MTM → STM.

    The summary item is inserted at the *front* of the MTM run (so the
    most recently compacted block sits closest to the conversation).
    """
    out: list[MemoryItem] = []
    inserted = False
    saw_mtm = False
    for it in other_items:
        if it.tier == MemoryTier.MTM:
            if not inserted:
                out.append(summary_item)
                inserted = True
            saw_mtm = True
            out.append(it)
        else:
            # If we had MTM rows and just left that run, the summary
            # should sit at the end of MTM if we didn't insert yet.
            if saw_mtm and not inserted:
                out.append(summary_item)
                inserted = True
            out.append(it)
    if not inserted:
        # No MTM rows existed; place the summary just before STM (i.e.
        # at the end of the non-STM run).
        out.append(summary_item)
    out.extend(recent)
    return out


def _rebuild_messages(ctx: ContextBundle, items: list[MemoryItem]) -> list[dict[str, str]]:
    """
    Reflect the new items order in ``messages``. Non-STM items use the
    ``"system"`` role (matching the Retriever's convention); STM items
    keep whatever role they had in metadata.
    """
    messages: list[dict[str, str]] = []
    for it in items:
        if it.tier == MemoryTier.STM:
            role = str(it.metadata.get("role", "user")) if it.metadata else "user"
        else:
            role = "system"
        messages.append({"role": role, "content": it.content})
    # Preserve any non-memory messages (e.g. injected system prompts) the
    # Retriever may have appended that don't correspond to items.
    extras = max(0, len(ctx.messages) - len(items))
    if extras:
        messages.extend(ctx.messages[-extras:])
    return messages


def _tier_breakdown(items: list[MemoryItem]) -> dict[str, int]:
    counts: dict[str, int] = {"stm": 0, "mtm": 0, "ltm": 0}
    for it in items:
        tokens = estimate_tokens_text(it.content)
        counts[it.tier.value] = counts.get(it.tier.value, 0) + tokens
    return counts


__all__ = ["StmTrim", "StmTrimConfig"]
