"""
continuum/optimizer/strategies/mtm_summarize.py
================================================
``MtmSummarize`` — collapse older MTM blocks into a single dense summary.

Strategy (in order)
-------------------
1. Filter the bundle items down to ``MemoryTier.MTM`` rows.
2. If there are ≤ ``keep_recent`` MTM rows, no-op.
3. Concatenate the older rows' text (oldest → newest), feed it through a
   summariser (extractive *or* abstractive), and emit one fresh MTM row
   carrying the result.
4. Rebuild the bundle: ``LTM → [summary, recent MTM rows] → STM``.

Summarisation methods
---------------------
``extractive`` (default, free, deterministic): tokenise into sentences,
                score each by **normalised term frequency** + a mild
                position bias, then greedily pick the highest-scoring
                sentences (in their original order) until ``max_summary_tokens``
                is reached. No ML deps.

``llm``      : delegate to ``summarizer.summarize(text, max_tokens=N)``
                — the collaborator returns a string. ``MtmSummarize``
                ships a fixed ``MTM_SYSTEM_PROMPT`` constant the
                collaborator can use as the system message; keeping the
                prompt literal lets Anthropic / OpenAI prompt caching
                hit on every call. Failures fall back to extractive.

Token reduction
---------------
Empirically the extractive path yields ~50–70 % reduction on multi-block
inputs; the test suite asserts a 30 % floor to stay robust under varied
text. The LLM path is bounded by ``max_summary_tokens`` directly.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
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

#: System prompt held constant across calls so prompt caching can hit.
MTM_SYSTEM_PROMPT = (
    "You are a precise summariser. Compress the user's MTM project "
    "context into ≤ {max_tokens} tokens, preserving every concrete "
    "fact, entity, decision, and deadline. Use terse declarative "
    "sentences. Do not invent details."
)

# Tiny English stopword set — enough to suppress noise in TF scoring
# without dragging in NLTK. Don't expand without a benchmark.
_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "me",
        "him",
        "us",
        "as",
    ]
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"\[])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MtmSummarizeConfig(BaseModel):
    """Configuration for :class:`MtmSummarize`."""

    model_config = ConfigDict(extra="forbid")

    keep_recent: int = Field(
        default=3,
        ge=0,
        description="Number of most-recent MTM blocks kept verbatim.",
    )
    method: Literal["extractive", "llm"] = Field(
        default="extractive",
        description=(
            "How to compress older MTM blocks. 'extractive' picks top-"
            "ranked sentences (free). 'llm' delegates to the injected "
            "summariser."
        ),
    )
    max_summary_tokens: int = Field(
        default=500,
        ge=32,
        description="Upper bound on the produced summary's token count.",
    )
    position_bias: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Extractive only: small boost for early sentences (news-"
            "summary heuristic). 0 disables; 0.15 is a sensible default."
        ),
    )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class MtmSummarize(BaseOptimizer):
    """
    Compress older MTM blocks into a single summary row.

    Parameters
    ----------
    keep_recent:
        Number of most-recent MTM blocks kept verbatim. Default 3.
    method:
        ``"extractive"`` (default, no LLM) or ``"llm"`` (delegates to
        *summarizer*).
    summarizer:
        Required only for ``method='llm'``. Must expose
        ``async summarize(text: str, *, max_tokens: int,
        system_prompt: str | None = None) -> str``. Failures fall back
        to extractive.
    model:
        Optional model hint forwarded to the summariser's ``model=``
        kwarg if it accepts one. Defaults to ``"gpt-4o-mini"`` to match
        the rest of the framework.
    max_summary_tokens:
        Token cap for the produced summary.
    config:
        Bundle the four scalars at once; explicit kwargs override.
    """

    def __init__(
        self,
        *,
        keep_recent: int | None = None,
        method: Literal["extractive", "llm"] | None = None,
        summarizer: Any | None = None,
        model: str = "gpt-4o-mini",
        max_summary_tokens: int | None = None,
        position_bias: float | None = None,
        config: MtmSummarizeConfig | None = None,
    ) -> None:
        base = config or MtmSummarizeConfig()
        self.keep_recent: int = keep_recent if keep_recent is not None else base.keep_recent
        self.method: Literal["extractive", "llm"] = method if method is not None else base.method
        self.max_summary_tokens: int = (
            max_summary_tokens if max_summary_tokens is not None else base.max_summary_tokens
        )
        self.position_bias: float = (
            position_bias if position_bias is not None else base.position_bias
        )
        self.summarizer = summarizer
        self.model = model

    # ── public API ──────────────────────────────────────────────────────────

    async def apply(self, ctx: ContextBundle, budget: TokenBudget) -> ContextBundle:
        mtm_items = [it for it in ctx.items if it.tier == MemoryTier.MTM]
        if len(mtm_items) <= self.keep_recent:
            return ctx

        # Order by created_at; oldest first so the summary reads
        # chronologically.
        ordered = sorted(mtm_items, key=_created_at_key)
        cutoff = len(ordered) - self.keep_recent
        older = ordered[:cutoff]
        recent = ordered[cutoff:]

        joined = "\n\n".join(it.content.strip() for it in older if it.content)
        try:
            summary_text = await self._summarise(joined)
        except Exception:
            log.exception("MtmSummarize summariser failed — falling back to extractive")
            summary_text = extractive_summary(
                joined,
                max_tokens=self.max_summary_tokens,
                position_bias=self.position_bias,
            )

        summary_item = _build_summary_item(summary_text, older, self.method)
        new_items = _rebuild(ctx.items, older, recent, summary_item)
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
                "mtm_summarize": {
                    "kept": len(recent),
                    "summarised": len(older),
                    "method": self.method,
                    "input_tokens": estimate_tokens_text(joined),
                    "output_tokens": estimate_tokens_text(summary_text),
                },
            },
        )

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """
        Projection: tokens drop by the older-block contribution minus
        ``max_summary_tokens``. Latency ~ 0 for extractive, 800 ms ceiling
        for LLM.
        """
        mtm_items = [it for it in ctx.items if it.tier == MemoryTier.MTM]
        current = sum(estimate_tokens_text(it.content) for it in ctx.items)
        if len(mtm_items) <= self.keep_recent:
            return Cost(tokens=current, latency_ms=0.0)
        older_tokens = sum(
            estimate_tokens_text(it.content)
            for it in sorted(mtm_items, key=_created_at_key)[: -self.keep_recent]
        )
        projected = current - older_tokens + self.max_summary_tokens
        return Cost(
            tokens=max(0, projected),
            latency_ms=800.0 if self.method == "llm" else 0.0,
        )

    # ── internals ───────────────────────────────────────────────────────────

    async def _summarise(self, text: str) -> str:
        if self.method == "llm" and self.summarizer is not None:
            return await self._llm_summary(text)
        return extractive_summary(
            text,
            max_tokens=self.max_summary_tokens,
            position_bias=self.position_bias,
        )

    async def _llm_summary(self, text: str) -> str:
        """Delegate; tolerate summarizers with simpler signatures."""
        assert self.summarizer is not None  # checked by caller
        system_prompt = MTM_SYSTEM_PROMPT.format(max_tokens=self.max_summary_tokens)
        # Try the rich signature first, then degrade gracefully so a
        # plain ``async summarize(text) -> str`` collaborator still works.
        try:
            return str(
                await self.summarizer.summarize(
                    text,
                    max_tokens=self.max_summary_tokens,
                    system_prompt=system_prompt,
                    model=self.model,
                )
            )
        except TypeError:
            try:
                return str(
                    await self.summarizer.summarize(text, max_tokens=self.max_summary_tokens)
                )
            except TypeError:
                return str(await self.summarizer.summarize(text))


# ---------------------------------------------------------------------------
# Extractive summarisation (module-level so it is testable directly)
# ---------------------------------------------------------------------------


def extractive_summary(
    text: str,
    *,
    max_tokens: int = 500,
    position_bias: float = 0.15,
) -> str:
    """
    Pick the highest-scoring sentences until *max_tokens* is reached.

    Scoring = normalised term frequency + a small linear bias toward
    early sentences (news-summary heuristic). Stopwords are filtered.
    Picked sentences are emitted in their **original** order so the
    summary reads naturally.
    """
    if not text or not text.strip():
        return ""

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return text[:max_tokens] if max_tokens > 0 else ""

    # Term frequencies (stopword-filtered, lowercased).
    tokens_per_sentence: list[list[str]] = [
        [w.lower() for w in _WORD_RE.findall(s) if w.lower() not in _STOPWORDS and len(w) > 2]
        for s in sentences
    ]
    flat = [w for sent in tokens_per_sentence for w in sent]
    if not flat:
        # Degenerate: nothing to score. Return the first sentence(s)
        # under budget — better than empty.
        return _truncate_to_tokens(" ".join(sentences), max_tokens)

    tf = Counter(flat)
    max_tf = max(tf.values()) or 1

    n_sent = len(sentences)
    scored: list[tuple[float, int, str]] = []
    for idx, (sent, toks) in enumerate(zip(sentences, tokens_per_sentence, strict=False)):
        if not toks:
            score = 0.0
        else:
            score = sum(tf[w] / max_tf for w in toks) / math.sqrt(len(toks))
        # Position bias: linear decay, max bonus on sentence 0.
        if position_bias > 0:
            score += position_bias * (1.0 - idx / max(1, n_sent - 1))
        scored.append((score, idx, sent))

    # Greedy selection: highest score first, but emit in original order.
    scored.sort(key=lambda x: x[0], reverse=True)
    picked_indices: list[int] = []
    used_tokens = 0
    for _score, idx, sent in scored:
        cost = estimate_tokens_text(sent)
        if used_tokens + cost > max_tokens and picked_indices:
            continue
        picked_indices.append(idx)
        used_tokens += cost
        if used_tokens >= max_tokens:
            break

    picked_indices.sort()
    return " ".join(sentences[i] for i in picked_indices)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens_text(text) <= max_tokens:
        return text
    # Coarse char-based truncation: cl100k_base averages ~4 chars/token.
    return text[: max_tokens * 4].rstrip() + "…"


# ---------------------------------------------------------------------------
# Bundle plumbing helpers
# ---------------------------------------------------------------------------


def _created_at_key(item: MemoryItem) -> Any:
    ts = getattr(item, "created_at", None)
    return (ts is None, ts)


def _build_summary_item(text: str, older: list[MemoryItem], method: str) -> MemoryItem:
    return MemoryItem(
        id=str(uuid4()),
        content=text,
        tier=MemoryTier.MTM,
        importance=0.6,
        confidence=1.0,
        metadata={
            "kind": "mtm_summary",
            "summarisation_method": method,
            "compacted_from": [str(it.id) for it in older],
            "compacted_count": len(older),
        },
    )


def _rebuild(
    items: list[MemoryItem],
    older: list[MemoryItem],
    recent: list[MemoryItem],
    summary: MemoryItem,
) -> list[MemoryItem]:
    """
    Drop *older* from the original order, inject *summary* in front of
    the surviving MTM run. Non-MTM items keep their position.
    """
    older_ids = {it.id for it in older}
    out: list[MemoryItem] = []
    inserted = False
    for it in items:
        if it.id in older_ids:
            continue
        if not inserted and it.tier == MemoryTier.MTM:
            out.append(summary)
            inserted = True
        out.append(it)
    if not inserted:
        # Original list had no surviving MTM rows (older filled the
        # whole MTM run). Place the summary just before the first STM,
        # falling back to the end of the list.
        first_stm = next(
            (i for i, it in enumerate(out) if it.tier == MemoryTier.STM),
            len(out),
        )
        out.insert(first_stm, summary)
        # We may have emitted *recent* implicitly via the loop above, but
        # _rebuild's caller passes recent as a subset of items, so they
        # are already present. Nothing else to do.
    _ = recent  # documented contract; kept for API symmetry
    return out


def _rebuild_messages(ctx: ContextBundle, items: list[MemoryItem]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for it in items:
        if it.tier == MemoryTier.STM:
            role = str(it.metadata.get("role", "user")) if it.metadata else "user"
        else:
            role = "system"
        messages.append({"role": role, "content": it.content})
    extras = max(0, len(ctx.messages) - len(items))
    if extras:
        messages.extend(ctx.messages[-extras:])
    return messages


def _tier_breakdown(items: list[MemoryItem]) -> dict[str, int]:
    counts: dict[str, int] = {"stm": 0, "mtm": 0, "ltm": 0}
    for it in items:
        counts[it.tier.value] = counts.get(it.tier.value, 0) + estimate_tokens_text(it.content)
    return counts


__all__ = [
    "MtmSummarize",
    "MtmSummarizeConfig",
    "MTM_SYSTEM_PROMPT",
    "extractive_summary",
]
