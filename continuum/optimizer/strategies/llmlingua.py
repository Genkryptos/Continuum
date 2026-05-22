"""
continuum/optimizer/strategies/llmlingua.py
============================================
``LLMLinguaCompress`` — token-level prompt compression via LLMLingua-2.

What it does
------------
Take every LTM row in the bundle, join them with a stable separator,
hand the joined text to LLMLingua-2 (paper: arXiv 2403.12968) for
compression, then either:

* split the compressed text back into the original number of items
  proportionally by length (so individual rows still address-correctly), or
* collapse the LTM run into a single ``llmlingua_compressed`` row if the
  split fails or the original rows have no natural separator structure.

LLMLingua-2 reports 2–5× compression with minimal quality loss on
GPT-class downstream tasks. The model is heavy (~500 MB) and CPU-slow
(seconds per 1 K tokens), so:

* The package is imported **lazily**.
* The compressor itself is **injectable** so unit tests don't load a
  real ML model.
* Compression is offloaded to a worker thread via
  :func:`asyncio.to_thread` — never blocks the event loop.
* Results are cached on the SHA-256 of the input + ratio, so repeat
  bundles (e.g. background promotion sweeps over the same context) pay
  the model cost once.
* The strategy short-circuits when the input is shorter than
  ``min_input_tokens`` — compression overhead dominates on small
  contexts.

Quality gate
------------
If an ``embedder`` collaborator is wired, ``LLMLinguaCompress`` computes
the cosine similarity between the original and compressed text
embeddings; when the similarity drops below ``quality_threshold`` the
compressed output is **rejected** and the bundle returned unchanged.
This guards against pathological compressions that drop key entities.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import replace
from typing import Any
from uuid import uuid4

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

#: Separator stitched between items before compression and used to split
#: the compressed output back. Chosen to be improbable in real text and
#: tokenizer-friendly (LLMLingua preserves rare strings).
_ITEM_SEP = " ⟦SEP⟧ "

DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LLMLinguaConfig(BaseModel):
    """Configuration for :class:`LLMLinguaCompress`."""

    model_config = ConfigDict(extra="forbid")

    ratio: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Target retained fraction. 0.5 = compress to 50 % of the "
            "original token count. LLMLingua-2 calls this ``rate``."
        ),
    )
    min_input_tokens: int = Field(
        default=500,
        ge=0,
        description=(
            "Skip compression below this many input tokens — model "
            "overhead dominates on short inputs."
        ),
    )
    quality_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Reject compressed output whose cosine similarity to the "
            "original (per the injected embedder) falls below this floor. "
            "Has no effect if no embedder is wired."
        ),
    )
    cache_size: int = Field(
        default=128,
        ge=0,
        description="Maximum entries in the in-memory compression cache.",
    )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class LLMLinguaCompress(BaseOptimizer):
    """
    Compress LTM text via LLMLingua-2.

    Parameters
    ----------
    ratio:
        Target retained fraction (0 < r < 1). Default 0.5.
    min_input_tokens:
        Skip if the joined LTM text is shorter than this. Default 500.
    quality_threshold:
        Cosine-similarity floor for the compressed output. Default 0.85.
        No-op unless ``embedder`` is wired.
    model_name:
        Override the default LLMLingua-2 model. Ignored when
        ``compressor`` is injected.
    compressor:
        Pre-built ``llmlingua.PromptCompressor`` (or a fake). When
        ``None`` the package is imported lazily on first call.
    embedder:
        Optional collaborator exposing
        ``async embed(text: str) -> list[float] | np.ndarray`` for the
        quality gate.
    config:
        Bundle the four scalars at once; explicit kwargs override.
    """

    def __init__(
        self,
        *,
        ratio: float | None = None,
        min_input_tokens: int | None = None,
        quality_threshold: float | None = None,
        model_name: str = DEFAULT_MODEL,
        compressor: Any | None = None,
        embedder: Any | None = None,
        cache_size: int | None = None,
        config: LLMLinguaConfig | None = None,
    ) -> None:
        base = config or LLMLinguaConfig()
        self.ratio: float = ratio if ratio is not None else base.ratio
        self.min_input_tokens: int = (
            min_input_tokens
            if min_input_tokens is not None
            else base.min_input_tokens
        )
        self.quality_threshold: float = (
            quality_threshold
            if quality_threshold is not None
            else base.quality_threshold
        )
        self.cache_size: int = (
            cache_size if cache_size is not None else base.cache_size
        )
        self.model_name = model_name
        self._compressor = compressor
        self._embedder = embedder
        self._cache: OrderedDict[str, str] = OrderedDict()

    # ── public API ──────────────────────────────────────────────────────────

    async def apply(
        self, ctx: ContextBundle, budget: TokenBudget
    ) -> ContextBundle:
        ltm_items = [it for it in ctx.items if it.tier == MemoryTier.LTM]
        if len(ltm_items) < 1:
            return ctx

        joined = _ITEM_SEP.join(it.content for it in ltm_items)
        input_tokens = estimate_tokens_text(joined)
        if input_tokens < self.min_input_tokens:
            return ctx

        try:
            compressed_text = await self._compress(joined)
        except Exception:
            # Lib missing, model load failed, runtime error … swallow so
            # the chain keeps going (it'll be logged loudly).
            log.exception(
                "LLMLingua compression failed — returning bundle unchanged"
            )
            return ctx

        if not compressed_text:
            return ctx

        if not await self._quality_ok(joined, compressed_text):
            log.info(
                "LLMLingua output failed quality gate (< %.2f); skipping",
                self.quality_threshold,
            )
            return ctx

        new_ltm_items = _split_back(
            compressed_text, ltm_items, separator=_ITEM_SEP
        )
        new_items = _splice(ctx.items, ltm_items, new_ltm_items)
        new_messages = _rebuild_messages(ctx, new_items)
        output_tokens = sum(
            estimate_tokens_text(i.content) for i in new_ltm_items
        )
        return replace(
            ctx,
            items=new_items,
            messages=new_messages,
            tier_breakdown=_tier_breakdown(new_items),
            tokens_used=sum(
                estimate_tokens_text(i.content) for i in new_items
            ),
            debug_info={
                **ctx.debug_info,
                "llmlingua": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "ratio": (
                        output_tokens / input_tokens if input_tokens else 0
                    ),
                    "target_ratio": self.ratio,
                    "cache_hit": self._last_was_cache_hit,
                },
            },
        )

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """
        Project tokens after compression. Latency is dominated by the
        forward pass — ~2 s per 1 K input tokens on CPU, ~200 ms on GPU.
        Use a 2 s/1K coarse upper bound here.
        """
        ltm = [it for it in ctx.items if it.tier == MemoryTier.LTM]
        current = sum(estimate_tokens_text(it.content) for it in ctx.items)
        ltm_tokens = sum(estimate_tokens_text(it.content) for it in ltm)
        if ltm_tokens < self.min_input_tokens:
            return Cost(tokens=current, latency_ms=0.0)
        compressed = int(ltm_tokens * self.ratio)
        return Cost(
            tokens=max(0, current - ltm_tokens + compressed),
            latency_ms=2.0 * ltm_tokens,  # 2 ms per token (CPU upper bound)
        )

    # ── compression internals ───────────────────────────────────────────────

    _last_was_cache_hit: bool = False

    async def _compress(self, text: str) -> str:
        """
        Cache → in-memory fast path; miss → ``asyncio.to_thread`` into
        the synchronous LLMLingua call.
        """
        key = self._cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._last_was_cache_hit = True
            return self._cache[key]
        self._last_was_cache_hit = False

        compressor = await self._get_compressor()
        if compressor is None:
            raise RuntimeError("LLMLingua compressor unavailable")

        result = await asyncio.to_thread(
            _call_compressor, compressor, text, self.ratio
        )
        compressed = _extract_compressed_text(result)
        if compressed:
            self._cache[key] = compressed
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return compressed

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(f"{self.ratio}:".encode())
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    async def _get_compressor(self) -> Any | None:
        """Return the injected compressor, or lazy-load the real one."""
        if self._compressor is not None:
            return self._compressor
        try:
            from llmlingua import PromptCompressor
        except ImportError:  # pragma: no cover - optional dep
            log.warning(
                "llmlingua not installed; LLMLinguaCompress is a no-op. "
                "Install with: pip install llmlingua"
            )
            return None
        try:
            self._compressor = await asyncio.to_thread(
                PromptCompressor,
                model_name=self.model_name,
                use_llmlingua2=True,
            )
        except Exception:
            log.exception("loading LLMLingua model failed")
            return None
        return self._compressor

    # ── quality gate ────────────────────────────────────────────────────────

    async def _quality_ok(self, original: str, compressed: str) -> bool:
        if self._embedder is None:
            return True
        try:
            v_orig = await self._embedder.embed(original)
            v_comp = await self._embedder.embed(compressed)
        except Exception:
            log.exception("embedder failed during quality gate; allowing")
            return True
        a = np.asarray(v_orig, dtype=np.float32)
        b = np.asarray(v_comp, dtype=np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return True
        cos = float(np.dot(a, b) / (na * nb))
        return cos >= self.quality_threshold


# ---------------------------------------------------------------------------
# Helpers — module-level for independent testing
# ---------------------------------------------------------------------------


def _call_compressor(compressor: Any, text: str, ratio: float) -> Any:
    """
    Run the synchronous LLMLingua call. Wrapped in a function so
    :func:`asyncio.to_thread` has a clean target and tests can replace
    it via the injected compressor.
    """
    return compressor.compress_prompt(text, rate=ratio, force_tokens=[])


def _extract_compressed_text(result: Any) -> str:
    """
    LLMLingua versions differ on return shape. Handle the common cases:
    a dict with ``compressed_prompt`` (current), or a bare string.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("compressed_prompt", "compressed", "output"):
            v = result.get(key)
            if isinstance(v, str):
                return v
    return ""


def _split_back(
    compressed: str, originals: list[MemoryItem], *, separator: str
) -> list[MemoryItem]:
    """
    Map *compressed* back onto the original item count.

    Preferred path: split on the literal separator. If the compressor
    preserved them, we get N pieces and can pair item-for-item.

    Fallback: replace all originals with a single ``llmlingua_compressed``
    item carrying the full compressed text. Better to ship one accurate
    row than N speculative splits.
    """
    pieces = [p.strip() for p in compressed.split(separator) if p.strip()]
    if len(pieces) == len(originals):
        return [
            replace(orig, content=piece, embedding=None)
            for orig, piece in zip(originals, pieces, strict=True)
        ]
    # Fallback: single fused row, provenance preserved.
    return [
        MemoryItem(
            id=str(uuid4()),
            content=compressed,
            tier=MemoryTier.LTM,
            importance=max(it.importance for it in originals),
            confidence=min(it.confidence for it in originals),
            metadata={
                "kind": "llmlingua_compressed",
                "compacted_from": [str(it.id) for it in originals],
                "compacted_count": len(originals),
            },
        )
    ]


def _splice(
    items: list[MemoryItem],
    old_ltm: list[MemoryItem],
    new_ltm: list[MemoryItem],
) -> list[MemoryItem]:
    """
    Replace the LTM run in *items* with *new_ltm*. Original list order
    is otherwise preserved: LTM rows live wherever they did, and the
    new rows are dropped in at the position of the first original.
    """
    old_ids = {it.id for it in old_ltm}
    out: list[MemoryItem] = []
    inserted = False
    for it in items:
        if it.id in old_ids:
            if not inserted:
                out.extend(new_ltm)
                inserted = True
            continue
        out.append(it)
    if not inserted:
        out.extend(new_ltm)
    return out


def _rebuild_messages(
    ctx: ContextBundle, items: list[MemoryItem]
) -> list[dict[str, str]]:
    """One message per item, role inferred from tier."""
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
        counts[it.tier.value] = counts.get(it.tier.value, 0) + estimate_tokens_text(
            it.content
        )
    return counts


__all__ = [
    "LLMLinguaCompress",
    "LLMLinguaConfig",
    "DEFAULT_MODEL",
]
