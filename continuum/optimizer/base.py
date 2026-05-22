"""
continuum/optimizer/base.py
===========================
Shared helpers for optimisation strategies.

* :func:`estimate_tokens_text` — count tokens in a single string using
  ``tiktoken`` (encoding ``cl100k_base`` by default), with a whitespace
  fallback if tiktoken is unavailable.
* :func:`estimate_tokens` — sum ``estimate_tokens_text`` across every
  :class:`MemoryItem` in a :class:`ContextBundle`. Counts STM + MTM + LTM
  uniformly: the bundle already deduplicates by item id and each item's
  ``content`` is the canonical text the LLM will see.
* :class:`BaseOptimizer` — ABC providing a sensible default
  ``cost_estimate`` so concrete strategies only override ``apply``.

The encoder is cached at module scope to avoid the cold-start hit on every
estimation call.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from continuum.core.types import ContextBundle, TokenBudget
from continuum.optimizer.protocol import Cost

if TYPE_CHECKING:  # pragma: no cover - type only
    pass

log = logging.getLogger(__name__)

#: Default tiktoken encoding — matches GPT-4 / GPT-3.5 / Claude tokenisers
#: closely enough for budget arithmetic.
_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=4)
def _get_encoder(encoding: str = _DEFAULT_ENCODING) -> Any | None:
    """Return a tiktoken encoder; ``None`` if tiktoken isn't installed."""
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - optional dep
        log.debug("tiktoken unavailable — falling back to whitespace counter")
        return None
    try:
        return tiktoken.get_encoding(encoding)
    except Exception:  # pragma: no cover - bad encoding name
        log.exception("tiktoken.get_encoding(%r) failed", encoding)
        return None


def estimate_tokens_text(text: str, *, encoding: str = _DEFAULT_ENCODING) -> int:
    """
    Count tokens in *text*.

    Uses tiktoken (``cl100k_base`` by default) when available. Falls back
    to a whitespace-split count so this function works without optional
    dependencies. Returns ``0`` for empty / non-string input.
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    enc = _get_encoder(encoding)
    if enc is None:
        return len(text.split())
    try:
        return len(enc.encode(text))
    except Exception:  # pragma: no cover - encoder edge cases
        log.debug("tiktoken encode failed; using whitespace fallback",
                  exc_info=True)
        return len(text.split())


def estimate_tokens(
    ctx: ContextBundle, *, encoding: str = _DEFAULT_ENCODING
) -> int:
    """
    Sum :func:`estimate_tokens_text` over every item in *ctx*.

    The bundle's ``items`` already contains STM + MTM + LTM in assembled
    order (no double counting) — the caller of the Retriever assembles
    them that way. We deliberately *don't* read ``ctx.tokens_used`` here
    because that field reflects whatever counter the Retriever used (often
    a whitespace estimate); the optimizer needs the real token count.
    """
    return sum(estimate_tokens_text(it.content, encoding=encoding)
               for it in ctx.items)


class BaseOptimizer(ABC):
    """
    Convenience ABC for strategies.

    Provides a default :meth:`cost_estimate` that returns the current
    bundle's token count and zero latency. LLM-backed strategies should
    override it with a smarter projection (e.g. expected post-compression
    size, model latency p50).
    """

    @abstractmethod
    async def apply(
        self, ctx: ContextBundle, budget: TokenBudget
    ) -> ContextBundle:
        """Apply the strategy. Concrete subclasses implement."""

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        return Cost(tokens=estimate_tokens(ctx), latency_ms=0.0)


__all__ = [
    "estimate_tokens",
    "estimate_tokens_text",
    "BaseOptimizer",
]
