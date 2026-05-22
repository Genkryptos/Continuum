"""
continuum/optimizer/chain.py
============================
``OptimizerChain`` — compose strategies into a single :class:`Optimizer`.

Each strategy in the chain is applied **in order**. After every step the
chain re-counts tokens; once the bundle fits under ``budget.context_available``
remaining strategies are skipped (the *short-circuit*). Strategies later
in the chain are therefore "more expensive / more lossy" by convention —
cheap structural trims first, then summarisation, dedup, compression, and
finally score-aware pruning as a last resort.

:func:`default_optimizer_chain` builds the canonical 5-strategy stack from
an :class:`OptimizerConfig`. The concrete strategy classes are imported
**lazily** so this module loads cheaply even before they're implemented —
callers that don't invoke ``default_optimizer_chain`` never pay the cost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from continuum.core.types import ContextBundle, TokenBudget
from continuum.optimizer.base import estimate_tokens
from continuum.optimizer.protocol import Cost, Optimizer

if TYPE_CHECKING:  # pragma: no cover - type only
    from continuum.core.config import OptimizerConfig

log = logging.getLogger(__name__)


class OptimizerChain:
    """
    Compose a list of :class:`Optimizer` strategies into one.

    The chain itself satisfies :class:`Optimizer` so it can be wired
    anywhere an individual strategy is expected (and even nested).
    """

    def __init__(self, strategies: list[Optimizer]) -> None:
        self.strategies: list[Optimizer] = list(strategies)

    async def apply(
        self, ctx: ContextBundle, budget: TokenBudget
    ) -> ContextBundle:
        """
        Apply each strategy in order. Short-circuit once the bundle fits.

        Errors raised by an individual strategy are logged and swallowed —
        a misbehaving optimiser must never break the retrieval path.
        """
        limit = budget.context_available
        for strategy in self.strategies:
            current = estimate_tokens(ctx)
            if current <= limit:
                log.debug(
                    "OptimizerChain short-circuit at %s/%s tokens",
                    current, limit,
                )
                return ctx
            try:
                ctx = await strategy.apply(ctx, budget)
            except Exception:
                log.exception(
                    "optimizer %s.apply failed — continuing chain",
                    type(strategy).__name__,
                )
        return ctx

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """
        Sum tokens through the chain by best estimate (cheap; no calls).

        Each strategy's ``cost_estimate`` is consulted in order and the
        running token count is replaced by its projection. Latency is
        summed because strategies run sequentially.
        """
        running = estimate_tokens(ctx)
        latency = 0.0
        for strategy in self.strategies:
            try:
                est = strategy.cost_estimate(ctx)
            except Exception:
                log.debug(
                    "cost_estimate failed for %s; skipping in projection",
                    type(strategy).__name__, exc_info=True,
                )
                continue
            running = min(running, est.tokens)
            latency += est.latency_ms
        return Cost(tokens=running, latency_ms=latency)


def default_optimizer_chain(config: OptimizerConfig) -> OptimizerChain:
    """
    Build the canonical 5-strategy chain from configuration.

    Order (cheap → expensive):
      1. StmTrim                 — keep only the last N STM turns
      2. MtmSummarize            — collapse old MTM summaries via an LLM
      3. SemanticDedupe          — cosine-near duplicates removed
      4. LLMLinguaCompress       — token-level compression
      5. ScoreAwareBudgetPrune   — last-resort lowest-score drop

    Strategy classes are imported lazily so the import graph stays small
    when only one (or none) of these is actually wired.
    """
    # Lazy imports: each module may pull in heavy ML deps (sentence-
    # transformers, llmlingua, etc.); avoid loading at framework import.
    from continuum.optimizer.strategies import (
        LLMLinguaCompress,
        MtmSummarize,
        ScoreAwareBudgetPrune,
        SemanticDedupe,
        StmTrim,
    )

    stm_keep = int(getattr(config, "stm_turns", 8))
    summariser_model = str(
        getattr(config, "summarizer_model", "gpt-4o-mini")
    )
    compress_ratio = float(getattr(config, "compress_ratio", 0.5))
    dedupe_threshold = float(getattr(config, "dedupe_threshold", 0.92))

    return OptimizerChain([
        StmTrim(keep_last=stm_keep),
        MtmSummarize(model=summariser_model),
        SemanticDedupe(threshold=dedupe_threshold),
        LLMLinguaCompress(ratio=compress_ratio),
        ScoreAwareBudgetPrune(),
    ])


__all__ = ["OptimizerChain", "default_optimizer_chain"]
