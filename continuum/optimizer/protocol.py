"""
continuum/optimizer/protocol.py
===============================
The Optimizer Protocol and its companion ``Cost`` estimate.

Every concrete strategy (``StmTrim``, ``MtmSummarize``, ``SemanticDedupe``,
``LLMLinguaCompress``, ``ScoreAwareBudgetPrune``, …) implements this
two-method surface. The Protocol is ``@runtime_checkable`` so callers can
``isinstance(obj, Optimizer)`` without pulling in concrete dependencies.

Design notes
------------
* ``apply`` is **async** because real strategies may call an LLM
  (summariser, LLMLingua, reranker). In-memory strategies just await
  nothing.
* ``cost_estimate`` is **sync** and cheap — it must never call an LLM.
  Used by :class:`OptimizerChain` for budget short-circuiting and by
  callers that want to display "before applying" projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from continuum.core.types import ContextBundle, TokenBudget


@dataclass(frozen=True)
class Cost:
    """
    Static, before-run projection of running an optimizer on a bundle.

    ``tokens``  – expected ``ContextBundle.tokens_used`` after ``apply``.
    ``latency_ms`` – wall-clock estimate. ``0.0`` for in-memory strategies;
                     LLM-backed strategies should set a realistic upper
                     bound so callers can reason about throughput budgets.
    """

    tokens: int
    latency_ms: float = 0.0


@runtime_checkable
class Optimizer(Protocol):
    """The contract every optimisation strategy fulfils."""

    async def apply(self, ctx: ContextBundle, budget: TokenBudget) -> ContextBundle:
        """Apply optimisation to reduce *ctx* under *budget*."""
        ...

    def cost_estimate(self, ctx: ContextBundle) -> Cost:
        """Estimate post-apply token count + latency without running."""
        ...


__all__ = ["Optimizer", "Cost"]
