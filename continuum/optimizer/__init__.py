"""
continuum.optimizer
===================
Token-budget optimisation: a small Protocol + composable strategy chain.

Public surface
--------------
Optimizer            — runtime-checkable Protocol every strategy implements.
Cost                 — (tokens, latency_ms) estimate returned by cost_estimate.
OptimizerChain       — applies strategies in order with short-circuit on budget.
BaseOptimizer        — convenience ABC with a sensible default cost_estimate.
estimate_tokens      — count tokens in a ContextBundle (tiktoken with fallback).
estimate_tokens_text — count tokens in a free-form string.
default_optimizer_chain — build the canonical strategy stack from OptimizerConfig.

Strategy implementations (StmTrim, MtmSummarize, SemanticDedupe,
LLMLinguaCompress, ScoreAwareBudgetPrune) are imported lazily by
``default_optimizer_chain`` — this package imports cheaply even before
they exist.
"""
from __future__ import annotations

from continuum.optimizer.base import (
    BaseOptimizer,
    estimate_tokens,
    estimate_tokens_text,
)
from continuum.optimizer.chain import OptimizerChain, default_optimizer_chain
from continuum.optimizer.protocol import Cost, Optimizer

__all__ = [
    "Optimizer",
    "Cost",
    "OptimizerChain",
    "BaseOptimizer",
    "estimate_tokens",
    "estimate_tokens_text",
    "default_optimizer_chain",
]
