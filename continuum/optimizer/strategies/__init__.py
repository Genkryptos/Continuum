"""
continuum.optimizer.strategies
==============================
Concrete :class:`continuum.optimizer.Optimizer` implementations.

Currently exports
-----------------
StmTrim        — keep the last N STM turns verbatim, summarise the rest
                  into a single MTM row.
MtmSummarize   — collapse older MTM blocks via extractive or LLM-based
                  summarisation; keeps the most-recent N verbatim.
SemanticDedupe — remove near-duplicate LTM rows by cosine similarity;
                  keeps the highest-scored row from each duplicate pair.
LLMLinguaCompress — token-level prompt compression via LLMLingua-2;
                  caches results and applies a cosine quality gate.
ScoreAwareBudgetPrune — last-resort lossy strategy: drops the lowest-
                  value items (LTM → MTM → optionally STM) until under
                  budget.

The full :func:`continuum.optimizer.chain.default_optimizer_chain` is
now satisfied.
"""
from __future__ import annotations

from continuum.optimizer.strategies.llmlingua import (
    DEFAULT_MODEL,
    LLMLinguaCompress,
    LLMLinguaConfig,
)
from continuum.optimizer.strategies.mtm_summarize import (
    MTM_SYSTEM_PROMPT,
    MtmSummarize,
    MtmSummarizeConfig,
    extractive_summary,
)
from continuum.optimizer.strategies.score_prune import (
    ScoreAwareBudgetPrune,
    ScorePruneConfig,
)
from continuum.optimizer.strategies.semantic_dedupe import (
    SemanticDedupe,
    SemanticDedupeConfig,
    cosine_similarity_matrix,
)
from continuum.optimizer.strategies.stm_trim import StmTrim, StmTrimConfig

__all__ = [
    "StmTrim",
    "StmTrimConfig",
    "MtmSummarize",
    "MtmSummarizeConfig",
    "MTM_SYSTEM_PROMPT",
    "extractive_summary",
    "SemanticDedupe",
    "SemanticDedupeConfig",
    "cosine_similarity_matrix",
    "LLMLinguaCompress",
    "LLMLinguaConfig",
    "DEFAULT_MODEL",
    "ScoreAwareBudgetPrune",
    "ScorePruneConfig",
]
