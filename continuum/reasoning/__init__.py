"""
continuum.reasoning
===================
Reasoning-time orchestration. Currently exports:

* :class:`IterativeReasoner` — a budget-capped loop that runs intent
  classification, decomposition, per-sub-question retrieval +
  verification (with a refine round when verification fails), and
  composer synthesis. Designed to be agnostic of the LongMemEval
  rig: every domain-specific step is injected as a callable.
"""

from __future__ import annotations

from continuum.reasoning.iterative_reasoner import (
    IterativeReasoner,
    ReasoningResult,
)

__all__ = ["IterativeReasoner", "ReasoningResult"]
