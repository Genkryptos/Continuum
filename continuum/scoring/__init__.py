"""
continuum.scoring
=================
The composite memory-scoring formula.

Exports
-------
Scorer  — ``ScorerProtocol`` implementation: weighted relevance / importance
          / reinforcement-aware recency / confidence, with a per-tier
          multiplicative boost.

Dependency-free (stdlib + core types); safe to import anywhere.
"""
from __future__ import annotations

from continuum.scoring.scorer import Scorer

__all__ = ["Scorer"]
