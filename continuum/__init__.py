"""Continuum — AI agent memory framework.

Tiered short / mid / long-term memory with first-class supersession
and bi-temporal queries. See the top-level README for the value-prop
and ``docs/`` for the four-page reference set.

The version constant below is the single source of truth used by
``pyproject.toml`` (via hatchling), the CHANGELOG, and any runtime
introspection (``continuum.__version__``).
"""

from __future__ import annotations

__version__ = "2.0.0"

# Public API. Heavy submodules stay lazy — importing `continuum` must be cheap
# and must not pull in optional backends (Postgres, embedders) until used.
from continuum.core.types import MemoryItem, MemoryTier, Query
from continuum.memory import Memory

__all__ = [
    "Memory",
    "MemoryItem",
    "MemoryTier",
    "Query",
    "__version__",
]
