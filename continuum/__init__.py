"""Continuum — AI agent memory framework.

Tiered short / mid / long-term memory with first-class supersession
and bi-temporal queries. See the top-level README for the value-prop
and ``docs/`` for the four-page reference set.

The version constant below is the single source of truth used by
``pyproject.toml`` (via hatchling), the CHANGELOG, and any runtime
introspection (``continuum.__version__``).
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
