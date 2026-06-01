"""
continuum.stores.in_memory
==========================
Process-local Continuum tier stores. No external infrastructure
(Postgres, pgvector, pg_trgm) required — handy for benchmarks,
unit tests, and ephemeral eval rigs.

The in-memory implementations are intentionally protocol-compliant
with their Postgres siblings (``continuum.stores.postgres.*``) so a
caller can swap backends by changing one constructor.
"""

from __future__ import annotations

from continuum.stores.in_memory.ltm import InMemoryLTM

__all__ = ["InMemoryLTM"]
