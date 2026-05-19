"""
continuum.stores.postgres
=========================
PostgreSQL-backed retrieval primitives.

Exports
-------
PostgresRetriever          — dense (halfvec) + lexical (pg_trgm) + hybrid (RRF).
reciprocal_rank_fusion     — pure RRF helper (no DB), unit-testable.
FusionResult               — per-id RRF outcome with component ranks.
PostgresMTM                — mid-term memory over memory_nodes (layer='MTM').
PostgresLTM                — bi-temporal long-term memory + knowledge graph.
count_tokens               — tiktoken cl100k_base counter (lazy, cached).

psycopg3 is imported lazily inside each store, so importing this package is
cheap and the pure helpers work without psycopg installed.
"""

from __future__ import annotations

from continuum.stores.postgres.ltm import PostgresLTM
from continuum.stores.postgres.mtm import PostgresMTM, count_tokens
from continuum.stores.postgres.retrieval import (
    FusionResult,
    PostgresRetriever,
    reciprocal_rank_fusion,
)

__all__ = [
    "PostgresRetriever",
    "reciprocal_rank_fusion",
    "FusionResult",
    "PostgresMTM",
    "PostgresLTM",
    "count_tokens",
]
