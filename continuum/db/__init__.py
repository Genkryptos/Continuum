"""
continuum.db
============
Database operational tooling (migration verification, index inspection).

Exports
-------
verify_upgrade        — async: confirm the pgvector >=0.8 / halfvec / HNSW
                        upgrade succeeded and measure size + recall.
UpgradeReport         — dataclass returned by ``verify_upgrade``.
parse_pgvector_version / meets_minimum / recall_at_k / build_knn_query
                      — pure helpers, usable without a DB connection.

psycopg3 is imported lazily, so importing this package is cheap and the
pure helpers can be unit-tested without psycopg installed.
"""

from __future__ import annotations

from continuum.db.pgvector_upgrade import (
    MIN_PGVECTOR,
    UpgradeReport,
    build_knn_query,
    meets_minimum,
    parse_pgvector_version,
    recall_at_k,
    verify_upgrade,
)

__all__ = [
    "verify_upgrade",
    "UpgradeReport",
    "parse_pgvector_version",
    "meets_minimum",
    "recall_at_k",
    "build_knn_query",
    "MIN_PGVECTOR",
]
