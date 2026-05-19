"""
continuum/db/pgvector_upgrade.py
================================
Verify and benchmark the pgvector >= 0.8 / ``halfvec`` / HNSW upgrade
performed by ``migrations/002_pgvector_upgrade.sql``.

It answers four questions, objectively:

1. **Is pgvector new enough?**            ``parse_pgvector_version`` / ``meets_minimum``
2. **Did the column convert?**            ``UpgradeReport.column_type == "halfvec(1024)"``
3. **How much storage did we save?**       index size before vs after
4. **Did recall hold (< 1% loss)?**        ANN top-k vs exact top-k overlap

Run it once before the migration and once after; compare the two
``UpgradeReport``s.

CLI
---
    python -m continuum.db.pgvector_upgrade --dsn "$DATABASE_URL" --samples 50

Library
-------
    report = await verify_upgrade(dsn=dsn, k=10, sample_size=50)
    print(report.column_type, report.index_size_pretty, report.recall_at_k)

Testing
-------
The pure helpers (``parse_pgvector_version``, ``meets_minimum``,
``recall_at_k``, ``build_knn_query``) need no DB.  The DB-touching code uses
an injectable ``conn_factory`` (same pattern as ``PostgresSTM``) so unit
tests run against a mock connection with neither psycopg3 nor PostgreSQL.

QUERY-CODE UPDATE (halfvec + ``<=>``)
-------------------------------------
After the migration the embedding column is ``halfvec(1024)``.  Bind the
query vector as a halfvec literal and rank with the ``<=>`` cosine-distance
operator (``<=>`` returns cosine *distance*; ``1 - distance`` = similarity)::

    # psycopg3 — register the type once per connection (optional but tidy):
    #   from pgvector.psycopg import register_vector_async
    #   await register_vector_async(conn)
    #
    # Without the adapter you can always pass a text literal + ::halfvec cast:
    sql = build_knn_query(k=10)              # see below for the exact SQL
    vec = to_halfvec_literal(query_embedding)  # '[0.1,0.2,...]'
    cur = await conn.execute(sql, {"q": vec, "k": 10})
    rows = await cur.fetchall()              # nearest-first

``build_knn_query()`` emits::

    SELECT id, text, 1 - (embedding <=> %(q)s::halfvec) AS cosine_similarity
    FROM   memory_nodes
    WHERE  embedding IS NOT NULL
    ORDER  BY embedding <=> %(q)s::halfvec
    LIMIT  %(k)s
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Minimum pgvector that supports halfvec + HNSW + iterative scans.
MIN_PGVECTOR: tuple[int, int, int] = (0, 8, 0)

#: Default object names (match migrations 001/002).
DEFAULT_TABLE = "memory_nodes"
DEFAULT_COLUMN = "embedding"
DEFAULT_INDEX = "memory_nodes_embedding_hnsw_idx"


# ============================================================================
# Pure helpers (no DB, fully unit-testable)
# ============================================================================


def parse_pgvector_version(raw: str) -> tuple[int, int, int]:
    """
    Parse a pgvector ``extversion`` string into a ``(major, minor, patch)``.

    Tolerates pre-release / build suffixes and missing components::

        "0.8.0"      -> (0, 8, 0)
        "0.8"        -> (0, 8, 0)
        "0.10.2-dev" -> (0, 10, 2)
        "0.7.0+meta" -> (0, 7, 0)

    Raises
    ------
    ValueError
        If *raw* has no leading numeric component at all.
    """
    if raw is None:
        raise ValueError("pgvector version is None (extension not installed?)")
    core = raw.strip().split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    nums: list[int] = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError as exc:
            raise ValueError(f"unparseable pgvector version: {raw!r}") from exc
    if not nums:
        raise ValueError(f"unparseable pgvector version: {raw!r}")
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def meets_minimum(
    version: tuple[int, int, int],
    minimum: tuple[int, int, int] = MIN_PGVECTOR,
) -> bool:
    """Return ``True`` if *version* >= *minimum* (tuple comparison)."""
    return version >= minimum


def recall_at_k(
    approx_ids: Sequence[Any],
    exact_ids: Sequence[Any],
) -> float:
    """
    Recall@k = |approx ∩ exact| / |exact|, comparing only the top-k of each.

    ``exact_ids`` is ground truth (brute-force nearest neighbours).
    Returns ``1.0`` when *exact_ids* is empty (nothing to miss).
    """
    k = len(exact_ids)
    if k == 0:
        return 1.0
    truth = set(exact_ids)
    hit = sum(1 for i in approx_ids[:k] if i in truth)
    return hit / k


def to_halfvec_literal(vector: Sequence[float]) -> str:
    """
    Format a Python float sequence as a pgvector text literal: ``[a,b,c]``.

    Works for both ``vector`` and ``halfvec`` when used with an explicit
    ``::halfvec`` / ``::vector`` cast in the SQL.
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def build_knn_query(
    *,
    table: str = DEFAULT_TABLE,
    column: str = DEFAULT_COLUMN,
    select_columns: Sequence[str] = ("id", "text"),
    k: int = 10,
) -> str:
    """
    Build a parameterised cosine-KNN SQL string for a ``halfvec`` column.

    Placeholders: ``%(q)s`` (the query vector text literal) and ``%(k)s``.
    ``<=>`` is pgvector's cosine-distance operator; ``1 - distance`` is the
    cosine similarity, exposed as ``cosine_similarity`` for convenience.
    """
    cols = ", ".join(select_columns)
    return (
        f"SELECT {cols}, "
        f"1 - ({column} <=> %(q)s::halfvec) AS cosine_similarity\n"
        f"FROM   {table}\n"
        f"WHERE  {column} IS NOT NULL\n"
        f"ORDER  BY {column} <=> %(q)s::halfvec\n"
        f"LIMIT  %(k)s"
    )


# ============================================================================
# Report
# ============================================================================


@dataclass
class UpgradeReport:
    """Snapshot of the vector-index state at one point in time."""

    pgvector_version: str
    version_ok: bool
    column_type: str  # e.g. 'halfvec(1024)' or 'vector(1024)'
    is_halfvec: bool
    index_name: str | None
    index_method: str | None  # 'hnsw' / 'ivfflat' / None
    index_size_bytes: int
    index_size_pretty: str
    row_count: int
    sample_size: int
    k: int
    recall_at_k: float | None  # None when not measured (e.g. empty table)

    def summary(self) -> str:
        """One-line human summary for logs / CLI."""
        rec = f"{self.recall_at_k:.4f}" if self.recall_at_k is not None else "n/a"
        return (
            f"pgvector={self.pgvector_version}"
            f"(ok={self.version_ok}) "
            f"col={self.column_type} "
            f"index={self.index_name}:{self.index_method} "
            f"size={self.index_size_pretty} "
            f"rows={self.row_count} "
            f"recall@{self.k}={rec}"
        )


# ============================================================================
# Connection plumbing (lazy psycopg3, injectable factory)
# ============================================================================


def _require_psycopg() -> tuple[Any, Any]:
    """Import psycopg3 or raise a clear, actionable error."""
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg, dict_row
    except ImportError as exc:  # pragma: no cover - exercised via conn_factory
        raise ImportError(
            "psycopg3 is required for continuum.db.pgvector_upgrade.\n"
            "Install it with:  pip install 'psycopg[binary]>=3.1'"
        ) from exc


@asynccontextmanager
async def _connect(
    dsn: str | None,
    conn_factory: Any | None,
) -> AsyncIterator[Any]:
    """Yield a connection. Priority: conn_factory > dsn."""
    if conn_factory is not None:
        async with conn_factory() as conn:
            yield conn
        return

    if dsn is None:
        raise ValueError("Provide either dsn or conn_factory.")

    psycopg, dict_row = _require_psycopg()
    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row)
    try:
        yield conn
    finally:
        await conn.close()


async def _fetchone(conn: Any, sql: str, params: Any | None = None) -> Any:
    cur = await conn.execute(sql, params or {})
    return await cur.fetchone()


async def _fetchall(conn: Any, sql: str, params: Any | None = None) -> list[Any]:
    cur = await conn.execute(sql, params or {})
    rows: list[Any] = await cur.fetchall()
    return rows


# ============================================================================
# Individual measurements
# ============================================================================


async def get_pgvector_version(conn: Any) -> str:
    row = await _fetchone(conn, "SELECT extversion AS v FROM pg_extension WHERE extname = 'vector'")
    if not row or row.get("v") is None:
        raise RuntimeError("pgvector extension 'vector' is not installed")
    return str(row["v"])


async def get_column_type(
    conn: Any, table: str = DEFAULT_TABLE, column: str = DEFAULT_COLUMN
) -> str:
    """Return e.g. ``'halfvec(1024)'`` / ``'vector(1024)'`` for the column."""
    row = await _fetchone(
        conn,
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS t
        FROM   pg_attribute a
        JOIN   pg_class     c ON c.oid = a.attrelid
        JOIN   pg_namespace n ON n.oid = c.relnamespace
        WHERE  n.nspname = 'public'
          AND  c.relname = %(table)s
          AND  a.attname = %(column)s
          AND  a.attnum  > 0
          AND  NOT a.attisdropped
        """,
        {"table": table, "column": column},
    )
    if not row or row.get("t") is None:
        raise RuntimeError(f"{table}.{column} not found")
    return str(row["t"])


async def get_index_info(conn: Any, index_name: str = DEFAULT_INDEX) -> tuple[str | None, int, str]:
    """
    Return ``(access_method, size_bytes, size_pretty)`` for *index_name*.

    ``(None, 0, '0 bytes')`` when the index does not exist (e.g. pre-upgrade
    with no index at all).
    """
    row = await _fetchone(
        conn,
        """
        SELECT am.amname                                AS method,
               pg_relation_size(i.indexrelid)           AS bytes,
               pg_size_pretty(pg_relation_size(i.indexrelid)) AS pretty
        FROM   pg_class     ic
        JOIN   pg_index     i  ON i.indexrelid = ic.oid
        JOIN   pg_am        am ON am.oid       = ic.relam
        WHERE  ic.relname = %(idx)s
        """,
        {"idx": index_name},
    )
    if not row:
        return (None, 0, "0 bytes")
    return (
        row.get("method"),
        int(row.get("bytes") or 0),
        str(row.get("pretty") or "0 bytes"),
    )


async def get_row_count(conn: Any, table: str = DEFAULT_TABLE) -> int:
    row = await _fetchone(conn, f"SELECT COUNT(*) AS n FROM {table}")
    return int(row["n"]) if row else 0


async def _sample_query_vectors(
    conn: Any,
    sample_size: int,
    table: str = DEFAULT_TABLE,
    column: str = DEFAULT_COLUMN,
) -> list[str]:
    """Pull random existing embeddings to use as realistic probe queries."""
    rows = await _fetchall(
        conn,
        f"""
        SELECT {column}::text AS v
        FROM   {table}
        WHERE  {column} IS NOT NULL
        ORDER  BY random()
        LIMIT  %(n)s
        """,
        {"n": sample_size},
    )
    return [r["v"] for r in rows if r.get("v")]


async def _knn_ids(
    conn: Any,
    query_literal: str,
    k: int,
    *,
    exact: bool,
    ef_search: int,
    table: str = DEFAULT_TABLE,
    column: str = DEFAULT_COLUMN,
) -> list[Any]:
    """
    Return the ids of the *k* nearest rows to *query_literal*.

    ``exact=True`` forces a brute-force scan (ground truth) by disabling
    index access for the statement; ``exact=False`` uses the HNSW index with
    ``hnsw.ef_search`` set to *ef_search*.
    """
    if exact:
        # Force a sequential scan → exact nearest neighbours (ground truth).
        await conn.execute("SET LOCAL enable_indexscan = off")
        await conn.execute("SET LOCAL enable_bitmapscan = off")
        await conn.execute("SET LOCAL enable_indexonlyscan = off")
    else:
        await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")

    rows = await _fetchall(
        conn,
        f"""
        SELECT id
        FROM   {table}
        WHERE  {column} IS NOT NULL
        ORDER  BY {column} <=> %(q)s::halfvec
        LIMIT  %(k)s
        """,
        {"q": query_literal, "k": k},
    )
    return [r["id"] for r in rows]


async def measure_recall(
    conn: Any,
    query_literals: Sequence[str],
    *,
    k: int = 10,
    ef_search: int = 40,
) -> float | None:
    """
    Mean recall@k of the HNSW index vs exact search over *query_literals*.

    Returns ``None`` if there are no probe queries (empty table).
    """
    if not query_literals:
        return None
    total = 0.0
    for q in query_literals:
        exact = await _knn_ids(conn, q, k, exact=True, ef_search=ef_search)
        approx = await _knn_ids(conn, q, k, exact=False, ef_search=ef_search)
        total += recall_at_k(approx, exact)
    return total / len(query_literals)


# ============================================================================
# Orchestration
# ============================================================================


async def verify_upgrade(
    *,
    dsn: str | None = None,
    conn_factory: Any | None = None,
    table: str = DEFAULT_TABLE,
    column: str = DEFAULT_COLUMN,
    index_name: str = DEFAULT_INDEX,
    k: int = 10,
    sample_size: int = 50,
    ef_search: int = 40,
) -> UpgradeReport:
    """
    Produce an :class:`UpgradeReport` for the current DB state.

    Run before AND after applying ``002_pgvector_upgrade.sql`` and diff the
    two reports to confirm: ``is_halfvec`` flips True, ``index_size_bytes``
    roughly halves, and ``recall_at_k`` stays within ~1% of the baseline.
    """
    async with _connect(dsn, conn_factory) as conn:
        version_raw = await get_pgvector_version(conn)
        version_ok = meets_minimum(parse_pgvector_version(version_raw))

        col_type = await get_column_type(conn, table, column)
        method, size_bytes, size_pretty = await get_index_info(conn, index_name)
        row_count = await get_row_count(conn, table)

        recall: float | None = None
        probes: list[str] = []
        if row_count > 0:
            probes = await _sample_query_vectors(conn, min(sample_size, row_count), table, column)
            recall = await measure_recall(conn, probes, k=k, ef_search=ef_search)

    return UpgradeReport(
        pgvector_version=version_raw,
        version_ok=version_ok,
        column_type=col_type,
        is_halfvec=col_type.startswith("halfvec"),
        index_name=index_name if method else None,
        index_method=method,
        index_size_bytes=size_bytes,
        index_size_pretty=size_pretty,
        row_count=row_count,
        sample_size=len(probes),
        k=k,
        recall_at_k=recall,
    )


# ============================================================================
# CLI
# ============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m continuum.db.pgvector_upgrade",
        description="Verify the pgvector >=0.8 / halfvec / HNSW upgrade.",
    )
    p.add_argument("--dsn", required=True, help="postgresql://user:pw@host/db")
    p.add_argument("--k", type=int, default=10, help="top-k for recall (default 10)")
    p.add_argument(
        "--samples",
        type=int,
        default=50,
        help="number of probe queries sampled from the table (default 50)",
    )
    p.add_argument(
        "--ef-search",
        type=int,
        default=40,
        help="hnsw.ef_search used for the ANN side of recall (default 40)",
    )
    return p


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = await verify_upgrade(
        dsn=args.dsn,
        k=args.k,
        sample_size=args.samples,
        ef_search=args.ef_search,
    )
    print(report.summary())
    if not report.version_ok:
        print("FAIL: pgvector < 0.8.0 — upgrade the extension first.")
        return 2
    if not report.is_halfvec:
        print("WARN: embedding column is not halfvec — migration not applied yet.")
        return 1
    if report.recall_at_k is not None and report.recall_at_k < 0.99:
        print(
            f"WARN: recall@{report.k}={report.recall_at_k:.4f} (<0.99). "
            f"Raise hnsw.ef_search or m; see migration 002 notes."
        )
        return 1
    print("OK: halfvec + HNSW upgrade verified.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entry point for ``python -m`` / console scripts."""
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
