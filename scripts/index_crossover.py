#!/usr/bin/env python3
"""
scripts/index_crossover.py
==========================
Find where the HNSW index stops costing recall and starts earning its latency.

At 45k the approximate index loses a quarter of retrievable memories (65% vs an
exact scan's 90%) to save 25ms — so exact scan is the better default *there*. But
a sequential scan is O(rows): eventually it is too slow and the index has to win.
This sweeps store sizes and measures both recall and latency for each, so the
crossover is a number rather than a guess.

Self-contained on purpose: earlier attempts orchestrated this across shell jobs
that contended on the embedder and died half-built. One process, one store per
size, results to stdout.

    CONTINUUM_DB_DSN=postgresql://…/throwaway python3 scripts/index_crossover.py \\
        --sizes 3000 10000 25000 45000

Needs a throwaway migrated database — it writes each size into its own namespace.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import pathlib
import statistics
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ras", _HERE / "recall_at_scale.py")
assert _spec and _spec.loader
ras = importlib.util.module_from_spec(_spec)
sys.modules["ras"] = ras
_spec.loader.exec_module(ras)


async def _build(dsn: str, rows: int, namespace: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_nodes WHERE namespace = %s", (namespace,))
        row = cur.fetchone()
        if row and row[0] >= rows:
            return  # already built
        cur.execute("DELETE FROM memory_nodes WHERE namespace = %s", (namespace,))
        c.commit()
    await ras.load(dsn, rows, "realistic", namespace)


def _measure(dsn: str, namespace: str, k: int, use_index: bool, reps: int) -> tuple[int, float]:
    """Recall over the 20 needles + median query latency, dense channel only.

    `use_index=False` forces a sequential (exact) scan by disabling index scans
    on the connection — the same lever the retriever would pull, measured
    directly so the recall/latency trade is not confounded by the sparse channel
    or RRF.
    """
    import psycopg

    from continuum.core.config import ContinuumConfig
    from continuum.db.pgvector_upgrade import to_halfvec_literal
    from continuum.embeddings import EmbeddingService

    async def _embed_queries() -> list[str]:
        e = EmbeddingService(ContinuumConfig.load().embedding)
        return [to_halfvec_literal(v) for v in await e.embed([q for _f, q in ras.NEEDLES])]

    qvecs = asyncio.run(_embed_queries())
    sql = (
        'SELECT "text" FROM memory_nodes '
        "WHERE invalidated_at IS NULL AND namespace = %s AND embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::halfvec LIMIT %s"
    )
    hits = 0
    latencies: list[float] = []
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 1000")  # give the index its best shot
        cur.execute(f"SET enable_indexscan = {'on' if use_index else 'off'}")
        cur.execute(f"SET enable_bitmapscan = {'on' if use_index else 'off'}")
        for (fact, _q), qv in zip(ras.NEEDLES, qvecs, strict=True):
            best = None
            for _ in range(reps):
                t = time.perf_counter()
                cur.execute(sql, (namespace, qv, k))
                rows = [r[0].strip() for r in cur.fetchall()]
                best = (
                    min(best, (time.perf_counter() - t) * 1000)
                    if best
                    else (time.perf_counter() - t) * 1000
                )
            latencies.append(best or 0.0)
            if fact.strip() in rows:
                hits += 1
    return hits, statistics.median(latencies)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[3000, 10000, 25000, 45000])
    p.add_argument("--k", type=int, default=24, help="candidate depth, matching the retriever")
    p.add_argument("--reps", type=int, default=3, help="best-of-N per query, to de-noise latency")
    args = p.parse_args(argv)

    dsn = os.environ.get("CONTINUUM_DB_DSN")
    if not dsn:
        print("[crossover] set CONTINUUM_DB_DSN to a THROWAWAY migrated db.", file=sys.stderr)
        return 1

    total = len(ras.NEEDLES)
    print(f"  dense channel only, top-{args.k}, {total} needles, best-of-{args.reps} latency\n")
    print(
        f"  {'rows':>8} {'exact recall':>14} {'exact p50':>11} {'index recall':>14} {'index p50':>11}"
    )
    for size in args.sizes:
        ns = f"x{size}"
        asyncio.run(_build(dsn, size, ns))
        ex_hits, ex_lat = _measure(dsn, ns, args.k, use_index=False, reps=args.reps)
        ix_hits, ix_lat = _measure(dsn, ns, args.k, use_index=True, reps=args.reps)
        flag = "  <- index loses recall" if ix_hits < ex_hits else ""
        slow = "  <- exact slower than index" if ex_lat > ix_lat * 1.5 else ""
        print(
            f"  {size:>8} {f'{ex_hits}/{total}':>14} {ex_lat:>9.1f}ms {f'{ix_hits}/{total}':>14}"
            f" {ix_lat:>9.1f}ms{flag}{slow}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
