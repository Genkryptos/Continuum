#!/usr/bin/env python3
"""
scripts/scale_test.py
=====================
How does Continuum's retrieval hold up past a handful of memories? Everything so
far was measured at ~62 rows; real use is thousands. This loads a throwaway
namespace to 1k / 10k / 50k rows and measures recall latency and index usage at
each size.

It does NOT embed through the model (that is ~85ms/row = 14min at 10k). Instead
it inserts random 1024-dim unit vectors directly, which is a fair stress of the
pgvector HNSW + pg_trgm QUERY path — the thing that actually scales. Semantic
*quality* at scale needs real embeddings and is a separate, smaller test.

    python3 scripts/scale_test.py --dsn postgresql://…/continuum_scale
    make scale-test SCALE_DSN=postgresql://…/continuum_scale

NEVER touches CONTINUUM_DB_DSN — it writes 10k+ rows. A throwaway DSN is required.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from typing import Any

_WORDS = [
    "postgres",
    "redis",
    "kafka",
    "deploy",
    "migration",
    "billing",
    "frontend",
    "backend",
    "latency",
    "index",
    "vector",
    "embedding",
    "recall",
    "memory",
    "session",
    "tenant",
    "namespace",
    "pruning",
    "schema",
    "query",
    "pipeline",
    "cache",
    "token",
    "budget",
    "scorer",
    "retriever",
    "sparse",
    "dense",
    "hybrid",
    "cluster",
    "shard",
    "pricing",
    "launch",
    "sprint",
    "incident",
    "vendor",
    "limit",
    "retention",
    "export",
    "admin",
    "onboarding",
]


def _sentence(rng: Any) -> str:
    n = rng.integers(6, 14)
    return " ".join(rng.choice(_WORDS, size=n)) + "."


def _unit_vec(rng: Any) -> list[float]:
    import numpy as np

    v = rng.standard_normal(1024)
    return (v / np.linalg.norm(v)).tolist()


def _bulk_insert(cur: Any, ns: str, n: int, rng: Any) -> None:
    from continuum.db.pgvector_upgrade import to_halfvec_literal

    batch = 500
    rows: list[tuple[Any, ...]] = []
    for _ in range(n):
        rows.append(
            (str(uuid.uuid4()), "LTM", _sentence(rng), to_halfvec_literal(_unit_vec(rng)), ns)
        )
        if len(rows) >= batch:
            cur.executemany(
                'INSERT INTO memory_nodes (id, layer, "text", embedding, namespace,'
                " created_at, updated_at) VALUES (%s, %s, %s, %s::halfvec, %s, now(), now())",
                rows,
            )
            rows.clear()
    if rows:
        cur.executemany(
            'INSERT INTO memory_nodes (id, layer, "text", embedding, namespace,'
            " created_at, updated_at) VALUES (%s, %s, %s, %s::halfvec, %s, now(), now())",
            rows,
        )


def _measure(dsn: str, ns: str, rng: Any, queries: int) -> tuple[float, float, str]:
    import asyncio

    from continuum.core.types import MemoryTier, Query
    from continuum.stores.postgres.ltm import PostgresLTM

    ltm = PostgresLTM(dsn=dsn, namespace=ns)

    async def run() -> tuple[list[float], str]:
        lat: list[float] = []
        for _ in range(queries):
            q = Query(
                text=rng.choice(_WORDS),
                embedding=_unit_vec(rng),
                top_k=8,
                tiers=[MemoryTier.LTM],
            )
            t0 = time.perf_counter()
            await ltm.search_hybrid(q, 8)
            lat.append((time.perf_counter() - t0) * 1000)
        # index usage: EXPLAIN one dense query
        import psycopg

        from continuum.db.pgvector_upgrade import to_halfvec_literal

        conn = await psycopg.AsyncConnection.connect(dsn)
        async with conn.cursor() as cur:
            await cur.execute(
                "EXPLAIN SELECT id FROM memory_nodes WHERE namespace=%s AND invalidated_at IS NULL "
                "ORDER BY embedding <=> %s::halfvec LIMIT 8",
                (ns, to_halfvec_literal(_unit_vec(rng))),
            )
            plan = " ".join(r[0] for r in await cur.fetchall())
        await conn.close()
        await ltm.aclose()
        idx = (
            "hnsw"
            if "hnsw" in plan.lower()
            else ("seq scan" if "seq scan" in plan.lower() else "?")
        )
        return lat, idx

    lat, idx = asyncio.run(run())
    lat.sort()
    return statistics.median(lat), lat[int(0.95 * len(lat)) - 1], idx


def main(argv: list[str]) -> int:
    import os

    p = argparse.ArgumentParser(prog="scale-test", description="Continuum retrieval at scale")
    p.add_argument("--dsn", default=os.environ.get("SCALE_DSN"), help="THROWAWAY Postgres DSN")
    p.add_argument("--sizes", default="1000,10000,50000", help="cumulative row counts")
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.dsn:
        print("scale-test: --dsn (or SCALE_DSN) required — this writes 10k+ rows.")
        return 2
    if args.dsn == os.environ.get("CONTINUUM_DB_DSN"):
        print("scale-test: refusing to run against CONTINUUM_DB_DSN — use a throwaway DB.")
        return 2

    import numpy as np
    import psycopg

    rng = np.random.default_rng(args.seed)
    ns = f"scale_{uuid.uuid4().hex[:8]}"
    sizes = [int(s) for s in args.sizes.split(",")]

    print(
        f"[scale] dsn={args.dsn.rsplit('/', 1)[-1]}  namespace={ns}  queries/size={args.queries}\n"
    )
    print(f"{'rows':>8s} {'insert':>9s} {'recall p50':>11s} {'recall p95':>11s} {'index':>9s}")
    print("-" * 52)
    conn = psycopg.connect(args.dsn, autocommit=True)
    try:
        prev = 0
        for target in sizes:
            add = target - prev
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                _bulk_insert(cur, ns, add, rng)
            ins = time.perf_counter() - t0
            prev = target
            p50, p95, idx = _measure(args.dsn, ns, rng, args.queries)
            print(f"{target:8d} {ins:7.1f}s {p50:9.1f}ms {p95:9.1f}ms {idx:>9s}")
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE namespace = %s", (ns,))
        conn.close()
    print("\n[scale] namespace cleaned up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
