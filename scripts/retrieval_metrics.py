#!/usr/bin/env python3
"""
scripts/retrieval_metrics.py
============================
Retrieval-only quality metrics — Recall@k, MRR, NDCG@k — decoupled from any
answerer, reported for two systems side by side:

  * **hybrid**  — Continuum's shipped pipeline (dense bge-m3 + BM25, fused with
    RRF), via ``Memory.recall``.
  * **cosine**  — a plain pgvector top-k cosine scan (dense only, no lexical
    channel, no fusion): the "just a vector DB" baseline the paper compares to.

Both are scored over the same needle set (single relevant document per query)
by recording the **rank** of the gold fact, so the metrics describe retrieval
quality alone — not whether an LLM then reasoned correctly over it.

    CONTINUUM_DB_DSN=postgresql://…/throwaway python3 scripts/retrieval_metrics.py \\
        --sizes 3000 25000 --k 20

Needs a throwaway migrated database — it writes each size into its own namespace.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import math
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ras", _HERE / "recall_at_scale.py")
assert _spec and _spec.loader
ras = importlib.util.module_from_spec(_spec)
sys.modules["ras"] = ras
_spec.loader.exec_module(ras)


async def _build(dsn: str, rows: int, namespace: str) -> None:
    """Load `rows` distractors + the needles into `namespace` (idempotent)."""
    import psycopg

    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memory_nodes WHERE namespace = %s", (namespace,)
        )
        row = cur.fetchone()
        if row and row[0] >= rows:
            return  # already built
        cur.execute("DELETE FROM memory_nodes WHERE namespace = %s", (namespace,))
        c.commit()
    await ras.load(dsn, rows, "realistic", namespace)


async def _hybrid_ranks(dsn: str, namespace: str, depth: int) -> list[int | None]:
    """Gold-fact rank per needle through the real product path (Memory.recall)."""
    from continuum.memory import Memory

    mem = Memory.from_postgres(dsn, embeddings=True, namespace=namespace)
    await mem.start()
    try:
        await mem.recall("warm up", k=1)
        ranks: list[int | None] = []
        for fact, query in ras.NEEDLES:
            found = await mem.recall(query, k=depth)
            ranks.append(_rank_of(fact, [(h.content or "") for h in found]))
        return ranks
    finally:
        await mem.aclose()


def _cosine_ranks(dsn: str, namespace: str, depth: int) -> list[int | None]:
    """Gold-fact rank per needle through a plain pgvector cosine top-k scan
    (dense only — no BM25, no RRF): the vector-DB baseline."""
    import psycopg

    from continuum.core.config import ContinuumConfig
    from continuum.db.pgvector_upgrade import to_halfvec_literal
    from continuum.embeddings import EmbeddingService

    async def _embed() -> list[str]:
        e = EmbeddingService(ContinuumConfig.load().embedding)
        return [to_halfvec_literal(v) for v in await e.embed([q for _f, q in ras.NEEDLES])]

    qvecs = asyncio.run(_embed())
    sql = (
        'SELECT "text" FROM memory_nodes '
        "WHERE invalidated_at IS NULL AND namespace = %s AND embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::halfvec LIMIT %s"
    )
    ranks: list[int | None] = []
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 1000")  # give the index its best shot
        for (fact, _q), qv in zip(ras.NEEDLES, qvecs, strict=True):
            cur.execute(sql, (namespace, qv, depth))
            ranks.append(_rank_of(fact, [r[0] for r in cur.fetchall()]))
    return ranks


def _rank_of(fact: str, texts: list[str]) -> int | None:
    gold = fact.strip()
    for i, t in enumerate(texts, start=1):
        if (t or "").strip() == gold:
            return i
    return None


def _metrics(ranks: list[int | None], cutoffs: list[int]) -> dict[str, float]:
    """Single-relevant-doc IR metrics from a list of gold ranks (None = miss)."""
    n = len(ranks)
    out: dict[str, float] = {}
    for k in cutoffs:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    out["mrr"] = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    kmax = max(cutoffs)  # NDCG@kmax; one relevant doc → IDCG = 1
    out[f"ndcg@{kmax}"] = (
        sum((1.0 / math.log2(r + 1)) if (r and r <= kmax) else 0.0 for r in ranks) / n
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[3000])
    p.add_argument("--k", type=int, default=20, help="retrieval depth (largest cutoff)")
    args = p.parse_args(argv)

    dsn = os.environ.get("CONTINUUM_DB_DSN")
    if not dsn:
        print("[metrics] set CONTINUUM_DB_DSN to a THROWAWAY migrated db.", file=sys.stderr)
        return 1

    cutoffs = sorted({c for c in (1, 5, 10, 20, args.k) if c <= args.k})
    total = len(ras.NEEDLES)
    print(
        f"  retrieval-only metrics · hybrid (Memory.recall) vs pgvector-cosine · "
        f"{total} needles · depth={args.k}\n"
    )
    header = f"  {'rows':>7} {'system':>8}" + "".join(f"{f'R@{c}':>8}" for c in cutoffs)
    header += f"{'MRR':>8}{f'NDCG@{args.k}':>9}"
    print(header)
    for size in args.sizes:
        ns = f"m{size}"
        asyncio.run(_build(dsn, size, ns))
        for label, ranks in (
            ("hybrid", asyncio.run(_hybrid_ranks(dsn, ns, args.k))),
            ("cosine", _cosine_ranks(dsn, ns, args.k)),
        ):
            m = _metrics(ranks, cutoffs)
            row = f"  {size:>7} {label:>8}" + "".join(
                f"{m[f'recall@{c}']:>8.3f}" for c in cutoffs
            )
            row += f"{m['mrr']:>8.3f}{m[f'ndcg@{args.k}']:>9.3f}"
            print(row)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
