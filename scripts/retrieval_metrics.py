#!/usr/bin/env python3
"""
scripts/retrieval_metrics.py
============================
Retrieval-only quality metrics — Recall@k, MRR, NDCG@k — through Continuum's
**real** hybrid pipeline (``Memory.recall``), decoupled from any answerer.

Every headline number Continuum has reported so far is *end-to-end* (retrieval
+ an LLM answerer + a judge), which confounds "did we retrieve the right
memory" with "did the model reason correctly over it". A research claim about
the retriever needs the retrieval axis on its own. This harness provides it:
for each needle it records the **rank** of the gold fact in the returned list,
then derives the standard IR metrics.

It reuses the needle set and bulk loader from ``recall_at_scale.py`` (single
relevant document per query), and scores through ``Memory.recall`` exactly as
``recall_at_scale.score`` does — so the metrics describe the shipped hybrid
(dense bge-m3 + BM25, fused with RRF), not a re-implementation.

    CONTINUUM_DB_DSN=postgresql://…/throwaway python3 scripts/retrieval_metrics.py \\
        --sizes 3000 --k 20

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


async def _ranks(dsn: str, namespace: str, depth: int) -> list[int | None]:
    """For each needle, the 1-indexed rank of the gold fact in recall(depth),
    or None if it never appears in the top-`depth`. Uses the real product path."""
    from continuum.memory import Memory

    mem = Memory.from_postgres(dsn, embeddings=True, namespace=namespace)
    await mem.start()
    try:
        await mem.recall("warm up", k=1)
        ranks: list[int | None] = []
        for fact, query in ras.NEEDLES:
            found = await mem.recall(query, k=depth)
            gold = fact.strip()
            rank: int | None = None
            for i, h in enumerate(found, start=1):
                if (h.content or "").strip() == gold:
                    rank = i
                    break
            ranks.append(rank)
        return ranks
    finally:
        await mem.aclose()


def _metrics(ranks: list[int | None], cutoffs: list[int]) -> dict[str, float]:
    """Single-relevant-doc IR metrics from a list of gold ranks (None = miss)."""
    n = len(ranks)
    out: dict[str, float] = {}
    for k in cutoffs:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    # MRR over the full retrieved depth (0 contribution for misses)
    out["mrr"] = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    # NDCG@k: one relevant doc → IDCG = 1, so NDCG = 1/log2(rank+1) if rank<=k
    kmax = max(cutoffs)
    out[f"ndcg@{kmax}"] = (
        sum((1.0 / math.log2(r + 1)) if (r and r <= kmax) else 0.0 for r in ranks) / n
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[3000])
    p.add_argument(
        "--k", type=int, default=20, help="retrieval depth (largest cutoff scored)"
    )
    args = p.parse_args(argv)

    dsn = os.environ.get("CONTINUUM_DB_DSN")
    if not dsn:
        print("[metrics] set CONTINUUM_DB_DSN to a THROWAWAY migrated db.", file=sys.stderr)
        return 1

    cutoffs = sorted({c for c in (1, 5, 10, 20, args.k) if c <= args.k})
    total = len(ras.NEEDLES)
    print(
        f"  retrieval-only metrics · real hybrid pipeline (Memory.recall) · "
        f"{total} needles · depth={args.k}\n"
    )
    header = f"  {'rows':>8}" + "".join(f"{f'R@{c}':>9}" for c in cutoffs)
    header += f"{'MRR':>9}{f'NDCG@{args.k}':>10}"
    print(header)
    for size in args.sizes:
        ns = f"m{size}"
        asyncio.run(_build(dsn, size, ns))
        ranks = asyncio.run(_ranks(dsn, ns, args.k))
        m = _metrics(ranks, cutoffs)
        row = f"  {size:>8}" + "".join(f"{m[f'recall@{c}']:>9.3f}" for c in cutoffs)
        row += f"{m['mrr']:>9.3f}{m[f'ndcg@{args.k}']:>10.3f}"
        print(row)
        misses = sum(1 for r in ranks if r is None)
        if misses:
            print(f"           ({misses}/{total} needles not in top-{args.k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
