#!/usr/bin/env python3
"""
scripts/rerank_ab.py
====================
A/B the cross-encoder reranker against the plain hybrid ranking, on the
**paraphrase** query set (no word shared with the target fact — the case a
reranker is supposed to win).

This exists because "enable the reranker and measure" (plan item 3.2) has a
counter-intuitive answer: on personal first-person facts both candidate
rerankers make recall@1 *worse*. That claim should be checkable, not asserted,
so this reproduces it end to end.

Reranking runs through the project's own ``Reranker`` — with
``skip_if_fewer_than=0``, because at the default of 10 it would skip a k=8
recall entirely and measure nothing.

    CONTINUUM_DB_DSN=postgresql://…/throwaway python3 scripts/rerank_ab.py
    python3 scripts/rerank_ab.py --model cross-encoder/ms-marco-MiniLM-L-6-v2

Requires a migrated, WRITEABLE database — it writes the scenario facts, so
point it at a throwaway, never your real store. Exit 0 always: this is a
measurement, not a gate.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

# The scenario and its paraphrase queries live with the retrieval eval.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_eval import FACTS, PARAPHRASE_QUERIES


async def _run(dsn: str, model: str, k: int) -> int:
    from continuum.core.config import RerankerConfig
    from continuum.memory import Memory
    from continuum.retrieval.reranker import Reranker

    mem = Memory.from_postgres(dsn, embeddings=True, session_id="rerank-ab")
    await mem.start()
    for f in FACTS:
        await mem.add(f["text"])

    t0 = time.perf_counter()
    reranker = Reranker(RerankerConfig(model_name=model, skip_if_fewer_than=0, rerank_top_n=k))
    base_hits = rr_hits = 0
    recall_ms: list[float] = []
    rerank_ms: list[float] = []
    changes: list[str] = []

    for q in PARAPHRASE_QUERIES:
        query, want = q["query"], q["expect"].lower()
        t = time.perf_counter()
        items = await mem.recall(query, k=k)
        recall_ms.append((time.perf_counter() - t) * 1000)
        if not items:
            continue

        before = (items[0].content or "").lower()
        t = time.perf_counter()
        ranked = await reranker.rerank(query, [_scored(i) for i in items])
        rerank_ms.append((time.perf_counter() - t) * 1000)
        after = (ranked[0].item.content or "").lower() if ranked else ""

        hit_before, hit_after = want in before, want in after
        base_hits += hit_before
        rr_hits += hit_after
        if hit_before != hit_after:
            verb = "FIXED" if hit_after else "BROKE"
            changes.append(f"  {verb}  {query!r}\n         {before[:44]!r} -> {after[:44]!r}")

    await mem.aclose()

    n = len(PARAPHRASE_QUERIES)
    print(f"model: {model}   (first call incl. load: {time.perf_counter() - t0:.1f}s)")
    print(f"scenario: {len(FACTS)} facts, {n} paraphrase queries, k={k}\n")
    print("\n".join(changes) if changes else "  (no rank-1 changes)")
    print(f"\n  recall@1 hybrid only = {base_hits}/{n}")
    print(f"  recall@1 + reranker  = {rr_hits}/{n}")
    if recall_ms and rerank_ms:
        print(
            f"\n  recall p50 = {statistics.median(recall_ms):.0f}ms"
            f"   reranking adds p50 = {statistics.median(rerank_ms):.0f}ms"
            f"  (max {max(rerank_ms):.0f}ms)"
        )
    verdict = "helps" if rr_hits > base_hits else ("neutral" if rr_hits == base_hits else "HURTS")
    print(f"\n[rerank-ab] verdict: reranking {verdict} on this set")
    return 0


def _scored(item: object) -> object:
    from continuum.core.types import ScoreBreakdown, ScoredItem

    zero = ScoreBreakdown(relevance=0.0, importance=0.0, recency=0.0, confidence=0.0, composite=0.0)
    return ScoredItem(item=item, scores=zero)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3", help="cross-encoder to test")
    p.add_argument("--k", type=int, default=8, help="candidates to recall then rerank")
    args = p.parse_args(argv)

    dsn = os.environ.get("CONTINUUM_DB_DSN")
    if not dsn:
        print("[rerank-ab] set CONTINUUM_DB_DSN to a THROWAWAY migrated database.")
        return 0
    return asyncio.run(_run(dsn, args.model, args.k))


if __name__ == "__main__":
    raise SystemExit(main())
