"""
bench/ann_vs_exact.py
=====================
Is the ANN index costing recall — and how much?

Runs the same needle queries twice against the same rows:

* **ANN** — the HNSW index, as it ships.
* **Exact** — planner forced off index scans, so pgvector computes the true
  cosine distance over every row.

The difference is what approximation is costing. This reproduces the
methodology of commit ``0e31ae4`` ("the HNSW index, not the embedder, is what
costs recall at scale") at a chosen corpus size, and is the only way to learn
this number — it depends on the data, so no vendor publishes it for you.

Usage::

    python3.12 -m bench.ann_vs_exact --dsn postgresql://localhost:5433/continuum_bench
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.scale_ingest_retrieval import NEEDLES, _pct
from continuum.db.pgvector_upgrade import to_halfvec_literal
from continuum.stores.postgres.ltm import PostgresLTM

RESULTS = Path(__file__).parent / "results"

#: Dense-only top-k. We bypass search_hybrid deliberately: mixing in the
#: lexical channel would mask what the vector index is doing, and the
#: question here is specifically about ANN recall.
_DENSE_SQL = """
    SELECT id, "text"
    FROM   memory_nodes
    WHERE  invalidated_at IS NULL AND embedding IS NOT NULL
    ORDER  BY embedding <=> %(q)s::halfvec
    LIMIT  %(k)s
"""


async def _probe(
    ltm: PostgresLTM, qvecs: list[list[float]], k: int, *, exact: bool,
    ef_search: int | None = None,
) -> dict[str, Any]:
    hits, latencies, missed = 0, [], []
    async with ltm._connect() as conn:
        # Connections are POOLED. A session-level SET survives release back to
        # the pool and leaks into whichever probe borrows it next — that is
        # exactly how an earlier revision got "ef=200 scores 90%": it inherited
        # enable_indexscan=off from the exact probe and silently ran a seq
        # scan. Start clean, and reset in the finally below.
        await conn.execute("RESET ALL")
        # NB: `SET LOCAL` outside an explicit transaction block is a silent
        # no-op in Postgres. An earlier revision used it and produced
        # non-monotonic nonsense (ef=200 scoring above ef=1000), which is
        # impossible for a beam-width parameter. Use session-level SET on a
        # single held connection, then READ THE SETTINGS BACK and assert.
        if exact:
            await conn.execute("SET enable_indexscan = off")
            await conn.execute("SET enable_bitmapscan = off")
            cur = await conn.execute("show enable_indexscan")
            row = await cur.fetchone()
            got = next(iter(row.values())) if isinstance(row, dict) else row[0]
            assert got == "off", f"enable_indexscan did not apply: {got!r}"
        else:
            # Force the index. Without this the planner may cost an HNSW scan
            # at high ef_search ABOVE a sequential scan and quietly pick the
            # seq scan — which then measures exact search under an "ANN" label.
            # (That planner preference is a real finding in its own right and
            # is recorded as `planner_would_seqscan` below.)
            await conn.execute("SET enable_seqscan = off")
            await conn.execute(f"SET hnsw.ef_search = {int(ef_search or 1000)}")
            cur = await conn.execute("show hnsw.ef_search")
            row = await cur.fetchone()
            got = next(iter(row.values())) if isinstance(row, dict) else row[0]
            assert int(got) == int(ef_search or 1000), \
                f"ef_search did not apply: wanted {ef_search}, got {got!r}"
        # Confirm the plan matches the intent, so "exact" really is a scan.
        cur = await conn.execute(
            "EXPLAIN " + _DENSE_SQL, {"q": to_halfvec_literal(qvecs[0]), "k": k})
        plan = " ".join(
            str(next(iter(r.values())) if isinstance(r, dict) else r[0])
            for r in await cur.fetchall())
        # Must be the HNSW index specifically. "Index Scan" alone also matches
        # a btree scan + Sort, which returns EXACT results at seq-scan latency
        # — an earlier revision scored that as "hnsw ef=200: 90%", which is how
        # a 74ms "ANN" result appeared next to a 4.4ms one.
        used_index = "memory_nodes_embedding_hnsw_idx" in plan
        if exact and used_index:
            raise AssertionError("exact probe still used the index: " + plan[:160])
        if not exact and not used_index:
            raise AssertionError(f"ANN probe (ef={ef_search}) did not use the HNSW index: " + plan[:200])
        # What would the planner have chosen on its own, unforced?
        planner_would_seqscan = False
        if not exact:
            await conn.execute("SET enable_seqscan = on")
            cur = await conn.execute(
                "EXPLAIN " + _DENSE_SQL, {"q": to_halfvec_literal(qvecs[0]), "k": k})
            free_plan = " ".join(
                str(next(iter(r.values())) if isinstance(r, dict) else r[0])
                for r in await cur.fetchall())
            planner_would_seqscan = "memory_nodes_embedding_hnsw_idx" not in free_plan
            await conn.execute("SET enable_seqscan = off")
        for needle, qvec in zip(NEEDLES, qvecs, strict=True):
            t0 = time.perf_counter()
            cur = await conn.execute(
                _DENSE_SQL, {"q": to_halfvec_literal(qvec), "k": k})
            rows = await cur.fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)
            texts = [
                (r["text"] if isinstance(r, dict) else r[1]) or "" for r in rows
            ]
            if any(t.strip() == needle.text for t in texts):
                hits += 1
            else:
                missed.append(needle.query)
        await conn.execute("RESET ALL")   # never hand a dirty session back
    return {
        "used_index": used_index,
        "planner_would_seqscan": planner_would_seqscan if not exact else None,
        "recall": hits / len(NEEDLES),
        "hits": hits, "n": len(NEEDLES),
        "p50_ms": _pct(latencies, 0.50), "p95_ms": _pct(latencies, 0.95),
        "mean_ms": statistics.fmean(latencies),
        "missed": missed,
    }


async def run(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer

    ltm = PostgresLTM(dsn=args.dsn)
    try:
        async with ltm._connect() as conn:
            cur = await conn.execute(
                "select count(*) as n from memory_nodes where embedding is not null")
            row = await cur.fetchone()
            n_rows = int(row["n"] if isinstance(row, dict) else row[0])
        print(f"corpus   : {n_rows:,} embedded rows")

        model = SentenceTransformer("BAAI/bge-m3")
        qvecs = [
            [float(x) for x in v]
            for v in model.encode([n.query for n in NEEDLES],
                                  normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
        ]

        out: dict[str, Any] = {"n_rows": n_rows, "k": args.k, "probes": {}}

        exact = await _probe(ltm, qvecs, args.k, exact=True)
        out["probes"]["exact"] = exact
        print(f"exact    : recall@{args.k} {exact['recall']:.0%}  "
              f"p50 {exact['p50_ms']:.1f}ms  p95 {exact['p95_ms']:.1f}ms")

        for ef in args.ef_search:
            ann = await _probe(ltm, qvecs, args.k, exact=False, ef_search=ef)
            out["probes"][f"hnsw_ef{ef}"] = ann
            delta = (ann["recall"] - exact["recall"]) * 100
            note = "  [planner would seq-scan instead]" if ann.get("planner_would_seqscan") else ""
            print(f"hnsw {ef:>4}: recall@{args.k} {ann['recall']:.0%}  "
                  f"p50 {ann['p50_ms']:.1f}ms  p95 {ann['p95_ms']:.1f}ms   "
                  f"({delta:+.0f}pp vs exact){note}")

        best_ann = max(
            (v for kk, v in out["probes"].items() if kk.startswith("hnsw")),
            key=lambda v: v["recall"])
        gap = (exact["recall"] - best_ann["recall"]) * 100
        out["index_recall_cost_pp"] = gap
        print()
        if gap > 0:
            print(f"VERDICT  : the ANN index costs {gap:.0f}pp of recall at this "
                  f"corpus size, saving {best_ann['p50_ms'] - exact['p50_ms']:+.1f}ms p50.")
        else:
            print("VERDICT  : the index is not costing recall here — the misses "
                  "are upstream of the index (embedding or corpus difficulty).")

        RESULTS.mkdir(parents=True, exist_ok=True)
        f = RESULTS / f"ann_vs_exact_{n_rows}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        out["finished_at"] = datetime.now(UTC).isoformat()
        f.write_text(json.dumps(out, indent=2, default=str))
        print(f"results  → {f}")
        return 0
    finally:
        await ltm.aclose()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", default="postgresql://localhost:5433/continuum_bench")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef-search", type=int, nargs="+", default=[40, 200, 1000])
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
