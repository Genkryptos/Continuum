#!/usr/bin/env python3
"""
scripts/lme_library_recall.py
=============================
Gold-turn recall of the **library** path (``Memory.recall``, which the MCP
``recall`` tool calls) on LongMemEval-S haystacks.

The LongMemEval accuracy numbers in the README come from the eval harness in
``evals/longmemeval/``, which has its own retrieval. This asks what the code a
``pip install continuum-mcp`` user actually runs retrieves on the same data:
every haystack turn is written with ``Memory.add`` (dated with its session's
date), the question is asked with ``Memory.recall``, and we count how many of
the turns LongMemEval flags ``has_answer`` come back in the top *k*.

No LLM and no judge, so it is deterministic and costs nothing to run. It is a
retrieval number, not an accuracy number: a reader still has to answer.

Needs a **throwaway**, migrated Postgres database (it writes ~500 rows per
question, each question in its own namespace)::

    createdb continuum_lme_recall
    python -m continuum.db.migrate --dsn postgresql://.../continuum_lme_recall
    python scripts/lme_library_recall.py --dsn postgresql://.../continuum_lme_recall \\
        --per-type 10 --k 8,24,48 --out results/lme_library_recall.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DATA = "evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json"


def _parse_date(raw: str) -> datetime | None:
    # LongMemEval dates look like "2023/05/20 (Sat) 02:21".
    try:
        return datetime.strptime(raw.split(" (")[0], "%Y/%m/%d")
    except (ValueError, AttributeError):
        return None


def _sample(rows: list[dict[str, Any]], per_type: int, seed: int) -> list[dict[str, Any]]:
    """*per_type* questions of each type, skipping abstention rows (no gold)."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["question_id"].endswith("_abs"):
            continue
        by_type[r["question_type"]].append(r)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for qtype in sorted(by_type):
        pool = by_type[qtype]
        out.extend(rng.sample(pool, min(per_type, len(pool))))
    return out


def _gold(row: dict[str, Any]) -> set[str]:
    return {
        t["content"].strip()
        for session in row["haystack_sessions"]
        for t in session
        if t.get("has_answer") and t.get("content", "").strip()
    }


async def _stored_rows(dsn: str, ns: str, *, need_vectors: bool) -> int:
    """Live rows already stored for *ns* (with a vector, if *need_vectors*)."""
    import psycopg

    cond = " AND embedding IS NOT NULL" if need_vectors else ""
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        cur = await conn.execute(
            f"SELECT count(*) FROM memory_nodes WHERE namespace = %s AND invalidated_at IS NULL{cond}",
            (ns,),
        )
        got = await cur.fetchone()
    return int(got[0]) if got else 0


def _free_device_cache() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # cache freeing is an optimisation, never a failure
        pass


_CPU_ENCODER: list[Any] = []


def _cpu_encoder() -> Any:
    if not _CPU_ENCODER:
        from sentence_transformers import SentenceTransformer

        from continuum.core.config import EmbeddingConfig

        _CPU_ENCODER.append(SentenceTransformer(EmbeddingConfig().model_name, device="cpu"))
    return _CPU_ENCODER[0]


async def _fill_vectors(mem: Any, encoder: Any) -> None:
    """Embed every vector-less row with *encoder*, as the library would.

    Mirrors ``EmbeddingService``: ``encode(texts, batch_size=...)`` then a
    float16 down-cast. Only the device differs, so a GPU can pay the one-time
    haystack cost; queries still go through the library's own embedder.
    """
    import numpy as np

    ltm = mem.session.ltm
    # Small chunks, and the device cache freed after each: long turns (bge-m3
    # takes 8192 tokens) otherwise accumulate allocations until MPS runs out.
    while pending := await ltm.rows_missing_embeddings(32):
        texts = [text for _id, text in pending]
        try:
            vecs = encoder.encode(texts, batch_size=2)
        except RuntimeError as exc:
            if "out of memory" not in str(exc):
                raise
            # The GPU shares memory with everything else on the machine; when
            # a chunk of long turns does not fit, encode just that chunk on CPU.
            _free_device_cache()
            vecs = _cpu_encoder().encode(texts, batch_size=2)
        arr = np.asarray(vecs, dtype=np.float32).astype(np.float16)
        await ltm.set_embeddings(
            {node_id: [float(x) for x in v] for (node_id, _t), v in zip(pending, arr, strict=True)}
        )
        _free_device_cache()


async def _one(
    row: dict[str, Any],
    dsn: str,
    ks: list[int],
    embeddings: bool,
    run_tag: str,
    encoder: Any = None,
) -> dict[str, Any]:
    from continuum import Memory

    ns = f"lme_{run_tag}_{row['question_id']}"
    t0 = time.perf_counter()
    # Distinct texts: the store keeps one live row per text (a repeat reinforces).
    n_turns = len(
        {(t.get("content") or "").strip() for s in row["haystack_sessions"] for t in s} - {""}
    )
    # A haystack is ingested once per tag; later runs with the same tag reuse
    # the stored (and embedded) rows, so retrieval variants are compared on
    # identical data and only the first run pays for embedding.
    if await _stored_rows(dsn, ns, need_vectors=False) < n_turns:
        # Write sparse, then embed in batches: Memory.add embeds one text per
        # call, which on CPU costs minutes per haystack. The rows are the same.
        writer = Memory.from_postgres(dsn, embeddings=False, namespace=ns, session_id=ns)
        await writer.start()
        try:
            for session, date in zip(row["haystack_sessions"], row["haystack_dates"], strict=True):
                when = _parse_date(date)
                for turn in session:
                    text = (turn.get("content") or "").strip()
                    if text:
                        await writer.add(text, occurred_at=when)
        finally:
            await writer.aclose()

    mem = Memory.from_postgres(dsn, embeddings=embeddings, namespace=ns, session_id=ns)
    await mem.start()
    try:
        if embeddings and encoder is not None:
            await _fill_vectors(mem, encoder)
        elif embeddings:
            while await mem.backfill_embeddings(limit=2000, batch=64):
                pass
        ingest_s = time.perf_counter() - t0

        gold = _gold(row)
        t1 = time.perf_counter()
        hits = await mem.recall(row["question"], k=max(ks))
        recall_ms = (time.perf_counter() - t1) * 1000
        got = [(h.content or "").strip() for h in hits]
        per_k = {}
        for k in ks:
            top = set(got[:k])
            found = len(gold & top)
            per_k[str(k)] = {
                "recall": found / len(gold) if gold else None,
                "complete": bool(gold) and gold <= top,
            }
        return {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "n_turns": n_turns,
            "n_gold": len(gold),
            "returned": len(got),
            "ingest_s": round(ingest_s, 2),
            "recall_ms": round(recall_ms, 1),
            "per_k": per_k,
        }
    finally:
        await mem.aclose()


def _summarise(results: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rows)}
        for k in ks:
            rec = [r["per_k"][str(k)]["recall"] for r in rows if r["n_gold"]]
            comp = [r["per_k"][str(k)]["complete"] for r in rows if r["n_gold"]]
            out[f"recall@{k}"] = round(sum(rec) / len(rec), 3) if rec else None
            out[f"complete@{k}"] = round(sum(comp) / len(comp), 3) if comp else None
        return out

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
    return {"overall": agg(results), "by_type": {t: agg(v) for t, v in sorted(by_type.items())}}


def _share_query_encoder() -> None:
    """Make every ``EmbeddingService`` reuse one model.

    Each question opens its own ``Memory``, and each ``EmbeddingService`` builds
    its own 2.3 GB model, so a 60-question run would load sixty copies. Query
    embedding still runs the library's own ``EmbeddingService.embed``; only the
    underlying model is shared.
    """
    import continuum.embeddings as emb

    base = emb.EmbeddingService
    cache: dict[str, Any] = {}

    def factory(device: str) -> Any:
        if device == "cpu":
            return _cpu_encoder()  # the same model the GPU fallback uses
        if device not in cache:
            from sentence_transformers import SentenceTransformer

            from continuum.core.config import EmbeddingConfig

            cache[device] = SentenceTransformer(EmbeddingConfig().model_name, device=device)
        return cache[device]

    class SharedEmbeddingService(base):
        def __init__(self, config: Any = None, **kw: Any) -> None:
            kw.setdefault("encoder_factory", factory)
            super().__init__(config, **kw)

    emb.EmbeddingService = SharedEmbeddingService


async def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", required=True, help="a THROWAWAY migrated database")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", default="8,24,48", help="comma-separated cut-offs")
    ap.add_argument("--embeddings", action="store_true", help="attach bge-m3 (slow on CPU)")
    ap.add_argument(
        "--embed-device",
        default=None,
        help="embed haystacks on this torch device (e.g. mps) instead of the library's CPU path",
    )
    ap.add_argument("--tag", default="run", help="namespace prefix, keeps runs apart")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    rows = json.loads(Path(args.data).read_text())
    sample = _sample(rows, args.per_type, args.seed)
    print(f"{len(sample)} questions, k={ks}, embeddings={args.embeddings}", file=sys.stderr)

    if args.embeddings:
        _share_query_encoder()

    encoder = None
    if args.embeddings and args.embed_device:
        from sentence_transformers import SentenceTransformer

        from continuum.core.config import EmbeddingConfig

        model_name = EmbeddingConfig().model_name
        encoder = SentenceTransformer(model_name, device=args.embed_device)

    results = []
    for i, row in enumerate(sample, 1):
        r = await _one(row, args.dsn, ks, args.embeddings, args.tag, encoder)
        results.append(r)
        print(
            f"[{i}/{len(sample)}] {r['question_type']:<26} gold={r['n_gold']} "
            f"returned={r['returned']} "
            + " ".join(f"@{k}={r['per_k'][str(k)]['recall']:.2f}" for k in ks)
            + f"  ingest={r['ingest_s']}s",
            file=sys.stderr,
        )

    summary = _summarise(results, ks)
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {"args": vars(args), "summary": summary, "rows": results}, indent=2, default=str
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
