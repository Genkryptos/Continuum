#!/usr/bin/env python3
"""
scripts/backfill_embeddings.py
==============================
Attach embeddings to memories that were stored without one.

The ``UserPromptSubmit`` capture hook runs sparse on purpose — a fresh process
per prompt cannot load a 2.3GB model — so every fact it captures lands with a
NULL embedding and **dense recall can never find it**. The memories are in the
store and semantically invisible, which is worse than absent: nothing looks
broken. Run this from a warm process, occasionally, to make them searchable.

    CONTINUUM_DB_DSN=postgresql://…/continuum python3 scripts/backfill_embeddings.py
    make backfill

Idempotent: it only fills NULLs and never overwrites an existing vector.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time


async def _run(dsn: str, namespace: str, limit: int) -> int:
    from continuum.memory import Memory

    mem = Memory.from_postgres(dsn, embeddings=True, namespace=namespace)
    await mem.start()
    try:
        return await mem.backfill_embeddings(limit=limit)
    finally:
        await mem.aclose()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default=os.environ.get("CONTINUUM_MCP_NAMESPACE", "default"))
    p.add_argument("--limit", type=int, default=500, help="most rows to embed in one run")
    args = p.parse_args(argv)

    dsn = os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("[backfill] set CONTINUUM_DB_DSN first.", file=sys.stderr)
        return 1

    t = time.perf_counter()
    n = asyncio.run(_run(dsn, args.namespace, args.limit))
    took = time.perf_counter() - t
    if n:
        print(f"[backfill] embedded {n} memory(ies) in namespace {args.namespace!r} ({took:.1f}s)")
        print("[backfill] they are now findable by semantic recall.")
    else:
        print(
            f"[backfill] nothing to do — every memory in {args.namespace!r} already has a vector."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
