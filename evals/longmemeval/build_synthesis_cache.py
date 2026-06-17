"""
evals/longmemeval/build_synthesis_cache.py
==========================================
Pre-build the v3 synthesis extraction cache for LongMemEval — **extraction only,
no reader**. This bills only the gpt-4o-mini extractor (via the OpenAI key), not
the gpt-oss-120b answerer, so it's the cheap way to get the full-500 extraction
banked. Once cached, the eval runs with ``--synthesis`` for free.

Key trick: the cache is content-addressed on (namespace, prompt). We keep the
namespace fixed at the OpenRouter id (``--cache-namespace``) even though we call
OpenAI here — gpt-4o-mini output is model-equivalent across providers, so the
~5,189 sessions already cached from the 118 runs are REUSED (28% free), and the
eval (which uses that namespace) finds everything we add.

Batched: run 50 questions at a time so you can watch spend and stop any time.

Usage
-----
.. code-block:: bash

    export OPENAI_API_KEY=$(grep '^OPEN_AI_KEY=' .env | cut -d= -f2-)
    python3.12 -m evals.longmemeval.build_synthesis_cache --offset 0 --limit 50
    python3.12 -m evals.longmemeval.build_synthesis_cache --offset 50 --limit 50
    ...                                                   --offset 450 --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from continuum.promotion.synthesis import disk_cached, extract_structured_facts

_DEFAULT_DATASET = "evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json"
# Keep the OpenRouter id so the 118-run cache (and the eval) share these files.
_DEFAULT_NS = "openrouter/openai/gpt-4o-mini"

# Rough gpt-4o-mini cost per extraction call (input ~1.3k tok @ $0.15/M + output
# ~0.3k tok @ $0.60/M) — for the running estimate only.
_COST_PER_CALL = 0.0005


def _openai_key() -> str:
    for name in ("OPENAI_API_KEY", "OPEN_AI_KEY", "OPEN_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    raise SystemExit(
        "No OpenAI key found. Export it first:\n"
        "  export OPENAI_API_KEY=$(grep '^OPEN_AI_KEY=' .env | cut -d= -f2-)"
    )


def _sessions_for(row: dict[str, Any]) -> list[str]:
    """Per-session user-turn text, exactly as the eval groups it for extraction."""
    out: list[str] = []
    for sess in row.get("haystack_sessions", []):
        turns = [
            str(m.get("content", "")).strip()
            for m in sess
            if m.get("role") == "user" and str(m.get("content", "")).strip()
        ]
        if turns:
            out.append("\n".join(turns))
    return out


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset", type=int, default=0, help="skip the first N questions")
    p.add_argument("--limit", type=int, default=50, help="questions in this batch")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id (direct, no provider prefix)")
    p.add_argument("--cache-dir", default=".synthesis_cache")
    p.add_argument("--cache-namespace", default=_DEFAULT_NS, help="cache key namespace (keep to reuse the 118 cache)")
    p.add_argument("--dataset", default=_DEFAULT_DATASET)
    p.add_argument("--concurrency", type=int, default=8, help="parallel extraction calls")
    args = p.parse_args()

    key = _openai_key()
    os.environ["OPENAI_API_KEY"] = key  # litellm reads the canonical name

    import litellm

    calls = {"new": 0, "hit": 0}
    cache_dir = Path(args.cache_dir)
    before = len(list(cache_dir.glob("*.txt"))) if cache_dir.exists() else 0

    async def _raw(prompt: str) -> str:
        calls["new"] += 1
        resp = await litellm.acompletion(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return str(resp["choices"][0]["message"]["content"])

    cached_fn = disk_cached(_raw, args.cache_dir, namespace=args.cache_namespace)

    with open(args.dataset) as fh:
        data = json.load(fh)
    batch = data[args.offset: args.offset + args.limit]
    # Unique session texts in this batch (dedupe within-batch repeats too).
    texts: list[str] = []
    seen: set[str] = set()
    for row in batch:
        for t in _sessions_for(row):
            if t not in seen:
                seen.add(t)
                texts.append(t)

    print(
        f"batch: questions [{args.offset}:{args.offset + len(batch)}] "
        f"→ {len(texts)} unique sessions | model={args.model} ns={args.cache_namespace}"
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def _one(t: str) -> None:
        async with sem:
            try:
                await extract_structured_facts(t, completion_fn=cached_fn)
            except Exception as e:
                print(f"  ! extraction error (skipped): {type(e).__name__}: {e}")

    done = 0
    for i in range(0, len(texts), 50):
        chunk = texts[i: i + 50]
        await asyncio.gather(*(_one(t) for t in chunk))
        done += len(chunk)
        print(
            f"  {done}/{len(texts)} sessions | new API calls so far: {calls['new']} "
            f"(~${calls['new'] * _COST_PER_CALL:.2f})"
        )

    after = len(list(cache_dir.glob("*.txt")))
    calls["hit"] = len(texts) - calls["new"]
    print(
        f"DONE batch [{args.offset}:{args.offset + len(batch)}]: "
        f"{calls['new']} new calls (~${calls['new'] * _COST_PER_CALL:.2f}), "
        f"{calls['hit']} cache hits | cache {before}→{after} files"
    )


if __name__ == "__main__":
    asyncio.run(main())
