"""
Gold-turn recall of each retrieval configuration over all of LongMemEval-S.

    python3.12 -m evals.audit.turn_selection_ablation [--sample N]

Zero LLM calls, zero network: retrieval only, with the local embedder. The
2026-11-07 gap audit found accuracy is 93.2% on rows where every gold turn
reached the reader and ~35-39% where some did not, so gold-turn recall is the
quantity to choose a configuration by, and it does not depend on the
answerer. Each haystack is embedded once and shared across configurations.

``legacy`` reproduces the structure of the A-BUDGET arm: the session-aware
retriever (global re-rank) wrapped in reciprocal-rank fusion with a top-k
BM25 retriever. The other rows fuse BM25 inside session selection, as the
pinned path now does.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from continuum.core.types import Query, TokenBudget
from continuum.retrieval.rrf import HybridRetriever
from evals.longmemeval.bootstrap_ollama import (
    BM25HaystackRetriever,
    SessionAwareSemanticRetriever,
    _Embedder,
    load_longmemeval_rows,
    make_adapter_factory,
)
from evals.longmemeval.evidence import gold_turn_recall, gold_turns_from_raw

DATA = Path("evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json")
RELEVANCE_ONLY = {"relevance": 1.0, "importance": 0.0, "recency": 0.0, "confidence": 0.0}
DEFAULT_WEIGHTS = {"relevance": 0.45, "importance": 0.25, "recency": 0.20, "confidence": 0.10}

# name: (structure, turn_rerank, top_k, session_top_k, turns_per_session, weights, user_first)
R, D = RELEVANCE_ONLY, DEFAULT_WEIGHTS
CONFIGS: dict[str, tuple[str, str, int, int, int, dict[str, float], bool]] = {
    "legacy 24/8/2 global":          ("wrapped", "global", 24, 8, 2, R, False),
    "fused 24/8/2 floor":            ("fused", "floor", 24, 8, 2, R, False),
    "fused 24/8/2 global +user":     ("fused", "global", 24, 8, 2, R, True),
    "fused 24/8/2 floor +user":      ("fused", "floor", 24, 8, 2, R, True),
    "fused 36/12/2 global +user":    ("fused", "global", 36, 12, 2, R, True),
    "fused 36/12/2 floor +user":     ("fused", "floor", 36, 12, 2, R, True),
    "fused 36/12/2 floor +user+rec": ("fused", "floor", 36, 12, 2, D, True),
    "fused 48/12/2 global +user":    ("fused", "global", 48, 12, 2, R, True),
    "fused 48/12/3 floor +user":     ("fused", "floor", 48, 12, 3, R, True),
    "fused 48/12/3 floor +user+rec": ("fused", "floor", 48, 12, 3, D, True),
}
_BUDGET = TokenBudget(10**7, 10**7, 0, 0, 0)


def _retriever(store: Any, embedder: Any, cfg: tuple, matrix: Any, items: list) -> Any:
    structure, rerank, top_k, stk, tps, weights, user_first = cfg
    dense = SessionAwareSemanticRetriever(
        store=store, embedder=embedder, top_k=top_k, session_top_k=stk,
        turns_per_session=tps, max_items=top_k, turn_rerank=rerank,
        lexical=structure == "fused", score_weights=weights, user_first=user_first,
    )
    # Share the haystack embedding: retrieve() only re-embeds when the
    # store's length differs from the cached length.
    dense._cached_matrix, dense._cached_items, dense._cached_len = matrix, items, len(items)
    if structure == "wrapped":
        return HybridRetriever(dense, BM25HaystackRetriever(store=store, top_k=top_k), k=60, top_k=top_k)
    return dense


async def _run(sample: int | None) -> None:
    raw = {r["question_id"]: r for r in json.loads(DATA.read_text())}
    rows = load_longmemeval_rows(DATA)
    if sample:
        # Evenly spaced, not the first N: the file is sorted by category.
        step = max(1, len(rows) // sample)
        rows = rows[::step][:sample]
    embedder = _Embedder()
    recall: dict[str, list[float]] = defaultdict(list)
    by_type: dict[tuple[str, str], list[float]] = defaultdict(list)
    chars: dict[str, list[int]] = defaultdict(list)
    for n, row in enumerate(rows, 1):
        gold = gold_turns_from_raw(raw[row.question_id])
        if not gold:
            continue
        adapter = make_adapter_factory(None, embedder, direct_answer=True)()
        await adapter.process_conversation(row.messages)
        store = adapter.session.stm
        items = list(store.items)
        matrix = embedder.encode([it.content for it in items])
        qtype = raw[row.question_id]["question_type"]
        for name, cfg in CONFIGS.items():
            ctx = await _retriever(store, embedder, cfg, matrix, items).retrieve(
                Query(text=row.question), _BUDGET,
            )
            picked = list(ctx.items)[: cfg[2]]
            r = gold_turn_recall(gold, [it.content for it in picked])["gold_turn_recall"]
            recall[name].append(r)
            by_type[(name, qtype)].append(r)
            chars[name].append(sum(len(it.content) for it in picked))
        if n % 20 == 0:
            print(f"  … {n}/{len(rows)} rows", flush=True)

    types = sorted({t for _n, t in by_type})
    print(f"\nrows with gold turns: {len(recall[next(iter(CONFIGS))])}")
    head = f"{'config':30} {'gold-turn recall':>16} {'complete':>9} {'ctx chars':>10}"
    print(head + "".join(f" {t[:14]:>15}" for t in types))
    for name in CONFIGS:
        vals = recall[name]
        mean = sum(vals) / len(vals)
        complete = sum(v >= 1.0 for v in vals) / len(vals)
        avg_chars = sum(chars[name]) / len(chars[name])
        cells = "".join(
            f" {sum(by_type[(name, t)]) / len(by_type[(name, t)]):>15.3f}" for t in types
        )
        print(f"{name:30} {mean:>16.3f} {complete:>9.3f} {avg_chars:>10.0f}{cells}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="evenly spaced subset of N rows (default: all 500)")
    asyncio.run(_run(p.parse_args().sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
