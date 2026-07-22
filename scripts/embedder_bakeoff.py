#!/usr/bin/env python3
"""
scripts/embedder_bakeoff.py
===========================
Compare embedding models on Continuum's own needle set.

Recall falls with store size — ~100% at tens of memories, 95% at 3k, 75% at 47k
— and every retrieval lever has been measured and exhausted: `ef_search` fixed a
real loss, widening the candidate pool did nothing, raising `k` did nothing, and
a cross-encoder reranker made it worse. What is left is the embedding model
itself. This measures that directly.

Deliberately **no database and no index**: embeddings are compared by exact
cosine in memory. HNSW build randomness moves recall by 1–2 needles on identical
data, which is the same size as the effect being measured — running this through
the index would drown the signal in build noise.

    python3 scripts/embedder_bakeoff.py --rows 3000
    python3 scripts/embedder_bakeoff.py --models BAAI/bge-m3 intfloat/multilingual-e5-large

Each model downloads 1–2 GB on first use.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent


def _needles_and_distractors(rows: int) -> tuple[list[tuple[str, str]], list[str]]:
    spec = importlib.util.spec_from_file_location("ras", _HERE / "recall_at_scale.py")
    assert spec and spec.loader
    ras = importlib.util.module_from_spec(spec)
    sys.modules["ras"] = ras
    spec.loader.exec_module(ras)
    return ras.NEEDLES, ras.distractors(rows)


#: Models that need an instruction prefix. Getting this wrong understates them
#: badly — e5 without "query:"/"passage:" is a materially different model, and a
#: bake-off that skipped it would reach a confident wrong conclusion.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "BAAI/bge-large-en-v1.5": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
}


def evaluate(model_name: str, rows: int, k: int) -> tuple[int, int, float, int]:
    """Return (hits, total, seconds, dimensions) for one model."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    needles, distractors = _needles_and_distractors(rows)
    q_prefix, d_prefix = _PREFIXES.get(model_name, ("", ""))

    started = time.perf_counter()
    model = SentenceTransformer(model_name)
    corpus = [d_prefix + t for t in distractors] + [d_prefix + fact for fact, _q in needles]
    corpus_vecs = model.encode(corpus, batch_size=64, normalize_embeddings=True)
    query_vecs = model.encode(
        [q_prefix + q for _f, q in needles], batch_size=32, normalize_embeddings=True
    )

    needle_start = len(distractors)
    hits = 0
    for i, _pair in enumerate(needles):
        sims = corpus_vecs @ query_vecs[i]
        top = np.argpartition(-sims, k)[:k]
        if needle_start + i in top:
            hits += 1
    dims = int(corpus_vecs.shape[1])
    elapsed = time.perf_counter() - started

    # Free it before the next candidate loads. These are 1-2GB each and holding
    # three at once got the first attempt OOM-killed mid-run — which looked
    # exactly like the script quietly deciding not to test the other models.
    del model, corpus_vecs, query_vecs
    _release_memory()
    return hits, len(needles), elapsed, dims


def _release_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--k", type=int, default=24, help="candidate depth, matching the retriever")
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "BAAI/bge-m3",  # what ships today
            "intfloat/multilingual-e5-large",  # multilingual peer
            "mixedbread-ai/mxbai-embed-large-v1",  # strong English-only
        ],
    )
    args = p.parse_args(argv)

    print(f"  {args.rows} distractors + 20 needles, exact cosine, top-{args.k}\n")
    print(f"  {'model':44} {'recall':>10} {'dims':>6} {'load+embed':>12}")
    results: list[tuple[str, int]] = []
    for name in args.models:
        try:
            hits, total, secs, dims = evaluate(name, args.rows, args.k)
        except Exception as exc:  # a model that will not load is a result too
            print(f"  {name:44} {'FAILED':>10}   {type(exc).__name__}: {str(exc)[:40]}")
            continue
        marker = "  <- current" if name == "BAAI/bge-m3" else ""
        print(f"  {name:44} {f'{hits}/{total}':>10} {dims:>6} {secs:>10.0f}s{marker}")
        results.append((name, hits))

    if len(results) > 1:
        best = max(results, key=lambda r: r[1])
        current = next((h for n, h in results if n == "BAAI/bge-m3"), None)
        print()
        if current is not None and best[1] > current:
            print(f"  {best[0]} beats the current model by {best[1] - current} needles.")
            print("  Switching means re-embedding every stored memory — worth it only if")
            print("  the gap holds at your store size and the model keeps the dimensions.")
        else:
            print("  No candidate beats the current model on this set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
