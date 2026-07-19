"""
tests/unit/test_embedding_query.py
==================================
`EmbeddingQueryRetriever` — the wrapper that embeds `query.text` before
delegating so the LTM dense channel fires. Hermetic: fake inner retriever + fake
embedder, no model, no DB.
"""

from __future__ import annotations

from typing import Any

import pytest

from continuum.core.types import Query
from continuum.retrieval.embedding_query import EmbeddingQueryRetriever

pytestmark = pytest.mark.unit


class _FakeInner:
    """Records the query it was handed; returns a sentinel bundle."""

    def __init__(self) -> None:
        self.seen: Query | None = None
        self.budget: Any = None

    async def retrieve(self, query: Query, budget: Any) -> Any:
        self.seen = query
        self.budget = budget
        return {"items": []}  # sentinel — the wrapper passes it straight through


class _FakeEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self.vec = vec
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vec for _ in texts]


async def test_embeds_query_text_when_missing() -> None:
    inner = _FakeInner()
    emb = _FakeEmbedder([0.1, 0.2, 0.3])
    out = await EmbeddingQueryRetriever(inner, emb).retrieve(  # type: ignore[arg-type]
        Query(text="where do I live", top_k=5), budget="BUDGET"
    )
    assert inner.seen is not None and inner.seen.embedding == [0.1, 0.2, 0.3]
    assert emb.calls == [["where do I live"]]  # embedded exactly once
    assert inner.budget == "BUDGET"  # budget passed through untouched
    assert out == {"items": []}  # bundle returned as-is


async def test_no_embedder_passes_through() -> None:
    inner = _FakeInner()
    await EmbeddingQueryRetriever(inner, None).retrieve(  # type: ignore[arg-type]
        Query(text="x", top_k=5), budget=None
    )
    assert inner.seen is not None and inner.seen.embedding is None


async def test_existing_embedding_not_recomputed() -> None:
    inner = _FakeInner()
    emb = _FakeEmbedder([9.9])
    await EmbeddingQueryRetriever(inner, emb).retrieve(  # type: ignore[arg-type]
        Query(text="x", embedding=[1.0, 2.0], top_k=5), budget=None
    )
    assert inner.seen is not None and inner.seen.embedding == [1.0, 2.0]  # untouched
    assert emb.calls == []  # embedder not called


async def test_embed_error_degrades_to_sparse() -> None:
    inner = _FakeInner()

    class _Boom:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model down")

    await EmbeddingQueryRetriever(inner, _Boom()).retrieve(  # type: ignore[arg-type]
        Query(text="x", top_k=5), budget=None
    )
    assert inner.seen is not None and inner.seen.embedding is None  # no crash, sparse path
