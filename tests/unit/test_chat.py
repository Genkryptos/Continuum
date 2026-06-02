"""
tests/unit/test_chat.py
======================
Hermetic tests for ``continuum.chat`` — the interactive REPL's pure pieces
(context formatting, the mock responder, and the embedding-retriever wrapper).
No Postgres, no network, no model: fakes throughout.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.chat import (
    EmbeddingRetriever,
    build_mock_responder,
    format_context,
)
from continuum.core.types import MemoryItem, MemoryTier, Query

pytestmark = pytest.mark.unit


def _item(content: str, role: str = "user") -> MemoryItem:
    return MemoryItem(content=content, tier=MemoryTier.LTM, metadata={"role": role})


def _ctx(*contents: str) -> SimpleNamespace:
    return SimpleNamespace(items=[_item(c) for c in contents])


# ── format_context ─────────────────────────────────────────────────────────


def test_format_context_lists_items_with_roles() -> None:
    out = format_context(_ctx("likes hiking", "lives in Pune"))
    assert "Remembered context:" in out
    assert "- [user] likes hiking" in out
    assert "- [user] lives in Pune" in out


def test_format_context_empty_or_none() -> None:
    assert "nothing relevant" in format_context(None)
    assert "nothing relevant" in format_context(_ctx())


def test_format_context_skips_the_current_message() -> None:
    out = format_context(_ctx("what is my name", "name is Mayank"), skip="what is my name")
    assert "name is Mayank" in out
    assert "what is my name" not in out


def test_format_context_respects_limit() -> None:
    out = format_context(_ctx(*[f"fact {i}" for i in range(20)]), limit=3)
    assert out.count("- [") == 3


# ── mock responder ───────────────────────────────────────────────────────────


async def test_mock_responder_echoes_and_counts() -> None:
    respond = build_mock_responder()
    reply = await respond("hello", _ctx("a", "b"))
    assert "hello" in reply
    assert "2 memory item" in reply


async def test_mock_responder_handles_none_context() -> None:
    respond = build_mock_responder()
    reply = await respond("hi", None)
    assert "0 memory item" in reply


# ── EmbeddingRetriever ───────────────────────────────────────────────────────


class _CapturingInner:
    def __init__(self) -> None:
        self.seen: Query | None = None

    async def retrieve(self, query: Query, budget: Any) -> str:
        self.seen = query
        return "BUNDLE"


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_embedding_retriever_sets_query_embedding() -> None:
    inner, emb = _CapturingInner(), _FakeEmbedder()
    r = EmbeddingRetriever(inner, emb)
    out = await r.retrieve(Query(text="hello"), budget=None)
    assert out == "BUNDLE"
    assert emb.calls == 1
    assert inner.seen is not None
    assert inner.seen.embedding == [0.1, 0.2, 0.3]  # embedding was injected


async def test_embedding_retriever_preserves_existing_embedding() -> None:
    inner, emb = _CapturingInner(), _FakeEmbedder()
    r = EmbeddingRetriever(inner, emb)
    await r.retrieve(Query(text="hello", embedding=[9.0]), budget=None)
    assert emb.calls == 0  # already had an embedding → no re-embed
    assert inner.seen is not None and inner.seen.embedding == [9.0]


async def test_embedding_retriever_degrades_when_embedder_fails() -> None:
    class _BoomEmbedder:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model down")

    inner = _CapturingInner()
    r = EmbeddingRetriever(inner, _BoomEmbedder())
    out = await r.retrieve(Query(text="hello"), budget=None)
    assert out == "BUNDLE"  # still retrieves (sparse-only)
    assert inner.seen is not None and inner.seen.embedding is None


async def test_embedding_retriever_with_no_embedder() -> None:
    inner = _CapturingInner()
    r = EmbeddingRetriever(inner, None)
    out = await r.retrieve(Query(text="hello"), budget=None)
    assert out == "BUNDLE"
    assert inner.seen is not None and inner.seen.embedding is None
