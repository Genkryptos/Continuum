from __future__ import annotations

import numpy as np
import pytest

from continuum.core.types import MemoryItem, MemoryTier, Query, TokenBudget
from evals.longmemeval.bootstrap_ollama import (
    FlatHaystackStore,
    WikiMemoryRetriever,
    make_adapter_factory,
)


class _KeywordEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            t = text.lower()
            v = np.zeros(5, dtype=np.float32)
            if "favorite" in t or "color" in t or "blue" in t:
                v[0] = 1.0
            if "fitbit" in t or "march" in t:
                v[1] = 1.0
            if "december" in t:
                v[2] = 1.0
            if "session" in t:
                v[3] = 1.0
            if not v.any():
                v[4] = 1.0
            rows.append(v)
        return np.stack(rows, axis=0)


class _NeedleDilutionEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            t = text.lower()
            v = np.zeros(3, dtype=np.float32)
            if t.startswith("what exact needle"):
                v[0] = 1.0
            elif "needle fact" in t and len(t.split()) <= 8:
                v[0] = 1.0
            elif "noise" in t:
                v[1] = 1.0
            else:
                v[2] = 1.0
            rows.append(v)
        return np.stack(rows, axis=0)


class _FlatEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 1), dtype=np.float32)


class _MisleadingCommuteEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            t = text.lower()
            v = np.zeros(2, dtype=np.float32)
            if "audiobooks during my daily commute" in t and "45 minutes" not in t:
                v[0] = 1.0
            elif "how long is my daily commute" in t:
                v[0] = 1.0
            else:
                v[1] = 1.0
            rows.append(v)
        return np.stack(rows, axis=0)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000,
        stm_reserved=500,
        mtm_reserved=500,
        ltm_reserved=2000,
        response_reserved=500,
    )


async def _append(
    store: FlatHaystackStore,
    *,
    item_id: str,
    session_id: str,
    content: str,
    role: str = "user",
) -> None:
    await store.append(
        MemoryItem(
            id=item_id,
            content=content,
            tier=MemoryTier.STM,
            metadata={"role": role, "session_id": session_id},
        )
    )


@pytest.mark.asyncio
async def test_wiki_memory_retriever_builds_markdown_pages_and_retrieves_by_query() -> None:
    store = FlatHaystackStore()
    await _append(
        store,
        item_id="s1-a",
        session_id="s1",
        content="My favorite color is blue.",
    )
    await _append(
        store,
        item_id="s2-a",
        session_id="s2",
        content="I started using the Fitbit Charge 3 in March.",
    )
    await _append(
        store,
        item_id="s3-a",
        session_id="s3",
        content="It is now December.",
    )

    retriever = WikiMemoryRetriever(
        raw_store=store,
        embedder=_KeywordEmbedder(),
        top_k=3,
    )

    ctx = await retriever.retrieve(Query(text="What is my favorite color?"), _budget())

    assert ctx.debug_info["retrieval_mode"] == "wiki_memory"
    assert ctx.debug_info["wiki_pages"] >= 4
    assert any(
        item.metadata.get("wiki_page") == "sessions/s1.md" for item in ctx.items
    )
    assert any("favorite color is blue" in item.content.lower() for item in ctx.items)
    retrieved_session_ids = {
        sid
        for item in ctx.items
        for sid in item.metadata.get("session_ids", [])
    }
    assert "s1" in retrieved_session_ids
    assert "s2" not in retrieved_session_ids
    assert "s3" not in retrieved_session_ids


def test_factory_uses_wiki_memory_retriever_when_enabled() -> None:
    class LLM:
        async def complete(self, *, prompt: str, max_tokens: int) -> str:
            return "answer"

    factory = make_adapter_factory(
        llm=LLM(),
        embedder=_KeywordEmbedder(),
        wiki_memory=True,
        wiki_top_k=5,
    )

    adapter = factory()

    assert isinstance(adapter.session.retriever, WikiMemoryRetriever)
    assert adapter.session.retriever.top_k == 5


@pytest.mark.asyncio
async def test_wiki_memory_uses_raw_turn_scores_to_rescue_source_session_pages() -> None:
    store = FlatHaystackStore()
    await _append(
        store,
        item_id="other-a",
        session_id="other-session",
        content="noise unrelated",
    )
    await _append(
        store,
        item_id="answer-a",
        session_id="answer-session",
        content="Needle fact.",
    )
    for idx in range(30):
        await _append(
            store,
            item_id=f"answer-noise-{idx}",
            session_id="answer-session",
            content=f"noise filler {idx}",
        )

    retriever = WikiMemoryRetriever(
        raw_store=store,
        embedder=_NeedleDilutionEmbedder(),
        top_k=3,
    )

    ctx = await retriever.retrieve(Query(text="What exact needle?"), _budget())

    assert ctx.items[0].metadata["wiki_page_type"] == "evidence_window"
    assert "answer-session" in ctx.items[0].metadata["session_ids"]
    assert any(
        item.metadata["wiki_page"] == "sessions/answer-session.md"
        for item in ctx.items
    )


@pytest.mark.asyncio
async def test_wiki_memory_prepends_query_focused_evidence_window() -> None:
    store = FlatHaystackStore()
    await _append(
        store,
        item_id="target-app",
        session_id="answer-session",
        content="I use the Target app for household coupons.",
    )
    await _append(
        store,
        item_id="assistant-context",
        session_id="answer-session",
        content="Target sends exclusive coupons to email subscribers.",
        role="assistant",
    )
    await _append(
        store,
        item_id="coupon",
        session_id="answer-session",
        content="I redeemed a $5 coupon on coffee creamer last Sunday.",
    )
    await _append(
        store,
        item_id="noise",
        session_id="other-session",
        content="noise unrelated",
    )

    retriever = WikiMemoryRetriever(
        raw_store=store,
        embedder=_NeedleDilutionEmbedder(),
        top_k=4,
    )

    ctx = await retriever.retrieve(
        Query(text="Where did I redeem a $5 coupon on coffee creamer?"),
        _budget(),
    )

    assert ctx.items[0].metadata["wiki_page_type"] == "evidence_window"
    assert "Target app" in ctx.items[0].content
    assert "$5 coupon on coffee creamer" in ctx.items[0].content
    assert ctx.items[0].metadata["session_ids"] == ["answer-session"]


@pytest.mark.asyncio
async def test_wiki_memory_evidence_window_uses_lexical_overlap_for_exact_user_fact() -> None:
    store = FlatHaystackStore()
    await _append(
        store,
        item_id="related-noise",
        session_id="answer-session",
        content="I enjoy audiobooks during my daily commute.",
    )
    for idx in range(4):
        await _append(
            store,
            item_id=f"filler-{idx}",
            session_id="answer-session",
            content=f"Unrelated filler turn {idx}.",
        )
    await _append(
        store,
        item_id="exact-fact",
        session_id="answer-session",
        content="I've been listening to audiobooks during my daily commute, which takes 45 minutes each way.",
    )

    retriever = WikiMemoryRetriever(
        raw_store=store,
        embedder=_FlatEmbedder(),
        top_k=4,
    )

    ctx = await retriever.retrieve(Query(text="How long is my daily commute?"), _budget())

    assert ctx.items[0].metadata["wiki_page_type"] == "evidence_window"
    assert "45 minutes each way" in ctx.items[0].content


@pytest.mark.asyncio
async def test_wiki_memory_lexical_overlap_can_override_misleading_semantic_match() -> None:
    store = FlatHaystackStore()
    await _append(
        store,
        item_id="semantic-noise",
        session_id="answer-session",
        content="I enjoy audiobooks during my daily commute.",
    )
    for idx in range(4):
        await _append(
            store,
            item_id=f"filler-misleading-{idx}",
            session_id="answer-session",
            content=f"Unrelated filler turn {idx}.",
        )
    await _append(
        store,
        item_id="exact-commute-duration",
        session_id="answer-session",
        content="My daily commute takes 45 minutes each way.",
    )

    retriever = WikiMemoryRetriever(
        raw_store=store,
        embedder=_MisleadingCommuteEmbedder(),
        top_k=4,
    )

    ctx = await retriever.retrieve(Query(text="How long is my daily commute?"), _budget())

    assert "45 minutes each way" in ctx.items[0].content
    assert "## Matched Turn" in ctx.items[0].content
    assert ctx.items[0].content.index("45 minutes each way") < ctx.items[0].content.index("## Nearby Context")
