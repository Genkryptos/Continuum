"""
tests/unit/evals/test_direct_rerank.py
======================================
WS-4 — the cross-encoder reranker is now wired into ``_DirectAnswerAdapter``
(the v1 *winning* path), not just the optimizing path.

These tests prove the wiring with fakes (no model, no network):
* when the retrieved pool is larger than ``top_k``, the reranker reorders it
  and the *reranked* top_k (not the raw retrieval order) reaches the prompt;
* reranking is skipped when the pool already fits in ``top_k``;
* a reranker failure falls back to retrieval order (never breaks answering).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import _DirectAnswerAdapter

pytestmark = pytest.mark.unit


def _item(tag: str) -> MemoryItem:
    return MemoryItem(content=f"turn-{tag}", tier=MemoryTier.STM, metadata={"role": "user"})


class _FakeRetriever:
    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items

    async def retrieve(self, query: Any, budget: Any) -> Any:
        return SimpleNamespace(items=list(self._items))


class _CaptureLLM:
    """Records the prompt it was asked to complete."""

    def __init__(self) -> None:
        self.prompt: str | None = None

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompt = prompt
        return "ok"


class _ReverseReranker:
    """Reorders the pool (reverse) and returns the top_k — a deterministic
    stand-in for the cross-encoder so we can assert ordering changed."""

    def __init__(self) -> None:
        self.called = False

    async def rerank(self, query: str, items: list[MemoryItem], *, top_k: int) -> list[MemoryItem]:
        self.called = True
        return list(reversed(items))[:top_k]


class _BoomReranker:
    async def rerank(self, query: str, items: list[MemoryItem], *, top_k: int) -> list[MemoryItem]:
        raise RuntimeError("model exploded")


def _adapter(retriever: _FakeRetriever, reranker: Any, *, top_k: int) -> _DirectAnswerAdapter:
    session = SimpleNamespace(retriever=retriever, stm=None, session_id="s")
    return _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=top_k,
        max_context_chars=10_000,
        reranker=reranker,
    )


async def test_rerank_reorders_pool_into_prompt() -> None:
    # pool A,B,C,D,E ; top_k=2. Raw order → A,B. Reverse reranker → E,D.
    pool = [_item(t) for t in "ABCDE"]
    rer = _ReverseReranker()
    adapter = _adapter(_FakeRetriever(pool), rer, top_k=2)
    await adapter.answer_question("q?")
    prompt = adapter.llm.prompt  # type: ignore[attr-defined]
    assert rer.called is True
    assert adapter.last_telemetry["reranked"] is True
    # The reranked winners (E, D) are in the prompt; the raw-order head (A, B) is not.
    assert "turn-E" in prompt and "turn-D" in prompt
    assert "turn-A" not in prompt and "turn-B" not in prompt


async def test_rerank_skipped_when_pool_fits_top_k() -> None:
    pool = [_item("A"), _item("B")]
    rer = _ReverseReranker()
    adapter = _adapter(_FakeRetriever(pool), rer, top_k=2)  # len(pool) == top_k → skip
    await adapter.answer_question("q?")
    assert rer.called is False
    assert adapter.last_telemetry["reranked"] is False


async def test_rerank_failure_falls_back_to_retrieval_order() -> None:
    pool = [_item(t) for t in "ABCDE"]
    adapter = _adapter(_FakeRetriever(pool), _BoomReranker(), top_k=2)
    answer = await adapter.answer_question("q?")
    prompt = adapter.llm.prompt  # type: ignore[attr-defined]
    # Reranker blew up → retrieval-order head (A, B) is used, answering still works.
    assert answer == "ok"
    assert adapter.last_telemetry["reranked"] is False
    assert "turn-A" in prompt and "turn-B" in prompt


async def test_no_reranker_uses_retrieval_order() -> None:
    pool = [_item(t) for t in "ABCDE"]
    adapter = _adapter(_FakeRetriever(pool), None, top_k=2)
    await adapter.answer_question("q?")
    prompt = adapter.llm.prompt  # type: ignore[attr-defined]
    assert adapter.last_telemetry["reranked"] is False
    assert "turn-A" in prompt and "turn-B" in prompt


async def test_rerank_to_controls_keep_count_not_top_k() -> None:
    # The bug WS-4 fixed: rerank kept top_k (so with top_k >= pool it was a
    # no-op). Now rerank keeps `rerank_to`. pool=10, top_k=8, rerank_to=3 →
    # reranks the pool and keeps the best 3 (reverse reranker → J, I, H).
    pool = [_item(t) for t in "ABCDEFGHIJ"]
    session = SimpleNamespace(retriever=_FakeRetriever(pool), stm=None, session_id="s")
    adapter = _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        reranker=_ReverseReranker(),
        rerank_to=3,
    )
    await adapter.answer_question("q?")
    prompt = adapter.llm.prompt  # type: ignore[attr-defined]
    assert adapter.last_telemetry["reranked"] is True
    assert adapter.last_telemetry["retrieved_items"] == 3
    assert "turn-J" in prompt and "turn-I" in prompt and "turn-H" in prompt
    assert "turn-A" not in prompt and "turn-G" not in prompt
