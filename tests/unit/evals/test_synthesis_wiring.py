"""
tests/unit/evals/test_synthesis_wiring.py
=========================================
v3 wiring: _DirectAnswerAdapter runs ingest-time synthesis and injects the
code-computed counts into the prompt for COUNTING questions only. Fakes
throughout (no LLM, no network, no eval run) — validates the plumbing added to
bootstrap_ollama, complementing the pure-logic tests in test_synthesis.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import _DirectAnswerAdapter

pytestmark = pytest.mark.unit


class _FakeSTM:
    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    async def append(self, item: MemoryItem) -> None:
        self.items.append(item)


class _FakeRetriever:
    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items

    async def retrieve(self, query: Any, budget: Any) -> Any:
        return SimpleNamespace(items=list(self._items))


class _CaptureLLM:
    def __init__(self) -> None:
        self.prompt: str | None = None

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompt = prompt
        return "answer"


def _triples_llm(reply: str) -> Any:
    async def fn(prompt: str) -> str:
        return reply

    return fn


def _turn(text: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.STM, metadata={"role": "user"})


def _adapter(synthesis_fn: Any) -> _DirectAnswerAdapter:
    session = SimpleNamespace(
        stm=_FakeSTM(),
        retriever=_FakeRetriever([_turn("I have a goldfish tank")]),
        session_id="s",
    )
    return _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        synthesis_fn=synthesis_fn,
    )


_THREE_TANKS = (
    '{"facts": ['
    '{"predicate": "tank", "object": "goldfish tank"},'
    '{"predicate": "tank", "object": "shrimp tank"},'
    '{"predicate": "tank", "object": "betta tank"}]}'
)

_MSGS = [
    {
        "role": "user",
        "content": "I set up a goldfish tank",
        "session_id": "s1",
        "date": "2023-01-01",
    },
    {"role": "user", "content": "added a shrimp tank", "session_id": "s1", "date": "2023-01-02"},
    {"role": "user", "content": "and a betta tank", "session_id": "s1", "date": "2023-01-03"},
]


async def test_synthesis_builds_counts_at_ingest() -> None:
    a = _adapter(_triples_llm(_THREE_TANKS))
    await a.process_conversation(_MSGS)
    assert [f.text for f in a._synthesis_facts] == ["User has 3 tanks."]
    assert len(a.session.stm.items) == 3  # normal ingest still happened


async def test_counting_question_injects_computed_facts() -> None:
    a = _adapter(_triples_llm(_THREE_TANKS))
    await a.process_conversation(_MSGS)
    await a.answer_question("How many tanks do I have?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "COMPUTED FACTS" in prompt
    assert "User has 3 tanks." in prompt
    assert a.last_telemetry["synthesis_injected"] is True


async def test_non_counting_question_does_not_inject() -> None:
    a = _adapter(_triples_llm(_THREE_TANKS))
    await a.process_conversation(_MSGS)
    await a.answer_question("What is my favorite tank?")  # not a counting question
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "COMPUTED FACTS" not in prompt
    assert a.last_telemetry["synthesis_injected"] is False


async def test_no_synthesis_fn_is_inert() -> None:
    a = _adapter(synthesis_fn=None)
    await a.process_conversation(_MSGS)
    await a.answer_question("How many tanks do I have?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert a._synthesis_facts == []
    assert "COMPUTED FACTS" not in prompt
    assert a.last_telemetry["synthesis_injected"] is False
