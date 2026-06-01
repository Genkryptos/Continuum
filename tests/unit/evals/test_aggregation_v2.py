"""
tests/unit/evals/test_aggregation_v2.py
=======================================
WS-2 — strengthened aggregation prompt in ``_DirectAnswerAdapter``.

Multi-session "how many / list all across sessions" questions fail by
counting without enumerating ("3 weddings" -> "1"). WS-2 (``aggregation_v2``)
swaps the aggregate branch's prompt for an explicit
"enumerate every distinct instance -> dedupe -> THEN count" version. Gated to
the same aggregate questions; A/B vs the current aggregation prompt. Fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import _DirectAnswerAdapter

pytestmark = pytest.mark.unit

_V2_MARKER = "enumerate EVERY distinct"
_V1_MARKER = "combine information that"


def _item(text: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.STM, metadata={"role": "user"})


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
        return "ok"


def _adapter(*, aggregation_v2: bool, qtype: str | None):
    session = SimpleNamespace(
        retriever=_FakeRetriever([_item("went to a wedding"), _item("another wedding")]),
        stm=None,
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        aggregation_v2=aggregation_v2,
    )
    a.dataset_question_type = qtype  # type: ignore[attr-defined]
    return a


async def test_v2_prompt_fires_on_aggregate_when_enabled() -> None:
    a = _adapter(aggregation_v2=True, qtype="multi-session")
    await a.answer_question("How many weddings did I attend?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert _V2_MARKER in prompt
    assert _V1_MARKER not in prompt
    assert a.last_telemetry["aggregation_v2"] is True
    assert a.last_telemetry["aggregate_prompt"] is True


async def test_v1_prompt_used_when_disabled() -> None:
    a = _adapter(aggregation_v2=False, qtype="multi-session")
    await a.answer_question("How many weddings did I attend?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert _V1_MARKER in prompt  # current aggregation prompt
    assert _V2_MARKER not in prompt
    assert a.last_telemetry["aggregate_prompt"] is True
    assert a.last_telemetry["aggregation_v2"] is False


async def test_v2_does_not_fire_on_non_aggregate_question() -> None:
    a = _adapter(aggregation_v2=True, qtype="single-session-user")
    await a.answer_question("What is my undergraduate major?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert _V2_MARKER not in prompt
    assert _V1_MARKER not in prompt  # plain default prompt
    assert a.last_telemetry["aggregate_prompt"] is False
    assert a.last_telemetry["aggregation_v2"] is False
