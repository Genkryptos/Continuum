"""
tests/unit/evals/test_ku_recency.py
===================================
WS-3 — knowledge-update recency conditioning in ``_DirectAnswerAdapter``.

Continuum's supersession already removes stale LTM facts (retrieved
``source="ltm_fact"`` items are current-only), but the old RAW TURN is still
in context and distracts the reader. WS-3 (``knowledge_update_conditioning``):
for KU questions, sort current LTM facts to the front, mark them
``[CURRENT FACT]``, and prompt the reader to prefer current state. Gated +
flagged. Fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import (
    _DirectAnswerAdapter,
    _is_knowledge_update_question,
)

pytestmark = pytest.mark.unit

_KU_MARKER = "outdated versions have already been removed"
_FACT_TAG = "[CURRENT FACT]"


def _turn(text: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.STM, metadata={"role": "user"})


def _fact(text: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.LTM, metadata={"source": "ltm_fact"})


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


def _adapter(*, ku: bool, qtype: str | None, items: list[MemoryItem]):
    session = SimpleNamespace(retriever=_FakeRetriever(items), stm=None, session_id="s")
    a = _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        knowledge_update_conditioning=ku,
    )
    a.dataset_question_type = qtype  # type: ignore[attr-defined]
    return a


# ── detector ─────────────────────────────────────────────────────────────────


def test_detector() -> None:
    assert _is_knowledge_update_question("What is my current job?")
    assert _is_knowledge_update_question("Where do I live these days?")
    assert _is_knowledge_update_question("Do I still own the blue car?")
    assert not _is_knowledge_update_question("What did I study in college?")


# ── branch behaviour ─────────────────────────────────────────────────────────


async def test_ku_marks_and_prioritises_current_fact() -> None:
    # raw stale turn first in retrieval order, current fact last → after WS-3
    # the fact should be marked AND moved ahead of the stale turn.
    items = [_turn("I live in NYC"), _fact("user lives in Boston")]
    a = _adapter(ku=True, qtype="knowledge-update", items=items)
    await a.answer_question("Where do I currently live?")
    p = a.llm.prompt  # type: ignore[attr-defined]
    assert _KU_MARKER in p
    assert f"{_FACT_TAG} user lives in Boston" in p
    assert a.last_telemetry["ku_prompt"] is True
    # current-first: the marked fact appears before the stale NYC turn
    assert p.index("user lives in Boston") < p.index("I live in NYC")


async def test_disabled_flag_leaves_context_unchanged() -> None:
    items = [_turn("I live in NYC"), _fact("user lives in Boston")]
    a = _adapter(ku=False, qtype="knowledge-update", items=items)
    await a.answer_question("Where do I currently live?")
    p = a.llm.prompt  # type: ignore[attr-defined]
    assert _KU_MARKER not in p
    assert _FACT_TAG not in p
    assert a.last_telemetry["ku_prompt"] is False
    # retrieval order preserved (NYC turn first, as returned)
    assert p.index("I live in NYC") < p.index("user lives in Boston")


async def test_non_ku_question_unaffected() -> None:
    items = [_turn("I live in NYC"), _fact("user lives in Boston")]
    a = _adapter(ku=True, qtype="single-session-user", items=items)
    await a.answer_question("What is my undergraduate major?")
    p = a.llm.prompt  # type: ignore[attr-defined]
    assert _KU_MARKER not in p
    assert _FACT_TAG not in p
    assert a.last_telemetry["ku_prompt"] is False
