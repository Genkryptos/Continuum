"""
tests/unit/evals/test_preference_conditioning.py
================================================
WS-7 — preference-conditioning prompt branch in ``_DirectAnswerAdapter``.

single-session-preference has ~93% recall but ~50% judged: the model sees
the preference but doesn't apply it. When ``preference_conditioning`` is on,
preference-type questions get a Reminder-style prompt (PrefEval's best
method) telling the model to identify and honour a stated preference.

Gated + feature-flagged: it must NEVER fire on factual/temporal/etc.
questions (the over-personalization risk is global injection). These tests
assert exactly that, with fakes — no model, no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import (
    _DirectAnswerAdapter,
    _is_preference_question,
)

pytestmark = pytest.mark.unit

# A sentinel only the preference prompt contains, so we can detect the branch.
_PREF_MARKER = "preference this user has stated"


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


def _adapter(*, preference_conditioning: bool, question_type: str | None) -> _DirectAnswerAdapter:
    session = SimpleNamespace(
        retriever=_FakeRetriever([_item("I edit in Adobe Premiere Pro")]),
        stm=None,
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        preference_conditioning=preference_conditioning,
    )
    a.dataset_question_type = question_type  # type: ignore[attr-defined]
    return a


# ── the wording detector ─────────────────────────────────────────────────────


def test_detector_fires_on_preference_intent() -> None:
    assert _is_preference_question("Can you recommend some video editing resources?")
    assert _is_preference_question("Which laptop should I buy for my work?")
    assert _is_preference_question("Suggest a hotel for my trip")
    # factual / temporal questions must NOT trip it
    assert not _is_preference_question("What year did I graduate?")
    assert not _is_preference_question("How many days between the two meetings?")
    assert not _is_preference_question("What did the assistant say about Postgres?")


# ── branch selection ─────────────────────────────────────────────────────────


async def test_preference_prompt_fires_on_type_when_enabled() -> None:
    a = _adapter(preference_conditioning=True, question_type="single-session-preference")
    await a.answer_question("Anything good to look at?")  # type alone gates it
    assert _PREF_MARKER in a.llm.prompt  # type: ignore[attr-defined]
    assert a.last_telemetry["preference_prompt"] is True


async def test_preference_prompt_fires_on_wording_without_type() -> None:
    a = _adapter(preference_conditioning=True, question_type=None)
    await a.answer_question("Can you recommend some resources to learn editing?")
    assert _PREF_MARKER in a.llm.prompt  # type: ignore[attr-defined]
    assert a.last_telemetry["preference_prompt"] is True


async def test_disabled_flag_never_uses_preference_prompt() -> None:
    # Even a textbook preference question must use the default prompt when off.
    a = _adapter(preference_conditioning=False, question_type="single-session-preference")
    await a.answer_question("Can you recommend some resources?")
    assert _PREF_MARKER not in a.llm.prompt  # type: ignore[attr-defined]
    assert a.last_telemetry["preference_prompt"] is False


async def test_enabled_but_factual_question_uses_default_prompt() -> None:
    # Flag on, but a factual question (no pref type, no pref wording) → default.
    a = _adapter(preference_conditioning=True, question_type="single-session-user")
    await a.answer_question("What is my undergraduate major?")
    assert _PREF_MARKER not in a.llm.prompt  # type: ignore[attr-defined]
    assert a.last_telemetry["preference_prompt"] is False


async def test_preference_prompt_preserves_raw_context() -> None:
    # Additive, not replacement: the retrieved turn still appears in the prompt.
    a = _adapter(preference_conditioning=True, question_type="single-session-preference")
    await a.answer_question("Suggest learning resources")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "Adobe Premiere Pro" in prompt  # raw retrieved context intact
