"""
tests/unit/evals/test_temporal_conditioning.py
==============================================
WS-1 — temporal-reasoning prompt branch in ``_DirectAnswerAdapter``.

Measured failure: the direct context built ``[role] content`` and DROPPED the
turn dates, so the model answered "the conversation doesn't include any dates"
/ "0". When ``temporal_conditioning`` is on, temporal questions get each turn
PREFIXED with its date + the reference "now", and a prompt to compute the
delta. Gated + flagged so non-temporal questions are untouched. Fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import (
    _DirectAnswerAdapter,
    _is_temporal_question,
)

pytestmark = pytest.mark.unit

_TEMPORAL_MARKER = "prefixed with the date"


def _item(text: str, date: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.STM, metadata={"role": "user", "date": date})


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


def _adapter(*, temporal_conditioning: bool, qtype: str | None, qdate: str | None = "2023-02-01"):
    session = SimpleNamespace(
        retriever=_FakeRetriever(
            [
                _item("visited MoMA", "2023-01-24"),
                _item("Ancient Civilizations exhibit", "2023-01-31"),
            ]
        ),
        stm=None,
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        temporal_conditioning=temporal_conditioning,
    )
    a.dataset_question_type = qtype  # type: ignore[attr-defined]
    a.dataset_question_date = qdate  # type: ignore[attr-defined]
    return a


# ── detector ─────────────────────────────────────────────────────────────────


def test_detector_fires_on_temporal_intent() -> None:
    assert _is_temporal_question("How many days passed between the two visits?")
    assert _is_temporal_question("How many weeks ago did I meet my aunt?")
    assert _is_temporal_question("Which three events happened in the order from first to last?")
    assert not _is_temporal_question("What is my undergraduate major?")
    assert not _is_temporal_question("Can you recommend a hotel?")


# ── branch + date surfacing ──────────────────────────────────────────────────


async def test_temporal_prompt_surfaces_dates_and_reference_now() -> None:
    a = _adapter(temporal_conditioning=True, qtype="temporal-reasoning")
    await a.answer_question("How many days between the two visits?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert _TEMPORAL_MARKER in prompt
    assert "Today's date is 2023-02-01" in prompt  # reference "now" injected
    assert "[2023-01-24]" in prompt and "[2023-01-31]" in prompt  # turn dates surfaced
    assert a.last_telemetry["temporal_prompt"] is True


async def test_temporal_fires_on_wording_without_type() -> None:
    a = _adapter(temporal_conditioning=True, qtype=None)
    await a.answer_question("How many weeks ago did the exhibit open?")
    assert _TEMPORAL_MARKER in a.llm.prompt  # type: ignore[attr-defined]


async def test_disabled_flag_drops_dates_like_before() -> None:
    a = _adapter(temporal_conditioning=False, qtype="temporal-reasoning")
    await a.answer_question("How many days between the two visits?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert _TEMPORAL_MARKER not in prompt
    assert "[2023-01-24]" not in prompt  # dates NOT surfaced when off (baseline)
    assert a.last_telemetry["temporal_prompt"] is False


async def test_enabled_but_non_temporal_question_unchanged() -> None:
    a = _adapter(temporal_conditioning=True, qtype="single-session-user")
    await a.answer_question("What is my undergraduate major?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert _TEMPORAL_MARKER not in prompt
    assert "[2023-01-24]" not in prompt  # no date prefix on non-temporal
    assert a.last_telemetry["temporal_prompt"] is False


async def test_missing_reference_date_still_surfaces_turn_dates() -> None:
    a = _adapter(temporal_conditioning=True, qtype="temporal-reasoning", qdate=None)
    await a.answer_question("How many days between the visits?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "Today's date is" not in prompt  # gracefully omitted
    assert "[2023-01-24]" in prompt  # turn dates still surfaced
