"""
tests/unit/evals/test_temporal_codemath.py
==========================================
WS-date-math — for temporal DELTA questions, the model emits a date SPEC and
CODE computes the delta deterministically (overriding its mental arithmetic),
in a single call, no agent loop. Graceful: a missing/invalid SPEC keeps the
model's free-text answer.

Tests cover the deterministic ``_compute_temporal_from_spec`` (the core) and
the adapter wiring (override vs graceful fallback). Fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import (
    _compute_temporal_from_spec,
    _DirectAnswerAdapter,
    _is_temporal_delta_question,
)

pytestmark = pytest.mark.unit


# ── the deterministic compute function ───────────────────────────────────────


def test_detector() -> None:
    assert _is_temporal_delta_question("How many days passed between A and B?")
    assert _is_temporal_delta_question("How many weeks ago did I meet my aunt?")
    assert not _is_temporal_delta_question("Which three events happened in order?")
    assert not _is_temporal_delta_question("What is my job?")


def test_between_days() -> None:
    out = _compute_temporal_from_spec(
        '7 days.\nSPEC: {"op":"between","unit":"days","dates":["2023-01-24","2023-01-31"]}'
    )
    assert out == "7 days"


def test_ago_weeks() -> None:
    out = _compute_temporal_from_spec(
        'SPEC: {"op":"ago","unit":"weeks","dates":["2023-03-04"],"now":"2023-04-01"}'
    )
    assert out == "4 weeks"  # 28 days → 4 weeks


def test_since_months() -> None:
    out = _compute_temporal_from_spec(
        'SPEC: {"op":"ago","unit":"months","dates":["2023-02-18"],"now":"2023-04-18"}'
    )
    assert out == "2 months"


def test_between_years() -> None:
    out = _compute_temporal_from_spec(
        'SPEC: {"op":"between","unit":"years","dates":["2020-05-01","2023-05-01"]}'
    )
    assert out == "3 years"


def test_no_spec_returns_none() -> None:
    assert _compute_temporal_from_spec("It was about 7 days, I think.") is None


def test_malformed_spec_returns_none() -> None:
    assert _compute_temporal_from_spec("SPEC: {not valid json") is None


def test_incomplete_spec_returns_none() -> None:
    # op=between needs two dates
    assert (
        _compute_temporal_from_spec('SPEC: {"op":"between","unit":"days","dates":["2023-01-01"]}')
        is None
    )


# ── adapter wiring ───────────────────────────────────────────────────────────


def _item(text: str, date: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.STM, metadata={"role": "user", "date": date})


class _FakeRetriever:
    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items

    async def retrieve(self, query: Any, budget: Any) -> Any:
        return SimpleNamespace(items=list(self._items))


class _ScriptLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompt: str | None = None

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompt = prompt
        return self.reply


def _adapter(*, codemath: bool, reply: str):
    session = SimpleNamespace(
        retriever=_FakeRetriever(
            [_item("visited MoMA", "2023-01-24"), _item("exhibit", "2023-01-31")]
        ),
        stm=None,
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session,
        llm=_ScriptLLM(reply),
        answer_max_tokens=64,
        top_k=8,
        max_context_chars=10_000,
        temporal_conditioning=True,
        temporal_codemath=codemath,
    )
    a.dataset_question_type = "temporal-reasoning"  # type: ignore[attr-defined]
    a.dataset_question_date = "2023-02-01"  # type: ignore[attr-defined]
    return a


async def test_code_overrides_model_arithmetic() -> None:
    # Model says "5 days" (wrong) but emits a correct SPEC → code computes 7.
    a = _adapter(
        codemath=True,
        reply='5 days.\nSPEC: {"op":"between","unit":"days","dates":["2023-01-24","2023-01-31"]}',
    )
    out = await a.answer_question("How many days between the two visits?")
    assert out == "7 days"  # code wins over the model's "5 days"
    assert "SPEC:" in a.llm.prompt  # type: ignore[attr-defined]  (prompt asked for a SPEC)
    assert a.last_telemetry["temporal_codemath"] is True


async def test_graceful_fallback_strips_spec_when_invalid() -> None:
    a = _adapter(codemath=True, reply="about 7 days\nSPEC: {broken")
    out = await a.answer_question("How many days between the two visits?")
    assert "SPEC" not in out  # the broken SPEC line is stripped
    assert out == "about 7 days"
    assert a.last_telemetry["temporal_codemath"] is False


async def test_codemath_off_does_not_request_spec() -> None:
    a = _adapter(codemath=False, reply="7 days")
    await a.answer_question("How many days between the two visits?")
    assert "SPEC:" not in a.llm.prompt  # type: ignore[attr-defined]
    assert a.last_telemetry["temporal_codemath"] is False
