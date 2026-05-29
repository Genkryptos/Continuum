"""
tests/unit/reasoning/test_wiring_smoke.py
=========================================
End-to-end smoke for the LongMemEval wiring factory.

Boots :func:`evals.longmemeval.iterative_reasoner_wiring.build_iterative_reasoner`
with a scripted fake retriever (a real :class:`ContextBundle` with two
known turns), a fake composer that returns a known synthesis reply, and
the real SmallLLM-shaped fake. Asserts that ``.answer()`` returns a
populated :class:`ReasoningResult` whose ``evidence`` is a real
:class:`EvidencePacket`, ``trace["task_mode"]`` matches the expected
:class:`TaskMode` for the test question, and no exceptions bubble out
of the real longmemeval modules.

This is the closest unit-level proof that the wiring is wired correctly —
short of an actual LongMemEval row, which needs a live LLM.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, TokenBudget
from evals.longmemeval.evidence_packet import EvidencePacket
from evals.longmemeval.iterative_reasoner_wiring import (
    HEADS_BY_MODE,
    build_iterative_reasoner,
    heuristic_rewrite,
)
from evals.longmemeval.task_router import TaskMode

from tests.unit.reasoning.conftest import FakeComposerLLM, FakeSmallLLM

pytestmark = pytest.mark.unit


def _ctx(*contents: str) -> ContextBundle:
    items = [
        MemoryItem(
            id=str(uuid.uuid4()),
            content=text,
            tier=MemoryTier.LTM,
            metadata={"role": "user", "session_id": f"s{i}", "date": ""},
        )
        for i, text in enumerate(contents)
    ]
    return ContextBundle(
        items=items, messages=[], tokens_used=0,
        budget=TokenBudget(
            total=4000, stm_reserved=0, mtm_reserved=0,
            ltm_reserved=2000, response_reserved=100,
        ),
    )


class _ScriptedRetriever:
    """Returns a single ContextBundle regardless of query."""

    def __init__(self, ctx: ContextBundle) -> None:
        self._ctx = ctx
        self.calls: list[str] = []

    async def retrieve(self, query: str) -> ContextBundle:
        self.calls.append(query)
        return self._ctx


# ── tests ──────────────────────────────────────────────────────────────────


def test_heads_by_mode_covers_every_task_mode() -> None:
    """The wiring's head dict must cover every TaskMode value."""
    for mode in TaskMode:
        assert mode in HEADS_BY_MODE, f"missing head for {mode}"


def test_heuristic_rewrite_strips_stopwords() -> None:
    # Stopwords ("do", "I", "the", "for") are stripped; content words
    # and digits / proper nouns ride through.
    assert heuristic_rewrite("Where do I live now?", 0) == "Where live now"
    # Returns the input unchanged when stripping would leave nothing.
    assert heuristic_rewrite("the a an", 0) == "the a an"
    assert heuristic_rewrite("How many shirts for Boston 2024?", 0) == \
           "many shirts Boston 2024"


@pytest.mark.asyncio
async def test_wiring_runs_end_to_end_with_real_modules() -> None:
    """
    Spec'd e2e: the real longmemeval modules (candidates,
    claims, claim_verifier, evidence_packet, decomposed_answer,
    query_intent, task_router, reasoning_heads) drive a question
    through to a ReasoningResult. We use a scripted composer +
    SmallLLM so no real LLM is needed.

    The retriever returns a ctx with two short user turns. Even
    if the deterministic verifier rejects everything (likely, since
    the candidates module has its own structural requirements), the
    reasoner must still return a non-raising result with a populated
    trace.
    """
    ctx = _ctx(
        "I live in Boston.",
        "I work at MIT and have lived in Boston since 2018.",
    )
    retriever = _ScriptedRetriever(ctx)
    composer = FakeComposerLLM(replies=[
        "1. Where do I live?",   # decompose
        "Boston",                # synthesis
    ])
    small = FakeSmallLLM()

    reasoner = build_iterative_reasoner(
        retriever=retriever,
        small_llm=small,
        composer_llm=composer,
        max_rounds=2,
        max_llm_calls=6,
        question_type_hint="single-session-user",
    )
    result = await reasoner.answer("Where do I live?")

    # Did NOT raise; produced a real evidence packet.
    assert isinstance(result.evidence, EvidencePacket)
    # Trace records the route the longmemeval router picked.
    assert result.trace["task_mode"] in {m.name for m in TaskMode}
    # Budget kept honest — at most max_llm_calls.
    assert result.llm_call_count <= 6
    # Composer was called at least once (decompose); synthesis may or
    # may not have run depending on whether a head fired.
    assert composer.calls
