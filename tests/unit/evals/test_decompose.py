"""
tests/unit/evals/test_decompose.py
==================================
Unit tests for ``evals.longmemeval.decompose`` — the question-
decomposition retriever (the multi-session / temporal lever).

All LLM + base-retriever calls are stubbed; tests run offline.
"""

from __future__ import annotations

import pytest

from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, TokenBudget
from evals.longmemeval.decompose import (
    DecompositionRetriever,
    parse_subquestions,
)

# ---------------------------------------------------------------------------
# parse_subquestions
# ---------------------------------------------------------------------------


def test_parse_plain_lines() -> None:
    out = parse_subquestions(
        "When did X start?\nWhen did Y start?",
        original="orig",
    )
    assert out == ["When did X start?", "When did Y start?"]


def test_parse_strips_numbering_and_bullets() -> None:
    reply = "1. First question?\n2) Second question?\n- Third?\n* Fourth?"
    out = parse_subquestions(reply, original="orig")
    assert out == [
        "First question?",
        "Second question?",
        "Third?",
        "Fourth?",
    ]


def test_parse_dedupes_case_insensitively() -> None:
    out = parse_subquestions(
        "What is X?\nwhat is x?\nWhat is Y?",
        original="orig",
    )
    assert out == ["What is X?", "What is Y?"]


def test_parse_caps_at_four() -> None:
    reply = "\n".join(f"q{i}?" for i in range(10))
    out = parse_subquestions(reply, original="orig")
    assert len(out) == 4


def test_parse_empty_falls_back_to_original() -> None:
    assert parse_subquestions("", original="the original") == ["the original"]
    assert parse_subquestions("   \n  \n", original="orig") == ["orig"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, reply: str = "", raise_: bool = False) -> None:
        self.reply = reply
        self.raise_ = raise_
        self.calls: list[str] = []

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.calls.append(prompt)
        if self.raise_:
            raise RuntimeError("llm down")
        return self.reply


class _FakeBaseRetriever:
    """
    Returns a fixed set of items per query text. Lets us verify the
    decomposer retrieves once per sub-question and merges/dedupes.
    """

    def __init__(self, table: dict[str, list[str]]) -> None:
        # table: query text -> list of item ids to return
        self.table = table
        self.session_id = "s"
        self.queries: list[str] = []

    async def retrieve(self, query, budget):
        self.queries.append(query.text)
        ids = self.table.get(query.text, [])
        items = [
            MemoryItem(id=i, content=f"content-{i}", tier=MemoryTier.STM, metadata={"role": "user"})
            for i in ids
        ]
        return ContextBundle(
            items=items,
            messages=[],
            tokens_used=0,
            budget=budget,
            tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
        )


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000,
        stm_reserved=500,
        mtm_reserved=500,
        ltm_reserved=2000,
        response_reserved=500,
    )


def _q(text: str):
    from continuum.core.types import Query

    return Query(text=text)


# ---------------------------------------------------------------------------
# DecompositionRetriever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieves_once_per_subquestion_and_merges() -> None:
    base = _FakeBaseRetriever(
        {
            "When did X start?": ["a", "b"],
            "When did Y start?": ["c", "d"],
        }
    )
    llm = _FakeLLM(reply="When did X start?\nWhen did Y start?")
    r = DecompositionRetriever(base=base, llm=llm)

    ctx = await r.retrieve(_q("Did X start before Y?"), _budget())

    # Base retriever was hit once per sub-question.
    assert base.queries == ["When did X start?", "When did Y start?"]
    # Merged union of both sub-question hits.
    assert {it.id for it in ctx.items} == {"a", "b", "c", "d"}
    assert ctx.debug_info["retrieval_mode"] == "decompose"
    assert ctx.debug_info["n_sub_questions"] == 2
    assert ctx.debug_info["per_subquestion_hits"] == [2, 2]


@pytest.mark.asyncio
async def test_merge_dedupes_overlapping_hits() -> None:
    # Both sub-questions retrieve item "shared".
    base = _FakeBaseRetriever(
        {
            "Q1?": ["shared", "x"],
            "Q2?": ["shared", "y"],
        }
    )
    llm = _FakeLLM(reply="Q1?\nQ2?")
    r = DecompositionRetriever(base=base, llm=llm)
    ctx = await r.retrieve(_q("compound"), _budget())
    ids = [it.id for it in ctx.items]
    assert ids.count("shared") == 1  # deduped
    assert set(ids) == {"shared", "x", "y"}


@pytest.mark.asyncio
async def test_simple_question_single_subquestion() -> None:
    # LLM decides the question is already atomic — returns it unchanged.
    base = _FakeBaseRetriever({"What is my name?": ["n1"]})
    llm = _FakeLLM(reply="What is my name?")
    r = DecompositionRetriever(base=base, llm=llm)
    ctx = await r.retrieve(_q("What is my name?"), _budget())
    assert base.queries == ["What is my name?"]  # retrieved once
    assert ctx.debug_info["n_sub_questions"] == 1


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_plain_retrieval() -> None:
    base = _FakeBaseRetriever({"the whole question": ["z"]})
    llm = _FakeLLM(raise_=True)
    r = DecompositionRetriever(base=base, llm=llm)
    ctx = await r.retrieve(_q("the whole question"), _budget())
    # Decomposition failed → retrieve once with the original question.
    assert base.queries == ["the whole question"]
    assert [it.id for it in ctx.items] == ["z"]


@pytest.mark.asyncio
async def test_max_items_caps_merged_context() -> None:
    base = _FakeBaseRetriever(
        {
            "Q1?": [f"a{i}" for i in range(10)],
            "Q2?": [f"b{i}" for i in range(10)],
        }
    )
    llm = _FakeLLM(reply="Q1?\nQ2?")
    r = DecompositionRetriever(base=base, llm=llm, max_items=12)
    ctx = await r.retrieve(_q("big compound"), _budget())
    assert len(ctx.items) == 12  # 20 unique → capped at 12


@pytest.mark.asyncio
async def test_base_retrieve_error_is_isolated() -> None:
    """One sub-question's retrieval failing shouldn't sink the others."""

    class _PartlyBroken(_FakeBaseRetriever):
        async def retrieve(self, query, budget):
            if query.text == "BOOM?":
                raise RuntimeError("retrieve broke")
            return await super().retrieve(query, budget)

    base = _PartlyBroken({"OK?": ["good"]})
    llm = _FakeLLM(reply="BOOM?\nOK?")
    r = DecompositionRetriever(base=base, llm=llm)
    ctx = await r.retrieve(_q("compound"), _budget())
    # The broken sub-question contributed nothing; the OK one survived.
    assert [it.id for it in ctx.items] == ["good"]
    assert ctx.debug_info["per_subquestion_hits"] == [0, 1]


@pytest.mark.asyncio
async def test_session_id_mirrors_base() -> None:
    base = _FakeBaseRetriever({})
    base.session_id = "user-42:chat-9"
    r = DecompositionRetriever(base=base, llm=_FakeLLM())
    assert r.session_id == "user-42:chat-9"
