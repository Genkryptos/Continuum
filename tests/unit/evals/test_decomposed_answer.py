from __future__ import annotations

import pytest

from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, Query, TokenBudget
from evals.longmemeval.bootstrap_ollama import (
    FlatHaystackStore,
    SessionAwareSemanticRetriever,
    _DecomposedAnsweringAdapter,
    make_adapter_factory,
)
from evals.longmemeval.decomposed_answer import (
    SubAnswer,
    build_final_synthesis_prompt,
    build_subanswer_prompt,
    extract_session_ids,
)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000,
        stm_reserved=500,
        mtm_reserved=500,
        ltm_reserved=2000,
        response_reserved=500,
    )


def _ctx() -> ContextBundle:
    return ContextBundle(
        items=[
            MemoryItem(
                id="a",
                content="I started using the Fitbit Charge 3 in March.",
                tier=MemoryTier.STM,
                metadata={"role": "user", "session_id": "s1"},
            ),
            MemoryItem(
                id="b",
                content="It is now December, so I have used it for 9 months.",
                tier=MemoryTier.STM,
                metadata={"role": "assistant", "session_id": "s2"},
            ),
        ],
        messages=[],
        tokens_used=20,
        budget=_budget(),
        tier_breakdown={"stm": 20, "mtm": 0, "ltm": 0},
    )


def test_extract_session_ids_dedupes_in_order() -> None:
    ctx = _ctx()
    ctx.items.append(
        MemoryItem(
            id="c",
            content="duplicate session",
            tier=MemoryTier.STM,
            metadata={"session_id": "s1"},
        )
    )

    assert extract_session_ids(ctx) == ["s1", "s2"]


def test_build_subanswer_prompt_uses_only_subquestion_and_context() -> None:
    prompt = build_subanswer_prompt("How long have I used the Fitbit?", _ctx())

    assert "Sub-question: How long have I used the Fitbit?" in prompt
    assert "I started using the Fitbit Charge 3 in March." in prompt
    assert "Answer the sub-question using only the evidence." in prompt
    assert "Original question:" not in prompt


def test_build_final_synthesis_prompt_contains_structured_subanswers() -> None:
    prompt = build_final_synthesis_prompt(
        "How long have I used the Fitbit?",
        [
            SubAnswer(
                subquestion="When did I start using the Fitbit?",
                answer="March",
                evidence_session_ids=["s1"],
                evidence_text="I started using the Fitbit Charge 3 in March.",
                hit_count=1,
            ),
            SubAnswer(
                subquestion="What month is it now?",
                answer="December",
                evidence_session_ids=["s2"],
                evidence_text="It is now December.",
                hit_count=1,
            ),
        ],
    )

    assert "Original question: How long have I used the Fitbit?" in prompt
    assert "Sub-question 1: When did I start using the Fitbit?" in prompt
    assert "Sub-answer 1: March" in prompt
    assert "Sessions 1: s1" in prompt
    assert "Return only the final answer" in prompt


def test_final_synthesis_prompt_uses_evidence_when_subanswers_miss_fact() -> None:
    prompt = build_final_synthesis_prompt(
        "What is the name of the playlist I created on Spotify?",
        [
            SubAnswer(
                subquestion="What is the name of the playlist?",
                answer="I don't know.",
                evidence_session_ids=["answer_3e012175"],
                evidence_text=(
                    "I've been listening to this one playlist on Spotify "
                    "that I created, called Summer Vibes."
                ),
                hit_count=1,
            ),
        ],
    )

    assert "Use the evidence blocks directly" in prompt
    assert "Sub-answers are intermediate notes and may be incomplete" in prompt
    assert 'If the sub-answers are insufficient, say "I don\'t know"' not in prompt


class _FakeRetriever:
    def __init__(self, table: dict[str, ContextBundle]) -> None:
        self.table = table
        self.queries: list[str] = []

    async def retrieve(self, query, budget):
        self.queries.append(query.text)
        return self.table.get(
            query.text,
            ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
            ),
        )


class _FakeSession:
    def __init__(self, retriever: _FakeRetriever) -> None:
        self.retriever = retriever


class _ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("unexpected LLM call")
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_decomposed_adapter_answers_subquestions_then_synthesizes() -> None:
    retriever = _FakeRetriever(
        {
            "When did I start using the Fitbit?": _ctx(),
            "What month is it now?": _ctx(),
        }
    )
    llm = _ScriptedLLM(
        [
            "When did I start using the Fitbit?\nWhat month is it now?",
            "March",
            "December",
            "9 months",
        ]
    )
    adapter = _DecomposedAnsweringAdapter(
        session=_FakeSession(retriever),
        llm=llm,
        answer_max_tokens=100,
        subanswer_max_tokens=40,
    )

    answer = await adapter.answer_question("How long have I used the Fitbit?")

    assert answer == "9 months"
    assert retriever.queries == [
        "When did I start using the Fitbit?",
        "What month is it now?",
    ]
    assert adapter.last_ctx is not None
    assert len(adapter.last_ctx.items) == 2
    assert adapter.last_decomposition_stats["n_sub_questions"] == 2
    assert adapter.last_decomposition_stats["sub_answers"] == ["March", "December"]


@pytest.mark.asyncio
async def test_decomposed_adapter_atomic_question_uses_normal_answer_path() -> None:
    retriever = _FakeRetriever({"What degree did I graduate with?": _ctx()})
    llm = _ScriptedLLM(
        [
            "What degree did I graduate with?",
            "Business Administration",
        ]
    )
    adapter = _DecomposedAnsweringAdapter(
        session=_FakeSession(retriever),
        llm=llm,
        answer_max_tokens=100,
    )

    answer = await adapter.answer_question("What degree did I graduate with?")

    assert answer == "Business Administration"
    assert retriever.queries == ["What degree did I graduate with?"]
    assert adapter.last_decomposition_stats["mode"] == "atomic_fallback"


class _FakeEmbedder:
    def encode(self, texts: list[str]):
        import numpy as np

        rows = []
        for idx, _ in enumerate(texts):
            row = np.zeros(4, dtype=np.float32)
            row[idx % 4] = 1.0
            rows.append(row)
        return np.stack(rows, axis=0)


def test_factory_can_build_decomposed_answering_adapter() -> None:
    llm = _ScriptedLLM(["What degree did I graduate with?", "Business Administration"])
    factory = make_adapter_factory(
        llm=llm,
        embedder=_FakeEmbedder(),
        decompose_answer=True,
    )

    adapter = factory()

    assert isinstance(adapter, _DecomposedAnsweringAdapter)
    assert isinstance(adapter.session.stm, FlatHaystackStore)


class _KeywordEmbedder:
    def encode(self, texts: list[str]):
        import numpy as np

        vectors = []
        for text in texts:
            t = text.lower()
            v = np.zeros(4, dtype=np.float32)
            if "fitbit" in t or "march" in t:
                v[0] = 1.0
            if "december" in t or "now" in t:
                v[1] = 1.0
            if "degree" in t:
                v[2] = 1.0
            if not v.any():
                v[3] = 1.0
            vectors.append(v)
        return np.stack(vectors, axis=0)


@pytest.mark.asyncio
async def test_session_aware_retriever_selects_turns_from_multiple_sessions() -> None:
    store = FlatHaystackStore()
    await store.append(
        MemoryItem(
            id="s1-a",
            content="I started using the Fitbit Charge 3 in March.",
            tier=MemoryTier.STM,
            metadata={"role": "user", "session_id": "s1"},
        )
    )
    await store.append(
        MemoryItem(
            id="s1-b",
            content="Unrelated filler from the same session.",
            tier=MemoryTier.STM,
            metadata={"role": "assistant", "session_id": "s1"},
        )
    )
    await store.append(
        MemoryItem(
            id="s2-a",
            content="It is now December.",
            tier=MemoryTier.STM,
            metadata={"role": "user", "session_id": "s2"},
        )
    )

    retriever = SessionAwareSemanticRetriever(
        store=store,
        embedder=_KeywordEmbedder(),
        session_top_k=2,
        turns_per_session=1,
        max_items=4,
    )

    ctx = await retriever.retrieve(
        Query(text="How long between Fitbit March and December?"),
        _budget(),
    )

    assert [item.id for item in ctx.items] == ["s1-a", "s2-a"]
    assert ctx.debug_info["retrieval_mode"] == "session_aware"
    assert ctx.debug_info["selected_sessions"] == ["s1", "s2"]


def test_factory_uses_session_aware_retriever_with_decomposed_answering() -> None:
    llm = _ScriptedLLM(["What degree did I graduate with?", "Business Administration"])
    factory = make_adapter_factory(
        llm=llm,
        embedder=_KeywordEmbedder(),
        decompose_answer=True,
        session_aware_retrieval=True,
        session_top_k=3,
        turns_per_session=2,
    )

    adapter = factory()

    assert isinstance(adapter, _DecomposedAnsweringAdapter)
    assert isinstance(adapter.session.retriever, SessionAwareSemanticRetriever)
    assert adapter.session.retriever.session_top_k == 3
    assert adapter.session.retriever.turns_per_session == 2
