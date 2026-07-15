"""
tests/unit/evals/test_reflect_wiring.py
=======================================
v3 Reflect: the single BOUNDED reasoning pass (not an agentic loop) wired into
_DirectAnswerAdapter. Covers the gate (which question types are eligible), the
"Answer:" final-line parsing, and the adapter plumbing — that Reflect uses
exactly ONE llm.complete with the larger token budget, emits the right
telemetry, is gated OFF simple fact recall, and STACKS with synthesis (a
counting multi-session question gets both the COMPUTED FACTS block and the CoT
prompt). Fakes throughout — no LLM, no network, no eval run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.types import MemoryItem, MemoryTier
from evals.longmemeval.bootstrap_ollama import (
    _DirectAnswerAdapter,
    _compute_temporal_order,
    _extract_reflect_answer,
    _is_distill_eligible,
    _is_reflect_eligible,
    _is_temporal_order_question,
    _majority_vote,
)

pytestmark = pytest.mark.unit


# ── the gate ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "qt",
    [
        "single-session-preference",  # clear semantic win (preference application)
        "knowledge-update",           # net +2
    ],
)
def test_eligible_default_categories(qt: str) -> None:
    assert _is_reflect_eligible(qt, "whatever the question is") is True


@pytest.mark.parametrize(
    "qt",
    [
        "multi-session",            # net −13 → dropped from default
        "temporal-reasoning",       # net −13 → dropped
        "single-session-assistant", # net −2 → dropped
    ],
)
def test_net_negative_categories_now_excluded(qt: str) -> None:
    # full-500 showed CoT over-reasons these → no longer in the default set.
    assert _is_reflect_eligible(qt, "whatever the question is") is False


def test_excluded_simple_fact_recall() -> None:
    # single-session-user plain recall is exactly where CoT over-thinks → off.
    assert _is_reflect_eligible("single-session-user", "What is my dog's name?") is False


def test_unknown_type_falls_back_to_wording() -> None:
    assert _is_reflect_eligible("", "What headphones should I buy? Recommend one.") is True  # preference
    assert _is_reflect_eligible("", "What is my dog's name?") is False        # plain


def test_known_but_not_allowed_type_is_off() -> None:
    # a type not in the allowed set and not excluded → don't reflect (no wording
    # fallback for a *known* type).
    assert _is_reflect_eligible("some-future-type", "How long ago?") is False


def test_custom_allowed_set_overrides_default() -> None:
    allowed = frozenset({"multi-session"})
    assert _is_reflect_eligible("multi-session", "q", allowed=allowed) is True
    assert _is_reflect_eligible("temporal-reasoning", "q", allowed=allowed) is False


# ── answer parsing ────────────────────────────────────────────────────────────


def test_parse_simple_answer_marker() -> None:
    cot = "Step 1: the user said X.\nStep 2: so the answer is...\nAnswer: Target"
    assert _extract_reflect_answer(cot) == "Target"


def test_parse_uses_last_marker() -> None:
    # the word "answer" appears mid-reasoning; take the final committed one.
    cot = "The answer: depends.\nReasoning...\nAnswer: five"
    assert _extract_reflect_answer(cot) == "five"


def test_parse_multiline_tail() -> None:
    cot = "Reasoning.\nAnswer: a list:\n- one\n- two"
    assert _extract_reflect_answer(cot) == "a list:\n- one\n- two"


def test_parse_fallback_to_last_line_when_no_marker() -> None:
    assert _extract_reflect_answer("just reasoning\nfinal thought") == "final thought"


def test_parse_empty() -> None:
    assert _extract_reflect_answer("   ") == ""


def test_parse_idk() -> None:
    assert _extract_reflect_answer("no evidence.\nAnswer: I don't know.") == "I don't know."


# ── adapter plumbing (fakes) ──────────────────────────────────────────────────


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
    def __init__(self, reply: str = "Answer: done") -> None:
        self.prompt: str | None = None
        self.max_tokens: int | None = None
        self._reply = reply

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.prompt = prompt
        self.max_tokens = max_tokens
        return self._reply


def _turn(text: str) -> MemoryItem:
    return MemoryItem(content=text, tier=MemoryTier.STM, metadata={"role": "user"})


def _adapter(
    *,
    reflect: bool,
    reply: str = "Answer: done",
    qt: str = "knowledge-update",  # a default reflect-eligible type
    synthesis_fn: Any = None,
    router: bool = False,
) -> _DirectAnswerAdapter:
    session = SimpleNamespace(
        stm=_FakeSTM(),
        retriever=_FakeRetriever([_turn("I have a goldfish tank and a shrimp tank")]),
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session,
        llm=_CaptureLLM(reply),
        answer_max_tokens=16,
        top_k=8,
        max_context_chars=10_000,
        reflect=reflect,
        reflect_max_tokens=400,
        synthesis_fn=synthesis_fn,
        router=router,
    )
    a.dataset_question_type = qt  # type: ignore[attr-defined]
    return a


async def test_reflect_uses_cot_prompt_and_big_budget() -> None:
    a = _adapter(reflect=True, reply="reasoning here\nAnswer: Target")
    out = await a.answer_question("Where did I redeem the coupon?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "Reason in a few SHORT steps" in prompt
    assert "Answer:" in prompt
    assert a.llm.max_tokens == 400          # the reflect budget, not 16
    assert out == "Target"                  # parsed from the final line
    assert a.last_telemetry["reflect_applied"] is True
    assert a.last_telemetry["llm_call_count"] == 1   # still ONE call


async def test_reflect_off_uses_direct_prompt() -> None:
    a = _adapter(reflect=False, reply="Target")
    out = await a.answer_question("Where did I redeem the coupon?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "Reason in a few SHORT steps" not in prompt
    assert a.llm.max_tokens == 16           # the direct budget
    assert out == "Target"
    assert a.last_telemetry["reflect_applied"] is False


async def test_reflect_gated_off_for_single_session_user() -> None:
    a = _adapter(reflect=True, reply="Answer: Rex", qt="single-session-user")
    await a.answer_question("What is my dog's name?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "Reason in a few SHORT steps" not in prompt
    assert a.last_telemetry["reflect_applied"] is False


async def test_reflect_stacks_with_synthesis() -> None:
    # multi-session counting question: Reflect prompt AND the COMPUTED FACTS block.
    a = _adapter(reflect=True, reply="reasoning\nAnswer: 3")
    # seed a synthesis fact directly (bypass ingest extraction).
    from continuum.promotion.synthesis import DerivedFact

    a._synthesis_facts = [  # type: ignore[attr-defined]
        DerivedFact(subject="user", predicate="tank", count=3, members=["a", "b", "c"])
    ]
    out = await a.answer_question("How many tanks do I have?")
    prompt = a.llm.prompt  # type: ignore[attr-defined]
    assert "COMPUTED FACTS" in prompt           # synthesis injected
    assert "User has 3 tanks." in prompt
    assert "Reason in a few SHORT steps" in prompt  # AND reflect
    assert out == "3"
    assert a.last_telemetry["reflect_applied"] is True
    assert a.last_telemetry["synthesis_injected"] is True


# ── deterministic router: counting questions bypass the reader entirely ────────


async def test_router_answers_count_without_calling_reader() -> None:
    a = _adapter(reflect=True, reply="reasoning\nAnswer: 99", router=True)
    from continuum.promotion.synthesis import DerivedFact

    a._synthesis_facts = [  # type: ignore[attr-defined]
        DerivedFact(subject="user", predicate="tank", count=3, members=["a", "b", "c"])
    ]
    out = await a.answer_question("How many tanks do I have?")
    # routed deterministically → the code answer, NOT the reader's "99".
    assert out == "3"
    assert a.llm.prompt is None  # type: ignore[attr-defined]  # reader never called
    assert a.last_telemetry["llm_call_count"] == 0
    assert a.last_telemetry["router_applied"] is True


async def test_router_falls_through_when_no_match() -> None:
    a = _adapter(reflect=True, reply="reasoning\nAnswer: blue", router=True)
    from continuum.promotion.synthesis import DerivedFact

    a._synthesis_facts = [  # type: ignore[attr-defined]
        DerivedFact(subject="user", predicate="tank", count=3, members=["a", "b", "c"])
    ]
    # not a counting question → router returns None → reader answers.
    out = await a.answer_question("What is my favorite colour?")
    assert out == "blue"
    assert a.llm.prompt is not None  # type: ignore[attr-defined]  # reader WAS called
    assert a.last_telemetry.get("router_applied") is None


# ── self-consistency (vote-of-N) ──────────────────────────────────────────────


def test_majority_vote_picks_consensus() -> None:
    assert _majority_vote(["3 months", "3 months.", "a month"]) == "3 months"


def test_majority_vote_normalizes_for_grouping() -> None:
    # "Target", "target.", "TARGET" all group → consensus over a singleton.
    assert _majority_vote(["Target", "target.", "Walmart"]) == "Target"


def test_majority_vote_tie_breaks_earliest() -> None:
    assert _majority_vote(["a", "b"]) == "a"


def test_majority_vote_skips_empty_unless_all_empty() -> None:
    # one real answer + two empties → the real answer wins (empties don't count).
    assert _majority_vote(["", "Target", ""]) == "Target"
    assert _majority_vote(["", "", ""]) == ""


class _CyclingLLM:
    """Returns a different reply each call (to exercise voting / distill)."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0
        self.last_prompt: str | None = None

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        self.last_prompt = prompt
        r = self._replies[self.calls % len(self._replies)]
        self.calls += 1
        return r


def _vote_adapter(replies: list[str], vote_n: int) -> _DirectAnswerAdapter:
    session = SimpleNamespace(
        stm=_FakeSTM(),
        retriever=_FakeRetriever([_turn("I have a goldfish tank")]),
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session, llm=_CyclingLLM(replies), answer_max_tokens=16,
        top_k=8, max_context_chars=10_000, vote_n=vote_n,
    )
    a.dataset_question_type = "single-session-user"  # type: ignore[attr-defined]
    return a


async def test_vote_of_three_takes_majority() -> None:
    # 2 of 3 samples say "Boston" → consensus, despite one "Chicago".
    a = _vote_adapter(["Boston", "Chicago", "Boston"], vote_n=3)
    out = await a.answer_question("Where do I live?")
    assert out == "Boston"
    assert a.llm.calls == 3  # type: ignore[attr-defined]
    assert a.last_telemetry["llm_call_count"] == 3
    assert a.last_telemetry["vote_n"] == 3


async def test_vote_n_one_is_single_call() -> None:
    a = _vote_adapter(["Boston", "Chicago", "Boston"], vote_n=1)
    out = await a.answer_question("Where do I live?")
    assert out == "Boston"
    assert a.llm.calls == 1  # type: ignore[attr-defined]
    assert a.last_telemetry["llm_call_count"] == 1


# ── Phase 1 evidence distillation ─────────────────────────────────────────────


@pytest.mark.parametrize("qt", ["multi-session", "knowledge-update"])
def test_distill_eligible_default(qt: str) -> None:
    assert _is_distill_eligible(qt, "how many tanks?") is True


@pytest.mark.parametrize("qt", ["single-session-user", "single-session-preference"])
def test_distill_not_eligible_other_types(qt: str) -> None:
    assert _is_distill_eligible(qt, "what is my dog's name?") is False


async def test_distill_filters_then_answers() -> None:
    # call 1 = distill (returns quotes), call 2 = answer over the quotes.
    llm = _CyclingLLM(["[user] I have a goldfish tank\n[user] and a shrimp tank", "2"])
    session = SimpleNamespace(
        stm=_FakeSTM(),
        retriever=_FakeRetriever([_turn("I have a goldfish tank"), _turn("random chatter")]),
        session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session, llm=llm, answer_max_tokens=16, top_k=8,
        max_context_chars=10_000, distill=True,
    )
    a.dataset_question_type = "multi-session"  # type: ignore[attr-defined]
    out = await a.answer_question("How many tanks do I have?")
    assert out == "2"
    assert a.llm.calls == 2  # type: ignore[attr-defined]  # distill + answer
    assert a.last_telemetry["distill_applied"] is True
    # the answer prompt must contain the DISTILLED quotes, not raw chatter
    assert "shrimp tank" in (a.llm.last_prompt or "")  # type: ignore[attr-defined]


async def test_distill_off_is_single_call() -> None:
    llm = _CyclingLLM(["answer"])
    session = SimpleNamespace(
        stm=_FakeSTM(), retriever=_FakeRetriever([_turn("I have a tank")]), session_id="s",
    )
    a = _DirectAnswerAdapter(
        session=session, llm=llm, answer_max_tokens=16, top_k=8,
        max_context_chars=10_000, distill=False,
    )
    a.dataset_question_type = "multi-session"  # type: ignore[attr-defined]
    await a.answer_question("How many tanks?")
    assert a.llm.calls == 1  # type: ignore[attr-defined]
    assert a.last_telemetry["distill_applied"] is False


# ── Phase 2 temporal ordering (sort in code) ──────────────────────────────────


@pytest.mark.parametrize(
    "q",
    [
        "What is the order of the six museums I visited from earliest to latest?",
        "Who graduated first, second and third among Emma, Rachel and Alex?",
        "Which event happened first, the concert or the festival?",
    ],
)
def test_is_temporal_order_positive(q: str) -> None:
    assert _is_temporal_order_question(q) is True


@pytest.mark.parametrize(
    "q",
    ["How many days ago did I move?", "What is my favorite museum?"],
)
def test_is_temporal_order_negative(q: str) -> None:
    assert _is_temporal_order_question(q) is False


def test_compute_temporal_order_sorts_by_date() -> None:
    a = (
        'The order is...\n'
        'ORDER: [{"event": "Metropolitan", "date": "2023-05-10"}, '
        '{"event": "Science Museum", "date": "2023-01-02"}, '
        '{"event": "Modern Art", "date": "2023-03-15"}]'
    )
    assert _compute_temporal_order(a) == "Science Museum, Modern Art, Metropolitan"


def test_compute_temporal_order_none_on_garbage() -> None:
    assert _compute_temporal_order("no spec here") is None
    assert _compute_temporal_order("ORDER: [not json]") is None


def test_compute_temporal_order_needs_two() -> None:
    assert _compute_temporal_order('ORDER: [{"event": "x", "date": "2023-01-01"}]') is None
