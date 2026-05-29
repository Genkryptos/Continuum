"""
tests/reasoning/test_iterative_reasoner.py
==========================================
Unit tests for :class:`continuum.reasoning.IterativeReasoner`.

Every test uses the fakes from ``conftest.py`` — no real LLM, no real
retriever, no embedders. Each test scripts the exact answers/decisions
the reasoner will see and asserts what flows through the trace,
the budget, and the abstain machinery.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from continuum.reasoning import IterativeReasoner

from tests.reasoning.conftest import (
    FakeCandidate,
    FakeClaim,
    FakeComposerLLM,
    FakeCtx,
    FakePacket,
    FakeRetriever,
    FakeSmallLLM,
    FakeVerifierResult,
    _StubIntent,
    _StubMode,
    _build_final_prompt,
    _build_packet,
    _decompose_empty,
    _decompose_two,
    _filter_passing,
    _identity_extract,
    _matched_intent,
    _passthrough_verify,
    _render_packet,
    _route_to,
    _unmatched_intent,
)

pytestmark = pytest.mark.unit


# ── shared helpers ─────────────────────────────────────────────────────────


def _make_reasoner(
    *,
    retriever: Any,
    composer: Any,
    small: Any,
    decompose_fn: Callable = _decompose_two,
    intent_fn: Callable = _matched_intent,
    route_to: Any = _StubMode,
    heads_by_mode: dict | None = None,
    verify_fn: Callable = _passthrough_verify,
    extract_candidates_fn: Callable = _identity_extract,
    extract_claims_fn: Callable = _identity_extract,
    rewrite_fn: Callable = lambda q, _a: q + " bare",
    max_rounds: int = 2,
    max_llm_calls: int = 6,
) -> IterativeReasoner:
    return IterativeReasoner(
        retriever=retriever,
        small_llm=small,
        composer_llm=composer,
        decompose_fn=decompose_fn,
        extract_candidates_fn=extract_candidates_fn,
        extract_claims_fn=extract_claims_fn,
        verify_fn=verify_fn,
        filter_passing_fn=_filter_passing,
        intent_fn=intent_fn,
        route_fn=_route_to(route_to),
        heads_by_mode=heads_by_mode or {},
        build_packet_fn=_build_packet,
        render_packet_fn=_render_packet,
        build_final_prompt_fn=_build_final_prompt,
        rewrite_query_fn=rewrite_fn,
        max_rounds=max_rounds,
        max_llm_calls=max_llm_calls,
    )


# ── 1. easy head short-circuit ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_easy_head_short_circuits_no_composer() -> None:
    """A head returns "Paris" → no synthesis call, abstained=False."""
    cand = FakeCandidate(value="Paris", confidence=0.9)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["1. only sub-q"])  # only decompose burns
    small = FakeSmallLLM()
    retriever = FakeRetriever(default_ctx=ctx)

    heads = {_StubMode: lambda claims, q: "Paris"}
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["only sub-q"]),
        heads_by_mode=heads,
    )
    res = await reasoner.answer("Where do I live?")
    assert res.abstained is False
    assert res.answer == "Paris"
    assert res.trace["head_short_circuit"] is True
    assert res.trace["head_fired"]  # set to the head's __name__ or repr
    assert res.trace["composer_calls"] == 1  # decompose only
    assert res.trace["small_llm_calls"] == 0
    assert res.llm_call_count == 1


# ── 2. typical two-sub-q synthesis ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_typical_two_subqs_synthesis() -> None:
    cand = FakeCandidate(value="X", confidence=0.7)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["1. a\n2. b", "FINAL ANSWER"])
    small = FakeSmallLLM()
    retriever = FakeRetriever(default_ctx=ctx)
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        heads_by_mode={},  # no head registered → composer runs
    )
    res = await reasoner.answer("What and why?")
    assert res.abstained is False
    assert res.answer == "FINAL ANSWER"
    assert res.trace["composer_calls"] == 2  # decompose + synthesis
    assert res.trace["small_llm_calls"] == 0
    assert len(res.trace["subquestions"]) == 2
    assert res.llm_call_count == 2


# ── 3. refine round 2 uses SmallLLM rewrite ────────────────────────────────


@pytest.mark.asyncio
async def test_refine_round_2_uses_small_llm_rewrite() -> None:
    """
    Round 0: verifier returns empty for the only sub-q.
    Round 1 (heuristic rewrite): the retriever returns ctx that's still
    empty after verification.
    Round 2: SmallLLM rewrites to a query that yields a verifiable
    candidate. Synthesis call then succeeds.
    """
    empty_ctx = FakeCtx(payload=[])
    good_cand = FakeCandidate(value="finally", confidence=0.8)
    good_ctx = FakeCtx(payload=[good_cand])
    # rewrite_query → "Q bare"; SmallLLM rewrite → "rewritten Q"
    retriever = FakeRetriever(
        default_ctx=empty_ctx,
        ctx_for_query={"rewritten Q": good_ctx},
    )
    composer = FakeComposerLLM(replies=["1. Q", "OK"])
    small = FakeSmallLLM(span_replies=["rewritten Q"])
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["Q"]),
        heads_by_mode={},  # force synthesis
        max_rounds=2,
    )
    res = await reasoner.answer("anchor question")
    assert res.abstained is False
    assert res.answer == "OK"
    assert res.trace["small_llm_calls"] == 1
    # The last round entry for sub-q "Q" records a SmallLLM rewrite.
    rounds = [r for r in res.trace["rounds"] if r["sub_q"] == "Q"]
    assert rounds
    assert any(
        r["rewrite"] and r["rewrite"].startswith("small_llm:")
        for r in rounds
    )
    # The SmallLLM rewrite was retried with the new query.
    assert "rewritten Q" in retriever.queries


# ── 4. budget exhaustion returns best-effort claim text ────────────────────


@pytest.mark.asyncio
async def test_budget_exhausted_returns_best_effort_abstain() -> None:
    """max_llm_calls=1 → decompose consumes the entire budget; synthesis is skipped."""
    cand = FakeCandidate(value="HIGH-CONF", confidence=0.95)
    weak = FakeCandidate(value="weak", confidence=0.1)
    ctx = FakeCtx(payload=[weak, cand])
    composer = FakeComposerLLM(replies=["1. Q"])  # only the decompose reply
    small = FakeSmallLLM()
    retriever = FakeRetriever(default_ctx=ctx)
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["Q"]),
        heads_by_mode={},  # composer would run if budget allowed
        max_llm_calls=1,
    )
    res = await reasoner.answer("anything")
    assert res.abstained is True
    assert res.trace["abstain_reason"] == "budget_exhausted"
    assert res.answer == "HIGH-CONF"  # highest-confidence verified candidate
    assert res.llm_call_count == 1
    # No synthesis call — only decompose consumed budget.
    assert composer.calls and len(composer.calls) == 1


# ── 5. no verified claims → empty answer ───────────────────────────────────


@pytest.mark.asyncio
async def test_no_verified_claims_abstains_with_empty_answer() -> None:
    """Verifier rejects everything; rewrites can't recover; composer is skipped."""
    cand = FakeCandidate(value="ignored", confidence=0.5)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["1. Q"])
    small = FakeSmallLLM(span_replies=["still-empty"])
    retriever = FakeRetriever(default_ctx=ctx)

    def reject_all(question: str, candidates: list[Any], **kw: Any) -> list[Any]:
        return [FakeVerifierResult(candidate=c, verdict="FAIL") for c in candidates]

    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["Q"]),
        verify_fn=reject_all,
        heads_by_mode={},
        max_llm_calls=4,
    )
    res = await reasoner.answer("q")
    assert res.abstained is True
    assert res.trace["abstain_reason"] == "no_verified_claims"
    # Composer ran the decompose call but NOT the synthesis call.
    assert len(composer.calls) == 1
    # Best-effort falls back to highest-confidence unverified candidate text.
    assert res.answer == "ignored"


# ── 6. SmallLLM intent fallback fires when deterministic returns unmatched ─


@pytest.mark.asyncio
async def test_intent_llm_fallback_when_deterministic_unknown() -> None:
    cand = FakeCandidate(value="X", confidence=0.8)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["1. Q", "ANSWER"])
    small = FakeSmallLLM(intent_replies=["compose"])
    retriever = FakeRetriever(default_ctx=ctx)
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["Q"]),
        intent_fn=_unmatched_intent,
        heads_by_mode={},
    )
    res = await reasoner.answer("q?")
    assert res.trace["intent"] == "compose"
    assert res.trace["intent_source"] == "small_llm"
    assert small.intent_calls and small.intent_calls[0]["cache_key"] == "intent:q?"
    assert res.trace["small_llm_calls"] == 1


# ── 7. decompose failure falls back to original question ───────────────────


@pytest.mark.asyncio
async def test_decompose_failure_falls_back_to_original_question() -> None:
    cand = FakeCandidate(value="ok", confidence=0.7)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["garbage that won't parse", "ANS"])
    small = FakeSmallLLM()
    retriever = FakeRetriever(default_ctx=ctx)
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=_decompose_empty,  # parser returns []
        heads_by_mode={},
    )
    res = await reasoner.answer("original Q?")
    assert res.trace["subquestions"] == ["original Q?"]
    assert res.abstained is False
    assert res.answer == "ANS"


# ── 8. trace records every (sub_q, round) attempt ─────────────────────────


@pytest.mark.asyncio
async def test_trace_records_all_rounds() -> None:
    """Two sub-qs, each going through 2 rounds because verify always fails."""
    cand = FakeCandidate(value="x", confidence=0.5)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["1. one\n2. two"])
    small = FakeSmallLLM(span_replies=["r1", "r2"])
    retriever = FakeRetriever(default_ctx=ctx)

    def reject_all(question: str, candidates: list[Any], **kw: Any) -> list[Any]:
        return [FakeVerifierResult(candidate=c, verdict="FAIL") for c in candidates]

    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        verify_fn=reject_all,
        heads_by_mode={},
        max_llm_calls=6,  # plenty for the rewrites
        max_rounds=2,
    )
    res = await reasoner.answer("Q")
    # max_rounds=2 → 3 attempts per sub-q (initial + heuristic + SmallLLM).
    # _decompose_two yields ["sub-q one", "sub-q two"].
    rounds_one = [r for r in res.trace["rounds"] if r["sub_q"] == "sub-q one"]
    rounds_two = [r for r in res.trace["rounds"] if r["sub_q"] == "sub-q two"]
    assert [r["round"] for r in rounds_one] == [0, 1, 2]
    assert [r["round"] for r in rounds_two] == [0, 1, 2]
    # Round-0 records the heuristic rewrite scheduled for round 1;
    # round-1 records the SmallLLM rewrite scheduled for round 2.
    assert rounds_one[0]["rewrite"] and rounds_one[0]["rewrite"].startswith("heuristic:")
    assert rounds_one[1]["rewrite"] and rounds_one[1]["rewrite"].startswith("small_llm:")


# ── 9. composer crash falls back to abstain (no raise into caller) ─────────


@pytest.mark.asyncio
async def test_composer_synthesis_exception_returns_abstain() -> None:
    class _BoomComposer:
        def __init__(self) -> None:
            self.calls = 0
        async def complete(self, prompt: str, max_tokens: int) -> str:
            self.calls += 1
            if self.calls == 1:
                return "1. Q"
            raise RuntimeError("boom")

    cand = FakeCandidate(value="last-resort", confidence=0.9)
    ctx = FakeCtx(payload=[cand])
    retriever = FakeRetriever(default_ctx=ctx)
    composer = _BoomComposer()
    small = FakeSmallLLM()
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["Q"]),
        heads_by_mode={},
    )
    res = await reasoner.answer("q")
    assert res.abstained is True
    assert res.answer == "last-resort"


# ── 10. head returns empty → falls through to composer synthesis ──────────


@pytest.mark.asyncio
async def test_head_returns_empty_falls_through_to_composer() -> None:
    cand = FakeCandidate(value="x", confidence=0.7)
    ctx = FakeCtx(payload=[cand])
    composer = FakeComposerLLM(replies=["1. Q", "synthesised"])
    small = FakeSmallLLM()
    retriever = FakeRetriever(default_ctx=ctx)
    heads = {_StubMode: lambda claims, q: ""}  # head abdicates
    reasoner = _make_reasoner(
        retriever=retriever, composer=composer, small=small,
        decompose_fn=lambda q: ("p", lambda r, original: ["Q"]),
        heads_by_mode=heads,
    )
    res = await reasoner.answer("Q?")
    assert res.trace["head_short_circuit"] is False
    assert res.answer == "synthesised"
    assert res.trace["composer_calls"] == 2
