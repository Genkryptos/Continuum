"""
continuum/reasoning/iterative_reasoner.py
==========================================
``IterativeReasoner`` — a budget-capped reasoning orchestrator.

The class wires together the steps the LongMemEval pipeline already
implements in scattered modules (intent classification, decomposition,
retrieval, claim verification, deterministic reasoning heads,
evidence-packet synthesis) into a single loop with a hard cap on
LLM calls.

Layer boundary
--------------
This module is in ``continuum/`` so it is reusable outside the eval
rig. To keep it free of ``evals/*`` imports — and so a future caller
can plug different reasoning heads / verifiers / extractors without
touching this file — every domain-specific step is a callable
injected via constructor kwargs. The thin wiring module
:mod:`evals.longmemeval.iterative_reasoner_wiring` provides the
LongMemEval-shaped defaults.

Two-tier LLM budget
-------------------
The reasoner takes two LLMs:

* ``small_llm`` — a :class:`continuum.extraction.small_llm.SmallLLM`
  instance (or anything matching its ``classify_intent`` /
  ``span_select`` API). Used for sqlite-cached intent fallback and
  the round-2 query rewrite.
* ``composer_llm`` — a larger answerer (gpt-4o-mini in the rig). Used
  for the one decompose call and the one final synthesis call.

Both share a single budget capped at ``max_llm_calls`` (default 6).
When the budget is exhausted mid-flight the reasoner returns the
highest-confidence claim it has seen with ``abstained=True`` —
better than empty for the substring + judge graders.

Refine loop
-----------
For each sub-question:

1. Retrieve.
2. Extract candidates + claims from the retrieved context.
3. Deterministic verification (``verify_fn``); any candidate marked
   ``PASS`` is "verified."
4. If empty: round 1 heuristic rewrite (no LLM) → retry.
5. If still empty: round 2 SmallLLM rewrite (1 LLM) → retry.
6. Otherwise abandon this sub-question (its contribution to the
   evidence packet is whatever the verifier produced — possibly
   nothing).

The deterministic reasoning heads short-circuit the composer call:
if the routed ``TaskMode`` has a head and the head produces a
non-empty answer from the union of all sub-questions' claims, that
answer is returned directly.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# ── Public result + trace shapes ────────────────────────────────────────────


@dataclass
class ReasoningResult:
    """
    Outcome of one :meth:`IterativeReasoner.answer` call.

    ``trace`` is a flat dict for easy JSONL logging; see the module
    docstring for its schema.
    """

    answer: str
    evidence: Any  # EvidencePacket-shaped; left as Any so this module has zero evals/* dep.
    trace: dict[str, Any]
    abstained: bool
    llm_call_count: int


# ── Budget bookkeeping ──────────────────────────────────────────────────────


@dataclass
class _Budget:
    cap: int
    spent: int = 0

    def can_afford(self, n: int = 1) -> bool:
        return self.spent + n <= self.cap

    def spend(self, n: int = 1) -> None:
        self.spent += n

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.cap


# ── Callable type aliases (advisory; constructor takes plain Callable) ──────


DecomposeFn = Callable[[str], tuple[str, Callable[..., list[str]]]]
ExtractFn = Callable[[Any], list[Any]]
VerifyFn = Callable[..., list[Any]]
IntentFn = Callable[[str], Any]
RouteFn = Callable[..., Any]
HeadFn = Callable[[list[Any], str], str]
PacketBuilder = Callable[..., Any]
PacketRenderer = Callable[[Any], str]
FinalPromptBuilder = Callable[..., str]
RewriteFn = Callable[[str, int], str]


# ── Default trace shape ─────────────────────────────────────────────────────


def _new_trace() -> dict[str, Any]:
    return {
        "intent": None,
        "intent_source": None,
        "task_mode": None,
        "subquestions": [],
        "rounds": [],
        "head_short_circuit": False,
        "head_fired": None,
        "composer_calls": 0,
        "small_llm_calls": 0,
        "abstain_reason": None,
    }


# ── Reasoner ────────────────────────────────────────────────────────────────


class IterativeReasoner:
    """Budget-capped intent → decompose → per-sub-q verify → synthesise loop."""

    def __init__(
        self,
        retriever: Any,
        small_llm: Any,
        composer_llm: Any,
        *,
        decompose_fn: DecomposeFn,
        extract_candidates_fn: ExtractFn,
        extract_claims_fn: ExtractFn,
        verify_fn: VerifyFn,
        filter_passing_fn: Callable[[list[Any]], list[Any]],
        intent_fn: IntentFn,
        route_fn: RouteFn,
        heads_by_mode: dict[Any, HeadFn],
        build_packet_fn: PacketBuilder,
        render_packet_fn: PacketRenderer,
        build_final_prompt_fn: FinalPromptBuilder,
        rewrite_query_fn: RewriteFn,
        mode_to_qtype_fn: Callable[[Any], Any] | None = None,
        max_rounds: int = 2,
        max_llm_calls: int = 6,
        question_type_hint: str | None = None,
        is_multi_session_hint: bool = False,
    ) -> None:
        self._retriever = retriever
        self._small_llm = small_llm
        self._composer_llm = composer_llm
        self._decompose_fn = decompose_fn
        self._extract_candidates_fn = extract_candidates_fn
        self._extract_claims_fn = extract_claims_fn
        self._verify_fn = verify_fn
        self._filter_passing_fn = filter_passing_fn
        self._intent_fn = intent_fn
        self._route_fn = route_fn
        self._heads_by_mode = dict(heads_by_mode)
        self._build_packet_fn = build_packet_fn
        self._render_packet_fn = render_packet_fn
        self._build_final_prompt_fn = build_final_prompt_fn
        self._rewrite_query_fn = rewrite_query_fn
        self._mode_to_qtype_fn = mode_to_qtype_fn
        self._max_rounds = max(1, max_rounds)
        self._max_llm_calls = max(1, max_llm_calls)
        self._question_type_hint = question_type_hint
        self._is_multi_session_hint = is_multi_session_hint

    # ── public API ──────────────────────────────────────────────────────────

    async def answer(self, question: str) -> ReasoningResult:
        budget = _Budget(cap=self._max_llm_calls)
        trace = _new_trace()

        # 1. Intent — deterministic first, SmallLLM fallback if unmatched.
        intent_obj = self._intent_fn(question)
        intent_str = _intent_string(intent_obj)
        intent_matched = _intent_matched(intent_obj)
        trace["intent_source"] = "deterministic"
        if not intent_matched and budget.can_afford(1):
            try:
                intent_str = (
                    await _maybe_await(
                        self._small_llm.classify_intent(question, cache_key=f"intent:{question}")
                    )
                    or intent_str
                )
                budget.spend()
                trace["small_llm_calls"] += 1
                trace["intent_source"] = "small_llm"
            except Exception:
                log.exception("SmallLLM.classify_intent failed; keeping deterministic intent")
        trace["intent"] = intent_str

        # 2. Route — pure.
        mode = self._route_fn(
            question,
            question_type_hint=self._question_type_hint,
            is_multi_session=self._is_multi_session_hint,
        )
        trace["task_mode"] = _enum_name(mode)

        # 3. Decompose — 1 composer call.
        if not budget.can_afford(1):
            return self._best_effort(
                [],
                trace,
                budget,
                packet=None,
                reason="budget_exhausted",
            )
        try:
            prompt, parser = self._decompose_fn(question)
            reply = await _maybe_await(self._composer_llm.complete(prompt=prompt, max_tokens=256))
            budget.spend()
            trace["composer_calls"] += 1
            subqs = parser(reply, original=question) or [question]
        except Exception:
            log.exception("decompose failed; using original question as sole sub-q")
            subqs = [question]
        trace["subquestions"] = list(subqs)

        # 4. Per-sub-q retrieve → extract → verify (with refine loop).
        sub_results: list[dict[str, Any]] = []
        for sub_q in subqs:
            result = await self._answer_subq(sub_q, mode, budget, trace)
            sub_results.append(result)
            if budget.exhausted:
                return self._best_effort(
                    sub_results,
                    trace,
                    budget,
                    packet=None,
                    reason="budget_exhausted",
                )

        # 5. Build evidence packet from all verified candidates.
        all_verified = [c for r in sub_results for c in r["verified"]]
        all_candidates = [c for r in sub_results for c in r["candidates"]]
        all_claims = [c for r in sub_results for c in r["claims"]]
        packet = self._build_packet_fn(
            question,
            all_verified,
            all_candidates=all_candidates,
            max_claims=6,
        )

        # 6. Heads-first short-circuit (0 LLM).
        head = self._heads_by_mode.get(mode)
        if head:
            try:
                head_answer = head(all_claims, question)
            except Exception:
                log.exception("head %r failed; falling through to composer", head)
                head_answer = ""
            if head_answer:
                trace["head_short_circuit"] = True
                trace["head_fired"] = getattr(head, "__name__", str(head))
                return ReasoningResult(
                    answer=head_answer,
                    evidence=packet,
                    trace=trace,
                    abstained=False,
                    llm_call_count=budget.spent,
                )

        # 7. Composer synthesis (1 call).
        if not budget.can_afford(1):
            return self._best_effort(
                sub_results,
                trace,
                budget,
                packet=packet,
                reason="budget_exhausted",
            )
        if not all_verified:
            # No verified evidence at all — synthesising would
            # hallucinate. Abstain with best-effort claim.
            return self._best_effort(
                sub_results,
                trace,
                budget,
                packet=packet,
                reason="no_verified_claims",
            )
        try:
            final_prompt = self._build_final_prompt_fn(
                question,
                subanswers=[],
                evidence_packet=packet,
            )
            text = await _maybe_await(
                self._composer_llm.complete(prompt=final_prompt, max_tokens=128)
            )
            budget.spend()
            trace["composer_calls"] += 1
        except Exception:
            log.exception("composer synthesis failed; abstaining with best-effort")
            return self._best_effort(
                sub_results,
                trace,
                budget,
                packet=packet,
                reason="no_verified_claims",
            )
        text = (text or "").strip()
        if not text:
            return self._best_effort(
                sub_results,
                trace,
                budget,
                packet=packet,
                reason="no_verified_claims",
            )
        return ReasoningResult(
            answer=text,
            evidence=packet,
            trace=trace,
            abstained=False,
            llm_call_count=budget.spent,
        )

    # ── private helpers ────────────────────────────────────────────────────

    async def _answer_subq(
        self,
        sub_q: str,
        mode: Any,
        budget: _Budget,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        query = sub_q
        verified: list[Any] = []
        candidates: list[Any] = []
        claims: list[Any] = []
        ctx: Any = None

        qtype = self._mode_to_qtype_fn(mode) if self._mode_to_qtype_fn is not None else None

        # ``max_rounds`` counts REFINE rounds, not total attempts. With
        # ``max_rounds=2`` we get up to 3 retrieve attempts: initial,
        # heuristic-rewrite, SmallLLM-rewrite. The trace's ``round`` field
        # is the 0-indexed attempt number (round 0 = initial).
        for attempt in range(self._max_rounds + 1):
            try:
                ctx = await _maybe_await(self._retriever.retrieve(query))
            except Exception:
                log.exception("retriever.retrieve failed for %r", query)
                ctx = None
            candidates = self._extract_candidates_fn(ctx) if ctx is not None else []
            claims = self._extract_claims_fn(ctx) if ctx is not None else []
            try:
                results = self._verify_fn(
                    sub_q,
                    candidates,
                    question_type=qtype,
                )
            except TypeError:
                # verify_fn may not accept question_type kwarg.
                results = self._verify_fn(sub_q, candidates)
            verified = self._filter_passing_fn(results)
            trace["rounds"].append(
                {
                    "sub_q": sub_q,
                    "round": attempt,
                    "query": query,
                    "n_candidates": len(candidates),
                    "n_verified": len(verified),
                    "rewrite": None,
                }
            )
            if verified:
                break
            if attempt >= self._max_rounds:
                break  # used every refine round
            # ── refine ────────────────────────────────────────────────────
            if attempt == 0:
                # Round 1: heuristic, no LLM call.
                rewritten = self._rewrite_query_fn(sub_q, 0)
                trace["rounds"][-1]["rewrite"] = f"heuristic:{rewritten}"
                query = rewritten or sub_q
            else:
                # Round 2: SmallLLM rewrite, 1 LLM call.
                if not budget.can_afford(1):
                    break
                try:
                    rewritten = await _maybe_await(
                        self._small_llm.span_select(
                            sub_q,
                            _rephrase_passage(sub_q),
                            cache_key=f"rewrite:{sub_q}",
                        )
                    )
                    budget.spend()
                    trace["small_llm_calls"] += 1
                    trace["rounds"][-1]["rewrite"] = f"small_llm:{rewritten}"
                    query = (rewritten or "").strip() or sub_q
                except Exception:
                    log.exception("SmallLLM.span_select rewrite failed")
                    break
        return {
            "sub_q": sub_q,
            "candidates": candidates,
            "claims": claims,
            "verified": verified,
            "ctx": ctx,
        }

    def _best_effort(
        self,
        sub_results: list[dict[str, Any]],
        trace: dict[str, Any],
        budget: _Budget,
        *,
        packet: Any,
        reason: str,
    ) -> ReasoningResult:
        """Highest-confidence verified candidate (or unverified fallback)."""
        best = _pick_highest_confidence(sub_results)
        trace["abstain_reason"] = reason
        return ReasoningResult(
            answer=best,
            evidence=packet,
            trace=trace,
            abstained=True,
            llm_call_count=budget.spent,
        )


# ── helpers ────────────────────────────────────────────────────────────────


async def _maybe_await(value: Any) -> Any:
    if isinstance(value, Awaitable):
        return await value
    return value


def _intent_string(intent_obj: Any) -> str:
    """
    Best-effort extraction of an intent label string from any
    object the deterministic classifier returned. The longmemeval
    ``QueryIntent`` carries ``answer_shape`` / ``relation``; the
    SmallLLM fallback returns a plain string. We accept both.
    """
    if intent_obj is None:
        return "unknown"
    if isinstance(intent_obj, str):
        return intent_obj or "unknown"
    for attr in ("answer_shape", "relation", "kind", "name"):
        v = getattr(intent_obj, attr, None)
        if v:
            return str(v)
    return "unknown"


def _intent_matched(intent_obj: Any) -> bool:
    if intent_obj is None:
        return False
    if isinstance(intent_obj, str):
        return bool(intent_obj.strip())
    m = getattr(intent_obj, "matched", None)
    if isinstance(m, bool):
        return m
    return bool(_intent_string(intent_obj) != "unknown")


def _enum_name(mode: Any) -> str:
    return getattr(mode, "name", str(mode))


def _pick_highest_confidence(sub_results: list[dict[str, Any]]) -> str:
    """
    Walk the verified-first then unverified candidates across every
    sub-result and return the highest-confidence ``value`` /
    ``claim`` / ``content`` string we can find. Returns "" when no
    candidate exists at all.
    """

    def _conf(c: Any) -> float:
        return float(getattr(c, "confidence", 0.0) or 0.0)

    def _text(c: Any) -> str:
        for attr in ("value", "claim", "content", "text"):
            v = getattr(c, attr, None)
            if v:
                return str(v).strip()
        return ""

    pool_verified: list[Any] = []
    pool_other: list[Any] = []
    for r in sub_results:
        pool_verified.extend(r.get("verified", []) or [])
        for c in r.get("candidates", []) or []:
            if c not in pool_verified:
                pool_other.append(c)
    pool = sorted(pool_verified, key=_conf, reverse=True) or sorted(
        pool_other, key=_conf, reverse=True
    )
    for c in pool:
        t = _text(c)
        if t:
            return t
    return ""


def _rephrase_passage(sub_q: str) -> str:
    """
    Tiny "passage" we hand to SmallLLM.span_select for query rewrite.
    The span selector returns the shortest span from the passage that
    answers the question — so the passage IS the rewrite instruction.
    """
    return (
        "Rewrite the question into a shorter, more retrieval-friendly form "
        "that keeps the key proper nouns and removes filler words. "
        f"Original: {sub_q}"
    )


__all__ = ["IterativeReasoner", "ReasoningResult"]
