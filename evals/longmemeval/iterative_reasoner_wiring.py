"""
evals/longmemeval/iterative_reasoner_wiring.py
==============================================
Thin factory that constructs a
:class:`continuum.reasoning.IterativeReasoner` wired with the
LongMemEval-shaped callables for intent classification, routing,
decomposition, claim extraction, deterministic verification,
reasoning heads, evidence-packet building, and final-answer prompt
construction.

This is the only file that imports across the layer boundary
(``continuum/`` ← ``evals/longmemeval/*``). The reasoner itself
stays generic.

Heuristic query rewriter (round 1)
----------------------------------
The reasoner's round-1 rewrite must be free of LLM calls. We use a
small token-set transformation: drop English stopwords, lowercase,
keep digits and proper nouns. This trades surface form for recall —
useful when the original wording matched a turn whose phrasing
diverged in the haystack.
"""
from __future__ import annotations

import re
from typing import Any

from continuum.reasoning import IterativeReasoner

from evals.longmemeval.candidates import extract_candidates_from_context
from evals.longmemeval.claim_verifier import filter_passing, verify_claims
from evals.longmemeval.claims import extract_claims_from_context
from evals.longmemeval.decompose import build_decompose_prompt, parse_subquestions
from evals.longmemeval.decomposed_answer import build_final_synthesis_prompt
from evals.longmemeval.evidence_packet import (
    build_evidence_packet,
    render_evidence_packet_prompt,
)
from evals.longmemeval.query_intent import parse_intent
from evals.longmemeval.question_type import QuestionType
from evals.longmemeval.reasoning_heads import (
    head_assistant_memory,
    head_fact_lookup,
    head_knowledge_update,
    head_multi_session_aggregate,
    head_preference_profile,
    head_temporal_reasoning,
)
from evals.longmemeval.task_router import TaskMode, route_task


# ── Heads keyed by TaskMode ────────────────────────────────────────────────


HEADS_BY_MODE: dict[TaskMode, Any] = {
    TaskMode.FACT_LOOKUP:              head_fact_lookup,
    TaskMode.ASSISTANT_MEMORY_LOOKUP:  head_assistant_memory,
    TaskMode.PREFERENCE_PROFILE:       head_preference_profile,
    TaskMode.KNOWLEDGE_UPDATE:         head_knowledge_update,
    TaskMode.MULTI_SESSION_AGGREGATE:  head_multi_session_aggregate,
    TaskMode.TEMPORAL_REASONING:       head_temporal_reasoning,
}


# ── TaskMode → QuestionType (for the verifier kwarg) ───────────────────────


_MODE_TO_QTYPE: dict[TaskMode, QuestionType] = {
    TaskMode.FACT_LOOKUP:              QuestionType.GENERIC_FACT,
    TaskMode.ASSISTANT_MEMORY_LOOKUP:  QuestionType.GENERIC_FACT,
    TaskMode.PREFERENCE_PROFILE:       QuestionType.PREFERENCE,
    TaskMode.KNOWLEDGE_UPDATE:         QuestionType.KNOWLEDGE_UPDATE,
    TaskMode.MULTI_SESSION_AGGREGATE:  QuestionType.MULTI_SESSION,
    TaskMode.TEMPORAL_REASONING:       QuestionType.TEMPORAL_REASONING,
}


def _mode_to_qtype(mode: TaskMode) -> QuestionType | None:
    return _MODE_TO_QTYPE.get(mode)


# ── Decompose callable ─────────────────────────────────────────────────────


def _decompose_fn(question: str):
    """Return (prompt, parser) so the reasoner can run the one LLM call itself."""
    return build_decompose_prompt(question), parse_subquestions


# ── Heuristic round-1 query rewrite ────────────────────────────────────────


_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "did", "does", "have", "had", "has", "of", "to", "for", "in",
    "on", "at", "by", "with", "from", "about", "and", "or", "but",
    "i", "my", "me", "you", "your", "he", "she", "they", "we", "us",
    "this", "that", "these", "those", "it", "its", "as", "if", "than",
    "what", "which", "who", "whom", "whose", "why", "how",
})

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*|\d+")


def heuristic_rewrite(sub_q: str, attempt: int) -> str:
    """
    Round-0 rewrite: strip stopwords + filler punctuation, keep
    proper nouns and digits. Returns the original on degenerate
    input so the reasoner never retries on an empty query.
    """
    tokens = _WORD_RE.findall(sub_q or "")
    if not tokens:
        return sub_q
    kept = [
        t for t in tokens
        if t.lower() not in _STOPWORDS
    ]
    rewritten = " ".join(kept).strip()
    return rewritten or sub_q


# ── Public factory ─────────────────────────────────────────────────────────


def build_iterative_reasoner(
    retriever: Any,
    small_llm: Any,
    composer_llm: Any,
    *,
    max_rounds: int = 2,
    max_llm_calls: int = 6,
    question_type_hint: str | None = None,
    is_multi_session_hint: bool = False,
) -> IterativeReasoner:
    """
    Construct an :class:`IterativeReasoner` with LongMemEval defaults.

    Parameters mirror the reasoner's surface; everything else is
    fixed by the module-level constants above (HEADS_BY_MODE,
    _MODE_TO_QTYPE).
    """
    return IterativeReasoner(
        retriever=retriever,
        small_llm=small_llm,
        composer_llm=composer_llm,
        decompose_fn=_decompose_fn,
        extract_candidates_fn=extract_candidates_from_context,
        extract_claims_fn=extract_claims_from_context,
        verify_fn=verify_claims,
        filter_passing_fn=filter_passing,
        intent_fn=parse_intent,
        route_fn=route_task,
        heads_by_mode=HEADS_BY_MODE,
        build_packet_fn=build_evidence_packet,
        render_packet_fn=render_evidence_packet_prompt,
        build_final_prompt_fn=build_final_synthesis_prompt,
        rewrite_query_fn=heuristic_rewrite,
        mode_to_qtype_fn=_mode_to_qtype,
        max_rounds=max_rounds,
        max_llm_calls=max_llm_calls,
        question_type_hint=question_type_hint,
        is_multi_session_hint=is_multi_session_hint,
    )


__all__ = ["build_iterative_reasoner", "heuristic_rewrite", "HEADS_BY_MODE"]
