"""
evals/longmemeval/decompose.py
==============================
``DecompositionRetriever`` — the evidence-backed answer to the
multi-session / temporal-reasoning collapse.

The problem (proven across 6 LongMemEval runs)
-----------------------------------------------
A single ``retrieve top-k → answer`` call cannot answer questions whose
facts are scattered across several sessions. top-k=4 can't *gather* a
5-fact answer; long-context (100 % recall) can't *compose* one. Both
plateau at ~32 %.

The fix
-------
**Decompose, then retrieve per sub-question.**

1. An LLM splits the question into 1–4 atomic, single-fact
   sub-questions ("When did X start?" + "When did Y start?" rather than
   "Did X start before Y?").
2. The base retriever runs *once per sub-question* — so a 3-part
   question gets 3 × top-k candidates instead of one top-k.
3. The union (deduplicated) becomes the context. A question needing
   facts from 3 different sessions now actually has all 3 retrieved.

This is the technique behind LongMemEval's published 55–80 % systems.
It directly attacks the ``missing_fact`` + ``wrong_retrieval`` failures
that single-shot retrieval cannot.

Cost: one extra (cheap, ~150-token) LLM call per question for the
decomposition step. On gpt-4o-mini that's ~$0.0001/question.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from continuum.core.types import ContextBundle, MemoryItem, Query, TokenBudget
from continuum.optimizer.base import estimate_tokens_text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decomposition prompt — module constant so prompt caching can hit.
# ---------------------------------------------------------------------------


DECOMPOSE_SYSTEM_PROMPT = (
    "You break a user's question into the minimal set of simple, "
    "self-contained lookup sub-questions needed to answer it.\n\n"
    "Rules:\n"
    "1. Each sub-question must ask for exactly ONE fact and stand alone "
    "— no pronouns or references to other sub-questions.\n"
    "2. If the question is already a single-fact lookup, return it "
    "unchanged as the only sub-question.\n"
    "3. Return between 1 and 4 sub-questions, one per line. Output "
    "ONLY the sub-questions — no numbering, no bullets, no explanation, "
    "no preamble.\n"
)

_NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
#: Hard cap on sub-questions — keeps cost + token bloat bounded.
_MAX_SUBQUESTIONS = 4


def _build_decompose_prompt(question: str) -> str:
    return (
        DECOMPOSE_SYSTEM_PROMPT
        + f"\nQuestion: {question}\nSub-questions:"
    )


def parse_subquestions(reply: str, *, original: str) -> list[str]:
    """
    Parse the decomposition LLM's reply into a clean sub-question list.

    Strips numbering / bullets, drops blank lines, de-dupes
    case-insensitively, caps at :data:`_MAX_SUBQUESTIONS`. Falls back to
    ``[original]`` when the reply is empty or unparseable — so the
    retriever degrades to plain single-shot retrieval rather than
    breaking.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in (reply or "").splitlines():
        line = _NUMBER_PREFIX.sub("", raw_line).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= _MAX_SUBQUESTIONS:
            break
    return out or [original]


# ---------------------------------------------------------------------------
# DecompositionRetriever
# ---------------------------------------------------------------------------


class DecompositionRetriever:
    """
    Retriever wrapper that decomposes the query, retrieves per
    sub-question, and merges the results.

    Satisfies the same ``async retrieve(query, budget) -> ContextBundle``
    surface as :class:`STMSemanticRetriever`, so it drops straight into
    the adapter's ``session.retriever`` slot.

    Parameters
    ----------
    base:
        The underlying single-shot retriever (a ``STMSemanticRetriever``
        in ``topk`` mode). Its ``top_k`` is the *per-sub-question* count.
    llm:
        Object with ``async complete(prompt, max_tokens) -> str`` — used
        for the one decomposition call per question.
    max_items:
        Cap on the merged context size. A 4-sub-question × top-k=4
        retrieval can surface 16 unique items; this bounds it.
    decompose_max_tokens:
        Token cap on the decomposition reply (sub-questions are short).
    """

    def __init__(
        self,
        *,
        base: Any,
        llm: Any,
        max_items: int = 16,
        decompose_max_tokens: int = 160,
    ) -> None:
        self.base = base
        self.llm = llm
        self.max_items = max_items
        self.decompose_max_tokens = decompose_max_tokens
        # Mirror the base retriever's session id so the adapter's
        # session-id plumbing keeps working.
        self.session_id = getattr(base, "session_id", "default")
        #: Last decomposition, exposed for debug / telemetry.
        self.last_subquestions: list[str] = []

    async def _decompose(self, question: str) -> list[str]:
        """Ask the LLM to split *question*; degrade to [question] on error."""
        try:
            reply = await self.llm.complete(
                prompt=_build_decompose_prompt(question),
                max_tokens=self.decompose_max_tokens,
            )
        except Exception:
            log.exception("decomposition LLM call failed — using question as-is")
            return [question]
        subqs = parse_subquestions(reply, original=question)
        self.last_subquestions = subqs
        return subqs

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        subqs = await self._decompose(query.text)

        # Retrieve once per sub-question; merge, deduplicating by id and
        # preserving first-seen order so the highest-priority hits for
        # the earliest sub-questions lead.
        seen: dict[str, MemoryItem] = {}
        order: list[str] = []
        per_subq: list[int] = []
        for sq in subqs:
            try:
                ctx = await self.base.retrieve(Query(text=sq), budget)
            except Exception:
                log.exception("base retrieve failed for sub-question %r", sq)
                per_subq.append(0)
                continue
            per_subq.append(len(ctx.items))
            for it in ctx.items:
                if it.id not in seen:
                    seen[it.id] = it
                    order.append(it.id)

        merged = [seen[i] for i in order][: self.max_items]
        messages = [
            {
                "role": str(it.metadata.get("role", "user"))
                if it.metadata else "user",
                "content": it.content,
            }
            for it in merged
        ]
        token_count = sum(estimate_tokens_text(it.content) for it in merged)
        return ContextBundle(
            items=merged,
            messages=messages,
            tokens_used=token_count,
            budget=budget,
            tier_breakdown={"stm": token_count, "mtm": 0, "ltm": 0},
            debug_info={
                "retrieval_mode": "decompose",
                "sub_questions": subqs,
                "n_sub_questions": len(subqs),
                "per_subquestion_hits": per_subq,
                "merged_items": len(merged),
            },
        )


__all__ = [
    "DecompositionRetriever",
    "parse_subquestions",
    "DECOMPOSE_SYSTEM_PROMPT",
]
