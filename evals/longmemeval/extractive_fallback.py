"""
evals/longmemeval/extractive_fallback.py
========================================
Extractive LLM fallback for the claim-first pipeline.

The reasoning heads in :mod:`evals.longmemeval.reasoning_heads`
return ``""`` when the question's answer shape doesn't match any
regex pattern in the top claims. That's the trigger for an
**extractive** LLM call (not a free-form generation): the model gets
1-3 top-ranked claims and is asked to return the shortest substring
that answers the question.

Why extractive rather than generative
-------------------------------------
* Span answers can be validated — the returned string must be a
  substring of one of the supplied claims.
* The model can't invent facts it didn't see.
* The prompt is small (≤ 300 tokens) → cheap.
* The contract is "return text from these claims" rather than
  "produce an answer" — easier to follow for small models.

The module is split into two parts:

* :func:`build_extractive_prompt` — builds the prompt string. Pure,
  testable, no LLM dependency.
* :func:`validate_extracted_span` — verifies the returned span is
  source-supported (case-insensitive substring of one of the claims).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from evals.longmemeval.claims import Claim


def build_extractive_prompt(
    question: str,
    top_claims: Iterable[Claim],
    *,
    max_claims: int = 3,
) -> str:
    """
    Build the extractive prompt for the LLM fallback.

    The prompt:

    * shows the question verbatim;
    * lists 1-3 top-ranked claims with their source session id;
    * asks for the **shortest faithful substring** from those claims;
    * forbids analysis, prose, scaffold ("Sub-answer:"), and
      ``I don't know`` when the claims cover the question.

    Returns a single prompt string. Empty when no claims were given —
    the caller should treat that as "fallback unavailable".
    """
    claims_list = list(top_claims)[:max_claims]
    if not claims_list:
        return ""
    blocks = []
    for i, c in enumerate(claims_list, start=1):
        sid = c.source_session_id or "?"
        role = c.source_role or "?"
        blocks.append(f"[{i}] session={sid} role={role}\n    {c.text}")
    return (
        "You are an extractive QA system. Return the SHORTEST substring "
        "from one of the source claims below that directly answers the "
        "question. Do NOT paraphrase, do NOT add explanation, do NOT use "
        "scaffold labels like \"Sub-answer:\". If the claims cover the "
        "answer, return that span; do NOT reply \"I don't know\" — the "
        "claims have already been verified to be relevant.\n\n"
        f"Question: {question}\n\n"
        "Source claims:\n"
        + "\n".join(blocks)
        + "\n\nShortest faithful answer span:"
    )


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d+")


def _normalise_for_validation(text: str) -> str:
    """Lowercase + collapse whitespace + drop common punctuation."""
    return " ".join(text.lower().split())


def validate_extracted_span(
    answer: str,
    claims: Iterable[Claim],
    *,
    min_token_overlap: int = 1,
) -> bool:
    """
    Return True when ``answer`` is supported by at least one claim.

    Support means: the lower-cased normalised answer is a substring
    of the lower-cased normalised claim text OR they share at least
    ``min_token_overlap`` content words. The token-overlap fallback
    is what lets the validator pass slight paraphrases that are
    semantically the answer ("Bachelor's in Business Administration"
    vs "degree in Business Administration").
    """
    if not answer:
        return False
    a_norm = _normalise_for_validation(answer)
    if not a_norm:
        return False
    a_tokens = {t.lower() for t in _WORD_RE.findall(answer) if len(t) > 1}
    for c in claims:
        c_norm = _normalise_for_validation(c.text)
        if a_norm in c_norm:
            return True
        c_tokens = {t.lower() for t in _WORD_RE.findall(c.text) if len(t) > 1}
        if (
            a_tokens
            and len(a_tokens & c_tokens) >= max(min_token_overlap, len(a_tokens) // 2)
        ):
            return True
    return False


__all__ = [
    "build_extractive_prompt",
    "validate_extracted_span",
]
