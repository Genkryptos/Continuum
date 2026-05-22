"""
evals/longmemeval/judge.py
==========================
``LLMJudgeScorer`` — replaces the brittle substring scorer with an LLM
that grades each (question, expected, actual) triple semantically.

Why this exists
---------------
The default substring/token-overlap scorer in :mod:`baseline.py` is
permissive on the easy direction (substring matching catches "Paris" /
"the answer is Paris") but **strict on paraphrases** ("she lives in
Paris" — model answers "her residence is in Paris" → false negative).

On the LongMemEval-S full run, the 17B model produces verbose, varied
phrasings that the substring matcher rejects even when the underlying
fact is correct. The LLM judge fixes that by reading both strings and
deciding whether the model's answer entails the expected one.

Design
------
* Uses any object exposing the same ``async complete(prompt, max_tokens)
  -> str`` contract as :class:`OllamaLLM` / :class:`GroqLLM`. Inject
  whichever model you prefer for judging — typically a cheap one
  (``llama-3.1-8b-instant`` for Groq, or local Ollama).
* The judge prompt is a fixed literal (so Groq / Anthropic prompt
  caching can hit) ending in a single-token "yes"/"no" classification.
* On parse failure or LLM error, falls back to a configurable safe
  default (defaults to **False** so failures don't silently inflate
  accuracy).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt — kept as a module constant so prompt caching can hit on every call.
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair answer evaluator. You compare a model's "
    "answer against the ground truth for a benchmark question and "
    "decide whether the model's answer is semantically correct.\n\n"
    "Rules:\n"
    "1. Reply ONLY with 'yes' or 'no' — no explanation, no punctuation.\n"
    "2. Say 'yes' if the model's answer conveys the same factual content "
    "as the expected answer, even if phrased differently or includes "
    "extra correct detail.\n"
    "3. Say 'yes' if the model's answer entails the expected answer "
    "(e.g. expected='Paris', model='She lives in Paris, France').\n"
    "4. Say 'no' if the model's answer is empty, hedged ('I don't "
    "know'), wrong, or factually contradicts the expected answer.\n"
    "5. Say 'no' if the model only restates the question without "
    "actually answering it.\n"
)


def _build_user_prompt(question: str, expected: str, actual: str) -> str:
    return (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Model's answer: {actual}\n"
        "Is the model's answer correct? Reply with just 'yes' or 'no'."
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class LLMJudgeScorer:
    """
    Semantic answer scorer backed by an LLM.

    Parameters
    ----------
    llm:
        Any object with ``async complete(prompt: str, max_tokens: int)
        -> str``. The judge model — use a cheap fast one
        (``llama-3.1-8b-instant`` on Groq is recommended at $0.08/M).
    fallback_on_error:
        Verdict returned when the LLM call fails or the response can't
        be parsed. Defaults to ``False`` to keep accuracy honest.
    max_tokens:
        Cap on the judge's reply. 5 is plenty for "yes"/"no" and
        suppresses any chain-of-thought rambling.
    """

    def __init__(
        self,
        *,
        llm: Any,
        fallback_on_error: bool = False,
        max_tokens: int = 5,
    ) -> None:
        self.llm = llm
        self.fallback_on_error = fallback_on_error
        self.max_tokens = max_tokens

    async def is_correct(
        self, question: str, expected: str, actual: str
    ) -> bool:
        if not actual or not expected:
            return False
        prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + _build_user_prompt(
            question, str(expected), str(actual)
        )
        try:
            reply = await self.llm.complete(
                prompt=prompt, max_tokens=self.max_tokens
            )
        except Exception:
            log.exception("LLM judge call failed")
            return self.fallback_on_error
        return _parse_yes_no(reply, fallback=self.fallback_on_error)


def _parse_yes_no(reply: str, *, fallback: bool = False) -> bool:
    """
    Conservative parser for the judge's reply.

    Accepts 'yes' / 'no' anywhere in the (lowercased, stripped) output.
    Both present or neither present → use *fallback*.
    """
    if not reply:
        return fallback
    text = reply.strip().lower()
    has_yes = "yes" in text
    has_no = "no" in text
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    # Ambiguous or empty — explicit fallback.
    log.warning("judge reply ambiguous: %r — using fallback=%s", reply, fallback)
    return fallback


__all__ = ["LLMJudgeScorer", "JUDGE_SYSTEM_PROMPT"]
