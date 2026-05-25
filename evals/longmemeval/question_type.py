"""
evals/longmemeval/question_type.py
==================================
Deterministic question-type classifier.

Used by the answerer before final synthesis so:

* the candidate extractor knows which structured fields to mine
  (numbers vs dates vs locations);
* the answer-shape validator knows what the final answer must look
  like (a number for COUNT, a place for LOCATION, a date for
  DATE_TIME, etc.);
* the don't-IDK guard knows which candidate types matter for refusing
  to refuse.

The classifier is pure regex over the question text — no LLM call, no
training. Patterns are ordered: the first matching type wins.
"""

from __future__ import annotations

import re
from enum import Enum


class QuestionType(str, Enum):
    """Closed set of question shapes the downstream pipeline knows about."""

    COUNT = "COUNT"
    DATE_TIME = "DATE_TIME"
    DURATION = "DURATION"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    ENTITY_NAME = "ENTITY_NAME"
    PREFERENCE = "PREFERENCE"
    TEMPORAL_REASONING = "TEMPORAL_REASONING"
    KNOWLEDGE_UPDATE = "KNOWLEDGE_UPDATE"
    MULTI_SESSION = "MULTI_SESSION"
    ABSTAIN = "ABSTAIN"
    GENERIC_FACT = "GENERIC_FACT"


# Patterns are evaluated top-to-bottom; first match wins. Order matters
# — knowledge-update wording ("changed", "used to") trumps temporal
# wording ("now"), which in turn trumps generic ENTITY_NAME.
_RULES: list[tuple[re.Pattern[str], QuestionType]] = [
    # ABSTAIN — explicit "did I mention", "did I ever say"
    (re.compile(r"\b(?:did i (?:ever )?(?:mention|say|tell|talk about)|"
                r"have i (?:ever )?(?:mentioned|said|told))\b", re.I),
     QuestionType.ABSTAIN),

    # COUNT — "how many", "how much"
    (re.compile(r"\bhow\s+(?:many|much)\b", re.I), QuestionType.COUNT),

    # DURATION — "how long", "take to", "duration"
    (re.compile(r"\b(?:how\s+long|duration|takes?\s+to|"
                r"(?:taken|spent|spend)\s+(?:to|on))\b", re.I),
     QuestionType.DURATION),

    # DATE_TIME — "when", "what date", "which day", "what time"
    (re.compile(r"\b(?:when|what\s+(?:date|day|month|year|time)|"
                r"which\s+(?:day|month|year))\b", re.I),
     QuestionType.DATE_TIME),

    # KNOWLEDGE_UPDATE — words signalling change
    (re.compile(r"\b(?:previously|used\s+to|change[ds]?|changed?|updated?|"
                r"old(?:er)?|new(?:er)?|(?:before|after)\s+(?:and|vs|or)|"
                r"current(?:ly)?\s+versus|was\s+\w+\s+now\s+is)\b", re.I),
     QuestionType.KNOWLEDGE_UPDATE),

    # TEMPORAL_REASONING — before/after/first/last/latest/this year/etc.
    (re.compile(r"\b(?:before|after|earliest|latest|first|last|"
                r"this\s+(?:year|month|week)|last\s+(?:year|month|week)|"
                r"(?:order|sequence|timeline|chronolog))\b", re.I),
     QuestionType.TEMPORAL_REASONING),

    # PREFERENCE — like/prefer/favorite/dislike/avoid
    (re.compile(r"\b(?:prefer(?:ence|red)?|favou?rite|like\s+to|"
                r"dislike|avoid|favou?r)\b", re.I),
     QuestionType.PREFERENCE),

    # LOCATION — "where"
    (re.compile(r"\bwhere\b", re.I), QuestionType.LOCATION),

    # PERSON — "who"
    (re.compile(r"\bwho\b", re.I), QuestionType.PERSON),

    # ENTITY_NAME — "what is the name", "what's it called"
    (re.compile(r"\b(?:what\s+(?:is|are|'s)\s+(?:the\s+)?name|"
                r"what(?:'s| is)\s+it\s+called|"
                r"name\s+of\s+(?:the\s+)?[a-z]+)\b", re.I),
     QuestionType.ENTITY_NAME),
]


# Multi-session is orthogonal to question shape — it's a "did this
# question's evidence span sessions?" tag. The eval harness sets it
# from dataset metadata; classify() returns it only when explicit
# multi-session signals appear in the text itself.
_MULTI_SESSION_HINT_RE = re.compile(
    r"\b(?:across|total\s+(?:over|across)|combined|in\s+total|"
    r"all\s+the\s+(?:sessions|conversations))\b",
    re.I,
)


def classify(question: str) -> QuestionType:
    """
    Return the most-specific :class:`QuestionType` matched by *question*.

    Falls back to :data:`QuestionType.GENERIC_FACT` when no pattern
    fires — that bucket is the safety net, not a real shape.
    """
    if not question:
        return QuestionType.GENERIC_FACT
    for pattern, qtype in _RULES:
        if pattern.search(question):
            return qtype
    return QuestionType.GENERIC_FACT


def is_multi_session_hint(question: str) -> bool:
    """Heuristic flag — True when the question explicitly aggregates across sessions."""
    return bool(_MULTI_SESSION_HINT_RE.search(question or ""))


__all__ = ["QuestionType", "classify", "is_multi_session_hint"]
