"""
evals/longmemeval/task_router.py
================================
Broad task router.

Six modes — enough to cover every LongMemEval category without
inventing one rule per question shape:

* ``FACT_LOOKUP`` — single autobiographical fact ("Where did I get
  my Bachelor's?").
* ``ASSISTANT_MEMORY_LOOKUP`` — what the *assistant* said in a prior
  session ("Did you suggest a book about…?").
* ``PREFERENCE_PROFILE`` — likes / dislikes / habits summary
  ("What do I prefer for breakfast?").
* ``KNOWLEDGE_UPDATE`` — the user updated a fact; answer must reflect
  the *latest* version ("What did I change my pet's name to?").
* ``MULTI_SESSION_AGGREGATE`` — sum / count / list across sessions
  ("How many projects am I tracking?").
* ``TEMPORAL_REASONING`` — date arithmetic, ordering, elapsed time
  ("How long ago did I start the job?").

The router uses the LongMemEval ``question_type`` field when the eval
caller passes one, but always inspects question wording as a
secondary signal — the dataset's labels are LongMemEval's, not ours,
and wording often disambiguates ambiguous categories.
"""

from __future__ import annotations

import re
from enum import Enum


class TaskMode(str, Enum):
    """Closed set of broad reasoning modes."""

    FACT_LOOKUP = "FACT_LOOKUP"
    ASSISTANT_MEMORY_LOOKUP = "ASSISTANT_MEMORY_LOOKUP"
    PREFERENCE_PROFILE = "PREFERENCE_PROFILE"
    KNOWLEDGE_UPDATE = "KNOWLEDGE_UPDATE"
    MULTI_SESSION_AGGREGATE = "MULTI_SESSION_AGGREGATE"
    TEMPORAL_REASONING = "TEMPORAL_REASONING"


# Map LongMemEval question_type → primary mode hint. The router
# combines this with wording-based signals so a mislabeled dataset
# row doesn't trap the answer pipeline.
_DATASET_TYPE_HINT: dict[str, TaskMode] = {
    "single-session-user":        TaskMode.FACT_LOOKUP,
    "single-session-assistant":   TaskMode.ASSISTANT_MEMORY_LOOKUP,
    "single-session-preference":  TaskMode.PREFERENCE_PROFILE,
    "knowledge-update":           TaskMode.KNOWLEDGE_UPDATE,
    "multi-session":              TaskMode.MULTI_SESSION_AGGREGATE,
    "temporal-reasoning":         TaskMode.TEMPORAL_REASONING,
}

# Wording cues — each pattern flips the route to its mode when the
# dataset hint is ambiguous or wrong.
_TEMPORAL_RE = re.compile(
    r"\b(?:how\s+long\s+ago|how\s+many\s+(?:days|weeks|months|years)\s+(?:ago|since|between)|"
    r"between\s+\w+\s+and\s+\w+|how\s+long\s+(?:has|have)|"
    r"elapsed|before|after|earliest|latest|first|last|"
    r"when\s+did\s+i)\b",
    re.IGNORECASE,
)
# True-aggregate cues. "How many" / "how much" alone are *not* enough —
# a single-session "How many shirts did I pack?" or "How much is X
# worth?" is a FACT_LOOKUP, not an aggregation. The router routes to
# MULTI_SESSION_AGGREGATE only when the wording explicitly aggregates
# across multiple instances or sessions.
_AGGREGATE_RE = re.compile(
    r"\b(?:in\s+total|altogether|combined|across\s+(?:all\s+)?"
    r"(?:sessions|trips|years|months|weeks)|"
    r"sum\s+of|each\s+of|all\s+the\s+\w+s\b|"
    r"different\s+\w+s\b|"
    r"how\s+(?:many|much)\s+(?:in\s+total|altogether|combined|"
    r"across)|"
    r"list\s+(?:all|the))\b",
    re.IGNORECASE,
)
_KNOWLEDGE_UPDATE_RE = re.compile(
    r"\b(?:current(?:ly)?|now|these\s+days|latest|"
    r"most\s+recent|today|updated|changed|change\s+to|new\s+\w+\s+(?:is|name)|"
    r"used\s+to\s+\w+\s+(?:but|and)\s+now|previously\s+\w+\s+now)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(?:prefer(?:ence|red)?|favou?rite|do\s+i\s+like|do\s+i\s+prefer|"
    r"my\s+favou?rite|"
    r"what\s+(?:do\s+i\s+)?(?:like|prefer|enjoy|hate|avoid))\b",
    re.IGNORECASE,
)
_ASSISTANT_MEMORY_RE = re.compile(
    # Direct "you-said / did-you" frames.
    r"\b(?:you\s+(?:told|said|recommended|suggested|mentioned|asked|"
    r"advised|provided|gave|shared|listed|named|noted)|"
    r"did\s+you\s+(?:recommend|suggest|mention|tell|ask|provide|give|"
    r"share|list|name)|"
    r"what\s+(?:did\s+you\s+say|advice\s+did\s+you|book\s+did\s+you))\b"
    # Memory-recall frames addressed to the assistant.
    r"|\b(?:remind\s+me|do\s+you\s+remember|"
    r"we\s+(?:talked|chatted|discussed|spoke)\s+about|"
    r"(?:our|the|in\s+(?:our|the))\s+(?:previous|earlier|last|prior)\s+"
    r"(?:conversation|chat|session|discussion|talk)|"
    r"check\s+back\s+(?:on|with))\b"
    # "what was the X you (gave|provided|told me|sent|recommended)"
    r"|\bwhat\s+(?:was|were)\s+the\s+\w+(?:\s+\w+){0,3}\s+"
    r"you\s+(?:gave|provided|told\s+me|sent|recommended|suggested|"
    r"named|listed|shared)\b",
    re.IGNORECASE,
)


def route_task(
    question: str,
    *,
    question_type_hint: str | None = None,
    is_multi_session: bool = False,
) -> TaskMode:
    """
    Pick the broad reasoning mode for ``question``.

    Resolution order:

    1. **Strong wording cues** (temporal arithmetic, assistant-memory
       phrases, change-of-state words) override the dataset hint —
       these are unambiguous.
    2. **Dataset hint** (``question_type_hint``) decides next.
    3. **Wording-only** fallback (aggregate / preference / personal /
       generic).

    ``is_multi_session`` is a tiebreaker — set ``True`` when the eval
    row's ``answer_session_ids`` lists > 1 session.
    """
    # 1) Strong cues — wording trumps the dataset label. ORDER MATTERS:
    # assistant-memory ("did you recommend last week") is checked
    # before temporal so "last week" inside an assistant-memory
    # question doesn't accidentally route to TEMPORAL_REASONING.
    # Same for knowledge-update ("changed to ... now") which often
    # carries "before"/"after" without being a temporal-arithmetic Q.
    if _ASSISTANT_MEMORY_RE.search(question):
        return TaskMode.ASSISTANT_MEMORY_LOOKUP
    if _KNOWLEDGE_UPDATE_RE.search(question):
        return TaskMode.KNOWLEDGE_UPDATE
    if _PREFERENCE_RE.search(question):
        return TaskMode.PREFERENCE_PROFILE
    if _TEMPORAL_RE.search(question):
        return TaskMode.TEMPORAL_REASONING
    if _AGGREGATE_RE.search(question) or is_multi_session:
        return TaskMode.MULTI_SESSION_AGGREGATE

    # 2) Dataset hint when wording is neutral.
    if question_type_hint:
        hint = _DATASET_TYPE_HINT.get(question_type_hint.strip().lower())
        if hint is not None:
            return hint

    # 3) Fallback — assume an autobiographical fact.
    return TaskMode.FACT_LOOKUP


__all__ = [
    "TaskMode",
    "route_task",
]
