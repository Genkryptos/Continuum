"""
evals/longmemeval/query_intent.py
=================================
Parse a benchmark question into a structured intent.

The intent carries:

* ``relation`` — the canonical relation name (matches
  :data:`evals.longmemeval.structured_claims._RELATION_EXTRACTORS` keys).
* ``constraints`` — qualifiers extracted from the question wording
  (``{"recipient": "sister", "occasion": "birthday"}``).
* ``answer_shape`` — what the answer should look like (one of a
  small closed set: ``entity_or_phrase``, ``occupation_title``,
  ``field_of_study``, ``comparative_phrase``, ``count_with_unit``,
  ``date``).
* ``subject`` — what the question is about. Defaults to ``"user"``
  for autobiographical questions; can be a specific entity for
  third-person facts.

The parser is deliberately small: one table, one walk, returns the
first match. Adding a new relation = one extractor in
``structured_claims.py`` + one row here.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any


@dataclasses.dataclass(frozen=True)
class QueryIntent:
    """Structured interpretation of a benchmark question."""

    relation: str
    constraints: dict[str, str]
    answer_shape: str
    subject: str = "user"

    @property
    def matched(self) -> bool:
        return bool(self.relation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "constraints": dict(self.constraints),
            "answer_shape": self.answer_shape,
            "subject": self.subject,
            "matched": self.matched,
        }


# ─── Constraint extractors (one per intent that needs them) ────────────────


def _bought_gift_constraints(question: str) -> dict[str, str]:
    """Extract ``recipient`` + ``occasion`` from the gift question."""
    out: dict[str, str] = {}
    # "for my sister's birthday" / "for my brother's graduation"
    m = re.search(
        r"\bfor\s+my\s+(?P<recipient>\w+)['']?s?\s+"
        r"(?P<occasion>birthday|anniversary|graduation|wedding|"
        r"christmas|housewarming)",
        question, re.IGNORECASE,
    )
    if m:
        out["recipient"] = m.group("recipient").lower()
        out["occasion"] = m.group("occasion").lower()
    return out


def _count_packed_constraints(question: str) -> dict[str, str]:
    """Extract ``item`` and optional ``destination`` from the count question."""
    out: dict[str, str] = {}
    # "How many shirts did I pack for Costa Rica?"
    m = re.search(
        r"\bhow\s+many\s+(?P<item>\w+)\s+(?:did\s+I\s+|i\s+)?"
        r"(?:pack|bring|take)",
        question, re.IGNORECASE,
    )
    if m:
        out["item"] = m.group("item").rstrip("s").lower()
    m = re.search(
        r"\b(?:for|to|on)\s+(?:the\s+|my\s+)?"
        r"(?P<destination>(?:trip|vacation|holiday)\s+to\s+\w+|"
        r"[A-Z][\w\- ]+?)(?=[.!?,;]|$)",
        question,
    )
    if m:
        out["destination"] = m.group("destination").strip().lower()
    return out


def _event_date_constraints(question: str) -> dict[str, str]:
    """Extract the event name from a When question."""
    out: dict[str, str] = {}
    # "When did I volunteer at the local animal shelter's fundraising dinner?"
    # → event = "fundraising dinner" (the trailing noun phrase)
    m = re.search(
        r"\b(?:the\s+|my\s+)?"
        r"(?P<event>[a-z][\w' ]*?(?:dinner|event|party|wedding|"
        r"graduation|conference|meeting|interview|appointment|"
        r"reunion|fundraiser|fundraising|ceremony|gathering))\b",
        question, re.IGNORECASE,
    )
    if m:
        out["event"] = m.group("event").strip().lower()
    return out


def _value_vs_paid_constraints(question: str) -> dict[str, str]:
    """Extract the asset whose value is being asked about."""
    out: dict[str, str] = {}
    m = re.search(
        r"\b(?:how\s+much\s+is\s+|what\s+is\s+)(?:the\s+|my\s+)?"
        r"(?P<asset>[a-z][\w' ]{2,40}?)\s+worth\b",
        question, re.IGNORECASE,
    )
    if m:
        out["asset"] = m.group("asset").strip().lower()
    return out


# ─── Intent table ─────────────────────────────────────────────────────────


#: One row per question shape. Matched top-to-bottom; first hit wins.
#: Each row carries a question-pattern regex, the canonical relation
#: name, an optional constraint-extractor, and the expected
#: ``answer_shape``.
_INTENT_TABLE: tuple[dict[str, Any], ...] = (
    # FACT_LOOKUP — previous occupation. Coverage:
    #   "What was my previous occupation?"
    #   "What was my previous job?"
    #   "What was my former role / past position?"
    #   "What role did I have before?"
    #   "What did I used to work as?"
    #   "What did I do for work before?"
    {
        "pattern": re.compile(
            r"\bwhat\s+(?:was|is)\s+my\s+"
            r"(?:previous|former|old|last|past)\s+"
            r"(?:occupation|job|role|position|career|profession)",
            re.IGNORECASE,
        ),
        "relation": "previous_occupation",
        "answer_shape": "occupation_title",
    },
    {
        "pattern": re.compile(
            r"\bwhat\s+(?:role|job|position|occupation|career)\s+"
            r"did\s+I\s+(?:have|hold|work|do)\s+(?:before|previously|in\s+the\s+past)",
            re.IGNORECASE,
        ),
        "relation": "previous_occupation",
        "answer_shape": "occupation_title",
    },
    {
        "pattern": re.compile(
            r"\bwhat\s+did\s+I\s+(?:used?\s+to|previously)\s+"
            r"(?:work|do)\s+(?:as|in)\b",
            re.IGNORECASE,
        ),
        "relation": "previous_occupation",
        "answer_shape": "occupation_title",
    },
    {
        "pattern": re.compile(
            r"\bwhat\s+did\s+I\s+do\s+for\s+(?:work|a\s+living)\s+"
            r"(?:before|previously|in\s+the\s+past)",
            re.IGNORECASE,
        ),
        "relation": "previous_occupation",
        "answer_shape": "occupation_title",
    },
    # FACT_LOOKUP — degree / field of study
    {
        "pattern": re.compile(
            r"\bwhat\s+(?:degree|field|major|discipline)\s+"
            r"(?:did\s+I\s+)?(?:get|graduate|earn|study|complete)",
            re.IGNORECASE,
        ),
        "relation": "degree_in",
        "answer_shape": "field_of_study",
    },
    # FACT_LOOKUP — birthday / occasion gift purchase
    {
        "pattern": re.compile(
            r"\bwhat\s+did\s+I\s+(?:buy|bought|get|pick\s+up|purchase|gift)\s+"
            r"(?:for\s+my\s+\w+['']?s?\s+"
            r"(?:birthday|anniversary|graduation|wedding|"
            r"christmas|housewarming))?",
            re.IGNORECASE,
        ),
        "relation": "bought_gift_for",
        "constraints_fn": _bought_gift_constraints,
        "answer_shape": "purchased_item",
    },
    # FACT_LOOKUP — value comparison vs paid
    {
        "pattern": re.compile(
            r"\bhow\s+much\s+is\s+.+?\s+worth\b"
            r".*?(?:paid|spent|in\s+terms\s+of)",
            re.IGNORECASE,
        ),
        "relation": "value_compared_to_paid",
        "constraints_fn": _value_vs_paid_constraints,
        "answer_shape": "comparative_phrase",
    },
    # FACT_LOOKUP — count of items packed
    {
        "pattern": re.compile(
            r"\bhow\s+many\s+\w+\s+(?:did\s+I\s+|i\s+)?"
            r"(?:pack|bring|take)",
            re.IGNORECASE,
        ),
        "relation": "count_packed",
        "constraints_fn": _count_packed_constraints,
        "answer_shape": "count_with_unit",
    },
    # FACT_LOOKUP — event date
    {
        "pattern": re.compile(
            r"\bwhen\s+(?:did|was|is)\s+(?:I\s+)?",
            re.IGNORECASE,
        ),
        "relation": "event_date",
        "constraints_fn": _event_date_constraints,
        "answer_shape": "date",
    },
)


def parse_intent(question: str) -> QueryIntent:
    """
    Return the :class:`QueryIntent` for ``question`` — or an unmatched
    intent (``intent.matched is False``) when no row in the table
    fires.

    Callers should treat unmatched intents as "no structured path
    available; fall through to the existing claim-first head".
    """
    if not question or not question.strip():
        return QueryIntent(relation="", constraints={}, answer_shape="")
    for row in _INTENT_TABLE:
        if row["pattern"].search(question):
            constraints_fn = row.get("constraints_fn")
            constraints = constraints_fn(question) if constraints_fn else {}
            return QueryIntent(
                relation=row["relation"],
                constraints=constraints,
                answer_shape=row["answer_shape"],
            )
    return QueryIntent(relation="", constraints={}, answer_shape="")


__all__ = [
    "QueryIntent",
    "parse_intent",
]
