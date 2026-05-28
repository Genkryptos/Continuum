"""
evals/longmemeval/structured_finalizer.py
=========================================
Match a parsed :class:`QueryIntent` to a list of
:class:`StructuredClaim` and return the answer string.

Design rules
------------
* **Evidence-locked**: the answer IS ``claim.object``. No LLM
  rewriting, no COUNT override, no validator second-guessing.
* **Relation-first ranking**: a claim whose relation matches the
  intent's relation always outranks a claim that only mentions the
  topic. This is the cure for "Stanford" winning when "Bachelor's"
  was asked.
* **Constraint match**: each constraint that aligns
  (``recipient: sister`` ↔ ``recipient: sister``) adds a meaningful
  boost. A constraint mismatch is a soft penalty — sometimes the
  question's constraint is broader than the claim's.
* **Soft fallback**: if no claim matches the relation, the finalizer
  returns ``None`` rather than guessing. The caller (the adapter)
  treats that as "no structured answer; try the existing
  claim-first head".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from evals.longmemeval.query_intent import QueryIntent
from evals.longmemeval.scaffold_filter import (
    is_label_metadata_line,
    is_scaffold_text,
)
from evals.longmemeval.structured_claims import StructuredClaim

# ─── Answerability gate ────────────────────────────────────────────────────


#: Per-``answer_shape`` predicates. The gate runs on each ranked
#: candidate's ``object`` field; the first candidate whose object
#: passes the shape check is returned. When no candidate passes, the
#: finalizer returns None so the caller falls through.
_KNOWN_STORE_NAMES: frozenset[str] = frozenset({
    "best buy", "amazon", "target", "walmart", "ikea", "costco",
    "walgreens", "cvs", "whole foods", "trader joe's", "kroger",
    "safeway", "publix", "apple store", "home depot", "lowe's",
    "starbucks", "mcdonald's", "macy's", "nordstrom", "zara", "h&m",
    "uniqlo", "gap", "nike", "adidas", "etsy", "ebay",
})

_OCCUPATION_TITLE_RE = re.compile(
    r"\b(?:specialist|engineer|manager|developer|analyst|director|"
    r"officer|consultant|designer|architect|scientist|coordinator|"
    r"representative|advisor|adviser|teacher|nurse|doctor|chef|"
    r"writer|editor|attorney|lawyer|accountant|broker|trader|"
    r"associate|principal|head|chief|lead|owner|founder|entrepreneur|"
    r"intern|assistant|technician|operator|administrator|supervisor|"
    r"clerk|cashier|server|barista|waiter|waitress|driver|pilot|"
    r"paramedic|firefighter|police|professor|instructor|tutor|"
    r"researcher|investigator|reporter|journalist|photographer|"
    r"musician|artist|painter|sculptor|actor|actress|programmer|"
    r"strategist|copywriter|recruiter|ceo|cto|cfo|coo|"
    r"president|vp|partner|salesperson|salesman|saleswoman|realtor|"
    r"agent|trainee|apprentice)\b",
    re.IGNORECASE,
)


def _answer_shape_ok(answer: str, answer_shape: str) -> bool:
    """
    Per-shape sanity check on the candidate object.

    Generic checks fire on every shape:
      * scaffold/heading/label is never acceptable
      * empty after strip is never acceptable

    Shape-specific:
      ``occupation_title``  must contain a known head-noun
                            (specialist/engineer/…).
      ``purchased_item``    must not be a known store name; must not
                            be a bare heading; should look like a
                            noun phrase, not a verb phrase.
      Others pass through with the generic checks only.
    """
    a = answer.strip()
    if not a:
        return False
    if is_scaffold_text(a):
        return False
    if is_label_metadata_line(a):
        return False
    # Bold/underscore-wrapped headings.
    if (a.startswith("**") and a.endswith("**")) or \
       (a.startswith("__") and a.endswith("__")):
        return False

    if answer_shape == "occupation_title":
        return bool(_OCCUPATION_TITLE_RE.search(a))
    if answer_shape == "purchased_item":
        # Strip leading determiner for the store-name check.
        bare = re.sub(r"^(?:a|an|the|some)\s+", "", a, flags=re.IGNORECASE).lower()
        if bare in _KNOWN_STORE_NAMES:
            return False
        # Must look like a noun phrase ("a yellow dress"), not just a
        # store / proper noun on its own. Allow if it starts with a
        # determiner OR has 2+ tokens.
        if re.match(r"^(?:a|an|the|some)\s+\w+", a, re.IGNORECASE):
            return True
        return len(a.split()) >= 2
    # Other shapes — let through after the generic checks.
    return True

# ─── Ranking ──────────────────────────────────────────────────────────────


def rank_structured_claims(
    intent: QueryIntent,
    claims: Iterable[StructuredClaim],
) -> list[StructuredClaim]:
    """
    Rank claims for ``intent``. Returns the sorted list — claims
    with score ≤ 0 are dropped.

    Scoring (relation-first):
      +5.0  relation matches the intent's relation
      +1.0  per matching constraint key/value
      -0.5  per intent-constraint key NOT present on the claim
      +0.5  source_role == "user" (autobiographical claims)
      +0.2  confidence (tiebreaker)
    """
    out: list[tuple[float, StructuredClaim]] = []
    for c in claims:
        score = 0.0
        if c.relation == intent.relation:
            score += 5.0
        if score <= 0:
            # Relation didn't match → drop. Mentioning the topic is
            # not enough — the claim must encode the asked relation.
            continue
        for key, want in intent.constraints.items():
            have = c.qualifiers.get(key, "").lower()
            if have and want.lower() == have:
                score += 1.0
            elif (have and want.lower() in have) or (have and have in want.lower()):
                score += 0.6
            else:
                # Intent specified the constraint but the claim is
                # silent on it — mild penalty, not a hard reject.
                score -= 0.3
        if c.source_role == "user":
            score += 0.5
        score += 0.2 * c.confidence
        out.append((score, c))
    out.sort(key=lambda pair: -pair[0])
    return [c for _s, c in out]


# ─── Finalizer ────────────────────────────────────────────────────────────


def finalize_fact_answer(
    intent: QueryIntent,
    claims: Iterable[StructuredClaim],
) -> tuple[str, dict[str, Any]] | None:
    """
    Return ``(answer, diagnostics)`` when a structured claim matches
    the intent. Returns ``None`` when no relation-matching claim was
    found — caller falls through.

    The answer is the claim's ``object`` field directly. The
    diagnostics dict carries the matched claim's metadata so the
    JSONL trace can show the structured-path decision.
    """
    if not intent.matched:
        return None
    ranked = rank_structured_claims(intent, claims)
    if not ranked:
        return None
    # Answerability gate — walk past candidates whose object doesn't
    # match the intent's answer_shape. A heading/store/scaffold that
    # somehow survived ranking is rejected here, and the next claim
    # is tried.
    rejected: list[dict[str, Any]] = []
    for cand in ranked:
        answer = cand.object_.strip() or cand.source_span.strip()
        if not answer:
            rejected.append({"claim": cand.to_dict(), "reason": "empty_object"})
            continue
        if not _answer_shape_ok(answer, intent.answer_shape):
            rejected.append({
                "claim": cand.to_dict(),
                "reason": f"shape_mismatch:{intent.answer_shape}",
            })
            continue
        diagnostics: dict[str, Any] = {
            "intent": intent.to_dict(),
            "matched_claim": cand.to_dict(),
            "n_candidate_claims": len(ranked),
            "all_candidate_claims": [c.to_dict() for c in ranked[:5]],
            "rejected_by_answerability_gate": rejected,
        }
        return answer, diagnostics
    return None


__all__ = [
    "finalize_fact_answer",
    "rank_structured_claims",
]
