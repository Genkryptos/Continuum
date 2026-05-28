"""
evals/longmemeval/decompose_guard.py
====================================
Intent-preserving guard for decomposed sub-questions.

The decomposer (see :func:`evals.longmemeval.decompose.build_decompose_prompt`)
asks an LLM to break the user's question into atomic sub-questions.
LLMs occasionally *change the question's intent* while decomposing —
the classic failure modes the 500-row analysis surfaced:

* Original ``"What is the painting worth?"`` becomes
  ``"What amount did I pay for the painting?"``. The answer-type
  flipped from a current valuation to a historical purchase amount.
* Original ``"Where did I complete my Bachelor's degree?"`` becomes
  ``"What is a good university?"``. The object lost its specificity
  ("Bachelor's degree" disappeared) and the question type drifted
  from a personal LOCATION lookup to a generic ENTITY_NAME.

This module compares each sub-question against the original on five
axes and **rejects** any sub-question that drifts. Rejected
sub-questions never reach the retriever / answerer; if every
sub-question is rejected, the original question is reinstated as the
sole lookup.

Drift axes
----------
1. **answer_type_changes** — :func:`question_type.classify` returns a
   different type than the original (and the original was not
   ``GENERIC_FACT``).
2. **relation_changes** — the question's *verb keyword* maps to a
   different canonical relation. ``"worth"`` (→ ``is_worth``) and
   ``"pay"`` (→ ``paid``) cannot coexist.
3. **object_changes** — the sub-question dropped the original's
   distinctive content noun (``"painting"``, ``"Bachelor's"``) or
   replaced it with a different concrete noun.
4. **temporal_scope_changes** — the original carried a year / date /
   ``"today"`` / ``"recently"`` phrase and the sub-question lost it
   or replaced it with an incompatible phrase.
5. **different_fact** — catch-all: the sub-question and original share
   no content words at all (i.e. the rewrite is on a completely
   different topic).

Telemetry
---------
:class:`DecompositionGuardResult` records the kept sub-questions, the
rejected ones with their reasons, and a ``fallback_to_original`` flag
so the runner can write the breakdown into the per-row JSONL trace.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from evals.longmemeval.question_type import QuestionType, classify

log = logging.getLogger(__name__)


# ─── Result dataclasses ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RejectedSubquestion:
    """One sub-question that failed the guard, with its drift reason."""

    subquestion: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"subquestion": self.subquestion, "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class DecompositionGuardResult:
    """
    Outcome of guarding a decomposition.

    Attributes
    ----------
    kept:
        Sub-questions that passed every drift check, in input order.
    rejected:
        Sub-questions that failed at least one drift check.
    fallback_to_original:
        ``True`` when ``kept`` is empty after guarding — the runner
        should use the original question as the only lookup.
    """

    kept: list[str]
    rejected: list[RejectedSubquestion]
    fallback_to_original: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": list(self.kept),
            "rejected": [r.to_dict() for r in self.rejected],
            "fallback_to_original": self.fallback_to_original,
            "n_kept": len(self.kept),
            "n_rejected": len(self.rejected),
        }


# ─── Keyword → canonical relation map (mirrors the verifier's) ─────────────


_RELATION_KEYWORDS: dict[str, str] = {
    "worth":     "is_worth",
    "value":     "is_worth",
    "valued":    "is_worth",
    "cost":      "is_worth",
    "costs":     "is_worth",
    "pay":       "paid",
    "paid":      "paid",
    "spent":     "paid",
    "spend":     "paid",
    "took":      "duration_of",
    "lasted":    "duration_of",
    "where":     "located_at",
    "got":       "got_at",
    "earned":    "got_at",
    "completed": "got_at",
    "finished":  "got_at",
    "graduated": "got_at",
    "when":      "date",
    "who":       "person",
    # NOTE: ``how many`` / ``how much`` are quantifier *shapes*, not
    # relations — "how much is X worth?" and "how much did I pay?"
    # both contain "how much" but mean different things. Leaving
    # them out of the map preserves the relation-drift signal.
}


def _relations_for_question(text: str) -> set[str]:
    """Return the set of canonical relations implied by *text*'s keywords."""
    out: set[str] = set()
    lower = text.lower()
    for kw, rel in _RELATION_KEYWORDS.items():
        if re.search(rf"\b{kw}\b", lower):
            out.add(rel)
    return out


# ─── Content-word + temporal parsing ───────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d+")
#: Stopwords mirror the ranker's. Question words / determiners / common
#: helpers go; concrete nouns stay.
_GUARD_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on",
    "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "i", "me", "my", "you", "your", "we", "us", "our", "they",
    "them", "this", "that", "it", "its", "if", "then",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "do", "does", "did", "have", "has", "had", "can", "could",
    "should", "would", "will", "shall",
    "many", "much", "few", "several", "long", "lot", "more", "less",
})


def _content_words(text: str) -> set[str]:
    return {
        t.lower() for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _GUARD_STOPWORDS and len(t) > 1
    }


_TEMPORAL_RE = re.compile(
    r"\b(?:\d{4}"                                       # year
    r"|today|yesterday|tomorrow|tonight"
    r"|recently|currently|now"
    r"|last\s+(?:week|month|year|sunday|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|night)"
    r"|next\s+(?:week|month|year)"
    r"|in\s+\d{4}"
    r"|since\s+\d{4}|before\s+\d{4}|after\s+\d{4}"
    r"|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r"|mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:rs(?:day)?)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)


def _temporal_phrases(text: str) -> set[str]:
    """Return lowercase set of temporal markers found in *text*."""
    return {m.group(0).lower() for m in _TEMPORAL_RE.finditer(text or "")}


# ─── The five drift checks ────────────────────────────────────────────────


def _check_answer_type_changes(
    original: str, sub: str,
) -> tuple[bool, str]:
    """Reject when classify(sub) drifts from a typed original."""
    orig_type = classify(original)
    sub_type = classify(sub)
    if orig_type == sub_type:
        return False, ""
    # GENERIC_FACT is the catch-all bucket — drifting INTO it from a
    # typed original is suspicious; drifting OUT of it is fine (the
    # decomposer often sharpens generic questions).
    if orig_type == QuestionType.GENERIC_FACT:
        return False, ""
    # TEMPORAL_REASONING / KNOWLEDGE_UPDATE / MULTI_SESSION are
    # compositional types — a sub-question that resolves to a more
    # primitive type (DATE_TIME, LOCATION) is expected and not drift.
    if orig_type in {
        QuestionType.TEMPORAL_REASONING,
        QuestionType.KNOWLEDGE_UPDATE,
        QuestionType.MULTI_SESSION,
    }:
        return False, ""
    return True, f"answer_type_changed:{orig_type.value}→{sub_type.value}"


def _check_relation_changes(
    original: str, sub: str,
) -> tuple[bool, str]:
    """
    Reject when the sub introduces a different canonical relation than
    the original. Symmetric: both sides must share at least one
    relation when the original had any.
    """
    orig_rels = _relations_for_question(original)
    sub_rels = _relations_for_question(sub)
    if not orig_rels:
        # Original was relation-neutral; any relation in the sub is OK.
        return False, ""
    if not sub_rels:
        # Sub has no recognised relation — neutral, don't reject.
        return False, ""
    if orig_rels & sub_rels:
        return False, ""
    return True, (
        f"relation_changed:{sorted(orig_rels)}→{sorted(sub_rels)}"
    )


#: Concrete content words that, when dropped during decomposition,
#: indicate the question's *object* slot changed. These are the
#: words that survive stopword filtering — proper nouns, distinctive
#: common nouns like "painting" / "bookshelf" / "Bachelor's".
def _check_object_changes(
    original: str, sub: str,
) -> tuple[bool, str]:
    """Reject when the sub dropped the original's distinctive nouns."""
    orig_words = _content_words(original)
    sub_words = _content_words(sub)
    if not orig_words:
        return False, ""
    # If the sub shares NONE of the original's content words, it's
    # talking about a completely different thing — strong reject.
    shared = orig_words & sub_words
    if not shared:
        return True, f"object_dropped:{sorted(orig_words)}→{sorted(sub_words)}"
    # If the original had several content words and the sub dropped
    # ALL but generic noise, treat that as object drift too. We require
    # that at least one of the original's "anchor" tokens (everything
    # except common verbs) survives in the sub.
    orig_anchors = orig_words - _COMMON_VERB_TOKENS
    if orig_anchors and not (orig_anchors & sub_words):
        return True, (
            f"object_anchor_dropped:{sorted(orig_anchors)}→{sorted(sub_words)}"
        )
    return False, ""


#: Common action verbs that don't anchor the question's object.
#: Dropping these is fine; dropping "painting" or "Bachelor's" is not.
_COMMON_VERB_TOKENS = frozenset({
    "get", "got", "buy", "bought", "make", "made", "pay", "paid",
    "spent", "spend", "took", "take", "give", "gave", "send", "sent",
    "find", "found", "earn", "earned", "complete", "completed",
    "finish", "finished", "graduate", "graduated", "attend", "attended",
    "visit", "visited", "use", "used", "go", "went", "come", "came",
    "see", "saw", "do", "done", "live", "lived", "work", "worked",
    "amount",  # asking "what amount" is a request shape, not an anchor noun
})


def _check_temporal_scope_changes(
    original: str, sub: str,
) -> tuple[bool, str]:
    """Reject when the sub adds, drops, or contradicts a temporal phrase."""
    orig_temporal = _temporal_phrases(original)
    sub_temporal = _temporal_phrases(sub)
    if orig_temporal and not sub_temporal:
        return True, f"temporal_dropped:{sorted(orig_temporal)}"
    if orig_temporal and sub_temporal and not (orig_temporal & sub_temporal):
        return True, (
            f"temporal_changed:{sorted(orig_temporal)}→{sorted(sub_temporal)}"
        )
    # Original was temporally neutral but the sub *adds* a specific
    # date/year — that narrows scope artificially.
    if not orig_temporal and sub_temporal:
        # Allow vague temporal additions (today/now/currently are
        # OK to add) but reject specific dates / years.
        specific = {
            t for t in sub_temporal
            if re.match(r"^\d{4}$", t) or "before" in t or "after" in t
            or "since" in t
        }
        if specific:
            return True, f"temporal_added_specific:{sorted(specific)}"
    return False, ""


def _check_different_fact(
    original: str, sub: str,
) -> tuple[bool, str]:
    """
    Catch-all: when the sub-question and original share *no* content
    words AND no relations, it's a question about something else.
    """
    if not _content_words(original) & _content_words(sub) and not (
        _relations_for_question(original) & _relations_for_question(sub)
    ):
        return True, "different_fact"
    return False, ""


# ─── Public entry point ───────────────────────────────────────────────────


def guard_decomposition(
    original_question: str,
    subquestions: list[str],
) -> DecompositionGuardResult:
    """
    Compare each sub-question against ``original_question`` and reject
    drifts. Returns a :class:`DecompositionGuardResult`.

    Falls back to the original question when every sub-question is
    rejected — the caller can then retrieve once on the original
    instead of going degraded with no lookup at all.

    When the decomposer returned a single sub-question identical to
    the original, this is a pass-through.
    """
    kept: list[str] = []
    rejected: list[RejectedSubquestion] = []
    norm_original = original_question.strip().lower()

    for sub in subquestions:
        sub_clean = sub.strip()
        if not sub_clean:
            continue
        # An identical-after-normalisation sub-question is the
        # decomposer's "this is already atomic" output — keep it.
        if sub_clean.lower() == norm_original:
            kept.append(sub_clean)
            continue

        for check in (
            _check_answer_type_changes,
            _check_relation_changes,
            _check_object_changes,
            _check_temporal_scope_changes,
            _check_different_fact,
        ):
            drifted, reason = check(original_question, sub_clean)
            if drifted:
                rejected.append(RejectedSubquestion(sub_clean, reason))
                log.info(
                    "decomp_guard reject %r → %r (%s)",
                    original_question[:80], sub_clean[:80], reason,
                )
                break
        else:
            kept.append(sub_clean)

    fallback = not kept
    if fallback:
        kept = [original_question.strip()]
        log.info(
            "decomp_guard fallback to original: all %d sub-questions rejected",
            len(rejected),
        )
    return DecompositionGuardResult(
        kept=kept,
        rejected=rejected,
        fallback_to_original=fallback,
    )


__all__ = [
    "DecompositionGuardResult",
    "RejectedSubquestion",
    "guard_decomposition",
]
