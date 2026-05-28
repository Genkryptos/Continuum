"""
evals/longmemeval/claim_verifier.py
===================================
Verifier stage that runs between candidate extraction and final answer
generation. Filters out candidates that survive the type-and-overlap
gate but don't actually answer the original question.

Why this exists
---------------
The grounded-claim extractor (see :mod:`evals.longmemeval.candidates`)
populates ``subject``/``relation``/``object_``/``source_span`` so the
*ranker* can pick the best one. But the ranker is a sorter, not a
gate — it always returns a top candidate, even when the top candidate
is wrong. Three observed failure modes:

* A ``count`` candidate (``value="1"`` from "I made 1 painting") leaks
  into the answer for a ``"what is the painting worth?"`` question
  because the subject happens to mention "painting".
* A hypothetical ``location`` candidate (``"Stanford"`` from "looking
  at Stanford for Master's") gets surfaced for a ``Bachelor's degree``
  question even though its subject is "Master's".
* A factual claim with the wrong answer-type (e.g. a ``date``
  candidate where the question wanted a ``location``) survives
  because the answerer is too literal about the model's hints.

Contract
--------
:func:`verify_claim` returns a :class:`VerifierResult` with::

    verdict: "PASS" | "FAIL"
    reason:  short string ("subject_mismatch", "wrong_relation", ...)
    candidate: the candidate that was checked

Five deterministic checks run in order; the first FAIL short-circuits:

1. **answer_in_source_span** — the question's content words must
   intersect with the candidate's source_span / claim. Zero overlap
   means the candidate is from an unrelated turn.
2. **subject_matches_question** — the candidate's ``subject`` must
   share at least one content word with the question's subject
   phrase. This kills "Stanford for Master's" answering a
   "Bachelor's" question.
3. **answer_type_compatible** — the candidate's ``answer_type`` /
   ``candidate_type`` must be in the question type's expected set.
   A ``count`` candidate can't answer a ``"worth"`` question.
4. **factual_relation** — when the question implies a completed event
   ("did", "got", "have"), candidates with a hypothetical relation
   (``looking_at``, ``considering``) are dropped.
5. **no_stronger_conflict** — if another candidate has a *higher*
   subject-overlap on the same subject AND the same answer_type,
   this candidate is dropped (the conflict-resolution check).

An optional :class:`LLMVerifier` runs after the deterministic stage
behind a config flag — handy for the long tail of edge cases that
regex-driven checks miss. The deterministic stage is the primary
defence; LLM verification is the safety net.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Awaitable, Iterable
from typing import Any, Literal, Protocol

from evals.longmemeval.candidates import (
    FACTUAL_RELATIONS,
    HYPOTHETICAL_RELATIONS,
    Candidate,
    _question_subject_tokens,
)
from evals.longmemeval.question_type import QuestionType
from evals.longmemeval.scaffold_filter import (
    is_prompt_echo,
    is_scaffold_text,
    is_user_request_sentence,
)

log = logging.getLogger(__name__)


# ─── Result dataclass ──────────────────────────────────────────────────────


VerifierVerdict = Literal["PASS", "FAIL"]


@dataclasses.dataclass(frozen=True)
class VerifierResult:
    """One verifier verdict on one candidate."""

    candidate: Candidate
    verdict: VerifierVerdict
    reason: str

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "candidate": self.candidate.to_dict(),
        }


# ─── Question-type → expected answer/candidate types ───────────────────────

#: Mirror of the ranker's table — kept local so this module can be
#: imported without coupling to the ranker's internals. When the
#: question type is unknown the check is skipped (PASS).
_EXPECTED_ANSWER_TYPES: dict[QuestionType, set[str]] = {
    QuestionType.COUNT:              {"count"},
    QuestionType.DURATION:           {"duration"},
    QuestionType.DATE_TIME:          {"date"},
    QuestionType.LOCATION:           {"location"},
    QuestionType.PERSON:             {"person"},
    QuestionType.ENTITY_NAME:        {"entity"},
    QuestionType.PREFERENCE:         {"preference"},
    QuestionType.TEMPORAL_REASONING: {"date", "duration"},
    QuestionType.KNOWLEDGE_UPDATE:   {"old_fact", "new_fact"},
}

#: Keywords that signal a completed/factual question — used by check 4
#: to demand a factual relation. ``"considering"`` / ``"looking"`` in
#: the question intentionally bypass this so a hypothetical question
#: can still surface a hypothetical answer.
_COMPLETED_QUESTION_RE = re.compile(
    r"\b(?:did|got|have|had|finished|completed|earned|received|graduated|"
    r"took|spent|paid|wrote|made|bought|won)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_QUESTION_RE = re.compile(
    r"\b(?:considering|thinking|might|planning|looking\s+at)\b",
    re.IGNORECASE,
)
#: Keyword → relation map for the type-compat soft check. When the
#: question explicitly uses "worth", a non-``is_worth`` ``value``
#: candidate (e.g. relation=``paid``) is downgraded.
_RELATION_KEYWORD_MAP: dict[str, str] = {
    "worth":  "is_worth",
    "value":  "is_worth",
    "valued": "is_worth",
    "pay":    "paid",
    "paid":   "paid",
    "spent":  "paid",
    "spend":  "paid",
    "cost":   "is_worth",
}

#: Keyword → required candidate ``answer_type`` set. When the question
#: contains a value/amount keyword, ANY non-value candidate (e.g. a
#: stray ``count`` or ``location``) is rejected. This is what kills
#: ``"1 made"`` for a ``"what is the painting worth?"`` question even
#: though the question type is GENERIC_FACT and the candidate's
#: subject literally is "the painting".
_KEYWORD_ANSWER_TYPES: dict[str, set[str]] = {
    "worth":   {"value"},
    "value":   {"value"},
    "valued":  {"value"},
    "cost":    {"value", "amount"},
    "costs":   {"value", "amount"},
    "pay":     {"amount", "value"},
    "paid":    {"amount", "value"},
    "spent":   {"amount", "value"},
    "spend":   {"amount", "value"},
    "long":    {"duration"},
    "lasted":  {"duration"},
    "took":    {"duration"},
    "where":   {"location"},
    "when":    {"date"},
    "who":     {"person", "entity"},
}


# ─── Deterministic checks ──────────────────────────────────────────────────


def _content_words(text: str) -> set[str]:
    return _question_subject_tokens(text)


def _check_answer_in_source_span(
    question: str, candidate: Candidate,
) -> tuple[bool, str]:
    """
    1. Does the candidate's *grounding text* share any content words
    with the question?

    The grounding text is the union of ``claim`` (full sentence) and
    ``source_span`` (literal regex match) — checking either alone is
    too narrow:

    * ``source_span="at UCLA"`` has no overlap with "Bachelor's
      degree" but the full claim does.
    * A candidate without a parsed claim still has ``source_text``,
      which we use as the last fallback.

    Zero overlap across all three means the candidate genuinely
    comes from an unrelated turn.
    """
    q_tokens = _content_words(question)
    if not q_tokens:
        return True, ""
    grounding = " ".join(filter(None, (
        candidate.claim,
        candidate.source_span,
        candidate.source_text,
    )))
    grounding_tokens = _content_words(grounding)
    if not (q_tokens & grounding_tokens):
        return False, "span_no_overlap"
    return True, ""


def _check_subject_matches(
    question: str, candidate: Candidate,
) -> tuple[bool, str]:
    """2. Is the candidate's subject aligned with the question's subject?"""
    q_tokens = _content_words(question)
    if not q_tokens:
        return True, ""
    if not candidate.subject:
        # No subject was extracted — fall back to the source span check
        # so we don't reject perfectly good claims that just lacked a
        # parseable subject phrase.
        return True, ""
    subj_tokens = _content_words(candidate.subject)
    if not (q_tokens & subj_tokens):
        return False, "subject_mismatch"
    return True, ""


def _check_answer_type_compatible(
    question: str, candidate: Candidate,
    question_type: QuestionType | None,
) -> tuple[bool, str]:
    """3. Is the candidate's answer_type compatible with the question type?"""
    # Type-gate: when the question type narrows the expected answer
    # types, the candidate must be in that set.
    if question_type is not None and question_type in _EXPECTED_ANSWER_TYPES:
        wanted = _EXPECTED_ANSWER_TYPES[question_type]
        types = {candidate.answer_type, candidate.candidate_type} - {""}
        if not (types & wanted):
            return False, f"answer_type_mismatch:{candidate.candidate_type}!={sorted(wanted)}"
    # Keyword answer-type check — when the question contains a verb
    # like "worth" / "took" / "where", the candidate's answer_type must
    # be in the required set. This catches GENERIC_FACT questions where
    # the question_type gate is open but the keyword is unambiguous.
    # It's what rejects ``"1 made"`` (answer_type="count") on a
    # ``"worth"`` question even when both candidates share the subject
    # "painting".
    q_lower = question.lower()
    cand_types = {candidate.answer_type, candidate.candidate_type} - {""}
    for kw, required_types in _KEYWORD_ANSWER_TYPES.items():
        if (
            re.search(rf"\b{re.escape(kw)}\b", q_lower)
            and cand_types
            and not (cand_types & required_types)
        ):
            return False, (
                f"answer_type_mismatch_for_keyword:{kw}→{sorted(required_types)}"
            )
    # Relation-keyword soft check — when the question contains "worth"
    # we reject a ``paid`` candidate (and vice versa), because the
    # answer-type is technically right (both are ``value``) but the
    # specific relation is wrong for the question's verb.
    relation_map_values = set(_RELATION_KEYWORD_MAP.values())
    for kw, required_rel in _RELATION_KEYWORD_MAP.items():
        # Only enforce when (a) the keyword fires in the question and
        # (b) the candidate carries a relation that is *within* the
        # keyword map's vocabulary — otherwise we'd reject neutral
        # relations like ``duration_of``.
        if (
            re.search(rf"\b{re.escape(kw)}\b", q_lower)
            and candidate.relation
            and candidate.relation in relation_map_values
            and candidate.relation != required_rel
        ):
            return False, f"wrong_relation_for_keyword:{kw}→{required_rel}"
    return True, ""


def _check_factual_relation(
    question: str, candidate: Candidate,
) -> tuple[bool, str]:
    """4. Reject hypothetical claims when the question implies a completed event."""
    # If the question itself is hypothetical, the check is bypassed.
    if _HYPOTHETICAL_QUESTION_RE.search(question):
        return True, ""
    # The question implies completion → demand a factual relation.
    if _COMPLETED_QUESTION_RE.search(question) and (
        candidate.relation in HYPOTHETICAL_RELATIONS
        or candidate.relation == "looking_at"
    ):
        return False, f"hypothetical_relation:{candidate.relation}"
    # Even outside completed-event questions, a hypothetical claim with
    # an extremely low confidence (< 0.35) gets dropped.
    if (
        candidate.relation in HYPOTHETICAL_RELATIONS
        and candidate.confidence < 0.35
    ):
        return False, "low_confidence_hypothetical"
    return True, ""


def _check_no_stronger_conflict(
    question: str,
    candidate: Candidate, all_candidates: Iterable[Candidate],
) -> tuple[bool, str]:
    """
    5. Drop the candidate if another candidate of the same answer_type
    is a *better* fit for the question.

    Two "better-fit" criteria, in priority order:

    * **Question-overlap**: another candidate's ``subject`` shares
      strictly more content words with the question than this one's.
      This is what makes ``"IKEA table"`` (overlap=1 with the
      bookshelf question) lose to ``"IKEA bookshelf"`` (overlap=2)
      even though both are factual ``duration_of`` claims.
    * **Factual vs hypothetical**: when overlap ties, a factual
      relation beats a hypothetical one (``looking_at`` etc.).
    * **Confidence**: last tiebreaker.
    """
    if not candidate.subject:
        return True, ""
    q_tokens = _content_words(question)
    cand_subj_tokens = _content_words(candidate.subject)
    if not cand_subj_tokens or not q_tokens:
        return True, ""
    cand_q_overlap = len(q_tokens & cand_subj_tokens)
    cand_factual = candidate.relation in FACTUAL_RELATIONS
    for other in all_candidates:
        if other is candidate:
            continue
        # Same answer-domain: compare ``candidate_type`` so a stray
        # ``count`` doesn't get compared against a ``duration``.
        if other.candidate_type != candidate.candidate_type:
            continue
        if not other.subject:
            continue
        other_subj_tokens = _content_words(other.subject)
        other_q_overlap = len(q_tokens & other_subj_tokens)
        # (1) Question-overlap dominates. Strictly more overlap means
        # the other candidate is the better answer to this question.
        if other_q_overlap > cand_q_overlap:
            return False, (
                f"weaker_question_overlap_vs:{other.object_!r}"
                f"({other_q_overlap}>{cand_q_overlap})"
            )
        # (2) Overlap ties — factual beats hypothetical, both subjects
        # must mention something concrete (cand_subj_tokens & other_subj_tokens)
        # so we don't punish unrelated candidates with the same type.
        if other_q_overlap == cand_q_overlap and (
            cand_subj_tokens & other_subj_tokens
        ):
            other_factual = other.relation in FACTUAL_RELATIONS
            if other_factual and not cand_factual:
                return False, f"weaker_relation_vs:{other.object_!r}"
            if (
                other_factual == cand_factual
                and other.confidence > candidate.confidence + 0.15
            ):
                return False, f"weaker_confidence_vs:{other.object_!r}"
    return True, ""


# ─── Public verifier entry points ──────────────────────────────────────────


def verify_claim(
    question: str,
    candidate: Candidate,
    all_candidates: Iterable[Candidate] | None = None,
    *,
    question_type: QuestionType | None = None,
) -> VerifierResult:
    """
    Run the 5-check ladder on ``candidate``. First FAIL short-circuits.

    ``all_candidates`` is the pool the conflict check considers; pass
    the entire candidate list when verifying any one of them.
    """
    pool = list(all_candidates) if all_candidates is not None else [candidate]

    # 0. Scaffold pre-filter — reject candidates whose value/object/claim
    # is prompt-template debris ("Sub-answer", "Matched Turn", etc.).
    # Cheaper than running the full five-check ladder, and prevents
    # the verifier from ever blessing scaffold output.
    for blob in (
        candidate.value, candidate.normalized_value, candidate.object_,
    ):
        if blob and is_scaffold_text(blob):
            return VerifierResult(
                candidate, "FAIL", f"scaffold_text:{blob[:40]!r}",
            )
        # Imperative user-prompt forms ("Brainstorm ideas...",
        # "Give me 100 prompt parameters...") — the v8 failure where
        # the user's question got returned verbatim as the
        # assistant-memory answer. is_scaffold_text doesn't see these
        # because they're well-formed sentences.
        if blob and is_user_request_sentence(blob):
            return VerifierResult(
                candidate, "FAIL", f"user_request:{blob[:40]!r}",
            )
        # Prompt-echo — candidate is a verbatim/near-verbatim
        # restatement of the question. Belt-and-suspenders for cases
        # where the imperative check missed and the wording is
        # different enough from the bundle's user turn.
        if blob and is_prompt_echo(blob, question):
            return VerifierResult(
                candidate, "FAIL", f"prompt_echo:{blob[:40]!r}",
            )

    # 1. answer_in_source_span
    ok, reason = _check_answer_in_source_span(question, candidate)
    if not ok:
        return VerifierResult(candidate, "FAIL", reason)
    # 2. subject_matches_question
    ok, reason = _check_subject_matches(question, candidate)
    if not ok:
        return VerifierResult(candidate, "FAIL", reason)
    # 3. answer_type_compatible
    ok, reason = _check_answer_type_compatible(question, candidate, question_type)
    if not ok:
        return VerifierResult(candidate, "FAIL", reason)
    # 4. factual_relation
    ok, reason = _check_factual_relation(question, candidate)
    if not ok:
        return VerifierResult(candidate, "FAIL", reason)
    # 5. no_stronger_conflict
    ok, reason = _check_no_stronger_conflict(question, candidate, pool)
    if not ok:
        return VerifierResult(candidate, "FAIL", reason)
    return VerifierResult(candidate, "PASS", "ok")


def verify_claims(
    question: str,
    candidates: Iterable[Candidate],
    *,
    question_type: QuestionType | None = None,
) -> list[VerifierResult]:
    """Run :func:`verify_claim` over every candidate. Pure / synchronous."""
    pool = list(candidates)
    return [
        verify_claim(question, c, pool, question_type=question_type)
        for c in pool
    ]


def filter_passing(results: Iterable[VerifierResult]) -> list[Candidate]:
    """Return just the candidates whose verdict is PASS, preserving order."""
    return [r.candidate for r in results if r.verdict == "PASS"]


# ─── Optional LLM verifier ────────────────────────────────────────────────


class LLMVerifier(Protocol):
    """
    Async protocol for an LLM-backed verifier stage.

    Implementations receive the question, the candidate's grounded
    fields, and return ``(verdict, reason)``. The :func:`verify_with_llm`
    helper calls this on candidates that *passed* the deterministic
    stage, layering one more sanity check before the answerer sees
    them. Use behind a config flag — the deterministic stage already
    handles the spec-cited failure modes.
    """

    async def verify(
        self, question: str, candidate: Candidate,
    ) -> tuple[VerifierVerdict, str]: ...


async def verify_with_llm(
    question: str,
    candidates: Iterable[Candidate],
    llm_verifier: LLMVerifier,
    *,
    question_type: QuestionType | None = None,
) -> list[VerifierResult]:
    """
    Two-stage verification: run deterministic checks, then ask the LLM
    verifier to confirm the passing candidates.

    A candidate that already failed the deterministic stage is returned
    unchanged (no LLM call). Passes are double-checked.
    """
    det_results = verify_claims(
        question, candidates, question_type=question_type,
    )
    out: list[VerifierResult] = []
    for det in det_results:
        if det.verdict == "FAIL":
            out.append(det)
            continue
        try:
            verdict, reason = await _await_or_value(
                llm_verifier.verify(question, det.candidate),
            )
        except Exception as exc:
            log.exception(
                "LLM verifier raised on candidate %r — keeping deterministic PASS",
                det.candidate.value,
            )
            out.append(VerifierResult(
                det.candidate, "PASS", f"llm_verifier_error:{exc!r}",
            ))
            continue
        out.append(VerifierResult(
            det.candidate, verdict, f"llm:{reason}",
        ))
    return out


async def _await_or_value(value: Any) -> Any:
    """Mirror of ``baseline._maybe_await`` for the LLM stub."""
    if isinstance(value, Awaitable):
        return await value
    return value


__all__ = [
    "LLMVerifier",
    "VerifierResult",
    "VerifierVerdict",
    "filter_passing",
    "verify_claim",
    "verify_claims",
    "verify_with_llm",
]
