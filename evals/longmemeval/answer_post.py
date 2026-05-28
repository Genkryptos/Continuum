"""
evals/longmemeval/answer_post.py
================================
Post-answer validators + the final-answer cleanup helper.

Two responsibilities:

1. **`clean_final_answer`** — strips internal scaffold tokens
   (``Matching facts:``, ``Sub-answer:``, JSON blobs, evidence
   bullets) from the user-facing string. Final answers must be
   benchmark-grade: concise, no debug bleed.

2. **`validate_answer_shape`** — checks the cleaned answer against
   the question type. Returns ``(ok, reason)``:

   * COUNT — must contain a digit or number-word.
   * LOCATION — must look like a place noun, not an event verb.
   * DATE_TIME — must mention a date/day/month/year/relative phrase.
   * DURATION — must include a time unit.
   * PREFERENCE — must echo a stated user preference, not generic advice.
   * KNOWLEDGE_UPDATE — when the question asks for change, the
     answer should expose both halves.

3. **`build_repair_prompt`** — when validation fails or the
   don't-IDK guard fires, this builds a short, focused prompt that
   asks the model to repair its answer using ONLY the extracted
   candidates (no retrieval rerun).
"""

from __future__ import annotations

import re

from evals.longmemeval.candidates import Candidate
from evals.longmemeval.question_type import QuestionType
from evals.longmemeval.scaffold_filter import (
    is_label_metadata_line,
    is_scaffold_text,
    is_user_request_sentence,
)

# ─── Final-answer cleanup ──────────────────────────────────────────────────

_MATCHING_FACTS_RE = re.compile(
    r"^\s*matching\s+facts?\s*:.*?(?=sub[-\s]?answer\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_SUBANSWER_TAIL_RE = re.compile(
    r"sub[-\s]?answer\s*:\s*(?P<body>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_JSON_BLOB_RE = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)
_BULLET_LINE_RE = re.compile(r"^\s*[-•*]\s+", re.MULTILINE)


def clean_final_answer(text: str) -> str:
    """
    Strip scaffold, JSON blobs, evidence bullets, and leading prose
    from a model reply so what's left is the benchmark-shaped answer.

    Falls back to the input untouched when nothing recognisable can
    be stripped (so non-scaffolded answers pass through unchanged).
    """
    if not text:
        return ""
    stripped = text.strip()
    # If the model returned a JSON object as the final answer, treat
    # the object's `final_answer` field if present, else drop the JSON
    # — but only if the JSON is the entire reply.
    if _JSON_BLOB_RE.match(stripped):
        try:
            import json
            data = json.loads(stripped)
            if isinstance(data, dict) and isinstance(data.get("final_answer"), str):
                stripped = data["final_answer"].strip()
        except Exception:
            pass

    # Pull the "Sub-answer: ..." tail when present.
    tail = _SUBANSWER_TAIL_RE.search(stripped)
    if tail:
        body = tail.group("body").strip()
        if body:
            stripped = body
    else:
        # Strip a leading "Matching facts: ..." block when no Sub-answer tail.
        stripped = _MATCHING_FACTS_RE.sub("", stripped).strip() or stripped

    # If what's left is dominated by bullet evidence lines, take the
    # last non-bullet line — usually the model's actual answer prose.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    non_bullet = [ln for ln in lines if not _BULLET_LINE_RE.match(ln + " ")]
    if non_bullet and len(non_bullet) < len(lines):
        stripped = non_bullet[-1]

    # Drop a trailing period if the rest is a number/short noun phrase.
    if stripped.endswith(".") and len(stripped) <= 60:
        stripped = stripped[:-1].rstrip() or stripped
    return stripped


# ─── Shape validators ──────────────────────────────────────────────────────

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "hundred",
}
_TIME_UNITS = {
    "second", "minute", "hour", "day", "week", "month", "year",
    "min", "hr", "sec", "yr",
}
_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
}
_RELATIVE_TIME = {
    "today", "yesterday", "tomorrow", "last", "next", "this", "ago",
    "morning", "afternoon", "evening", "night", "weekend",
}

_IDK_PHRASES = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "no information", "not mentioned", "unclear", "no idea",
)


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _has_number_word(text: str) -> bool:
    return any(w in _NUMBER_WORDS for w in re.findall(r"[a-z]+", text.lower()))


def _has_time_unit(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return any(w.rstrip("s") in _TIME_UNITS for w in words)


def _has_date_phrase(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", lower):
        return True
    if re.search(r"\b\d{1,2}[-/]\d{1,2}\b", lower):
        return True
    if any(m in lower for m in _MONTH_NAMES):
        return True
    if any(t in lower for t in _RELATIVE_TIME):
        return True
    if re.search(r"\b(?:mon|tues?|wed|thurs?|fri|sat|sun)(?:day)?\b", lower):
        return True
    return False


def _looks_like_location(text: str) -> bool:
    # Reject obvious event-verb answers.
    if re.match(r"^\s*(?:redeemed|bought|attended|saw|did|spent)\b", text, re.I):
        return False
    # Accept anything with a capitalised noun, a store-like keyword,
    # or a "city, state" pattern.
    if re.search(r"\b[A-Z][a-zA-Z]+\b", text):
        return True
    location_hints = (
        "store", "shop", "app", "website", "venue", "park", "library",
        "office", "school", "university", "college", "restaurant",
    )
    return any(h in text.lower() for h in location_hints)


def is_idk(text: str) -> bool:
    """True if the cleaned answer is one of the recognised refusals."""
    if not text:
        return True
    lower = text.lower().strip().rstrip(".!?,")
    return any(lower.startswith(p) or p in lower for p in _IDK_PHRASES)


_ANALYSIS_PROSE_RE = re.compile(
    r"^\s*(?:based on|looking at|after reviewing|considering|reviewing|"
    r"the (?:answer|evidence) (?:is|suggests|indicates)|"
    r"according to|it (?:seems|appears|looks)|i (?:would|think|believe|am))\b",
    re.IGNORECASE,
)


def _is_analysis_prose(answer: str) -> bool:
    """True when the answer reads like analysis instead of a direct value."""
    if not answer:
        return False
    if _ANALYSIS_PROSE_RE.match(answer):
        return True
    # Many words, no concrete noun-ish anchor → likely prose.
    words = answer.split()
    return len(words) > 25


def validate_answer_shape(
    answer: str, question_type: QuestionType,
) -> tuple[bool, str]:
    """
    Check the cleaned ``answer`` matches the shape ``question_type`` demands.

    Returns ``(ok, reason)``. ``ok=False`` triggers a one-shot
    regeneration via :func:`build_repair_prompt`, OR a direct fallback
    to the top verified candidate when the LLM stage isn't available.
    """
    if not answer:
        return False, "empty_answer"
    # Scaffold rejection — catches "Sub-answer", "Candidates:", bare
    # role labels, prompt-echo, anything from
    # :mod:`evals.longmemeval.scaffold_filter`.
    if is_scaffold_text(answer):
        return False, "answer_is_scaffold"
    if _is_analysis_prose(answer):
        return False, "answer_is_analysis_prose"
    if is_idk(answer):
        return False, "answer_is_idk"

    if question_type == QuestionType.COUNT:
        if _has_digit(answer) or _has_number_word(answer):
            return True, ""
        return False, "count_missing_number"

    if question_type == QuestionType.DURATION:
        if (_has_digit(answer) or _has_number_word(answer)) and _has_time_unit(answer):
            return True, ""
        return False, "duration_missing_unit_or_number"

    if question_type == QuestionType.DATE_TIME:
        if _has_date_phrase(answer) or _has_digit(answer):
            return True, ""
        return False, "datetime_missing_date_phrase"

    if question_type == QuestionType.LOCATION:
        if _looks_like_location(answer):
            return True, ""
        return False, "location_not_place_shaped"

    if question_type in (
        QuestionType.PERSON, QuestionType.ENTITY_NAME,
    ):
        # A reasonable name has a capital letter and isn't a full sentence.
        if re.search(r"\b[A-Z][a-zA-Z]+\b", answer) and len(answer.split()) <= 12:
            return True, ""
        return False, "entity_or_person_not_named"

    if question_type == QuestionType.PREFERENCE:
        # Reject generic-advice phrasing.
        bad_starts = (
            "you should", "it is recommended", "consider ", "try ",
            "i recommend", "you could", "you might",
        )
        if any(answer.lower().startswith(s) for s in bad_starts):
            return False, "preference_is_generic_advice"
        return True, ""

    # KNOWLEDGE_UPDATE / TEMPORAL / MULTI_SESSION / GENERIC — no
    # hard shape rule, just non-empty + non-IDK (already checked above).
    return True, ""


# ─── Don't-IDK guard ───────────────────────────────────────────────────────


def should_block_idk(
    answer: str, question_type: QuestionType,
    relevant_candidates: list[Candidate],
) -> bool:
    """True when the model said "I don't know" but candidates exist."""
    if not is_idk(answer):
        return False
    return any(c.confidence >= 0.5 for c in relevant_candidates)


# ─── Repair prompt for one-shot regeneration ───────────────────────────────


def build_repair_prompt(
    question: str,
    question_type: QuestionType,
    candidates: list[Candidate],
    failed_answer: str,
    reason: str,
) -> str:
    """
    Build a small prompt that asks the LLM to repair its answer
    using only the structured candidates we already extracted.

    No retrieval is rerun — the candidates contain every fact the
    pipeline had access to. The prompt explicitly forbids "I don't
    know" when candidates are present.
    """
    candidate_lines: list[str] = []
    for c in candidates[:12]:
        suffix = f" [unit={c.unit}]" if c.unit else ""
        candidate_lines.append(
            f"  - {c.value} (type={c.candidate_type}, "
            f"confidence={c.confidence:.2f}, "
            f"source_session={c.source_session_id or '?'}){suffix}"
        )
    candidates_block = "\n".join(candidate_lines) or "  (no structured candidates)"

    shape_hint = {
        QuestionType.COUNT:        "Output ONLY the number, optionally with the unit ('20', '3 projects').",
        QuestionType.DURATION:     "Output the duration with its time unit ('45 minutes each way', '3.5 weeks').",
        QuestionType.DATE_TIME:    "Output the most specific date phrase available ('February 14th, 2024' beats 'February').",
        QuestionType.LOCATION:     "Output ONLY the location name (store, city, venue) — not an event description.",
        QuestionType.PERSON:       "Output ONLY the person's name.",
        QuestionType.ENTITY_NAME:  "Output ONLY the entity's name or title (no surrounding sentence).",
        QuestionType.PREFERENCE:   "Output the user's stated preference, not generic advice.",
        QuestionType.KNOWLEDGE_UPDATE: "If the question asks for change, output 'previously X, now Y'.",
        QuestionType.TEMPORAL_REASONING: "Output the relative time, ordering, or duration as asked.",
    }.get(question_type, "Output a short, direct answer.")

    return (
        "Your previous answer failed validation. Repair it using ONLY the "
        "structured candidates below. Do NOT rerun retrieval. Do NOT say "
        "\"I don't know\" if any candidate matches the question.\n\n"
        f"Question: {question}\n"
        f"Question type: {question_type.value}\n"
        f"Previous answer: {failed_answer!r}\n"
        f"Failure reason: {reason}\n\n"
        f"Extracted candidates ({len(candidates)}):\n{candidates_block}\n\n"
        f"Answer shape: {shape_hint}\n\n"
        "Repaired final answer (one short line, no preamble, no scaffold):"
    )


#: Numeric-timestamp transcript junk. Matches answer strings like
#: "110 for", "02 share", "04 principles" — small-number+stopword
#: patterns observed in ShareGPT/UltraChat transcript scrapes that
#: leaked into the candidate pool. Used as a hard reject in
#: :func:`validate_against_claims`.
_TIMESTAMP_JUNK_ANSWER_RE = re.compile(
    r"\b\d{1,3}\s+(?:for|share|principles|every|worked|applied|"
    r"completely|as|of|is|in|on|to|with|from|by|the|and|or|but)\b",
    re.IGNORECASE,
)

#: Stopwords stripped before counting "supporting" content-word
#: overlap between answer and claim. Sharing only stopwords ("for",
#: "the", "is") does NOT make a claim support an answer.
_OVERLAP_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on",
    "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "i", "me", "my", "you", "your", "we", "us", "our", "they",
    "them", "this", "that", "it", "its", "if", "then",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "do", "does", "did", "have", "has", "had", "can", "could",
    "should", "would", "will", "shall",
    "many", "much", "few", "several", "long", "lot", "more", "less",
})


def validate_against_claims(
    answer: str,
    claim_texts: list[str],
    *,
    require_support: bool = True,
) -> tuple[bool, str]:
    """
    Return ``(ok, reason)`` for ``answer`` against the source claims.

    Used by the claim-first pipeline as a stricter check than
    :func:`validate_answer_shape` — the answer must be either a
    substring or a high-token-overlap match against at least one
    source claim. Fails ``"unsupported_by_claim"`` when not.

    ``require_support=False`` skips the support check and only
    runs the scaffold / IDK / prose checks.
    """
    if not answer:
        return False, "empty_answer"
    # IDK is checked before the generic scaffold filter — "I don't know"
    # is technically scaffold, but the reason string should be the more
    # specific ``answer_is_idk_with_claims`` when claims exist.
    if is_idk(answer):
        return False, "answer_is_idk_with_claims" if claim_texts else "answer_is_idk"
    if is_scaffold_text(answer):
        return False, "answer_is_scaffold"
    # Markdown headings / label-metadata answers (e.g. "**Sister's
    # Birthday**", "Occasion: ..."). These slip past the basic scaffold
    # check because they have body content, but they don't ANSWER any
    # question — they label the topic.
    answer_stripped = answer.strip()
    if (
        (answer_stripped.startswith("**") and answer_stripped.endswith("**"))
        or (answer_stripped.startswith("__") and answer_stripped.endswith("__"))
        or is_label_metadata_line(answer_stripped)
    ):
        return False, "answer_is_heading"
    # User-request / question sentences are never the answer to a
    # remembered-fact question. ``"Can you help me organize my sales
    # data from previous markets?"`` shares "previous" with "What was
    # my previous occupation?" but is clearly not the occupation.
    if is_user_request_sentence(answer_stripped):
        return False, "answer_is_user_request"
    # Numeric-timestamp transcript junk ("110 for", "02 share") is
    # never a valid answer regardless of overlap. The hard reject fires
    # before the substring/overlap check so a transcript scrape can't
    # accidentally pass via stopword coincidence.
    if _TIMESTAMP_JUNK_ANSWER_RE.search(answer):
        return False, "answer_is_timestamp_junk"
    if _is_analysis_prose(answer):
        return False, "answer_is_analysis_prose"
    if not require_support:
        return True, ""
    if not claim_texts:
        return False, "no_supporting_claims"
    a_norm = " ".join(answer.lower().split())
    # Content tokens only — stopwords don't count toward "support".
    # "110 for" sharing the word "for" with the claim is NOT support.
    a_tokens = {
        t for t in re.findall(r"[A-Za-z0-9'\-]+", a_norm)
        if t not in _OVERLAP_STOPWORDS and len(t) > 1
    }
    for claim in claim_texts:
        c_norm = " ".join(claim.lower().split())
        if a_norm in c_norm:
            return True, ""
        c_tokens = {
            t for t in re.findall(r"[A-Za-z0-9'\-]+", c_norm)
            if t not in _OVERLAP_STOPWORDS and len(t) > 1
        }
        # Need at least one CONTENT-word overlap, and at least half
        # of the answer's content tokens must be in the claim.
        if (
            a_tokens
            and (a_tokens & c_tokens)
            and len(a_tokens & c_tokens) >= max(1, len(a_tokens) // 2)
        ):
            return True, ""
    return False, "unsupported_by_claim"


def best_candidate_fallback_answer(
    candidates: list[Candidate],
) -> str:
    """
    When validation fails AND we have verified candidates, return the
    best-source-backed value directly — skip the LLM repair entirely.

    This is the fix for the "empty final answer when candidates
    exist" failure mode (rows 58ef2f1c / 5d3d2817 — Gemma 4B
    returned an empty string from the repair LLM call when the
    candidates were already known).

    Ranking is deterministic: highest confidence first, falling back
    to the first candidate's ``object_`` (or ``value`` if object
    isn't set). Returns empty string when there are no candidates —
    the caller should treat that as a real abstain.
    """
    if not candidates:
        return ""
    # Drop any candidate whose value is itself scaffold (defensive).
    clean = [
        c for c in candidates
        if not is_scaffold_text(c.value)
        and not is_scaffold_text(c.object_)
    ]
    if not clean:
        return ""
    clean.sort(key=lambda c: -c.confidence)
    top = clean[0]
    return top.object_.strip() or top.value.strip()


__all__ = [
    "best_candidate_fallback_answer",
    "build_repair_prompt",
    "clean_final_answer",
    "is_idk",
    "should_block_idk",
    "validate_against_claims",
    "validate_answer_shape",
]
