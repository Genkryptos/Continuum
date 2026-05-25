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


def validate_answer_shape(
    answer: str, question_type: QuestionType,
) -> tuple[bool, str]:
    """
    Check the cleaned ``answer`` matches the shape ``question_type`` demands.

    Returns ``(ok, reason)``. ``ok=False`` triggers a one-shot
    regeneration via :func:`build_repair_prompt`.
    """
    if not answer:
        return False, "empty_answer"
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


__all__ = [
    "build_repair_prompt",
    "clean_final_answer",
    "is_idk",
    "should_block_idk",
    "validate_answer_shape",
]
