"""
evals/longmemeval/candidates.py
===============================
Structured candidate extraction over retrieved evidence.

The answerer needs typed candidates — numbers, dates, durations,
locations — to do four things the previous stack got wrong:

1. **Numeric-count override.** If evidence says ``"I have 20
   playlists"`` and the question is ``"how many playlists?"``, the
   answer is ``20``, not the cardinality of the candidate list.
2. **Don't-IDK guard.** If a candidate of the right type exists, the
   final answer cannot be ``"I don't know"``.
3. **Shape validator.** Compare the final answer to the question
   type — a LOCATION answer must look like a place, etc. — and
   regenerate from candidates on mismatch.
4. **Telemetry.** Every candidate carries its source session id and
   confidence so failed-row debugging shows where the answer should
   have come from.

The extractor is regex-driven (no LLM). False positives are tolerable
because downstream validators score by question type; false negatives
are not, so the patterns deliberately over-match — extra noise is
filtered by confidence, not by tighter regexes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from continuum.core.types import ContextBundle
from evals.longmemeval.question_type import QuestionType

CandidateType = Literal[
    "count", "date", "duration", "location", "person",
    "entity", "preference", "old_fact", "new_fact",
]


@dataclass(frozen=True)
class Candidate:
    """One structured fact mined from evidence text."""

    value: str
    normalized_value: str
    candidate_type: CandidateType
    unit: str = ""
    source_session_id: str = ""
    source_text: str = ""
    confidence: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "candidate_type": self.candidate_type,
            "unit": self.unit,
            "source_session_id": self.source_session_id,
            "source_text": self.source_text[:200],
            "confidence": round(self.confidence, 3),
            "extras": self.extras,
        }


# ---------------------------------------------------------------------------
# Number-word normalisation — keeps "two and a half" comparable to "2.5"
# ---------------------------------------------------------------------------

_WORD_NUMBERS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}
_HALF_PHRASE_RE = re.compile(
    r"\b(?P<base>" + "|".join(_WORD_NUMBERS) + r")\s+and\s+a\s+half\b",
    re.IGNORECASE,
)
_NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE,
)


def normalize_numeric(text: str) -> str:
    """Convert number words to digits; preserve '.5' for ``x and a half``."""
    if not text:
        return ""

    def _half(m: re.Match[str]) -> str:
        base = _WORD_NUMBERS.get(m.group("base").lower(), m.group("base"))
        return f"{base}.5"

    out = _HALF_PHRASE_RE.sub(_half, text)
    out = _NUMBER_WORD_RE.sub(
        lambda m: _WORD_NUMBERS.get(m.group(0).lower(), m.group(0)),
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Numeric / count extraction
# ---------------------------------------------------------------------------

#: Matches "<num> <noun>" or "I have/own/bought <num> <noun>".
_COUNT_RE = re.compile(
    r"\b"
    r"(?:i\s+(?:have|own|bought|brought|got|hold|keep|maintain|attended|visited|"
    r"adopted|read|watched|saw|played|completed|did|tried|made)\s+)?"
    r"(?P<num>\d+(?:\.\d+)?|[a-z]+(?:\s+and\s+a\s+half)?)\s+"
    r"(?P<unit>[a-zA-Z][a-zA-Z\-]*?s?)"
    r"\b",
    re.IGNORECASE,
)
_INT_RE = re.compile(r"^\d+(?:\.\d+)?$")
#: Tokens that *look* like a unit (4+ chars) but mean nothing as the
#: noun in "<n> <noun>" — keep this list short and focused on observed
#: regex false positives, not a generic stopword dump.
_COUNT_NOISE_UNITS = {
    "thing", "things", "stuff", "item", "items", "type", "types",
    "kind", "kinds", "rule", "rules", "way", "ways", "time", "times",
    "place", "places", "part", "parts", "side", "sides", "fact",
    "facts", "scale", "scales", "fold", "folds", "version", "versions",
    "level", "levels", "stage", "stages", "step", "steps", "more",
    "less", "other", "others", "another", "also", "even", "once",
    "twice", "such", "etc", "etc.",
}


def _maybe_count(num_text: str) -> str | None:
    """Return the digit form of *num_text* if it is a finite number, else None."""
    norm = normalize_numeric(num_text).strip()
    if _INT_RE.match(norm):
        return norm
    return None


def extract_counts(text: str, *, source_session_id: str = "") -> list[Candidate]:
    """Mine `<number> <noun>` phrases out of *text*."""
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for match in _COUNT_RE.finditer(text):
        num_raw = match.group("num").strip()
        unit_raw = match.group("unit").strip().lower().rstrip("s")
        num = _maybe_count(num_raw)
        if num is None:
            continue
        # Avoid pure-number traps like "12 hours" where the unit is a
        # time word — those become DURATION candidates instead.
        if unit_raw in _DURATION_UNITS:
            continue
        # Skip common false positives and units too short to mean a thing.
        if len(unit_raw) < 4:
            continue
        if unit_raw in _COUNT_NOISE_UNITS:
            continue
        key = (num, unit_raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            value=f"{num} {match.group('unit').strip()}",
            normalized_value=num,
            candidate_type="count",
            unit=match.group("unit").strip(),
            source_session_id=source_session_id,
            source_text=text,
            # Higher confidence when an "I have/own" frame anchored the match.
            confidence=0.8 if "i " in match.group(0).lower() else 0.5,
        ))
    return out


# ---------------------------------------------------------------------------
# Duration extraction
# ---------------------------------------------------------------------------

_DURATION_UNITS = {
    "second", "minute", "hour", "day", "week", "month", "year",
    "min", "hr", "sec", "yr",
}
_DURATION_RE = re.compile(
    r"\b(?P<num>\d+(?:\.\d+)?|[a-z]+(?:\s+and\s+a\s+half)?)\s*"
    r"-?\s*(?P<unit>seconds?|minutes?|mins?|hours?|hrs?|"
    r"days?|weeks?|months?|years?|yrs?)\b",
    re.IGNORECASE,
)


def extract_durations(text: str, *, source_session_id: str = "") -> list[Candidate]:
    """Find ``<n> <time-unit>`` spans."""
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for match in _DURATION_RE.finditer(text):
        num_raw = match.group("num").strip()
        unit_raw = match.group("unit").strip().lower().rstrip("s")
        norm = normalize_numeric(num_raw).strip()
        if not _INT_RE.match(norm):
            continue
        canonical_unit = unit_raw.rstrip(".")
        # Map abbreviated units onto their full form for normalisation.
        canonical_unit = {
            "min": "minute", "hr": "hour", "sec": "second", "yr": "year",
        }.get(canonical_unit, canonical_unit)
        key = (norm, canonical_unit)
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            value=f"{norm} {canonical_unit}{'s' if float(norm) != 1 else ''}",
            normalized_value=norm,
            candidate_type="duration",
            unit=canonical_unit,
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.7,
        ))
    return out


# ---------------------------------------------------------------------------
# Date / time extraction
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_PATTERNS = (
    # 2024-03-15, 2024/3/15
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    # March 15, 2024 — with optional day-suffix
    re.compile(
        rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
        re.IGNORECASE,
    ),
    # 15 March 2024
    re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})(?:\s+\d{{4}})?\b",
        re.IGNORECASE,
    ),
    # Month-only (less specific — lower confidence)
    re.compile(rf"\b(?:{_MONTH_NAMES})\b", re.IGNORECASE),
    # Day-of-week
    re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|"
               r"sunday|mon|tue|wed|thu|fri|sat|sun)\b", re.IGNORECASE),
    # Relative phrases
    re.compile(r"\blast\s+(?:week|month|year|sunday|monday|tuesday|"
               r"wednesday|thursday|friday|saturday)\b", re.IGNORECASE),
)


def extract_dates(text: str, *, source_session_id: str = "") -> list[Candidate]:
    """Find date phrases with confidence proportional to specificity."""
    out: list[Candidate] = []
    seen: set[str] = set()
    confidences = (0.95, 0.9, 0.9, 0.55, 0.5, 0.6)
    for pattern, conf in zip(_DATE_PATTERNS, confidences, strict=True):
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(
                value=raw,
                normalized_value=raw,
                candidate_type="date",
                source_session_id=source_session_id,
                source_text=text,
                confidence=conf,
            ))
    return out


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------

_KNOWN_STORES = {
    "target", "walmart", "costco", "kroger", "safeway", "trader joe's",
    "whole foods", "aldi", "publix", "wegmans", "amazon", "best buy",
    "apple store", "ikea", "home depot", "lowe's", "starbucks", "mcdonald's",
    "chipotle", "zara", "h&m", "uniqlo", "gap", "nike",
}
_LOCATION_FRAME_RE = re.compile(
    r"\b(?:at|from|to|in|near|visited|shopped\s+at|went\s+to)\s+"
    r"(?P<loc>(?:the\s+)?[A-Z][A-Za-z0-9'&\.\- ]+?)"
    r"(?=\s+(?:on|for|today|yesterday|last|next|in|after|before|because|"
    r"so|when|with|and|but|while|since|every|every|to)|[.,!;?]|$)",
)


def extract_locations(text: str, *, source_session_id: str = "") -> list[Candidate]:
    """Find location-shaped noun phrases."""
    out: list[Candidate] = []
    seen: set[str] = set()
    lower = text.lower()
    # Known stores — highest confidence
    for store in _KNOWN_STORES:
        if store in lower:
            key = store
            if key in seen:
                continue
            seen.add(key)
            # Recover canonical capitalisation from the original text
            idx = lower.find(store)
            canonical = text[idx:idx + len(store)] if idx >= 0 else store.title()
            out.append(Candidate(
                value=canonical.title(),
                normalized_value=canonical.title(),
                candidate_type="location",
                source_session_id=source_session_id,
                source_text=text,
                confidence=0.9,
            ))
    # Generic "at <Place>" / "in <Place>"
    for match in _LOCATION_FRAME_RE.finditer(text):
        loc = match.group("loc").strip()
        # Skip common false positives
        if loc.lower() in {"the", "a", "an", "this", "that"}:
            continue
        if len(loc.split()) > 5:  # too long to be a location
            continue
        key = loc.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            value=loc,
            normalized_value=loc,
            candidate_type="location",
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.55,
        ))
    return out


# ---------------------------------------------------------------------------
# Person / entity extraction (heuristic — favours recall)
# ---------------------------------------------------------------------------

_PERSON_FRAME_RE = re.compile(
    r"\b(?:my\s+(?:friend|partner|wife|husband|boss|colleague|doctor|"
    r"teacher|son|daughter|mom|dad|mother|father|brother|sister)\s+"
    r"(?:is\s+called\s+|named\s+|is\s+)?|"
    r"(?:dr|mr|mrs|ms|miss|prof|professor)\.?\s+)"
    r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)


def extract_persons(text: str, *, source_session_id: str = "") -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for match in _PERSON_FRAME_RE.finditer(text):
        name = match.group("name").strip()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(Candidate(
            value=name,
            normalized_value=name,
            candidate_type="person",
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.75,
        ))
    return out


# Quoted titles and capitalised multi-word names ("Summer Vibes",
# "The Glass Menagerie") — highest signal for ENTITY_NAME questions.
_QUOTED_TITLE_RE = re.compile(
    r"[\"“‘]([A-Z][\w' ,.\-:!?]{1,80}?)[\"”’]"
)
_CAPITAL_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-z']+\s+){1,4}[A-Z][a-z']+\b"
)


def extract_entities(text: str, *, source_session_id: str = "") -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for match in _QUOTED_TITLE_RE.finditer(text):
        ent = match.group(1).strip()
        if ent.lower() in seen:
            continue
        seen.add(ent.lower())
        out.append(Candidate(
            value=ent, normalized_value=ent,
            candidate_type="entity",
            source_session_id=source_session_id,
            source_text=text, confidence=0.85,
        ))
    for match in _CAPITAL_PHRASE_RE.finditer(text):
        ent = match.group(0).strip()
        if ent.lower() in seen:
            continue
        seen.add(ent.lower())
        out.append(Candidate(
            value=ent, normalized_value=ent,
            candidate_type="entity",
            source_session_id=source_session_id,
            source_text=text, confidence=0.5,
        ))
    return out


# ---------------------------------------------------------------------------
# Preference extraction
# ---------------------------------------------------------------------------

_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+(?:like|love|prefer|enjoy|favou?r|dislike|hate|avoid|"
    r"don't\s+like|am\s+a\s+fan\s+of)|"
    r"my\s+favou?rite\s+(?:thing|food|color|colour|hobby|movie|book|"
    r"music|sport|band|artist)\s+is)"
    r"\s+(?P<object>[A-Za-z0-9 ,'&\-]+?)(?=[.,!;?]|$)",
    re.IGNORECASE,
)


def extract_preferences(text: str, *, source_session_id: str = "") -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for match in _PREFERENCE_RE.finditer(text):
        obj = match.group("object").strip().rstrip(".,!;?")
        if not obj or obj.lower() in seen or len(obj) > 80:
            continue
        seen.add(obj.lower())
        polarity = "like"
        m0 = match.group(0).lower()
        if "dislike" in m0 or "hate" in m0 or "avoid" in m0 or "don't" in m0:
            polarity = "dislike"
        out.append(Candidate(
            value=obj,
            normalized_value=obj.lower(),
            candidate_type="preference",
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.8,
            extras={"polarity": polarity},
        ))
    return out


# ---------------------------------------------------------------------------
# Old / new fact extraction (knowledge-update)
# ---------------------------------------------------------------------------

_OLD_FACT_RE = re.compile(
    r"\b(?:used\s+to|previously|formerly|originally|back\s+then|"
    r"before(?:hand)?|i\s+(?:was|had)|until\s+(?:recently|last))\s+"
    r"(?P<fact>[^.]+?)(?=[.,!;?]|$)",
    re.IGNORECASE,
)
_NEW_FACT_RE = re.compile(
    r"\b(?:now|currently|these\s+days|just\s+(?:started|moved|switched)|"
    r"as\s+of\s+(?:today|now)|i\s+just|i\s+(?:am|'m)\s+now)\s+"
    r"(?P<fact>[^.]+?)(?=[.,!;?]|$)",
    re.IGNORECASE,
)


def extract_change_facts(
    text: str, *, source_session_id: str = "",
) -> list[Candidate]:
    out: list[Candidate] = []
    for match in _OLD_FACT_RE.finditer(text):
        fact = match.group("fact").strip()
        if not fact or len(fact) > 200:
            continue
        out.append(Candidate(
            value=fact, normalized_value=fact.lower(),
            candidate_type="old_fact",
            source_session_id=source_session_id,
            source_text=text, confidence=0.7,
        ))
    for match in _NEW_FACT_RE.finditer(text):
        fact = match.group("fact").strip()
        if not fact or len(fact) > 200:
            continue
        out.append(Candidate(
            value=fact, normalized_value=fact.lower(),
            candidate_type="new_fact",
            source_session_id=source_session_id,
            source_text=text, confidence=0.7,
        ))
    return out


# ---------------------------------------------------------------------------
# Top-level extractor — drives all type-specific extractors
# ---------------------------------------------------------------------------


def extract_candidates_from_text(
    text: str, *, source_session_id: str = "",
) -> list[Candidate]:
    """Run every typed extractor over a single chunk of text."""
    if not text:
        return []
    out: list[Candidate] = []
    out.extend(extract_counts(text,        source_session_id=source_session_id))
    out.extend(extract_durations(text,     source_session_id=source_session_id))
    out.extend(extract_dates(text,         source_session_id=source_session_id))
    out.extend(extract_locations(text,     source_session_id=source_session_id))
    out.extend(extract_persons(text,       source_session_id=source_session_id))
    out.extend(extract_entities(text,      source_session_id=source_session_id))
    out.extend(extract_preferences(text,   source_session_id=source_session_id))
    out.extend(extract_change_facts(text,  source_session_id=source_session_id))
    return out


def extract_candidates_from_context(
    ctx: ContextBundle | None,
) -> list[Candidate]:
    """Mine every retrieved evidence item for typed candidates."""
    if ctx is None or not getattr(ctx, "items", None):
        return []
    out: list[Candidate] = []
    for item in ctx.items:
        sid = str((item.metadata or {}).get("session_id") or "")
        out.extend(extract_candidates_from_text(
            item.content, source_session_id=sid,
        ))
    return out


def filter_candidates_for_question(
    candidates: Iterable[Candidate], question_type: QuestionType,
) -> list[Candidate]:
    """Return only the candidates whose type is relevant to *question_type*."""
    relevant_types: dict[QuestionType, set[CandidateType]] = {
        QuestionType.COUNT:              {"count"},
        QuestionType.DURATION:           {"duration"},
        QuestionType.DATE_TIME:          {"date"},
        QuestionType.LOCATION:           {"location"},
        QuestionType.PERSON:             {"person"},
        QuestionType.ENTITY_NAME:        {"entity"},
        QuestionType.PREFERENCE:         {"preference"},
        QuestionType.TEMPORAL_REASONING: {"date", "duration"},
        QuestionType.KNOWLEDGE_UPDATE:   {"old_fact", "new_fact"},
        QuestionType.MULTI_SESSION:     set(),
        QuestionType.ABSTAIN:           set(),
        QuestionType.GENERIC_FACT:      set(),
    }
    wanted = relevant_types.get(question_type, set())
    if not wanted:
        return list(candidates)
    return [c for c in candidates if c.candidate_type in wanted]


def best_count_for_object(
    candidates: Iterable[Candidate], question: str,
) -> Candidate | None:
    """
    For COUNT questions, pick the most-confident count whose ``unit``
    actually appears in the question text.

    Returns ``None`` when no candidate's unit matches — no arbitrary
    fallback. The caller (the COUNT override path) is responsible for
    leaving the model's answer alone when this returns None.

    Why this is strict:
        ``extract_counts`` is intentionally over-permissive (matches
        any ``<n> <word>`` phrase, e.g. ``"22 Marvel"`` from "22 Marvel
        movies"). If we allowed an unmatched fallback, the override
        would replace correct synthesis answers with arbitrary numbers
        from noisy candidates. Requiring an explicit unit-in-question
        match keeps the override surgical.

    Confidence floor (0.6) further trims regex-only matches that
    weren't anchored by an "I have/own/bought" frame.
    """
    if not question:
        return None
    q_lower = question.lower()
    counts = [c for c in candidates if c.candidate_type == "count"]
    if not counts:
        return None

    def matches_object(unit: str) -> bool:
        unit = unit.strip().lower().rstrip("s")
        if not unit or len(unit) < 3:
            return False
        # Strict containment: the unit must literally appear in the
        # question (singular or plural). No stem matching, no synonyms.
        return unit in q_lower or (unit + "s") in q_lower

    matches = [
        c for c in counts
        if matches_object(c.unit) and c.confidence >= 0.6
    ]
    if not matches:
        return None
    return max(matches, key=lambda c: (c.confidence, len(c.unit)))


__all__ = [
    "Candidate",
    "CandidateType",
    "best_count_for_object",
    "extract_candidates_from_context",
    "extract_candidates_from_text",
    "extract_change_facts",
    "extract_counts",
    "extract_dates",
    "extract_durations",
    "extract_entities",
    "extract_locations",
    "extract_persons",
    "extract_preferences",
    "filter_candidates_for_question",
    "normalize_numeric",
]
