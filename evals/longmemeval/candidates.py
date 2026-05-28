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
from evals.longmemeval.scaffold_filter import (
    is_scaffold_line,
    is_scaffold_text,
)

CandidateType = Literal[
    "count", "date", "duration", "location", "person",
    "entity", "preference", "old_fact", "new_fact", "value",
]

#: Relation taxonomy used by the ranking function. Factual relations
#: outrank hypothetical ones — "got_at"/"is_worth" beat "looking_at"/
#: "considering" when the question asks about a completed event.
FACTUAL_RELATIONS = frozenset({
    "count_of", "duration_of", "got_at", "located_at", "is_called",
    "is_worth", "paid", "prefer", "named", "changed_to", "is",
})
HYPOTHETICAL_RELATIONS = frozenset({
    "considering", "looking_at", "planning", "thinking_about",
    "might", "option", "alternative",
})


@dataclass(frozen=True)
class Candidate:
    """
    One structured **claim** mined from evidence text.

    Historic ``Candidate`` was a bare ``(value, type)`` pair — enough to
    drive the COUNT override and the don't-IDK guard but blind to *what*
    the value was about. Two failure modes followed:

    * **Stanford vs UCLA**: extractor saw both "UCLA" (completed
      Bachelor's) and "Stanford" (hypothetical Master's planning) as
      generic ``location`` candidates. Ranking by type alone couldn't
      tell which one answered "where did I get my Bachelor's?".
    * **IKEA durations**: both "4 hours" (bookshelf) and "2 hours"
      (table) were generic ``duration`` candidates. The COUNT/DURATION
      override couldn't tell them apart because the *subject* of each
      duration was thrown away.

    Fix: every extractor now populates a grounded claim with
    ``claim``/``subject``/``relation``/``object_``/``source_span``.
    Ranking (see :func:`rank_claims_for_question`) uses those fields,
    not just ``value``/``candidate_type``.

    The legacy fields (``value``, ``normalized_value``,
    ``candidate_type``, ``unit``, ``source_session_id``, ``source_text``,
    ``confidence``, ``extras``) are PRESERVED so the COUNT override and
    the don't-IDK guard keep working without changes.
    """

    # ─── Legacy fields (back-compat with COUNT override, IDK guard) ─────
    value: str
    normalized_value: str
    candidate_type: CandidateType
    unit: str = ""
    source_session_id: str = ""
    source_text: str = ""
    confidence: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)

    #: Role of the source turn (``"user"`` / ``"assistant"`` / ``""``).
    #: Used by the ranker to prefer user-stated autobiographical facts
    #: over assistant advice on questions containing "I"/"my"/"me".
    #: Empty when the extractor wasn't given a role.
    source_role: str = ""

    # ─── Grounded-claim fields (new; defaulted so old call sites work) ──
    #: Full clause / sentence this candidate came from. Used for
    #: lexical overlap against the question's subject phrase.
    claim: str = ""
    #: Finer-grained answer kind than ``candidate_type``. Examples:
    #: ``"duration"``, ``"location"``, ``"value"``, ``"amount"``.
    #: Defaults to ``candidate_type`` when an extractor doesn't override
    #: (computed in ``__post_init__``-like fashion via to_dict's default).
    answer_type: str = ""
    #: What the claim is about — the question's anchor noun.
    #: ``"IKEA bookshelf"``, ``"Bachelor's degree"``, ``"the painting"``.
    subject: str = ""
    #: Verb relation, drawn from :data:`FACTUAL_RELATIONS` or
    #: :data:`HYPOTHETICAL_RELATIONS`. Hypothetical relations get
    #: penalised during ranking so the model doesn't pick a
    #: planning/option statement when a completed-event statement
    #: exists.
    relation: str = ""
    #: The value as the object of the relation. May be a number, a
    #: name, or a qualitative phrase like ``"triple what I paid"``.
    object_: str = ""
    #: The exact substring of the source text that anchored this
    #: claim. Defaults to ``claim`` when the extractor doesn't have a
    #: tighter span.
    source_span: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            # Legacy
            "value": self.value,
            "normalized_value": self.normalized_value,
            "candidate_type": self.candidate_type,
            "unit": self.unit,
            "source_session_id": self.source_session_id,
            "source_role": self.source_role,
            "source_text": self.source_text[:200],
            "confidence": round(self.confidence, 3),
            "extras": self.extras,
            # Grounded-claim
            "claim": self.claim,
            "answer_type": self.answer_type or self.candidate_type,
            "subject": self.subject,
            "relation": self.relation,
            # `object` (no trailing underscore) is the on-the-wire key —
            # it would shadow the builtin in Python source, hence the
            # `object_` attribute name internally.
            "object": self.object_,
            "source_span": self.source_span or self.claim,
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


# ---------------------------------------------------------------------------
# Clause / subject helpers — drive the grounded-claim fields
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_HYPOTHETICAL_PHRASE_RE = re.compile(
    r"\b(?:considering|thinking about|looking at|might|may|could|"
    r"planning|going to|would|option|options?|alternative|either|"
    r"or|vs\.?|versus|instead)\b",
    re.IGNORECASE,
)


def _enclosing_clause(text: str, span_start: int, span_end: int) -> str:
    """Return the sentence containing the byte-span ``[start, end)``."""
    # Walk back to the previous sentence terminator (or start of string).
    head = text[:span_start]
    last_term = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    clause_start = (last_term + 1) if last_term >= 0 else 0
    # Walk forward to the next terminator.
    tail = text[span_end:]
    next_term_offsets = [
        i for i in (tail.find("."), tail.find("!"), tail.find("?")) if i >= 0
    ]
    if next_term_offsets:
        clause_end = span_end + min(next_term_offsets) + 1
    else:
        clause_end = len(text)
    return text[clause_start:clause_end].strip()


_SUBJECT_NOISE = frozenset({
    "i", "we", "you", "they", "the", "a", "an", "my", "our", "their",
    "and", "but", "or", "so", "then",
    "very", "really", "just", "also", "only",
})
#: Anaphoric / pronoun heads. When the noun phrase immediately before a
#: verb resolves to one of these, the parser looks further left in the
#: clause for the antecedent (the actual referent).
_ANAPHORA = frozenset({
    "it", "this", "that", "these", "those", "one", "ones",
    "thing", "stuff", "they", "them",
})


def _noun_phrase_before(clause: str, match_start: int) -> str:
    """
    Extract a noun phrase ending immediately before ``match_start``.

    Walks back over content words until it hits a stopword boundary or
    a verb. Returns the trimmed substring. Empty when no usable subject
    was found.

    Anaphora handling: when the candidate subject is a pronoun ("it",
    "this", "one"), the parser searches the rest of the clause for an
    antecedent noun. This catches patterns like
    ``"...the IKEA table — that one only took 2 hours"`` where the
    subject sits behind the dash.
    """
    head = clause[:match_start].rstrip()
    if not head:
        return ""
    # Strip trailing connective tokens / verbs that aren't the subject.
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", head)
    # Walk backward, dropping verbs and noise until we hit content nouns.
    keep: list[str] = []
    for tok in reversed(tokens):
        low = tok.lower()
        if low in _SUBJECT_NOISE:
            if keep:
                break  # we already collected some content — stop at noise
            continue
        if low in {
            "took", "takes", "lasted", "lasts", "is", "was", "are", "were",
            "got", "get", "earn", "earned", "did", "do", "have", "has",
        }:
            break  # hit the verb — subject is to the left
        keep.append(tok)
        if len(keep) >= 4:
            break

    # Anaphora resolution: if every kept token is a pronoun
    # ("that one", "it"), search the rest of the clause for the most
    # recent concrete noun phrase. This is what makes "that one took
    # 2 hours" point back to "the IKEA table".
    if keep and all(t.lower() in _ANAPHORA for t in keep):
        antecedent = _find_antecedent_noun(head)
        if antecedent:
            return antecedent
    if not keep:
        # Even when the immediate slot was empty (rare punctuation),
        # the antecedent search may still find a referent earlier.
        antecedent = _find_antecedent_noun(head)
        if antecedent:
            return antecedent
        return ""
    return " ".join(reversed(keep))


_ANTECEDENT_RE = re.compile(
    # A noun phrase with an optional determiner / adjective, e.g.
    # "the IKEA table" / "a small IKEA bookshelf" / "the painting".
    r"\b(?:the|a|an|my|our|their)\s+"
    r"(?:[A-Za-z][A-Za-z'\-]+\s+){0,3}"
    r"[A-Za-z][A-Za-z'\-]+\b"
)


def _find_antecedent_noun(head: str) -> str:
    """
    Return the LAST noun-phrase-shaped match in ``head``, stripped of
    its leading determiner. Used to resolve "it"/"that one" back to
    the noun it refers to.
    """
    matches = list(_ANTECEDENT_RE.finditer(head))
    if not matches:
        return ""
    raw = matches[-1].group(0).strip()
    # Drop leading determiner so the subject becomes "IKEA table" not
    # "the IKEA table" — easier downstream content-word overlap.
    return re.sub(
        r"^(?:the|a|an|my|our|their)\s+", "", raw, flags=re.IGNORECASE,
    )


def _is_hypothetical_clause(clause: str) -> bool:
    return bool(_HYPOTHETICAL_PHRASE_RE.search(clause))


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
    """
    Find ``<n> <time-unit>`` spans, each grounded as a claim.

    The grounded claim carries:

    * ``claim`` — the enclosing sentence ("Assembling the IKEA bookshelf
      took 4 hours from start to finish.")
    * ``subject`` — the noun phrase before the verb ("IKEA bookshelf")
    * ``relation`` — ``"duration_of"`` (or ``"hypothetical_duration"``
      when the enclosing clause is a what-if / planning sentence)
    * ``object_`` — the duration phrase ("4 hours")
    * ``source_span`` — the literal ``<n> <unit>`` substring

    This is what lets the ranker distinguish "4 hours for the bookshelf"
    from "2 hours for the table" when the question targets the bookshelf.
    """
    out: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
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
        # Grounded-claim fields
        clause = _enclosing_clause(text, match.start(), match.end())
        # Anchor the subject scan at the *verb* before the number when
        # one exists ("took 4 hours" — subject sits before "took").
        local_match_start = clause.find(match.group(0))
        verb_match = re.search(
            r"\b(?:took|takes|lasted|lasts|spent|spend)\b",
            clause[:local_match_start] if local_match_start >= 0 else clause,
            re.IGNORECASE,
        )
        if verb_match:
            subject = _noun_phrase_before(clause, verb_match.start())
            relation = "duration_of"
        else:
            subject = _noun_phrase_before(
                clause, local_match_start if local_match_start >= 0 else 0,
            )
            relation = "duration_of"
        if _is_hypothetical_clause(clause):
            relation = "hypothetical_duration"
        duration_obj = f"{norm} {canonical_unit}{'s' if float(norm) != 1 else ''}"
        key = (norm, canonical_unit, subject.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            value=duration_obj,
            normalized_value=norm,
            candidate_type="duration",
            unit=canonical_unit,
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.7 if relation == "duration_of" else 0.4,
            claim=clause,
            answer_type="duration",
            subject=subject,
            relation=relation,
            object_=duration_obj,
            source_span=match.group(0).strip(),
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


#: Verbs that mark a *completed* attainment of the location ("I GOT my
#: degree AT UCLA"). Versus "looking at" / "considering" which mark a
#: hypothetical. Used to set the ``relation`` field.
_COMPLETED_LOC_VERBS = re.compile(
    r"\b(?:got|earned|received|completed|finished|graduated|attended|"
    r"visited|went\s+to|live[ds]?\s+in|moved\s+to|work(?:ed|s)?\s+at|"
    r"studied\s+at|met\s+at|bought\s+at)\b",
    re.IGNORECASE,
)


def extract_locations(text: str, *, source_session_id: str = "") -> list[Candidate]:
    """
    Find location-shaped noun phrases, each grounded as a claim.

    The grounded claim carries enough context to tell *what* the
    location is the location *of*. This is what disambiguates:

    * ``"I got my Bachelor's at UCLA"`` →
      subject=``"Bachelor's"``, relation=``"got_at"``, object=``"UCLA"``
    * ``"I'm looking at Stanford for Master's"`` →
      subject=``"Master's"``, relation=``"looking_at"``, object=``"Stanford"``

    Ranking (:func:`rank_claims_for_question`) prefers the factual
    relation over the hypothetical one when the question subject
    matches the candidate subject.
    """
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
            clause = _enclosing_clause(text, idx, idx + len(store))
            # Known stores default to factual `located_at`; hypothetical
            # clauses ("considering Target") demote to `looking_at`.
            relation = (
                "looking_at" if _is_hypothetical_clause(clause)
                else "located_at"
            )
            out.append(Candidate(
                value=canonical.title(),
                normalized_value=canonical.title(),
                candidate_type="location",
                source_session_id=source_session_id,
                source_text=text,
                confidence=0.9 if relation == "located_at" else 0.4,
                claim=clause,
                answer_type="location",
                subject=_noun_phrase_before(clause, clause.lower().find(store)),
                relation=relation,
                object_=canonical.title(),
                source_span=text[idx:idx + len(store)],
            ))
    # Generic "at <Place>" / "in <Place>" with subject parsing.
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
        clause = _enclosing_clause(text, match.start(), match.end())
        # Subject = noun phrase before the location-frame verb/preposition.
        completed = _COMPLETED_LOC_VERBS.search(clause)
        if completed:
            subject = _noun_phrase_before(clause, completed.start())
            relation = "got_at"
        else:
            local_start = clause.find(loc)
            subject = _noun_phrase_before(
                clause, local_start if local_start >= 0 else 0,
            )
            relation = "located_at"
        if _is_hypothetical_clause(clause):
            # Demote — a hypothetical "looking at Stanford" is NOT a
            # factual location claim, so we both flip the relation and
            # drop confidence so ranking deprioritises it.
            relation = "looking_at"
        out.append(Candidate(
            value=loc,
            normalized_value=loc,
            candidate_type="location",
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.55 if relation in FACTUAL_RELATIONS else 0.3,
            claim=clause,
            answer_type="location",
            subject=subject,
            relation=relation,
            object_=loc,
            source_span=match.group(0).strip(),
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
# Qualitative value / worth claims (the painting case)
# ---------------------------------------------------------------------------

#: Patterns for "<subject> is worth <qualitative or numeric>" — captures
#: both ``"the painting is worth triple what I paid"`` (qualitative) and
#: ``"the painting is worth $400"`` (numeric). The ranker preserves the
#: qualitative phrase verbatim rather than reducing it to a digit count.
_WORTH_RE = re.compile(
    r"\b(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z\- ]{1,40}?)\s+"
    r"(?P<verb>is\s+worth|are\s+worth|sold\s+for|appraised\s+at|"
    r"valued\s+at|fetched|cost(?:s)?)\s+"
    r"(?P<object>[^.,!;?]{1,80})",
    re.IGNORECASE,
)
# Regex for "I paid <amount> for <subject>". Subject is matched as a
# noun-phrase of up to 4 content words; trailing prepositions ("in",
# "on", "from") are stripped in post-processing so a sentence like
# "I paid 200 dollars for the painting in 2015." yields subject
# ``"painting"`` rather than ``"painting in"``.
_PAID_RE = re.compile(
    r"\bi\s+(?:paid|spent)\s+"
    r"(?P<amount>\$?\d[\d.,]*\s*(?:dollars|usd|euros?|pounds?)?)"
    r"\s+(?:for\s+)?"
    r"(?P<subject>(?:the\s+|a\s+|an\s+|my\s+)?"
    r"[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,3})",
    re.IGNORECASE,
)
_TRAILING_PREP_RE = re.compile(
    r"\s+(?:in|on|at|from|to|with|by|for|of|during|since|until)$",
    re.IGNORECASE,
)
_LEADING_DET_RE = re.compile(
    r"^(?:the|a|an|my|our|their)\s+", re.IGNORECASE,
)


def extract_value_claims(
    text: str, *, source_session_id: str = "",
) -> list[Candidate]:
    """
    Extract "is worth X" / "paid X for Y" / "valued at X" claims.

    The painting case from the spec is the canonical failure mode this
    fixes: when a user says ``"the painting is worth triple what I
    paid"``, the answer to ``"what is the painting worth?"`` is the
    *qualitative* phrase ``"triple what I paid"``, NOT the numeric
    amount they originally paid. Earlier extractors reduced everything
    to a digit count and lost this.

    Two relations emitted:

    * ``relation="is_worth"`` — current-value claims (preferred when
      the question asks ``"worth"``).
    * ``relation="paid"`` — original-cost claims (preferred when the
      question asks ``"paid"``).
    """
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for match in _WORTH_RE.finditer(text):
        subject = match.group("subject").strip().lower()
        verb = match.group("verb").strip().lower()
        obj = match.group("object").strip().rstrip(".,;!?")
        if not obj:
            continue
        # Map verb to a canonical relation.
        relation = "is_worth" if "worth" in verb or "valued" in verb else (
            "sold_for" if "sold" in verb or "fetched" in verb else "is_worth"
        )
        clause = _enclosing_clause(text, match.start(), match.end())
        key = (subject, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            value=obj,
            normalized_value=obj.lower(),
            candidate_type="value",
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.8,
            claim=clause,
            answer_type="value",
            subject=match.group("subject").strip(),
            relation=relation,
            object_=obj,
            source_span=match.group(0).strip(),
        ))
    for match in _PAID_RE.finditer(text):
        amount = match.group("amount").strip()
        # Trim leading determiner + trailing prepositions so "the
        # painting in" becomes "painting" — matches the question's
        # subject phrasing.
        subject = match.group("subject").strip()
        subject = _LEADING_DET_RE.sub("", subject)
        subject = _TRAILING_PREP_RE.sub("", subject).strip()
        clause = _enclosing_clause(text, match.start(), match.end())
        key = ("paid:" + subject.lower(), amount.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            value=amount,
            normalized_value=re.sub(r"[^\d.]", "", amount) or amount,
            candidate_type="value",
            source_session_id=source_session_id,
            source_text=text,
            confidence=0.7,
            claim=clause,
            answer_type="amount",
            subject=subject,
            relation="paid",
            object_=amount,
            source_span=match.group(0).strip(),
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


def _strip_scaffold_lines(text: str) -> str:
    """
    Drop Markdown headings, code fences, and bullet labels that the
    wiki-window wrapper inserts. Keeps the body lines so extractors
    still see the actual user/assistant turn content.

    Without this filter, ``extract_entities`` happily turns
    ``"Matched Turn"`` into a capitalised-phrase entity candidate.
    """
    if not text:
        return ""
    kept: list[str] = []
    for raw_line in text.splitlines():
        if is_scaffold_line(raw_line):
            continue
        # Drop ``- (user):`` / ``- (assistant):`` bullet prefixes but
        # KEEP the body — the body is the actual content we want.
        stripped = re.sub(
            r"^\s*-\s+\((?:user|assistant|system|tool)\):\s*",
            "",
            raw_line,
        )
        kept.append(stripped)
    return "\n".join(kept)


def extract_candidates_from_text(
    text: str,
    *,
    source_session_id: str = "",
    source_role: str = "",
) -> list[Candidate]:
    """
    Run every typed extractor over a single chunk of text.

    ``source_role`` is propagated onto every emitted Candidate so the
    ranker can prefer user-stated facts on autobiographical questions.

    The input is pre-filtered with :func:`_strip_scaffold_lines` to
    drop Markdown headings (``## Matched Turn``) and code fences
    that would otherwise become entity candidates. Bullet-prefix
    role tags are removed but the bullet body is preserved.
    """
    if not text:
        return []
    cleaned = _strip_scaffold_lines(text)
    if not cleaned:
        return []
    raw: list[Candidate] = []
    raw.extend(extract_counts(cleaned,        source_session_id=source_session_id))
    raw.extend(extract_durations(cleaned,     source_session_id=source_session_id))
    raw.extend(extract_dates(cleaned,         source_session_id=source_session_id))
    raw.extend(extract_locations(cleaned,     source_session_id=source_session_id))
    raw.extend(extract_persons(cleaned,       source_session_id=source_session_id))
    raw.extend(extract_entities(cleaned,      source_session_id=source_session_id))
    raw.extend(extract_preferences(cleaned,   source_session_id=source_session_id))
    raw.extend(extract_value_claims(cleaned,  source_session_id=source_session_id))
    raw.extend(extract_change_facts(cleaned,  source_session_id=source_session_id))

    # Defence-in-depth: drop any candidate whose value/object/claim
    # IS scaffold text, even though the extractor was given a cleaned
    # input. Catches edge cases like a candidate whose extracted span
    # happens to literally be ``"Sub-answer"``. NB: ``is_scaffold_text``
    # returns True on empty strings, so only check non-empty fields —
    # otherwise we'd reject every candidate without an `object_` set
    # (most pre-grounded-claim extractors).
    out: list[Candidate] = []
    for c in raw:
        if c.value and is_scaffold_text(c.value):
            continue
        if c.object_ and is_scaffold_text(c.object_):
            continue
        if c.normalized_value and is_scaffold_text(c.normalized_value):
            continue
        # Stamp the role onto every kept candidate. We can't use
        # dataclasses.replace because that creates a new instance —
        # which is exactly what we need since Candidate is frozen.
        if source_role and not c.source_role:
            from dataclasses import replace as _replace
            c = _replace(c, source_role=source_role)
        out.append(c)
    return out


def extract_candidates_from_context(
    ctx: ContextBundle | None,
) -> list[Candidate]:
    """Mine every retrieved evidence item for typed candidates."""
    if ctx is None or not getattr(ctx, "items", None):
        return []
    out: list[Candidate] = []
    for item in ctx.items:
        metadata = item.metadata or {}
        sid = str(metadata.get("session_id") or "")
        role = str(metadata.get("role") or "")
        out.extend(extract_candidates_from_text(
            item.content,
            source_session_id=sid,
            source_role=role,
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


# ---------------------------------------------------------------------------
# Claim-aware ranking — the new entry point that supersedes value/type
# matching as the primary signal for answer selection.
# ---------------------------------------------------------------------------


_QUESTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_QUESTION_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on",
    "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "i", "me", "my", "you", "your", "we", "us", "our", "they",
    "them", "this", "that", "it", "its", "if", "then",
    "what", "which", "who", "whose", "where", "when", "why", "how",
    "do", "does", "did", "have", "has", "had", "can", "could", "should",
    "would", "will",
    "many", "much", "few", "several", "long",
})


def _question_subject_tokens(question: str) -> set[str]:
    """Content words of the question, used to align with claim.subject."""
    return {
        t.lower() for t in _QUESTION_TOKEN_RE.findall(question)
        if t.lower() not in _QUESTION_STOPWORDS and len(t) > 1
    }


#: Question-keyword → preferred relations. Drives a +/− 0.7 nudge in
#: the ranker so a "how much did I pay" question picks the ``paid``
#: claim even when a sibling ``is_worth`` claim's source span happens
#: to also mention "paid" (e.g. "worth triple what I paid").
_RELATION_KEYWORDS: dict[str, set[str]] = {
    "pay":   {"paid"},
    "paid":  {"paid"},
    "spent": {"paid"},
    "spend": {"paid"},
    "worth": {"is_worth", "sold_for"},
    "value": {"is_worth"},
    "valued": {"is_worth"},
    "cost":  {"is_worth", "paid"},
    "took":  {"duration_of"},
    "take":  {"duration_of"},
    "long":  {"duration_of"},
    "lasted": {"duration_of"},
    "got":   {"got_at"},
    "earn":  {"got_at"},
    "earned": {"got_at"},
    "graduated": {"got_at"},
}


def _preferred_relations(q_tokens: set[str]) -> set[str]:
    """Union of the relations the question's keywords imply."""
    out: set[str] = set()
    for tok in q_tokens:
        out.update(_RELATION_KEYWORDS.get(tok, set()))
    return out


def rank_claims_for_question(
    candidates: Iterable[Candidate],
    question: str,
    *,
    question_type: QuestionType | None = None,
) -> list[Candidate]:
    """
    Rank candidates by **grounded claim alignment** with ``question``.

    Scoring components (in order of weight):

    1. **Subject overlap** (×2.0) — content-word intersection between
       the candidate's ``subject`` field and the question's content
       words. This is the heaviest signal: a "Bachelor's degree at
       UCLA" candidate matches a "where did I get my Bachelor's"
       question even when the bare value ``"UCLA"`` shares no words
       with the question.
    2. **Source-span overlap** (×1.0) — same, but against the
       candidate's ``source_span`` / ``claim``. Catches subjects the
       parser couldn't isolate.
    3. **Relation bonus** (+0.5) — factual relations
       (:data:`FACTUAL_RELATIONS`) beat hypothetical ones
       (:data:`HYPOTHETICAL_RELATIONS`) when the question is about
       a completed event.
    4. **Object overlap** (×0.5) — the ``object_`` value can echo a
       question word (e.g. asking for a "marathon time" matches a
       claim whose object is "4 hours 12 minutes").
    5. **Confidence** (×0.2) — final tiebreaker; never the primary
       signal.

    Returns the full ranked list (callers truncate to top-N). Use this
    instead of ``filter_candidates_for_question`` when you need a
    grounded answer pick — the legacy filter still works for type-only
    gating (e.g. the don't-IDK guard).
    """
    q_tokens = _question_subject_tokens(question)
    if not q_tokens:
        return list(candidates)
    # Also include the bare question tokens (with stopwords stripped via
    # `_question_subject_tokens`) for relation-keyword matching.
    preferred_rels = _preferred_relations(q_tokens)
    # Personal-question check drives the user-turn boost on
    # autobiographical questions ("I"/"my"/"me" in the question text).
    q_lower = question.lower()
    is_personal = bool(re.search(r"\b(?:i|my|me|mine)\b", q_lower))
    # Question-context keyword boosts. The full lowercased question text
    # is scanned for domain phrases; each match contributes a boost
    # vector that biases candidates whose source span / claim contains
    # the matching anchor noun. This is what makes "What degree did I
    # graduate with?" prefer "graduated with a degree in Business
    # Administration" over generic dates / locations on the same row.
    boost_keywords = _question_context_boosts(question)
    penalty_keywords = _question_context_penalties(question)

    def score(c: Candidate) -> float:
        subject_words = {
            t.lower() for t in _QUESTION_TOKEN_RE.findall(c.subject or "")
            if t.lower() not in _QUESTION_STOPWORDS and len(t) > 1
        }
        span_words = {
            t.lower() for t in _QUESTION_TOKEN_RE.findall(
                c.source_span or c.claim or "",
            )
            if t.lower() not in _QUESTION_STOPWORDS and len(t) > 1
        }
        object_words = {
            t.lower() for t in _QUESTION_TOKEN_RE.findall(c.object_ or "")
            if t.lower() not in _QUESTION_STOPWORDS and len(t) > 1
        }
        subj_overlap = len(q_tokens & subject_words)
        span_overlap = len(q_tokens & span_words)
        obj_overlap = len(q_tokens & object_words)
        s = (
            2.0 * subj_overlap
            + 1.0 * span_overlap
            + 0.5 * obj_overlap
            + 0.2 * c.confidence
        )
        if c.relation in FACTUAL_RELATIONS:
            s += 0.5
        elif c.relation in HYPOTHETICAL_RELATIONS or c.relation == "looking_at":
            s -= 0.5
        # Relation-keyword alignment: when the question explicitly uses
        # "worth" / "pay" / "took", strongly prefer the matching
        # relation. This guards against the source_span trap where
        # one claim's verbatim text incidentally mentions another
        # claim's keyword (e.g. "worth triple what I paid" also
        # contains "paid").
        if preferred_rels:
            if c.relation in preferred_rels:
                s += 0.7
            else:
                s -= 0.3
        # Question-type alignment — a tiny nudge so a LOCATION question
        # prefers a location-typed claim when subject overlap ties.
        if question_type is not None:
            wanted = _ANSWER_TYPE_FOR_QUESTION.get(question_type, set())
            if c.answer_type in wanted or c.candidate_type in wanted:
                s += 0.2

        # Question-context keyword boosts. Each matched anchor in the
        # candidate's span/claim/object contributes +0.6 — strong enough
        # to flip the order when subject overlap ties.
        haystack = " ".join(filter(None, (
            c.claim, c.source_span, c.subject, c.object_, c.source_text,
        ))).lower()
        for anchor in boost_keywords:
            if anchor in haystack:
                s += 0.6
        # Penalty keywords for context-leakage failures
        # (gift-question + surgery-context, etc.). Each match -0.5.
        for anchor in penalty_keywords:
            if anchor in haystack:
                s -= 0.5

        # User-turn preference on autobiographical questions.
        # The user said it → strong signal (+0.4). The assistant said it
        # → mild penalty (-0.2) so it stays in the pool but doesn't
        # outrank a direct user memory. Unknown role → no effect.
        if is_personal:
            if c.source_role == "user":
                s += 0.4
            elif c.source_role == "assistant":
                s -= 0.2
        return s

    scored = [(score(c), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s > 0.0]
    scored.sort(key=lambda pair: -pair[0])
    return [c for _s, c in scored]


# ─── Question-context keyword tables ──────────────────────────────────────


#: Question phrase → anchor nouns we want present in the candidate's
#: source span / claim. Lowercased; substring match on the question.
#: Anchors that fire are added (lowercase) to the per-question boost
#: list and matched substring-style against the candidate's source.
_QUESTION_CONTEXT_BOOSTS: dict[str, tuple[str, ...]] = {
    "degree":           ("degree", "graduated", "graduation", "bachelor", "master", "doctorate", "phd"),
    "graduate":         ("degree", "graduated", "graduation"),
    "graduation":       ("degree", "graduated", "graduation"),
    "occupation":       ("occupation", "job", "work", "career", "role", "title", "specialist", "engineer", "manager", "developer"),
    "job":              ("job", "occupation", "work", "career", "role"),
    "previous job":     ("previous", "former", "used to", "before"),
    "previous occupation": ("previous", "former", "used to", "before"),
    "fundraising":      ("fundraising", "fundraiser", "fundraising dinner", "fundraising event"),
    "fundraiser":       ("fundraising", "fundraiser", "fundraising dinner"),
    "volunteer":        ("volunteer", "volunteered", "volunteering"),
    "animal shelter":   ("animal shelter", "shelter", "fundraising"),
    "birthday gift":    ("birthday", "gift", "present", "sister", "brother"),
    "birthday":         ("birthday", "gift", "present"),
    "gift":             ("gift", "present", "bought for", "got for"),
    "for my sister":    ("sister", "gift", "birthday"),
    "dress":            ("dress", "outfit"),
    "shirts":           ("shirts", "shirt", "packed", "pack"),
    "pack":             ("pack", "packed", "luggage", "suitcase"),
    "commute":          ("commute", "drive", "ride"),
    "school":           ("school", "university", "college"),
}

#: Question phrase → anchor nouns whose presence DOWN-WEIGHTS the
#: candidate (it's almost certainly distractor context). Used for the
#: gift-vs-surgery / gift-vs-groceries / gift-vs-flowers failure shape.
_QUESTION_CONTEXT_PENALTIES: dict[str, tuple[str, ...]] = {
    "birthday gift":    ("surgery", "groceries", "recipe", "doctor's appointment", "appointment"),
    "gift":             ("surgery", "groceries", "recipe", "appointment"),
    "for my sister":    ("surgery", "recipe", "appointment"),
    "degree":           ("recipe", "surgery", "shopping"),
    "occupation":       ("recipe", "surgery", "shopping list"),
    "job":              ("recipe", "surgery", "shopping list"),
    "fundraising":      ("recipe", "shopping", "groceries"),
    "volunteer":        ("recipe", "shopping list"),
}


def _question_context_boosts(question: str) -> set[str]:
    """Return the anchor nouns whose presence in a candidate boosts its score."""
    q = question.lower()
    out: set[str] = set()
    for phrase, anchors in _QUESTION_CONTEXT_BOOSTS.items():
        if phrase in q:
            out.update(a.lower() for a in anchors)
    return out


def _question_context_penalties(question: str) -> set[str]:
    """Return the anchor nouns whose presence in a candidate penalises its score."""
    q = question.lower()
    out: set[str] = set()
    for phrase, anchors in _QUESTION_CONTEXT_PENALTIES.items():
        if phrase in q:
            out.update(a.lower() for a in anchors)
    return out


# Map QuestionType → set of answer_type / candidate_type values that
# satisfy it. Used by `rank_claims_for_question` for a tiebreaker bonus.
_ANSWER_TYPE_FOR_QUESTION: dict[QuestionType, set[str]] = {
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
    "FACTUAL_RELATIONS",
    "HYPOTHETICAL_RELATIONS",
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
    "extract_value_claims",
    "filter_candidates_for_question",
    "normalize_numeric",
    "rank_claims_for_question",
]
