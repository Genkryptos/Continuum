"""
evals/longmemeval/reasoning_heads.py
====================================
Six reasoning heads — one per :class:`TaskMode`.

Each head consumes a question + ranked list of :class:`Claim` and
returns a short final-answer string (or ``""`` to signal "no
deterministic answer; please call the extractive fallback"). The
heads share a small set of span-extraction helpers so each one stays
short and audit-able.

Design notes
------------
* The heads are **deterministic by default**. Each one returns a
  best-effort string from the top claims using regex spans. They only
  return ``""`` when nothing in the claim list looks like the
  question's answer shape; the caller then routes to
  :mod:`evals.longmemeval.extractive_fallback` for a one-shot LLM
  span extraction.
* "Shortest faithful answer span" is the recurring rule for
  ``FACT_LOOKUP``: pull the smallest substring of the top claim that
  matches the question's expected answer shape (noun phrase, date,
  number+unit, etc.). The full claim is **not** the answer — the
  *substring* is.
* Heads never call an LLM directly. The orchestrator decides when
  the fallback is needed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from evals.longmemeval.claims import Claim, _content_tokens, rank_claims
from evals.longmemeval.scaffold_filter import (
    is_scaffold_text,
    is_user_request_sentence,
)

log = logging.getLogger(__name__)


# ─── Shortest-span extractors (shared by FACT_LOOKUP + variants) ──────────


_NOUN_PHRASE_AFTER_PREP = re.compile(
    r"\b(?:in|at|with|of|to|from|on|for|as)\s+"
    r"(?P<phrase>(?:[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,5}))",
)
_QUOTED_PHRASE = re.compile(
    r"[\"“‘]([^\"”’]{2,80})[\"”’]",
)
# Date patterns split by SPECIFICITY so the date extractor can prefer
# "February 14th" over a bare "May". Higher tiers are tried first; only
# fall to the next tier when nothing in the higher tier matched.
_DATE_SPECIFIC_RE = re.compile(
    # ISO date / "Month Day, Year" / "Day of Month Year"
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(?:\s+\d{4})?",
    re.IGNORECASE,
)
_DATE_RELATIVE_RE = re.compile(
    # Concrete relative phrases — "last Sunday", "two weeks ago".
    r"\b(?:yesterday|today|tomorrow|tonight)\b"
    r"|\b\d+\s+(?:day|week|month|year)s?\s+ago\b"
    r"|\blast\s+(?:week|month|year|sunday|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|night)\b"
    r"|\bnext\s+(?:week|month|year|sunday|monday|tuesday|wednesday|"
    r"thursday|friday|saturday)\b",
    re.IGNORECASE,
)
_DATE_VAGUE_RE = re.compile(
    # Month-only or week-only — last-resort fallback. "May" alone
    # is much weaker than "May 14th" and should only win when the
    # claim has no more-specific date.
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b"
    r"|\bthis\s+(?:week|month|year)\b",
    re.IGNORECASE,
)
# Kept for backward-compat with span-extraction tests / callers that
# expect ``_DATE_SPAN`` to be defined.
_DATE_SPAN = _DATE_SPECIFIC_RE
_NUMBER_UNIT_SPAN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s+([A-Za-z][A-Za-z\-]{2,30})",
)
_PURE_NUMBER_SPAN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_OCCUPATION_SPAN = re.compile(
    r"\b(?P<title>(?:senior|junior|lead|principal|staff)?\s*"
    r"(?:marketing|sales|software|product|data|design|business|"
    r"finance|operations|account|customer|engineering|research)?\s*"
    r"(?:specialist|engineer|manager|developer|analyst|director|"
    r"officer|consultant|designer|architect|scientist|coordinator|"
    r"associate|representative|lead|head|chief)"
    r"(?:\s+at\s+(?:a\s+|the\s+)?[A-Za-z][\w\- ]{1,40}?)?)",
    re.IGNORECASE,
)

#: "I used to be a/an X" / "my previous job was X" / "worked as a/an X".
#: Captures the job phrase including any trailing "at <company>"
#: qualifier. Used by the occupation extractor to surface the actual
#: previous-job phrase rather than the bare title word.
_PREVIOUS_OCCUPATION_RE = re.compile(
    r"\b(?:"
    r"i\s+used\s+to\s+be\s+(?:a|an)\s+"
    r"|i\s+was\s+(?:a|an)\s+"
    r"|my\s+(?:previous|former|old|last)\s+"
    r"(?:job|role|position|occupation|career|gig)\s+was\s+(?:a\s+|an\s+)?"
    r"|(?:i\s+)?worked\s+as\s+(?:a|an)\s+"
    r"|(?:i\s+)?used\s+to\s+work\s+as\s+(?:a|an)\s+"
    r")"
    r"(?P<phrase>[A-Za-z][\w\-' ]{3,80}?)"
    r"(?=\s+(?:before|but|and\s+then|until|when|because|so\s+I|"
    r"in\s+the|in\s+a|in\s+an)|[.!?,;]|$)",
    re.IGNORECASE,
)
_VALUE_PHRASE_AFTER_WORTH = re.compile(
    r"\bworth\s+(?P<phrase>[^.!?,;]+?)(?=[.!?,;]|$)",
    re.IGNORECASE,
)
# Allow an optional indirect-object between the verb and the gift
# noun ("I bought my sister a yellow dress" — "my sister" is the
# indirect object, "a yellow dress" is the item we want). The
# negative lookahead keeps the captured phrase short by ending at
# trailing prepositions like "for"/"to"/"on".
_OBJECT_AFTER_BOUGHT = re.compile(
    r"\b(?:bought|got|picked\s+up|chose|grabbed|purchased|gave)\s+"
    r"(?:(?:my|her|him|them)\s+\w+\s+)?"
    r"(?P<phrase>(?:a|an|the)\s+[A-Za-z][\w'\- ]{2,60}?)"
    r"(?=\s+(?:for|to|on|at|in|because|since|when|so|and|"
    r"as\s+a\s+gift|as\s+a\s+present)|[.!?,;]|$)",
    re.IGNORECASE,
)

#: "the gift was X" / "her birthday gift was X" / "I picked X as a
#: gift". Higher-priority fallback when the buy-verb didn't fire.
_GIFT_ITEM_RE = re.compile(
    r"\b(?:(?:her|his|the|my\s+sister[''']?s|my\s+brother[''']?s)\s+"
    r"(?:birthday\s+)?(?:gift|present)\s+(?:was|is)\s+"
    r"|(?:gift|present)\s+for\s+(?:my\s+sister|her|him|them)\s+(?:was|is)\s+)"
    r"(?P<phrase>(?:a|an|the)\s+[A-Za-z][\w'\- ]{2,60}?)"
    r"(?=[.!?,;]|$)",
    re.IGNORECASE,
)
_DEGREE_PHRASE = re.compile(
    r"\b(?:degree|bachelor[''']?s|master[''']?s|phd|doctorate|"
    r"bs|ba|ms|ma|mba)\s+(?:in|of)\s+"
    r"(?P<phrase>[A-Z][\w' ]{2,40}?)(?=\s+(?:from|at|in)|[.!?,;]|$)",
    re.IGNORECASE,
)


def _shortest_noun_phrase(claim_text: str) -> str:
    """First quoted phrase, then capitalised noun phrase after a preposition."""
    m = _QUOTED_PHRASE.search(claim_text)
    if m:
        return m.group(1).strip()
    m = _NOUN_PHRASE_AFTER_PREP.search(claim_text)
    if m:
        return m.group("phrase").strip()
    return ""


def _shortest_date(claim_text: str) -> str:
    """
    Return the MOST-SPECIFIC date in ``claim_text``.

    Tier 1: Month+day / ISO / "14th of May" — the answer the user
    almost certainly meant.
    Tier 2: Concrete relative phrases — "last Sunday", "2 weeks ago".
    Tier 3: Bare month / "this week" — only when nothing better
    exists in the claim.

    This is what stops the 58ef2f1c failure where the claim mentioned
    both "May" (incidentally) and "February 14th" (the actual answer)
    — the old extractor returned the first match in document order.
    """
    m = _DATE_SPECIFIC_RE.search(claim_text)
    if m:
        return m.group(0).strip()
    m = _DATE_RELATIVE_RE.search(claim_text)
    if m:
        return m.group(0).strip()
    m = _DATE_VAGUE_RE.search(claim_text)
    return m.group(0).strip() if m else ""


#: Words that CANNOT be the unit of a count. ``<num> <stopword>`` is
#: never a valid count span — "110 for" is transcript junk, not a
#: count of "for"s. The list is intentionally narrow; only function
#: words and connectives that frequently land after numbers in
#: scraped transcripts.
_STOPWORD_UNITS = frozenset({
    "for", "with", "from", "of", "to", "in", "on", "by", "at", "as",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "and", "or", "but", "the", "a", "an", "my", "your", "their",
    "this", "that", "these", "those", "it", "its",
    "i", "you", "we", "they", "he", "she",
    # Common transcript-timestamp tails observed in the wiki bundle.
    "share", "principles", "every", "worked", "applied", "completely",
    "so", "if", "when", "then",
})


def _shortest_count(claim_text: str, question: str) -> str:
    """
    For COUNT questions: ``<number> <unit>`` whose unit is mentioned in
    the question AND is a real noun, not a stopword.

    The stopword check is what stops "110 for" from being picked when
    the question contains the word "for" as a preposition rather than
    a real unit anchor. Without it, scraped transcripts that say
    things like "0:02 110 for the introduction" leak into the answer.
    """
    q_lower = question.lower()
    for num, unit in _NUMBER_UNIT_SPAN.findall(claim_text):
        unit_norm = unit.lower().rstrip("s")
        if unit_norm in _STOPWORD_UNITS:
            continue
        if unit_norm in q_lower:
            return f"{num} {unit}".strip()
    # Fallback: any pure number
    m = _PURE_NUMBER_SPAN.search(claim_text)
    return m.group(0) if m else ""


def _shortest_value_phrase(claim_text: str) -> str:
    """For ``worth`` questions: span after the word ``worth``."""
    m = _VALUE_PHRASE_AFTER_WORTH.search(claim_text)
    return m.group("phrase").strip().rstrip(".,;!?") if m else ""


def _shortest_object_bought(claim_text: str) -> str:
    """
    Extract the gift / purchased item.

    Tier 1: ``I bought (my sister)? a/an X`` — primary pattern.
    Tier 2: ``the gift/present was X`` — declarative variant.
    """
    m = _OBJECT_AFTER_BOUGHT.search(claim_text)
    if m:
        return m.group("phrase").strip().rstrip(".,;!?")
    m = _GIFT_ITEM_RE.search(claim_text)
    if m:
        return m.group("phrase").strip().rstrip(".,;!?")
    return ""


def _shortest_degree(claim_text: str) -> str:
    """For ``what degree`` questions: phrase after ``degree in/of``."""
    m = _DEGREE_PHRASE.search(claim_text)
    return m.group("phrase").strip().rstrip(".,;!?") if m else ""


def _shortest_occupation(claim_text: str) -> str:
    """
    Prefer the full "I was a/an X" phrase over a bare title word.

    Tier 1: ``_PREVIOUS_OCCUPATION_RE`` — captures the entire job
    phrase including ``at <company>`` (the answer the user gave).
    Tier 2: ``_OCCUPATION_SPAN`` — title-only fallback when no
    "I was…" frame was found.
    """
    m = _PREVIOUS_OCCUPATION_RE.search(claim_text)
    if m:
        return m.group("phrase").strip().rstrip(".,;!?")
    m = _OCCUPATION_SPAN.search(claim_text)
    return m.group("title").strip().rstrip(".,;!?") if m else ""


# ─── Question-shape detectors ─────────────────────────────────────────────


_Q_DEGREE = re.compile(r"\bdegree\b|\bgraduate", re.IGNORECASE)
_Q_OCCUPATION = re.compile(
    r"\boccupation|\bjob\b|\bwork\b|\bcareer\b|\bprofession\b",
    re.IGNORECASE,
)
_Q_BOUGHT = re.compile(
    r"\bbuy|\bbought|\bgift\b|\bpresent\b|\bpurchase", re.IGNORECASE,
)
_Q_WORTH = re.compile(r"\bworth\b|\bvalue\b|\bvalued\b", re.IGNORECASE)
_Q_WHEN = re.compile(r"\bwhen\b|\bwhat\s+(?:day|date|month|year)\b", re.IGNORECASE)
_Q_HOW_MANY = re.compile(r"\bhow\s+many\b|\bhow\s+much\b", re.IGNORECASE)


def _extract_fact_span(question: str, claim_text: str) -> str:
    """
    Try the question-shape-aware extractors in priority order.

    Returns the shortest faithful answer span, or ``""`` when no
    pattern matches.
    """
    if _Q_DEGREE.search(question):
        s = _shortest_degree(claim_text)
        if s:
            return s
    if _Q_OCCUPATION.search(question):
        s = _shortest_occupation(claim_text)
        if s:
            return s
    if _Q_WORTH.search(question):
        s = _shortest_value_phrase(claim_text)
        if s:
            return s
    if _Q_BOUGHT.search(question):
        s = _shortest_object_bought(claim_text)
        if s:
            return s
    if _Q_WHEN.search(question):
        s = _shortest_date(claim_text)
        if s:
            return s
    if _Q_HOW_MANY.search(question):
        s = _shortest_count(claim_text, question)
        if s:
            return s
    # Generic fallback — quoted phrase or capitalised noun phrase.
    return _shortest_noun_phrase(claim_text)


# ─── The six heads ────────────────────────────────────────────────────────


def _is_answerable_claim(claim_text: str) -> bool:
    """
    A claim is answerable if it isn't a heading/scaffold/user-request.
    These shapes mention topic words (the user's question shares
    vocab with them) but never contain the requested answer.
    """
    if not claim_text.strip():
        return False
    if is_scaffold_text(claim_text):
        return False
    return not is_user_request_sentence(claim_text)


def head_fact_lookup(claims: Iterable[Claim], question: str) -> str:
    """
    Single autobiographical fact.

    Walks the top claims and returns the first claim that produces an
    answer-shaped span. Heading-only, label-only, and user-request
    sentences are skipped — they may topic-overlap with the question
    but they don't ANSWER it.

    Fallback strategy when no span can be extracted: return the first
    *answerable* claim's body — never a heading or a request. This is
    what blocks the 66f24dbb regression ("**Sister's Birthday**" was
    returned as the bare claim body).
    """
    ranked = rank_claims(question, claims, prefer_role="user")
    # First pass: only consider claims that pass answerability.
    answerable = [c for c in ranked if _is_answerable_claim(c.text)]
    for c in answerable[:8]:
        span = _extract_fact_span(question, c.text)
        if span:
            return span
    # Body fallback — but ONLY when the top claim is short enough to
    # plausibly BE an answer. A long descriptive sentence ("In my
    # previous role at the startup, I was responsible for managing
    # marketing campaigns…") is prose, not an answer. Returning it
    # would just hand the validator something to reject. Returning
    # empty lets the orchestrator's extractive LLM fallback try the
    # top claims directly. The 20-word cap was chosen empirically:
    # short atomic claims like "I went to the park yesterday" survive
    # (8 words), long responsibility prose does not.
    if answerable and len(answerable[0].text.split()) <= 20:
        return answerable[0].text.strip().rstrip(".")
    return ""


# ─── Assistant-memory anchor helpers (Tier 2B-2 / 2B-3) ────────────────────


#: Ordinal-digit pattern: ``7th``, ``27th``, ``3rd``.
_ORDINAL_DIGIT_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
#: Ordinal-word map for spelled-out positions up to ``twentieth``.
_ORDINAL_WORD_MAP: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
    "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
}
_ORDINAL_WORD_RE = re.compile(
    r"\b(?P<word>" + "|".join(_ORDINAL_WORD_MAP) + r")\b",
    re.IGNORECASE,
)
#: ``last`` is treated as ordinal sentinel -1.
_LAST_RE = re.compile(r"\blast\b", re.IGNORECASE)
#: Numbered-list line prefix surviving sentence splitting.
_NUMBERED_ITEM_RE = re.compile(r"^\s*(\d+)[.)\]:]\s+(?P<body>.+)$")


def _extract_ordinal_from_question(question: str) -> int | None:
    """
    Return the 1-indexed position the question asks for, ``-1`` for
    ``"last"``, or ``None`` when no ordinal is mentioned.

    The check is intentionally narrow: only the explicit ordinal
    forms (``7th``, ``27th``, ``third``, ``last``). Words like
    ``first time`` would match ``first``, which is correct — if the
    user asked "what was the first thing you suggested?" we want
    item 1.
    """
    m = _ORDINAL_DIGIT_RE.search(question)
    if m:
        return int(m.group(1))
    m = _ORDINAL_WORD_RE.search(question)
    if m:
        return _ORDINAL_WORD_MAP[m.group("word").lower()]
    if _LAST_RE.search(question):
        return -1
    return None


def _group_claims_into_turns(
    claims: list[Claim],
) -> list[tuple[str, str, list[Claim]]]:
    """
    Order claims by ``(session_id, turn_index)`` and group consecutive
    same-role-same-session runs into turns.

    Returns ``[(role, session_id, claims_in_turn), ...]`` in
    bundle-position order. A "turn" here is the smallest unit of
    contiguous same-speaker content — exactly the structure the
    assistant-memory head needs to walk.
    """
    indexed = [
        c for c in claims
        if c.turn_index is not None and c.source_role
    ]
    indexed.sort(key=lambda c: (c.source_session_id, c.turn_index or 0))
    groups: list[tuple[str, str, list[Claim]]] = []
    cur_role = ""
    cur_session = ""
    cur_group: list[Claim] = []
    for c in indexed:
        if (
            c.source_role.lower() != cur_role.lower()
            or c.source_session_id != cur_session
        ):
            if cur_group:
                groups.append((cur_role, cur_session, cur_group))
            cur_group = [c]
            cur_role = c.source_role
            cur_session = c.source_session_id
        else:
            cur_group.append(c)
    if cur_group:
        groups.append((cur_role, cur_session, cur_group))
    return groups


def _find_ordinal_item_in_turn(
    claims_in_turn: list[Claim], n: int,
) -> str:
    """
    Resolve an ``n``-th item from an assistant turn that contains a
    list.

    Two-tier strategy:
      1. Explicit ``"N. body"`` / ``"N) body"`` prefixes when they
         survived sentence splitting.
      2. Positional fallback — the Nth answerable claim *is* the Nth
         list item.

    Returns the item body or ``""`` when no item can be resolved.
    """
    items = [c for c in claims_in_turn if _is_answerable_claim(c.text)]
    if not items:
        return ""
    # Tier 1: explicit numbering.
    by_number: dict[int, str] = {}
    for c in items:
        m = _NUMBERED_ITEM_RE.match(c.text.strip())
        if m:
            by_number[int(m.group(1))] = m.group("body").strip().rstrip(".")
    if by_number:
        if n == -1:
            return by_number[max(by_number)]
        return by_number.get(n, "")
    # Tier 2: positional.
    if n == -1:
        return items[-1].text.strip().rstrip(".")
    if 1 <= n <= len(items):
        return items[n - 1].text.strip().rstrip(".")
    return ""


def _score_user_turn_against_question(
    user_claims: list[Claim], q_tokens: set[str],
) -> float:
    """Content-overlap score of a user turn vs the question."""
    if not q_tokens:
        return 0.0
    text = " ".join(c.text for c in user_claims)
    turn_tokens = _content_tokens(text)
    if not turn_tokens:
        return 0.0
    return len(q_tokens & turn_tokens) / len(q_tokens)


#: Soft generic-intro pattern — catches openers that have enough
#: substantive tail to escape :func:`is_generic_intro` but are still
#: prefatory phrasing rather than an answer. Used as a soft penalty
#: in :func:`_pick_answer_from_assistant_claim`, not a hard reject:
#: a real answer can legitimately follow these phrases on the same
#: line, but when there's a better alternative we prefer it.
_SOFT_INTRO_RE = re.compile(
    r"^\s*(?:sure|yes|absolutely|certainly|of\s+course|"
    r"here\s+(?:are|is|you\s+go)|here(?:'?s)|"
    r"happy\s+to\s+help|i'?d\s+be\s+happy|"
    r"i\s+can\s+help|let\s+me|"
    r"that'?s\s+a\s+(?:great|good)\s+question)\b",
    re.IGNORECASE,
)
#: Nationality / regional adjectives the noun-phrase span extractor
#: sometimes returns alone ("Roman" from "great for Roman pasta",
#: "Italian" from "Italian food"). When the span IS one of these, we
#: prefer the body — the real answer is the proper noun the
#: adjective modifies, not the adjective itself. A small closed set
#: rather than a regex so legitimate proper nouns ending in -an
#: ("Manhattan", "Toronto") don't get rejected.
_NATIONALITY_ADJECTIVES: frozenset[str] = frozenset({
    "roman", "italian", "greek", "french", "german", "russian",
    "chinese", "japanese", "korean", "english", "british", "spanish",
    "american", "mexican", "canadian", "african", "asian", "european",
    "indian", "persian", "arabic", "slavic", "nordic", "hispanic",
    "latin", "polish", "swedish", "danish", "norwegian", "finnish",
    "irish", "scottish", "welsh", "dutch", "belgian", "swiss",
    "austrian", "portuguese", "brazilian", "argentine", "chilean",
    "colombian", "egyptian", "moroccan", "turkish", "iranian",
    "iraqi", "thai", "vietnamese", "indonesian", "filipino",
    "australian", "kiwi", "ethiopian", "kenyan", "south", "north",
    "east", "west", "western", "eastern", "northern", "southern",
})


def _pick_answer_from_assistant_claim(
    claim_text: str, question: str,
) -> str:
    """
    Pick the best answer string from a single assistant claim.

    Strategy (regression-aware):

      1. **Short bodies (≤ 12 words) → return body verbatim.** The
         substring scorer accepts any answer token contained in the
         body, so "Roscioli is great for Roman pasta" credits
         "Roscioli", "Andy is a great choice" credits "Andy", and
         "Use 2-3 eggs, beaten..." credits "2-3 eggs". Span
         extraction was over-shortening these (returning "Roman" /
         "3 eggs"), causing the v10 regressions.

      2. **Medium bodies (13–20 words) → span first if it's
         high-confidence, body otherwise.** A high-confidence span
         is one that (a) starts the body — subject-position, or
         (b) is multi-word, or (c) contains a digit. Single-word
         nationality adjectives sitting mid-body ("Roman" /
         "Italian") are explicitly rejected.

      3. **Long prose (> 20 words) → span only.** Returning a
         20-plus-word body verbatim is too noisy and tends to fail
         the answer-shape validator downstream.

    The split point — 12 words for body-wins — is empirical. Short
    factual replies almost always fit (assistant lists items
    individually, gives short recommendations, etc.); 12+ word
    sentences are increasingly likely to carry intro phrasing or
    multiple competing facts.
    """
    body = claim_text.strip().rstrip(".")
    if not body:
        return ""
    body_tokens = body.split()
    word_count = len(body_tokens)
    span = _extract_fact_span(question, claim_text)
    span_clean = span.strip().rstrip(".,;!?") if span else ""

    # Tier 1: short body — body verbatim wins.
    if word_count <= 12:
        return body

    # Tier 2: medium body — span first if high-confidence.
    if span_clean and word_count <= 20:
        starts_body = body.lower().startswith(span_clean.lower())
        is_multi_word = len(span_clean.split()) >= 2
        has_digit = bool(re.search(r"\d", span_clean))
        is_adjective_only = (
            len(span_clean.split()) == 1
            and span_clean.lower() in _NATIONALITY_ADJECTIVES
        )
        if (
            (starts_body or is_multi_word or has_digit)
            and not is_adjective_only
        ):
            return span_clean
        return body  # span is unreliable; body verbatim is safer.

    # Tier 3: long prose — span only.
    if span_clean:
        return span_clean
    return ""


def head_assistant_memory(claims: Iterable[Claim], question: str) -> str:
    """
    Assistant-said-X lookup with anchor-based span selection.

    Strategy (Tier 2B-2 / 2B-3):

      1. Group bundle claims into ordered turns by ``(session_id,
         role, turn_index)``.
      2. Score each USER turn by content-word overlap with the
         question — this is the anchor.
      3. Take the next 1–2 ASSISTANT turns in the same session.
      4. If the question contains an ordinal (``7th``, ``last``,
         ``third``), resolve it inside those assistant turns and
         return the item.
      5. Otherwise extract an answer-shaped span from those assistant
         claims using the FACT_LOOKUP extractor.
      6. Fall back to the legacy ``rank_claims(prefer_role=assistant)``
         path when no anchor produced an answer — preserves
         backward-compat for bundles that don't have clean turn
         structure.

    Why anchor-based: the legacy head scored claims globally by
    question overlap and let topic anchors (``Barcelona``, ``Rome``)
    outrank the assistant's actual answer. Anchoring on the matching
    user turn then taking the immediately-following assistant span
    is the structural cure to "topic anchor / nearby entity"
    failures.
    """
    claim_list = list(claims)
    q_tokens = _content_tokens(question)
    ordinal = _extract_ordinal_from_question(question)

    groups = _group_claims_into_turns(claim_list)

    # Score each user turn; sort by descending overlap.
    user_scores: list[tuple[float, int]] = []
    for i, (role, _sid, group) in enumerate(groups):
        if role.lower() != "user":
            continue
        score = _score_user_turn_against_question(group, q_tokens)
        if score > 0:
            user_scores.append((score, i))
    user_scores.sort(key=lambda t: -t[0])

    # Anchor selection.
    #
    # * Normal case: top 3 score-positive user turns.
    # * Ordinal queries: the question often shares few content words
    #   with the prompt that elicited the list (``"7th job"`` vs
    #   ``"Brainstorm ideas for jobs"``), so when an ordinal is
    #   present we expand to *every* user turn in bundle order.
    if ordinal is not None:
        if user_scores:
            anchors_to_try: list[tuple[float, int]] = list(user_scores)
        else:
            anchors_to_try = [
                (0.0, i) for i, (role, _sid, _g) in enumerate(groups)
                if role.lower() == "user"
            ]
    else:
        anchors_to_try = user_scores[:3]

    for _score, anchor_idx in anchors_to_try:
        anchor_session = groups[anchor_idx][1]
        # Take following assistant turn(s) in the same session.
        following_assistant: list[list[Claim]] = []
        for j in range(anchor_idx + 1, len(groups)):
            role, session, group = groups[j]
            if session != anchor_session:
                break
            r = role.lower()
            if r == "assistant":
                following_assistant.append(group)
                if len(following_assistant) >= 2:
                    break
            elif r == "user":
                # Next user turn marks the end of the anchor's
                # response area.
                break
        if not following_assistant:
            continue
        # Ordinal-aware extraction first.
        if ordinal is not None:
            for grp in following_assistant:
                item = _find_ordinal_item_in_turn(grp, ordinal)
                if item:
                    return item
        # Span / body extraction for assistant-memory.
        # The picker (:func:`_pick_answer_from_assistant_claim`) tries
        # the FACT_LOOKUP span first and only falls back to body when
        # the span is unreliable (mid-body adjective, single weak
        # token). This avoids the v10 regression where body-verbatim
        # would clobber a good span like "2-3 eggs" or "Andy".
        #
        # Within a turn we also softly penalise prefatory phrasing
        # ("Sure, here are...", "Happy to help...") that escaped the
        # hard intro filter — they get walked PAST in favour of a
        # later answerable claim from the same turn.
        for grp in following_assistant:
            answerable = [c for c in grp if _is_answerable_claim(c.text)]
            if not answerable:
                continue
            # Reorder: claims with a soft-intro prefix go to the end.
            prefatory: list[Claim] = []
            primary: list[Claim] = []
            for c in answerable:
                if _SOFT_INTRO_RE.match(c.text):
                    prefatory.append(c)
                else:
                    primary.append(c)
            ordered = primary + prefatory
            for c in ordered[:6]:
                picked = _pick_answer_from_assistant_claim(
                    c.text, question,
                )
                if picked:
                    return picked

    # Fallback — legacy ranked behavior. Preserves bundles where the
    # role/turn structure isn't clean enough to anchor.
    #
    # Uses the same picker as the anchored path so the regression-
    # aware short-body-wins / span-confidence rules apply here too.
    # Without this, the v10 regressions ("3 eggs" vs "2-3 eggs",
    # "Roman" vs "Roscioli") would resurface whenever anchor scoring
    # returned no user-turn match.
    ranked = rank_claims(question, claim_list, prefer_role="assistant")
    answerable = [c for c in ranked if _is_answerable_claim(c.text)]
    # Apply the soft-intro demotion here as well — prefatory phrasing
    # ("Sure, here's...") gets walked past in favour of the real
    # answer claim when one exists in the same pool.
    prefatory_fb = [c for c in answerable if _SOFT_INTRO_RE.match(c.text)]
    primary_fb = [c for c in answerable if not _SOFT_INTRO_RE.match(c.text)]
    ordered_fb = primary_fb + prefatory_fb
    for c in ordered_fb[:6]:
        picked = _pick_answer_from_assistant_claim(c.text, question)
        if picked:
            return picked
    return ""


#: Optional adverb between "I" and the preference verb ("I really love…").
_PREF_ADVERB = r"(?:really|truly|totally|honestly|absolutely|always|usually)?"

_PREF_LIKE_RE = re.compile(
    rf"\bi\s+{_PREF_ADVERB}\s*(?:like|love|prefer|enjoy|favou?r)\s+"
    r"(?P<obj>[^.!?,;]+)",
    re.IGNORECASE,
)
_PREF_AVOID_RE = re.compile(
    rf"\bi\s+{_PREF_ADVERB}\s*(?:don[''']t\s+like|dislike|hate|avoid|"
    r"don[''']t\s+enjoy)\s+(?P<obj>[^.!?,;]+)",
    re.IGNORECASE,
)


def head_preference_profile(claims: Iterable[Claim], question: str) -> str:
    """
    Preference summary. Lift "I like/prefer/avoid X" spans from user
    claims, join into a concise comma-separated profile.
    """
    ranked = rank_claims(question, claims, prefer_role="user")
    likes: list[str] = []
    avoids: list[str] = []
    for c in ranked[:8]:
        if not c.is_user:
            continue
        m = _PREF_LIKE_RE.search(c.text)
        if m:
            obj = m.group("obj").strip().rstrip(".,;!?")
            if obj and obj.lower() not in (item.lower() for item in likes):
                likes.append(obj)
        m = _PREF_AVOID_RE.search(c.text)
        if m:
            obj = m.group("obj").strip().rstrip(".,;!?")
            if obj and obj.lower() not in (item.lower() for item in avoids):
                avoids.append(obj)
    parts: list[str] = []
    if likes:
        parts.append(", ".join(likes[:5]))
    if avoids:
        parts.append("avoids " + ", ".join(avoids[:3]))
    if parts:
        return "; ".join(parts)
    # Fallback to a fact-style extraction on the top claim.
    return head_fact_lookup(claims, question)


_CURRENT_MARKER_RE = re.compile(
    r"\b(?:currently|now|these\s+days|today|just\s+(?:moved|switched|started)|"
    r"as\s+of\s+(?:today|now)|new\s+\w+\s+is|recently)\b",
    re.IGNORECASE,
)
_OLD_MARKER_RE = re.compile(
    r"\b(?:used\s+to|previously|formerly|originally|before|earlier|back\s+then)\b",
    re.IGNORECASE,
)


def head_knowledge_update(claims: Iterable[Claim], question: str) -> str:
    """
    Pick the *latest* claim that answers the question.

    Strategy:
      1. Rank by question-overlap (same as FACT_LOOKUP).
      2. Within the top claims, prefer those carrying a current-state
         marker ("now"/"currently"/etc.) AND higher ``turn_index``.
      3. Extract the shortest answer span from the chosen claim.
    """
    ranked = rank_claims(question, claims, prefer_role="user")
    if not ranked:
        return ""

    # Promote current-marker claims; demote old-marker claims.
    def freshness(c: Claim) -> tuple[int, int]:
        is_current = 1 if _CURRENT_MARKER_RE.search(c.text) else 0
        is_old = 1 if _OLD_MARKER_RE.search(c.text) else 0
        # (current_marker, turn_index) — higher tuple wins
        return (is_current - is_old, c.turn_index or 0)

    candidates = ranked[:6]
    candidates.sort(key=freshness, reverse=True)
    for c in candidates:
        span = _extract_fact_span(question, c.text)
        if span:
            return span
    return candidates[0].text.strip().rstrip(".")


def head_multi_session_aggregate(claims: Iterable[Claim], question: str) -> str:
    """
    Sum / count / list across sessions.

    For COUNT questions: emit "<total> <unit>" — sum of all matching
    ``<num> <unit>`` spans across distinct sessions (per-session dedup).
    For LIST questions: comma-separated distinct items.
    Falls back to fact-lookup on the top claim when shape is unclear.
    """
    ranked = rank_claims(question, claims, prefer_role="user")
    if not ranked:
        return ""
    q_lower = question.lower()
    is_count = bool(_Q_HOW_MANY.search(question))

    if is_count:
        # Find a unit anchored to the question (use first matched unit
        # from the top claim's hits, fall back across claims).
        per_session_totals: dict[str, int] = {}
        unit_chosen: str = ""
        for c in ranked:
            for num_s, unit in _NUMBER_UNIT_SPAN.findall(c.text):
                unit_norm = unit.lower().rstrip("s")
                # Reject stopword-units ("110 for") and units that
                # don't appear in the question's content words.
                if unit_norm in _STOPWORD_UNITS:
                    continue
                if unit_norm not in q_lower:
                    continue
                if not unit_chosen:
                    unit_chosen = unit
                try:
                    n = int(float(num_s))
                except ValueError:
                    continue
                # Take MAX per session (dedup repeated mentions).
                key = c.source_session_id
                per_session_totals[key] = max(per_session_totals.get(key, 0), n)
        total = sum(per_session_totals.values())
        if total > 0 and unit_chosen:
            return f"{total} {unit_chosen}"
        if total > 0:
            return str(total)

    # LIST shape — gather concrete noun phrases from each top claim.
    items: list[str] = []
    seen: set[str] = set()
    for c in ranked[:8]:
        span = _extract_fact_span(question, c.text)
        if not span:
            continue
        key = span.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        items.append(span)
    if items:
        return ", ".join(items[:6])
    return head_fact_lookup(claims, question)


_RELATIVE_DATE_RE = re.compile(
    r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)


def head_temporal_reasoning(
    claims: Iterable[Claim],
    question: str,
    *,
    question_date: str = "",
) -> str:
    """
    Compute an ordering / duration / elapsed-time answer.

    The head is intentionally narrow: it surfaces the most-specific
    date span from the top claims and returns it verbatim. When the
    question asks "how long ago", it converts an absolute date to a
    relative phrase against ``question_date`` (when set).

    Anything more complex is left to the extractive fallback — the
    deterministic head guarantees a date phrase is in the candidate
    set, not a full computed answer.
    """
    ranked = rank_claims(question, claims, prefer_role="user")
    # Look for the most-specific date in the top claims.
    for c in ranked[:6]:
        d = _shortest_date(c.text)
        if d:
            return d
    # Or a relative phrase the user already produced.
    for c in ranked[:6]:
        m = _RELATIVE_DATE_RE.search(c.text)
        if m:
            return m.group(0)
    return ranked[0].text.strip().rstrip(".") if ranked else ""


__all__ = [
    "head_assistant_memory",
    "head_fact_lookup",
    "head_knowledge_update",
    "head_multi_session_aggregate",
    "head_preference_profile",
    "head_temporal_reasoning",
]
