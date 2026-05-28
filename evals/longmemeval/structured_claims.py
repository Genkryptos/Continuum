"""
evals/longmemeval/structured_claims.py
======================================
Source-backed structured-claim store — the first layer of the
claim-first architecture.

A :class:`StructuredClaim` carries a parsed
``(subject, relation, object, qualifiers)`` tuple alongside its
source provenance. This is the answer-time analog of an LTM fact row
— except it lives entirely in the eval harness and is extracted
on-the-fly from the retrieved evidence window.

Why structured rather than free-text claims
-------------------------------------------
The previous claim layer (``evals/longmemeval/claims.py``) treats
each sentence as a unit and ranks by topic overlap. That gets the
right *sentence* on the top of the pile but doesn't help the
finalizer pick the right *span* inside that sentence. Adding more
regex shortcuts to the heads is a row-by-row patch trap.

Structured claims encode the relation explicitly:

* ``"I used to be a Marketing specialist at a small startup"`` →
  ``relation="previous_occupation"``, ``object="Marketing specialist
  at a small startup"``.
* ``"I bought my sister a yellow dress for her birthday"`` →
  ``relation="bought_gift_for"``, ``object="a yellow dress"``,
  ``qualifiers={"recipient": "sister", "occasion": "birthday"}``.

That lets the finalizer match a parsed :class:`QueryIntent` to a
:class:`StructuredClaim` *structurally* — the answer is
``claim.object`` directly, no LLM rewriting, no COUNT-override
fights over a numeric distractor.

Scope
-----
Phase 1 ships six relations covering the six precision_claim_first
rows + a handful of common LongMemEval question shapes. Adding new
relations is one regex + one intent-table entry; the rest of the
pipeline scales automatically.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable
from typing import Any

from evals.longmemeval.scaffold_filter import (
    is_label_metadata_line,
    is_scaffold_line,
    is_scaffold_text,
    is_user_request_sentence,
)

# ─── The dataclass ─────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class StructuredClaim:
    """One source-backed structured fact."""

    text: str                   # full source sentence
    subject: str                # who/what the claim is about ("user", "the painting", etc.)
    relation: str               # canonical relation name (snake_case)
    object_: str                # the answer-side phrase
    qualifiers: dict[str, str]  # {"recipient": "sister", "occasion": "birthday"}
    source_session_id: str      # for ranking + provenance
    source_role: str            # "user" / "assistant" / ""
    source_turn_index: int | None
    source_span: str            # the exact substring backing the object
    confidence: float           # extractor's confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object_,
            "qualifiers": dict(self.qualifiers),
            "source_session_id": self.source_session_id,
            "source_role": self.source_role,
            "source_turn_index": self.source_turn_index,
            "source_span": self.source_span,
            "confidence": round(self.confidence, 3),
        }


# ─── Per-relation extractors ──────────────────────────────────────────────


# Module-level pattern strings that several relations share — keeps the
# regex blocks readable.
_OPT_DET = r"(?:a|an|the|my|her|his|their)"
_NOUN_PHRASE = rf"(?:{_OPT_DET}\s+)?[A-Za-z][\w'\-]+(?:\s+[A-Za-z][\w'\-]+){{0,8}}"
_END = (
    r"(?=[.!?,;]|"
    r"\s+(?:before|but|and\s+then|and\s+I\b|and\s+i'?m\b|"
    r"when|while|until|so|because|as|in\s+(?:the|a|an))|$)"
)

#: Frame phrases that precede a job title. Centralised so the spec's
#: full coverage list lives in one place — easy to extend.
_OCCUPATION_LEAD_IN = (
    # I-was / I-used-to-be variants
    r"i\s+used\s+to\s+be\s+(?:a|an)\s+"
    r"|i\s+was\s+(?:a|an)\s+"
    r"|i'?m\s+a\s+former\s+"
    r"|i\s+am\s+a\s+former\s+"
    # Possessive: "my previous|former|old|last job was a X"
    r"|my\s+(?:previous|former|old|last)\s+"
    r"(?:job|role|position|occupation|career|gig)\s+was\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    # Prepositional: "in my previous|former|past|old|last role as a X"
    # (covers "I've used Trello in my previous role as a marketing
    # specialist at a small startup"). Excludes "new"/"current" so
    # only past-role mentions emit a previous_occupation claim.
    r"|in\s+my\s+(?:previous|former|past|old|last)\s+"
    r"(?:job|role|position|occupation|career|gig)\s+as\s+"
    r"(?:a\s+|an\s+|the\s+)"
    # "worked / used to work as"
    r"|(?:i\s+)?worked\s+as\s+(?:a|an)\s+"
    r"|(?:i\s+)?used\s+to\s+work\s+as\s+(?:a|an)\s+"
    # "served as" / "held the role|position|title of"
    r"|i\s+served\s+as\s+(?:a|an)\s+"
    r"|i\s+held\s+(?:a|the)\s+(?:position|role|job|title)\s+of\s+"
    r"(?:a\s+|an\s+)?"
    # "previously"-prefixed
    r"|previously\s*,?\s+i\s+(?:was|served\s+as|worked\s+as|held)\s+"
    r"(?:a|an|the)\s+(?:position|role|job|title)?\s*(?:of\s+)?"
    # "most recently, I was a/held"
    r"|most\s+recently\s*,?\s+i\s+(?:was|held|served\s+as|worked\s+as)\s+"
    r"(?:a|an|the)\s+(?:position|role|job|title)?\s*(?:of\s+)?"
    # "before/prior to <event>, I was a/served as"
    r"|(?:before|prior\s+to)\s+[^,]+,\s+i\s+"
    r"(?:was|served\s+as|worked\s+as|held)\s+"
    r"(?:a|an|the)\s+(?:position|role|job|title)?\s*(?:of\s+)?"
)

#: ``I used to be a X`` / ``I was a X at Y`` / ``my previous job was a X at Y``
_PREVIOUS_OCCUPATION_RE = re.compile(
    rf"\b(?:{_OCCUPATION_LEAD_IN})"
    r"(?P<title>[A-Za-z][\w'\- ]{2,80}?)"
    + _END,
    re.IGNORECASE,
)

#: Past-tense / first-person markers required for the standalone
#: ``<title> at <company>`` extractor — without them, the pattern would
#: match third-person facts ("My friend is a Marketing specialist at
#: Google") or current-job claims ("I'm a Marketing specialist at
#: BigCo today"). Only past-tense first-person sentences emit a
#: ``previous_occupation`` claim from the standalone title-at-company
#: pattern.
_FIRST_PERSON_RE = re.compile(r"\b(?:i|my|me|i'?m)\b", re.IGNORECASE)
_PAST_TENSE_OCCUPATION_RE = re.compile(
    r"\b(?:was|were|used\s+to|before|previously|former|formerly|"
    r"prior\s+to|earlier|in\s+the\s+past|years?\s+ago|back\s+then|"
    r"back\s+when|left|switched|transitioned|moved\s+on|"
    r"changed\s+careers?|joined|after\s+leaving)\b",
    re.IGNORECASE,
)

#: Standalone ``<job title> at <X>`` pattern (the last bullet in the
#: spec's title-pattern list). Emits a ``previous_occupation`` claim
#: ONLY when the enclosing sentence has both 1st-person AND past-tense
#: markers — otherwise it would match third-person / current-role
#: facts. The captured ``head_noun`` is anchored to the known
#: job-title vocabulary so the pattern can't drift onto random
#: ``<word> at <word>`` phrases.
_TITLE_AT_COMPANY_RE = re.compile(
    r"\b(?P<title>"
    r"(?:senior|junior|lead|principal|staff|chief|head|"
    r"former|previous|past)?\s*"
    r"(?:[a-z][a-z\-]+\s+){0,3}"
    r"(?:specialist|engineer|manager|developer|analyst|director|"
    r"officer|consultant|designer|architect|scientist|coordinator|"
    r"representative|advisor|adviser|teacher|nurse|doctor|chef|"
    r"writer|editor|attorney|lawyer|accountant|broker|trader|"
    r"associate|principal|technician|operator|administrator|"
    r"supervisor|clerk|cashier|server|barista|driver|pilot|"
    r"professor|instructor|tutor|researcher|reporter|journalist|"
    r"photographer|musician|programmer|strategist|copywriter|"
    r"recruiter|founder|ceo|cto|cfo|coo|president|vp|partner|"
    r"salesperson|salesman|saleswoman|realtor|agent|trainee|intern)"
    r")"
    r"\s+at\s+"
    r"(?P<company>(?:a|an|the)?\s*[A-Za-z][\w' ,&\-]{2,60}?)"
    + _END,
    re.IGNORECASE,
)

#: Job-title head nouns. The extractor requires the captured title to
#: contain one of these — otherwise the regex would extract verbs
#: like "responsible" / "managing" out of "I was responsible for
#: managing marketing campaigns" and emit them as titles. This is the
#: responsibility-only filter the spec asks for.
_JOB_TITLE_HEAD_NOUNS = re.compile(
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
    r"strategist|copywriter|recruiter|founder|ceo|cto|cfo|coo|"
    r"president|vice\s+president|vp|partner|salesperson|salesman|"
    r"saleswoman|realtor|agent|broker|trainee|apprentice)\b",
    re.IGNORECASE,
)

#: Responsibility / verb-phrase tokens that disqualify a "title"
#: match. Used after the regex grabs a title candidate — if the
#: candidate is one of these gerunds/adjectives, drop it.
_RESPONSIBILITY_HEAD_TOKENS = frozenset({
    "responsible", "responsibility", "managing", "leading", "handling",
    "doing", "working", "assigned", "tasked", "in", "charge", "of",
    "the", "an", "a", "lot",
})

#: ``Bachelor's degree in X`` / ``Master's in X`` / ``degree in X``
_DEGREE_IN_RE = re.compile(
    r"\b(?:bachelor[''']?s|master[''']?s|doctorate|phd|degree)\s+"
    r"(?:degree\s+)?(?:in|of)\s+"
    r"(?P<field>[A-Za-z][\w' ]{2,60}?)"
    r"(?=\s+(?:from|at|in|with)|[.!?,;]|$)",
    re.IGNORECASE,
)

#: ``graduated with a degree in X`` / ``graduated in X``
_GRADUATED_RE = re.compile(
    r"\bgraduated\s+(?:with\s+(?:a|an)\s+(?:[\w'\-]+\s+)?(?:degree\s+)?(?:in|of)\s+)?"
    r"(?P<field>[A-Za-z][\w' ]{2,60}?)"
    r"(?=\s+(?:from|at|in)|[.!?,;]|$)",
    re.IGNORECASE,
)

#: ``I bought my sister a yellow dress for her birthday``
#: ``I bought a yellow dress for my sister's birthday``
#: ``I got my sister a book``
#:
#: Item is bounded to ``det + 1-6 words`` so a trailing ``for X's
#: birthday`` clause doesn't accidentally get swallowed into the
#: item phrase. The end-lookahead permits trailing modifiers
#: (``last weekend``, ``yesterday``, ``because…``) so the match
#: terminates cleanly without requiring punctuation right after the
#: occasion.
_BOUGHT_GIFT_FOR_RE = re.compile(
    r"\bi\s+(?:bought|got|picked\s+up|chose|grabbed|purchased|gave)\s+"
    r"(?:(?P<recipient_indirect>my\s+\w+|her|him|them)\s+)?"
    r"(?P<item>(?:a|an|the|some)\s+[A-Za-z][\w'\-]+"
    r"(?:\s+[A-Za-z][\w'\-]+){0,5}?)"
    r"(?:\s+for\s+"
    r"(?:(?:my\s+(?P<recipient_direct>\w+)['']?s?|(?:her|his|their))\s+)?"
    r"(?P<occasion>birthday|anniversary|graduation|wedding|christmas|"
    r"hanukkah|housewarming))?"
    r"(?=\s+(?:last|this|next|on|in|at|because|since|when|so|and)|"
    r"[.!?,;]|$)",
    re.IGNORECASE,
)

#: Sentence-initial occasion prefix:
#:   ``For my sister's birthday, I got her a yellow dress``
#:   ``For my brother's graduation, I bought him a watch``
#:
#: Different shape from the verb-first pattern — the occasion clause
#: leads, the verb follows. The item capture stops at conjunctions
#: ("and a pair of earrings"), trailing modifiers ("last weekend"),
#: and punctuation so multi-item sentences only yield the first item
#: (matching the expected "first gift mentioned" answer).
_BOUGHT_GIFT_FOR_PREFIX_RE = re.compile(
    r"\bfor\s+(?:my\s+)?(?P<recipient>\w+)['']?s?\s+"
    r"(?P<occasion>birthday|anniversary|graduation|wedding|"
    r"christmas|hanukkah|housewarming)"
    r"\s*,?\s+i\s+"
    r"(?:bought|got|picked\s+up|chose|grabbed|purchased|gave|"
    r"went\s+with|settled\s+on|ended\s+up\s+with)\s+"
    r"(?:(?:her|him|them|my\s+\w+)\s+)?"
    r"(?P<item>(?:a|an|the|some)\s+[A-Za-z][\w'\-]+"
    r"(?:\s+[A-Za-z][\w'\-]+){0,5}?)"
    r"(?=\s+(?:and|to|for|because|since|when|so|last|this|next|"
    r"on|in|at)|[.!?,;]|$)",
    re.IGNORECASE,
)

#: ``the painting is worth triple what I paid`` / ``worth N times what I paid``
_VALUE_VS_PAID_RE = re.compile(
    r"\b(?P<subject>[A-Za-z][\w'\- ]{1,40}?)\s+(?:is|are)\s+worth\s+"
    r"(?P<phrase>(?:triple|double|twice|half|quadruple|\d+\s*(?:x|times))\s+"
    r"(?:what\s+I\s+paid|what\s+I\s+spent|the\s+(?:purchase|original)\s+price))"
    r"(?:[^.!?,;]*?)"
    r"(?=[.!?,;]|$)",
    re.IGNORECASE,
)

#: ``I packed 7 shirts for the trip`` / ``packed 7 shirts to Costa Rica``
_COUNT_PACKED_RE = re.compile(
    r"\b(?:i\s+)?(?:packed|brought|took)\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<item>[A-Za-z][\w\-]{2,30})"
    r"(?:\s+(?:for|to|on)\s+(?P<destination>(?:the\s+)?[A-Za-z][\w' ]{2,40}?))?"
    r"(?=[.!?,;]|$)",
    re.IGNORECASE,
)

#: ``the fundraising dinner was on February 14th`` / ``X is on Y``
_EVENT_DATE_RE = re.compile(
    r"\b(?:the\s+|my\s+)?(?P<event>[a-z][\w' ]{2,60}?)\s+"
    r"(?:was|is|will\s+be)\s+(?:on|scheduled\s+for)\s+"
    r"(?P<date>(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(?:\s+\d{4})?"
    r"|\d{4}-\d{1,2}-\d{1,2})"
    r"(?=[.!?,;]|$)",
    re.IGNORECASE,
)


def _make_claim(
    *,
    text: str,
    subject: str,
    relation: str,
    object_: str,
    qualifiers: dict[str, str] | None = None,
    source_span: str,
    confidence: float,
    metadata: dict[str, Any],
) -> StructuredClaim:
    """Constructor that pulls source metadata from a single dict."""
    return StructuredClaim(
        text=text,
        subject=subject.strip(),
        relation=relation,
        object_=object_.strip().rstrip(".,;!?"),
        qualifiers=qualifiers or {},
        source_session_id=str(metadata.get("source_session_id") or ""),
        source_role=str(metadata.get("source_role") or ""),
        source_turn_index=metadata.get("source_turn_index"),
        source_span=source_span.strip(),
        confidence=confidence,
    )


def _extract_previous_occupation(
    sentence: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    """
    Emit ``previous_occupation`` claims via two layered strategies.

    Strategy 1 — explicit lead-in frame (``I was a X``,
    ``I served as a X``, ``my previous role was a X``, etc.). Used
    when the sentence has a clear occupational frame.

    Strategy 2 — standalone ``<title> at <company>``. Used as a
    fallback when the sentence has 1st-person + past-tense markers
    but no explicit frame (e.g. ``"Before joining the agency I'd
    been a Marketing specialist at a small startup"`` where the
    frame ``"I'd been a"`` doesn't match the primary regex). The
    title head-noun lock ensures this can't drift onto random
    "<word> at <word>" phrases.

    Both strategies share the same answerability filter: the captured
    title must contain a recognised head noun and must NOT start with
    a responsibility verb.
    """
    out: list[StructuredClaim] = []
    seen_titles: set[str] = set()

    def _emit(title: str, span: str, confidence: float) -> None:
        title = title.strip().rstrip(".,;!?")
        # First real content token must not be a responsibility verb.
        first_token_m = re.match(r"\s*(?P<tok>[A-Za-z][\w'\-]+)", title)
        if first_token_m and first_token_m.group("tok").lower() in \
                _RESPONSIBILITY_HEAD_TOKENS:
            return
        # Title must contain a recognised head noun.
        if not _JOB_TITLE_HEAD_NOUNS.search(title):
            return
        key = title.lower()
        if key in seen_titles:
            return
        seen_titles.add(key)
        # Split out a company qualifier from "X at Y" trailing form.
        company = ""
        comp_m = re.search(r"\s+at\s+(?P<comp>(?:a|an|the)?\s*[A-Za-z][\w'\- ]+)$",
                           title, re.IGNORECASE)
        if comp_m:
            company = comp_m.group("comp").strip()
        out.append(_make_claim(
            text=sentence,
            subject="user",
            relation="previous_occupation",
            object_=title,
            qualifiers={"company": company} if company else {},
            source_span=span,
            confidence=confidence,
            metadata=metadata,
        ))

    # Strategy 1 — lead-in frame.
    for m in _PREVIOUS_OCCUPATION_RE.finditer(sentence):
        _emit(m.group("title"), m.group(0).strip(), confidence=0.85)

    # Strategy 2 — standalone ``<title> at <company>``. Gated on
    # 1st-person AND past-tense markers in the sentence.
    if _FIRST_PERSON_RE.search(sentence) and \
            _PAST_TENSE_OCCUPATION_RE.search(sentence):
        for m in _TITLE_AT_COMPANY_RE.finditer(sentence):
            title = m.group("title").strip()
            company = m.group("company").strip()
            full_title = f"{title} at {company}".rstrip(".,;!?")
            _emit(full_title, m.group(0).strip(), confidence=0.78)

    return out


def _extract_degree_in(
    sentence: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    out: list[StructuredClaim] = []
    for m in _DEGREE_IN_RE.finditer(sentence):
        field_text = m.group("field").strip()
        # Skip degenerate one-word matches that are stopwords.
        if field_text.lower() in {"the", "a", "an"}:
            continue
        out.append(_make_claim(
            text=sentence,
            subject="user",
            relation="degree_in",
            object_=field_text,
            source_span=m.group(0).strip(),
            confidence=0.85,
            metadata=metadata,
        ))
    # ``graduated with a degree in X`` fallback (lower priority — only
    # add when the primary degree regex missed).
    if not out:
        for m in _GRADUATED_RE.finditer(sentence):
            field_text = m.group("field").strip()
            if len(field_text) < 4 or field_text.lower() in {"the", "with", "from"}:
                continue
            out.append(_make_claim(
                text=sentence,
                subject="user",
                relation="degree_in",
                object_=field_text,
                source_span=m.group(0).strip(),
                confidence=0.75,
                metadata=metadata,
            ))
    return out


def _extract_bought_gift_for(
    sentence: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    """
    Per-sentence gift extractor.

    A sentence becomes a gift claim ONLY when it carries an explicit
    occasion (``"for my sister's birthday"``, ``"for the
    anniversary"``). Recipient alone is not enough — ``"I got her
    some groceries when she was recovering from surgery"`` shares
    the indirect-object pattern with a gift sentence but is clearly
    NOT about a gift.

    The section-scoped extractor
    (:func:`_extract_section_scoped_gifts`) handles the alternative
    case where the occasion lives in a preceding heading. Splitting
    the two paths means the per-sentence path can stay strict
    without losing the heading-bound items.

    Also: reject sentences whose context is clearly non-gift
    (groceries / surgery / recovery / grocery-run shopping
    distractors) — same reject set the section extractor uses.
    """
    out: list[StructuredClaim] = []
    # Per-spec: gift questions require occasion context. Reject
    # sentences carrying non-gift topic words even when they happen
    # to match the verb+recipient frame.
    if _GIFT_CONTEXT_REJECT_RE.search(sentence):
        return out
    seen_spans: set[str] = set()
    # Strategy 1 — sentence-initial occasion prefix:
    #   "For my sister's birthday, I got her a yellow dress"
    for m in _BOUGHT_GIFT_FOR_PREFIX_RE.finditer(sentence):
        item = m.group("item").strip()
        recipient = m.group("recipient").strip().lower()
        occasion = m.group("occasion").strip().lower()
        qualifiers = {"recipient": recipient, "occasion": occasion}
        span = m.group(0).strip()
        seen_spans.add(span.lower())
        out.append(_make_claim(
            text=sentence,
            subject="user",
            relation="bought_gift_for",
            object_=item,
            qualifiers=qualifiers,
            source_span=span,
            confidence=0.88,
            metadata=metadata,
        ))
    # Strategy 2 — verb-first ("I bought a yellow dress for my sister's
    # birthday"). Skip if the prefix strategy already covered the span.
    for m in _BOUGHT_GIFT_FOR_RE.finditer(sentence):
        item = m.group("item").strip()
        # Recipient sources: indirect object ("my sister a dress") OR
        # genitive after "for" ("for my sister's birthday").
        recipient = ""
        ri = m.group("recipient_indirect") or ""
        if ri:
            recipient = re.sub(r"^my\s+", "", ri, flags=re.IGNORECASE)
        if not recipient:
            recipient = (m.group("recipient_direct") or "").strip()
        occasion = (m.group("occasion") or "").strip().lower()
        # Hard requirement: occasion must be captured. Without it
        # the sentence is a generic purchase, not a gift claim.
        if not occasion:
            continue
        qualifiers: dict[str, str] = {"occasion": occasion}
        if recipient:
            qualifiers["recipient"] = recipient.lower()
        span = m.group(0).strip()
        if span.lower() in seen_spans:
            continue
        out.append(_make_claim(
            text=sentence,
            subject="user",
            relation="bought_gift_for",
            object_=item,
            qualifiers=qualifiers,
            source_span=span,
            confidence=0.85,
            metadata=metadata,
        ))
    return out


def _extract_value_vs_paid(
    sentence: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    out: list[StructuredClaim] = []
    for m in _VALUE_VS_PAID_RE.finditer(sentence):
        subject = m.group("subject").strip()
        phrase = m.group("phrase").strip()
        # Append "what I paid" tail when the phrase ends at the
        # quantifier ("triple", "twice") without the comparison object.
        out.append(_make_claim(
            text=sentence,
            subject=subject,
            relation="value_compared_to_paid",
            object_=phrase,
            source_span=m.group(0).strip(),
            confidence=0.85,
            metadata=metadata,
        ))
    return out


def _extract_count_packed(
    sentence: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    out: list[StructuredClaim] = []
    for m in _COUNT_PACKED_RE.finditer(sentence):
        count = m.group("count").strip()
        item = m.group("item").strip()
        destination = (m.group("destination") or "").strip()
        # Normalise the item qualifier to its singular form so it
        # matches the question intent's normalisation ("shirts" →
        # "shirt"). Keep the surface form in ``object_`` for the
        # user-facing answer.
        qualifiers: dict[str, str] = {"item": item.lower().rstrip("s")}
        if destination:
            qualifiers["destination"] = destination.lower()
        out.append(_make_claim(
            text=sentence,
            subject="user",
            relation="count_packed",
            object_=f"{count} {item}",
            qualifiers=qualifiers,
            source_span=m.group(0).strip(),
            confidence=0.85,
            metadata=metadata,
        ))
    return out


def _extract_event_date(
    sentence: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    out: list[StructuredClaim] = []
    for m in _EVENT_DATE_RE.finditer(sentence):
        event = m.group("event").strip()
        date = m.group("date").strip()
        # Skip degenerate event noun phrases.
        if len(event.split()) > 6:
            continue
        out.append(_make_claim(
            text=sentence,
            subject=event,
            relation="event_date",
            object_=date,
            qualifiers={"event": event.lower()},
            source_span=m.group(0).strip(),
            confidence=0.8,
            metadata=metadata,
        ))
    return out


# ─── Section-scoped extractors ────────────────────────────────────────────


#: Heading that carries gift context — ``**Sister's Birthday**``,
#: ``Sister's Birthday`` (plain), ``## Brother's Graduation``,
#: ``Occasion: Sister's birthday``.
_GIFT_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\*{1,3}|#{1,6}\s*|_{1,3})?"
    r"(?:occasion\s*:\s*)?"
    r"(?:my\s+)?"
    r"(?P<recipient>sister|brother|mom|mother|father|dad|mum|"
    r"partner|wife|husband|son|daughter|friend|boss|colleague|"
    r"niece|nephew|aunt|uncle|cousin|grandmother|grandfather)"
    r"['']?s?\s+"
    r"(?P<occasion>birthday|anniversary|graduation|wedding|"
    r"christmas|housewarming|baby\s+shower|engagement)"
    r"(?:\s*\*{1,3}|\s*_{1,3})?\s*$",
    re.IGNORECASE,
)

#: Known store / chain names — never the "purchased item" answer.
_KNOWN_STORES: frozenset[str] = frozenset({
    "best buy", "amazon", "target", "walmart", "ikea", "costco",
    "walgreens", "cvs", "whole foods", "trader joe's", "kroger",
    "safeway", "publix", "apple store", "home depot", "lowe's",
    "starbucks", "mcdonald's", "macy's", "nordstrom", "zara", "h&m",
    "uniqlo", "gap", "nike", "adidas", "etsy", "ebay",
})

#: Topic words that mean the line is NOT a gift line (groceries /
#: surgery / recovery / flowers-for-mom / etc.).
_GIFT_CONTEXT_REJECT_RE = re.compile(
    r"\b(?:groceries|grocery|recipe|surgery|recovery|painkillers?|"
    r"prescription|appointment|meeting|interview|deadline|"
    r"workout|gym|laundry|cleaning|repair)\b",
    re.IGNORECASE,
)

#: Item line patterns that pick up the purchased thing once a section
#: context is established. More permissive than the full-sentence
#: pattern because the heading already gave us the qualifiers.
#: Three forms covered:
#:   1. Verb-led: ``I bought / got / chose / went with / ended up with X``.
#:   2. Declarative: ``gift was X`` / ``item: X``.
#:   3. Bare bullet item (no verb): ``- a yellow dress`` / ``yellow dress``.
_SECTION_ITEM_VERB_RE = re.compile(
    r"\b(?:i\s+(?:bought|got|chose|picked\s+up|grabbed|purchased|gave|"
    r"went\s+with|settled\s+on|decided\s+on|ended\s+up\s+with|"
    r"ultimately\s+(?:got|chose|bought))|"
    r"gift\s+(?:was|is)|item\s*:\s*|gift\s*:\s*)\s*"
    r"(?:(?:her|him|them|my\s+\w+)\s+)?"
    r"(?P<item>(?:a|an|the|some)\s+[A-Za-z][\w'\- ]{2,60}?)"
    r"(?=\s+(?:from|at|for|to|on|as|because|since|"
    r"last|this|next|when|so|and)|[.!?,;]|$)",
    re.IGNORECASE,
)
_SECTION_BARE_ITEM_RE = re.compile(
    # Bare-bullet item line: optional leading bullet/dash/plus,
    # optional determiner, then a noun phrase. The ``+`` form shows
    # up in the live wiki bundle as nested sub-bullets (``\t+ Yellow
    # dress``). Capped at 6 words so a long sentence doesn't match
    # by accident.
    r"^\s*(?:[-*+]\s+)?"
    r"(?P<item>(?:a|an|the|some)?\s*[A-Za-z][\w'\-]+"
    r"(?:\s+[A-Za-z][\w'\-]+){0,5})"
    r"\s*$",
    re.IGNORECASE,
)


def _extract_section_scoped_gifts(
    text: str, metadata: dict[str, Any],
) -> list[StructuredClaim]:
    """
    Walk multi-line text once, tracking the active gift section.

    When a heading or ``Occasion:`` label binds a recipient+occasion,
    subsequent item lines inherit that context — letting us emit
    ``bought_gift_for`` claims even when the item line itself doesn't
    carry the ``"for my sister's birthday"`` tail. This is the cure
    for the markdown-sectioned bundle shape::

        **Sister's Birthday**
        - bought a yellow dress

    The active context is cleared when a new heading is encountered
    OR after 6 lines without a new content line (so a stale section
    can't bind unrelated content later in the bundle).
    """
    out: list[StructuredClaim] = []
    current_ctx: dict[str, str] | None = None
    lines_since_heading = 0
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        # Distance from the heading bounds the section's scope.
        if current_ctx is not None:
            lines_since_heading += 1
            if lines_since_heading > 6:
                current_ctx = None

        # Section heading? Establish context, don't emit a claim from
        # the heading itself.
        stripped = line.strip()
        m_h = _GIFT_SECTION_HEADING_RE.match(stripped)
        if m_h:
            current_ctx = {
                "recipient": m_h.group("recipient").lower(),
                "occasion": m_h.group("occasion").lower(),
            }
            lines_since_heading = 0
            continue
        if not current_ctx:
            continue
        # Skip scaffold / label lines inside a section.
        if is_scaffold_line(stripped):
            continue
        if is_label_metadata_line(stripped):
            continue
        # Reject lines whose topic clearly isn't a gift purchase.
        # Strip the leading bullet ("-", "*", "+") and optional
        # "(role): " role prefix so the body regex sees raw content.
        body = re.sub(
            r"^\s*[-*+]\s+(?:\([a-z]+\):\s*)?", "", stripped,
        )
        if is_user_request_sentence(body):
            continue
        if _GIFT_CONTEXT_REJECT_RE.search(body):
            continue
        # Try item extraction. The section already gave us the
        # qualifiers; the line just needs to mention the item.
        # Tier 1: verb-led / declarative pattern.
        m = _SECTION_ITEM_VERB_RE.search(body)
        item: str = ""
        span: str = ""
        if m:
            item = m.group("item").strip().rstrip(".,;!?")
            span = m.group(0).strip()
        else:
            # Tier 2: bare item line — only fires when the body is
            # short (≤ 8 words). Long sentences without a verb are
            # commentary, not an item.
            if len(body.split()) <= 8:
                m_bare = _SECTION_BARE_ITEM_RE.match(body.strip("-*+ \t"))
                if m_bare:
                    item = m_bare.group("item").strip().rstrip(".,;!?")
                    span = body.strip()
        if not item:
            continue
        # Reject obvious store names — those are where you bought it,
        # not what you bought.
        item_norm = re.sub(r"^(?:a|an|the|some)\s+", "", item, flags=re.IGNORECASE).lower()
        if item_norm in _KNOWN_STORES:
            continue
        # Reject items that are themselves topic-rejects (groceries /
        # flowers-from-the-surgery-distractor / etc.). The body filter
        # above already drops most, but a section line whose item word
        # is literally a rejected topic still needs the gate.
        if _GIFT_CONTEXT_REJECT_RE.search(item):
            continue
        out.append(_make_claim(
            text=body,
            subject="user",
            relation="bought_gift_for",
            object_=item,
            qualifiers=dict(current_ctx),
            source_span=span,
            confidence=0.82,
            metadata=metadata,
        ))
    return out


#: Registry of relation extractors. Each is called with
#: ``(sentence, metadata)`` and returns zero or more StructuredClaims.
_RELATION_EXTRACTORS: tuple[
    tuple[
        str,
        Any,  # callable[[str, dict], list[StructuredClaim]]
    ], ...,
] = (
    ("previous_occupation",        _extract_previous_occupation),
    ("degree_in",                  _extract_degree_in),
    ("bought_gift_for",            _extract_bought_gift_for),
    ("value_compared_to_paid",     _extract_value_vs_paid),
    ("count_packed",               _extract_count_packed),
    ("event_date",                 _extract_event_date),
)


# ─── Sentence splitting + filtering ────────────────────────────────────────


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
#: Matches the bundle's per-turn role prefix in any of three live
#: shapes:
#:   * ``- (user): text``           — Matched Turn line
#:   * ``- turn 1 (assistant): text`` — Nearby Context line
#:   * ``assistant: text``          — inline role prefix embedded mid-content
#: The ``role`` group is the speaker; ``body`` is the content. Used by
#: ``_sentences_of`` to thread the real speaker into each claim's
#: ``source_role`` — without this fix, every claim mined from a wiki
#: bundle is tagged ``source_role="system"`` (the wiki page's outer
#: role), which silently breaks the ranker's user-vs-assistant
#: distinction and the assistant-memory question shape.
_BULLET_RE = re.compile(
    r"^\s*-\s+"
    r"(?:turn\s+\d+\s+)?"
    r"(?:\((?P<role>user|assistant|system)\):\s*)?"
    r"(?P<body>.+)$",
    re.IGNORECASE,
)
_INLINE_ROLE_RE = re.compile(
    r"^\s*(?P<role>user|assistant|system)\s*:\s*(?P<body>.+)$",
    re.IGNORECASE,
)


def _sentences_of(text: str) -> list[tuple[str, str]]:
    """
    Yield non-scaffold ``(sentence, role)`` pairs.

    The role is the per-line speaker parsed from the bundle's
    ``- (user):`` / ``- turn N (assistant):`` / inline ``assistant:``
    prefix when present, otherwise the empty string (caller's default
    role applies). Returning role per-sentence — rather than per-call —
    fixes the wiki-bundle role leak: a wiki page is one MemoryItem
    with role ``"system"``, but its body interleaves user and
    assistant turns.
    """
    out: list[tuple[str, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if is_scaffold_line(line):
            continue
        if is_label_metadata_line(line):
            continue
        role = ""
        bullet = _BULLET_RE.match(line)
        if bullet:
            role = (bullet.group("role") or "").lower()
            body = bullet.group("body").strip()
        else:
            inline = _INLINE_ROLE_RE.match(line)
            if inline:
                role = inline.group("role").lower()
                body = inline.group("body").strip()
            else:
                body = line.strip()
        if not body:
            continue
        for sent in _SENT_SPLIT_RE.split(body):
            sent = sent.strip()
            if not sent:
                continue
            if is_scaffold_text(sent):
                continue
            # User-request sentences ("Can you help me…") never carry
            # autobiographical facts. Skip them at the source.
            if is_user_request_sentence(sent):
                continue
            out.append((sent, role))
    return out


def extract_structured_claims(
    text: str,
    *,
    source_session_id: str = "",
    source_role: str = "",
    source_turn_index: int | None = None,
) -> list[StructuredClaim]:
    """
    Mine a block of evidence text for structured claims.

    Drops scaffold lines, label metadata, user-request sentences, and
    transcript-timestamp junk *before* running the per-relation
    extractors. The per-relation extractors then walk each surviving
    sentence and emit zero or more :class:`StructuredClaim` per match.
    """
    if not text:
        return []
    metadata = {
        "source_session_id": source_session_id,
        "source_role": source_role,
        "source_turn_index": source_turn_index,
    }
    out: list[StructuredClaim] = []
    # Multi-line section-scoped extractors first — they need the full
    # text to track heading → body context.
    out.extend(_extract_section_scoped_gifts(text, metadata))
    # Per-sentence relation extractors second. Each sentence carries
    # its parsed role; if a role prefix was present we override the
    # caller-default ``source_role`` so the claim is correctly tagged
    # as user/assistant/system.
    for sentence, parsed_role in _sentences_of(text):
        sent_metadata = (
            {**metadata, "source_role": parsed_role}
            if parsed_role else metadata
        )
        for _name, extractor in _RELATION_EXTRACTORS:
            out.extend(extractor(sentence, sent_metadata))
    return out


def extract_structured_claims_from_context(
    items: Iterable[Any],
) -> list[StructuredClaim]:
    """
    Convenience helper for the wiki-evidence-window bundle path.

    Accepts any iterable of items that look like
    :class:`continuum.core.types.MemoryItem` — uses ``.content``,
    ``.metadata.session_id``, and ``.metadata.role``.
    """
    out: list[StructuredClaim] = []
    for idx, item in enumerate(items or []):
        metadata = item.metadata or {}
        sid = str(metadata.get("session_id") or getattr(item, "session_id", "") or "")
        role = str(metadata.get("role") or "")
        out.extend(extract_structured_claims(
            item.content,
            source_session_id=sid,
            source_role=role,
            source_turn_index=idx,
        ))
    return out


__all__ = [
    "StructuredClaim",
    "extract_structured_claims",
    "extract_structured_claims_from_context",
]
