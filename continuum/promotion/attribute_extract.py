"""
continuum.promotion.attribute_extract
=====================================
Best-effort, deterministic extraction of *what a fact is about* — its attribute
— so :meth:`continuum.Memory.current` works without the caller tagging every
write by hand.

``current(subject, attribute)`` is only exact for facts stored with an
``attribute`` (else it falls back to fuzzy retrieval, which ranks a stale value
above its correction). Tagging on write is the reliable path; doing it
automatically removes the burden.

Deliberately **conservative**. A *wrong* tag is worse than none — it makes
``current`` answer with a fact about something else — so this only fires on
high-confidence, canonical patterns and returns ``None`` otherwise:

    "I live in Boston"           -> residence
    "I moved to NYC"             -> residence
    "I work at Stripe"           -> employer
    "My name is Mayank"          -> name
    "I drive a red Tesla"        -> vehicle
    "My favorite editor is Vim"  -> favorite editor   (phrase, lower value)

No LLM, no network. Off by default in ``Memory.add`` (``auto_attribute=…``); the
MCP ``remember`` tool turns it on, since it is fed whole spoken sentences.
"""

from __future__ import annotations

import re

__all__ = ["extract_attribute"]

# Canonical attributes. Precision over recall: where the object is a proper noun
# (a place, a company, a vehicle brand) the pattern REQUIRES a capitalised token,
# which is what separates "I live in Boston" (residence) from "I live in fear"
# and "I drive a Tesla" from "I drive a hard bargain". Case-sensitive on purpose.
_CANONICAL: tuple[tuple[str, re.Pattern[str]], ...] = (
    # place after live/reside/moved must be capitalised
    ("residence", re.compile(r"\bI\s+(?:live|reside)\s+in\s+(?=[A-Z])")),
    ("residence", re.compile(r"\bI\s+(?:moved|relocated)\s+to\s+(?=[A-Z])")),
    ("residence", re.compile(r"\bI\s+(?:moved|relocated)\s+from\s+\w+\s+to\s+(?=[A-Z])")),
    # employer: "work at/for" (kept broad — low harm), and the specific
    # "left X and joined Y". Bare "joined" was dropped ("I joined the tables").
    ("employer", re.compile(r"\bI\s+work\s+(?:at|for)\s+(?=[A-Z])")),
    # Company names are usually more than one word ("I left Nimbus Data and
    # joined Stripe"), and a single \w+ silently missed every one of them.
    ("employer", re.compile(r"\bI\s+left\s+[\w\s]{1,40}?\s+and\s+joined\s+(?=[A-Z])")),
    ("employer", re.compile(r"\bI\s+joined\s+(?=[A-Z])")),  # "I joined Infosys"
    ("name", re.compile(r"\b(?:my\s+name\s+is|I\s+am\s+called|call\s+me)\s+(?=[A-Z])", re.I)),
    # vehicle: a brand (capital) or an explicit vehicle noun in the object.
    (
        "vehicle",
        re.compile(
            r"\bI\s+drive\s+(?:a|an|the)\s+[\w\s]*?"
            r"(?:[A-Z]|\b(?:car|truck|suv|van|bike|motorcycle|scooter|sedan|jeep)\b)"
        ),
    ),
)

# Generic "My <attribute> is/are <value>". Lower value (caller must know the
# phrase) but safe. The attribute is the phrase between "my" and the copula.
_POSSESSIVE = re.compile(
    r"^\s*my\s+(?P<attr>[a-z][a-z '\-]{1,40}?)\s+(?:is|are|was|were)\s+(?P<val>.+)$",
    re.I,
)

# Possessive heads that name an opinion/mental state, not a stable attribute.
_POSSESSIVE_STOP = {
    "opinion",
    "guess",
    "point",
    "concern",
    "worry",
    "question",
    "problem",
    "understanding",
    "take",
    "sense",
    "feeling",
    "thought",
    "view",
    "hope",
    "impression",
    "assumption",
    "belief",
    "hunch",
    "bet",
    "prediction",
    "expectation",
    "plan",
    "read",
    "instinct",
    "suspicion",
    "theory",
}

# A value that begins like a clause ("... is we should ...", "... is that it ships")
# is an opinion/statement, not an attribute value.
_CLAUSE_VALUE = re.compile(r"^(?:that|we|i|it|he|she|they|you|there)\b", re.I)


def extract_attribute(text: str) -> str | None:
    """A canonical attribute for *text*, or ``None`` when nothing is confident.

    Canonical patterns win over the generic possessive so "I live in X" is
    ``residence`` rather than being missed.
    """
    stripped = text.strip()
    if not stripped:
        return None

    for attribute, pattern in _CANONICAL:
        if pattern.search(stripped):
            return attribute

    m = _POSSESSIVE.match(stripped)
    if m and not _CLAUSE_VALUE.match(m.group("val").strip()):
        attr = re.sub(r"\s+", " ", m.group("attr").strip().lower())
        head = attr.split()[-1]
        if head not in _POSSESSIVE_STOP and len(attr) >= 3:
            return attr

    return None
