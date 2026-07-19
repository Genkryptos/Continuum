"""
continuum/promotion/local_extractor.py
======================================
A dependency-free, **$0** membership-fact extractor — the no-API alternative to
the gpt-4o-mini / gpt-5-mini synthesis extractor. Pure stdlib regex: no spaCy, no
GLiNER, no model download, no network. Runs free at write time and ships in the
product (no per-session API tax).

It produces the SAME ``{"facts": [...]}`` shape the LLM extractor does, so it
plugs straight into ``extract_structured_facts`` via a ``completion_fn``-shaped
wrapper (``local_completion_fn``) — same disk cache, same ``aggregate()``.

Design bias: **precision over recall.** The deterministic router only fires on a
confident single match and otherwise defers to the reader, so a missed fact is
cheap but a *spurious* fact causes a wrong count. We therefore emit a triple only
on clean acquisition/possession patterns and reject non-noun heads.

Honest limits (vs an LLM): no semantic normalization — "lemon" and "lime" stay
distinct (the LLM groups them as "citrus fruit"); implicit membership is missed.
Good on explicit "I bought/added/got/own a <noun>" — which is most counting.
"""

from __future__ import annotations

import json
import re

# Verbs that introduce a countable item the user acquired / owns / did.
_ACQUIRE = (
    r"bought|buy|buying|purchased|purchase|got|gotten|added|add|acquired|acquire"
    r"|adopted|adopt|received|receive|made|make|built|build|set\s+up|fixed|fix"
    r"|replaced|replace|serviced|service|installed|install|ordered|order"
    r"|attended|attend|completed|complete|collected|collect|own|owns|owned"
    r"|have|has|had|visited|visit|joined|join|signed\s+up\s+for"
)
# Singular determiners only — we want DISTINCT items, so we skip bare numbers
# ("3 tanks" is a count statement, not three listable members).
_DET = r"a|an|another|one|my|the|this|that"
# A short noun phrase: 1–5 tokens. Tokens may be alphanumeric ("1909 penny",
# "70-200mm lens"); the head is resolved to the last ALPHABETIC word below.
_NP = r"([A-Za-z0-9][\w'\-]*(?:\s+[A-Za-z0-9][\w'\-]*){0,4})"

_MEMBER_RE = re.compile(rf"\b(?:{_ACQUIRE})\s+(?:{_DET})\s+{_NP}", re.IGNORECASE)

# Measurement: a number followed by a time/money unit (for SUM aggregation).
_UNIT = r"hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|miles?|km|dollars?"
_MEAS_RE = re.compile(rf"\b(\d+(?:\.\d+)?)\s+({_UNIT})\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s+dollars?\b", re.IGNORECASE)

# Prepositions that end the head of a noun phrase ("top FROM h&m" → head "top").
_PREPS = {"from", "of", "for", "in", "at", "on", "with", "to", "about", "by"}

# Heads that are never a countable collection — reject to avoid spurious predicates.
_STOP_HEADS = {
    "it",
    "them",
    "one",
    "ones",
    "thing",
    "things",
    "lot",
    "lots",
    "bit",
    "couple",
    "few",
    "some",
    "any",
    "time",
    "times",
    "today",
    "yesterday",
    "tomorrow",
    "week",
    "weekend",
    "day",
    "morning",
    "night",
    "year",
    "month",
    "while",
    "now",
    "then",
    "here",
    "there",
    "way",
    "ways",
    "idea",
    "ideas",
    "look",
    "go",
    "plan",
    "chance",
    "moment",
    "place",
    "home",
    "house",
    "work",
    "job",
    "life",
    "fun",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
}

# Words that shouldn't appear mid-noun-phrase — trim the NP at the first one.
_TRIM_AFTER = {
    "yesterday",
    "today",
    "tomorrow",
    "when",
    "while",
    "because",
    "so",
    "since",
    "after",
    "before",
    "during",
    "last",
    "this",
    "that",
    "which",
    "who",
    "and",
    "but",
    "then",
    "now",
    "again",
    "also",
    "too",
    "very",
    "really",
    "just",
}


def _singularize(word: str) -> str:
    """Crude but safe singularizer for the predicate (collection noun)."""
    w = word.lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _head_and_object(np: str) -> tuple[str, str] | None:
    """From a captured noun phrase, return (predicate_head, object) or None.

    Object = the phrase trimmed at the first adverb/conjunction. Head = the last
    noun word, or the word before a preposition ("top from h&m" → "top").
    Returns None if the head is a stopword (not a real collection).
    """
    words = np.split()
    # Trim at the first preposition OR continuation word, so the object is the
    # core noun phrase ("gaming session for 40 hours" → "gaming session";
    # "top from h&m" → "top").
    kept: list[str] = []
    for w in words:
        if w.lower() in _TRIM_AFTER or w.lower() in _PREPS:
            break
        kept.append(w)
    if not kept:
        return None
    obj = " ".join(kept)
    # Head = last ALPHABETIC token (skip a trailing number/measurement token).
    head_idx = len(kept) - 1
    while head_idx >= 0 and not re.search(r"[A-Za-z]{2,}", kept[head_idx]):
        head_idx -= 1
    if head_idx < 0:
        return None
    head = _singularize(re.sub(r"[^A-Za-z'-]", "", kept[head_idx]))
    if head in _STOP_HEADS or len(head) < 3:
        return None
    return head, obj.lower()


def extract_facts_local(text: str) -> list[dict[str, object]]:
    """Rule-based membership + measurement extraction → fact dicts.

    Same shape as the LLM extractor's ``facts`` array, so downstream
    ``extract_structured_facts`` / ``aggregate`` consume it unchanged.
    """
    if not (text or "").strip():
        return []
    facts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for m in _MEMBER_RE.finditer(text):
        ho = _head_and_object(m.group(1))
        if ho is None:
            continue
        head, obj = ho
        key = (head, obj)
        if key in seen:
            continue
        seen.add(key)
        fact: dict[str, object] = {"predicate": head, "object": obj}
        # Attach a measurement from the matched phrase + a short tail (best
        # effort) — covers "a gaming session for 40 hours".
        span = m.group(1) + " " + text[m.end() : m.end() + 40]
        meas = _MEAS_RE.search(span)
        if meas:
            fact["quantity"] = meas.group(1)
            fact["unit"] = _singularize(meas.group(2).lower()) + "s"  # normalize plural unit
        facts.append(fact)
    return facts


# ── completion_fn-shaped wrapper so it plugs into the existing pipeline ────────

# Recover the raw text the prompt embedded (between the Text: """ … """ markers).
_TEXT_RE = re.compile(r'Text:\s*"""\s*(.*?)\s*"""\s*\Z', re.DOTALL)


async def local_completion_fn(prompt: str) -> str:
    """A ``completion_fn`` (prompt -> JSON str) backed by the local extractor.

    Recovers the session text from the synthesis prompt, runs the rule-based
    extractor, and returns ``{"facts": [...]}`` — identical to what the LLM
    extractor returns, so ``extract_structured_facts`` + the disk cache work
    unchanged. Set ``--synthesis-model local`` to use it (zero API cost).
    """
    m = _TEXT_RE.search(prompt or "")
    text = m.group(1) if m else (prompt or "")
    return json.dumps({"facts": extract_facts_local(text)})
