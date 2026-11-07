"""
continuum.promotion.scalar_supersession
=======================================
A **deterministic** supersession rule for restated scalar facts.

Why this exists
---------------
``Mem0Promoter`` sends the contradiction decision to an LLM. It already
short-circuits the two easy cases without one — best-neighbour cosine below
``add_threshold`` is an ADD, above ``noop_threshold`` is a NOOP — but the case
in the middle, *"this contradicts that"*, always costs a model call. That is
the decision `docs/ENGINEERING_PLAN.md` calls the moat ("deterministic
correctness with **no LLM in the critical path**"), and today it is the one
decision we delegate.

The two deterministic paths that exist cannot help on real conversation:

======================================================  ==========
detector, measured on the 70 LongMemEval-S              recall
knowledge-update questions that have a gold pair
======================================================  ==========
``_UPDATE_MARKERS`` regex ("no longer", "used to", …)          3%
``attribute_extract.extract_attribute`` key collision          **0%**
**this rule**                                                **51%**
======================================================  ==========

``extract_attribute`` scores zero because it keys on canonical forms — "I live
in X", "I work at X", "my X is Y" — and none of these turns are shaped that
way. The regex scores 3% because a restatement carries no update language:

    2023/08/11  "have you tried any good Korean restaurants … I've tried
                 three different ones recently"
    2023/09/30  "have you tried any good Korean restaurants … I've tried
                 four different ones so far"

Nothing says "no longer". What *is* present is the signal this module uses:
**the same topic frame, restated, with a different scalar.** Same words, new
number, later date.

The rule
--------
A candidate supersedes a live neighbour when

1. their content-word Jaccard overlap is at least ``min_overlap`` — the same
   thing is being talked about; and
2. each carries a numeric value the other does not — they disagree on a
   scalar rather than merely both containing numbers.

Numbers inside URLs and long opaque identifiers are stripped first: a YouTube
id is not a fact, and it was the largest single source of false positives.

Measured cost of being wrong
----------------------------
Against 420 *hard* negatives — the highest-overlap non-gold user-turn pairs
drawn from the same haystacks, i.e. the pairs most likely to fool it — the
rule fires on **3.1%**. Against the real gold pairs it fires on **51%**.

So this is a **fast path, not a replacement**: it removes the LLM from about
half the supersession decisions on this data and defers the rest, exactly as
the cosine short-circuits do. It is deliberately biased toward abstention —
a false supersession retires a fact the reader needs, which is worse than
paying for one more model call.

Reproduce with ``evals/audit/scalar_supersession_coverage.py``.
"""

from __future__ import annotations

import re

__all__ = ["scalar_values", "frame_overlap", "supersedes", "find_superseded"]

#: URLs and long opaque tokens (media ids, hashes). Numbers inside these are
#: identifiers, not claims about the world.
_URLISH = re.compile(
    r"https?://\S+"
    r"|\b[\w.-]+\.(?:com|org|net|io|be|co|uk)/\S*"
    r"|\b[A-Za-z0-9_-]{8,}\b"
)

#: Grouped thousands ("400,000"), clock times ("7:30"), decimals, bare ints.
_NUMERIC = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?::\d{2})?(?:\.\d+)?")

_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

#: Function words carry no topic identity. Small on purpose: the rule compares
#: *overlap*, so a word shared by both texts helps rather than hurts, and one
#: shared by neither is irrelevant.
_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "have",
        "been",
        "from",
        "they",
        "them",
        "there",
        "their",
        "would",
        "could",
        "should",
        "about",
        "into",
        "your",
        "mine",
        "what",
        "when",
        "where",
        "which",
        "will",
        "also",
        "just",
        "really",
        "very",
        "much",
        "many",
        "thing",
        "things",
        "time",
        "times",
        "going",
        "know",
        "think",
        "like",
        "want",
        "need",
        "make",
        "made",
        "take",
        "took",
        "over",
        "back",
        "down",
        "here",
        "than",
        "then",
        "most",
        "some",
    }
)

#: Minimum content-word Jaccard for two texts to count as the same frame.
#: Swept over 0.10-0.40 on the LongMemEval knowledge-update gold pairs: 0.10
#: gives 51% recall at 3.1% on hard negatives, and every higher threshold
#: trades recall away far faster than it buys precision (0.20 -> 20% recall).
DEFAULT_MIN_OVERLAP = 0.10


def scalar_values(text: str) -> set[str]:
    """Numeric values in *text*, with number words folded onto their digits.

    Identifier-bearing spans (URLs, long opaque tokens) are removed first.
    """
    stripped = _URLISH.sub(" ", str(text or "")).lower()
    out: set[str] = set()
    for m in _NUMERIC.finditer(stripped):
        v = m.group(0).replace(",", "")
        out.add(v)
        if re.fullmatch(r"\d+\.0+", v):
            out.add(v.split(".")[0])
    for word, digit in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", stripped):
            out.add(digit)
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", str(text or "").lower()) if w not in _STOPWORDS}


def frame_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' content words, in ``[0, 1]``."""
    left, right = _content_words(a), _content_words(b)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def supersedes(candidate: str, existing: str, *, min_overlap: float = DEFAULT_MIN_OVERLAP) -> bool:
    """True when *candidate* restates *existing* with a different scalar.

    Symmetric by construction — it detects that two texts disagree, not which
    came first. **The caller supplies the ordering**: only call this with a
    candidate that is genuinely newer, or a correction will retire its own
    correction.
    """
    if frame_overlap(candidate, existing) < min_overlap:
        return False
    new, old = scalar_values(candidate), scalar_values(existing)
    # Each must carry a value the other lacks. "both mention 3" is agreement;
    # "one says 3, one says 4" is a contradiction.
    return bool(new and old and (new - old) and (old - new))


def find_superseded(
    candidate: str,
    neighbours: list[str],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> int | None:
    """Index of the neighbour *candidate* supersedes, or ``None``.

    When several qualify, the highest-overlap one wins — that is the fact the
    candidate is most plausibly a restatement of. ``None`` means "undecided",
    and the caller should fall back to the LLM promoter rather than treat it
    as "no contradiction".
    """
    best: tuple[float, int] | None = None
    for i, text in enumerate(neighbours):
        if supersedes(candidate, text, min_overlap=min_overlap):
            score = frame_overlap(candidate, text)
            if best is None or score > best[0]:
                best = (score, i)
    return None if best is None else best[1]
