"""
continuum.promotion.clause_split
================================
Split a statement into atomic facts before it is embedded.

One embedding covers a whole string, so a sentence carrying two ideas lands
between them and matches neither well. Measured with bge-m3, the query
*"what database are we using?"* against:

    "We're going with Postgres 16, pgvector needs 0.8 or newer."   cos 0.4792
    "We're going with Postgres 16."                                cos 0.6135

The second clause about version numbers drags the vector far enough that the
compound form ranks *below* unrelated memories about frontend ownership
(cos 0.5217). Splitting on write is what recovers it.

Deliberately conservative — it is far worse to shred one fact into fragments
than to leave a compound one intact:

* sentence terminators (``. ! ? ;``) always split — unambiguous.
* a comma splits **only** when every part is an independent clause: three or
  more words, a verb, and either a subject pronoun or enough length. Parts
  opening with a subordinator (``which``, ``because``, ``and``, ``so``, …) are
  never independent, which is what protects lists ("Python, Rust, and Go"),
  appositives ("Bhilai, India") and relative clauses.

No LLM, no network — this runs on every write.
"""

from __future__ import annotations

import re

__all__ = ["split_facts"]

_ENDERS = re.compile(r"(?<=[.!?;])\s+")

#: A part opening with one of these continues the previous clause rather than
#: standing alone, so its presence vetoes the split.
_SUBORDINATOR = re.compile(
    r"^(which|who|whom|whose|where|when|while|although|though|because|since|so|and|or|but|"
    r"then|thus|hence|especially|including|such as|e\.g\.|i\.e\.)\b",
    re.I,
)

_SUBJECT = re.compile(r"\b(i|we|you|he|she|it|they|there|that|this|my|our|its|their)\b", re.I)

_VERB = re.compile(
    r"(\b(is|are|am|was|were|be|been|has|have|had|do|does|did|will|can|should|"
    r"need|needs|use|uses|using|handle|handles|handling|cap|caps|prefer|prefers|"
    r"run|runs|live|lives|moved|going|built|building|require|requires|hold|holds|"
    r"work|works|skip|skipped|pick|picked|set|sets|added|added)\b)|('m|'s|'re|'ve|'ll)",
    re.I,
)


def _is_independent(clause: str) -> bool:
    """Could this stand alone as its own fact?"""
    words = clause.split()
    if len(words) < 3 or _SUBORDINATOR.search(clause.strip()):
        return False
    return bool(_VERB.search(clause)) and bool(_SUBJECT.search(clause) or len(words) >= 4)


def split_facts(text: str) -> list[str]:
    """*text* as one or more atomic facts.

    Returns ``[text]`` unchanged whenever splitting would be unsafe, so callers
    can apply it unconditionally.
    """
    stripped = text.strip()
    if not stripped:
        return [text]

    facts: list[str] = []
    for sentence in _ENDERS.split(stripped):
        sentence = sentence.strip()
        if not sentence:
            continue
        parts = [p.strip() for p in sentence.split(",")]
        if len(parts) > 1 and all(_is_independent(p) for p in parts):
            facts.extend(p if p[-1] in ".!?" else p + "." for p in parts)
        else:
            facts.append(sentence)
    return facts or [text]
