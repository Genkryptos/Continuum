"""
continuum.extraction.retraction
===============================
Detect a **retraction** — an explicit statement that a *prior* claim was never
true ("I was never in Bengaluru", "that was wrong", "scratch that"), as opposed
to a temporal *change* ("I moved to Boston").

The distinction matters for memory correctness. A change *supersedes* the old
value but keeps it as historically valid; a retraction says the value was
**never valid** and should be retired from the timeline entirely. Confusing the
two corrupts "what was true before now?" reasoning — e.g. answering "you
returned from Bengaluru" when the user explicitly said they were never there.

This is a deliberately **conservative, deterministic** matcher (regex, no LLM):
it targets phrasings that negate a prior *experience/location/statement*, and
intentionally does **not** fire on standing negative facts like "I never eat
meat" (a genuine preference worth storing). Precision over recall — a missed
retraction degrades to normal handling; a false positive could wrongly delete a
real memory, so the downstream caller also gates on retrieval similarity.
"""

from __future__ import annotations

import re

# Each alternative is a phrasing that negates a *prior* claim. Notably absent:
# a bare "I never <verb>" for arbitrary verbs — that catches standing
# preferences ("I never eat meat"), which are facts, not retractions. We only
# match "never" with experiential/locational/declarative verbs.
_RETRACTION_RE = re.compile(
    r"""(?ix)            # case-insensitive, verbose
    \b(?:
        never \s+ (?: been | went | gone | visited | lived | stayed
                    | was | were | did | saw | met | said )       # "never been to", "never said"
      | (?: was | were ) \s+ never                                # "I was never in ..."
      | did (?: n['’]?t | \s+ not ) \s+ actually                  # "didn't actually"
      | actually ,? \s+ i \s+ (?: did (?:n['’]?t|\s+not)
                                 | never
                                 | was \s* n['’]? t )             # "actually I didn't / never / wasn't"
      | that (?: ['’]s | \s+ was | \s+ is )? \s+
            (?: wrong | incorrect | a \s+ mistake
              | not \s+ (?: true | right | correct ) )            # "that was wrong / not true"
      | (?: scratch | forget | ignore | disregard ) \s+
            (?: that | what \s+ i \s+ said )                      # "scratch that"
    )\b
    """,
)


def is_retraction(text: str) -> bool:
    """True if *text* explicitly retracts a prior claim (never-true), not a change."""
    return bool(text) and _RETRACTION_RE.search(text) is not None


__all__ = ["is_retraction"]
