"""
evals.longmemeval.evidence
==========================
Turn-level evidence diagnostics — what ``partial_recall`` cannot see.

Why this exists
---------------
The harness has one retrieval metric, ``partial_recall``: the fraction of
*answer-bearing sessions* that reached retrieval. On the same-setup control
(``results/v3_baseline_control/judged.json``) it reads **1.000 on
knowledge-update while accuracy on that category is 64.1%** — all 78 gold
sessions retrieved, 28 answers wrong. A metric that is saturated on the
category the store was built for cannot discriminate between the failures
that matter, and the failure taxonomy inherits the blindness: 130 of the
control's 131 errors land in ``wrong_span``, and ``stale_fact`` reads **0 on
every run ever recorded** because it keys off marker words in the matched
claim rather than off the gold evidence itself.

Two metrics at the granularity the failures actually live at:

``gold_turn_recall``
    LongMemEval flags the individual turns that carry the answer
    (``has_answer``). Session recall counts a session as recalled when *any*
    of its turns survives; a question whose evidence is two turns in one
    session scores 1.000 with one of them dropped. This counts the turns.

``supersession_verdict``
    When the gold turns disagree because a later one supersedes an earlier
    one — "I've tried three Korean restaurants" (Aug 11) then "I've tried
    four" (Sep 30) — did the answer take the **superseded** value? This is
    the failure bi-temporal supersession exists to prevent, and until now the
    harness could not count it.

The second one reframes a standing conclusion. ``docs/limitations.md`` files
these questions under "counting / aggregation is the floor … counting is
genuinely reader-bound", and v3 built a numeric aggregator to enumerate set
members for them. But the gold evidence for
``How many Korean restaurants have I tried?`` is two turns, each stating a
complete count outright. Nothing needs enumerating and nothing needs
counting: the answer is whichever turn is *later*. The aggregator was solving
a problem these questions do not pose, which is a better explanation of why
it came out net-negative than "counting is hard".

Multi-session counting is a genuinely different population — its gold turns
each contribute *one member* ("I saw Dr. Patel" … "I saw an ENT specialist"),
with coreference to resolve and duplicates to drop — and the reader-bound
verdict may well hold there. Separating the two populations is the point.

Conservative by construction
----------------------------
Both detectors return ``None`` rather than guess, in the spirit of
``continuum.promotion.attribute_extract``: a wrong verdict is worse than no
verdict, because this metric exists to be trusted as evidence. Concretely,
``supersession_verdict`` fires only when the *expected* answer confirms the
later gold turn is the right one — so a pair that merely differs, rather than
superseding, is skipped. Measured against the control run: **15 stale
verdicts on wrong answers, 0 on correct ones** (76 of 500 questions receive
any verdict at all). Treat the stale count as a **lower bound** on
supersession failures, never as the total.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = [
    "GoldTurn",
    "gold_turns_from_raw",
    "gold_turn_recall",
    "supersession_verdict",
]


@dataclasses.dataclass(frozen=True)
class GoldTurn:
    """One turn LongMemEval flags as carrying the answer (``has_answer``)."""

    session_id: str
    #: Raw haystack date string ("2023/08/11 (Fri) 09:09"). Compared only for
    #: ordering, and the format sorts lexicographically the same way it sorts
    #: chronologically, so it is kept verbatim rather than parsed.
    date: str
    role: str
    content: str
    #: Position within its session — orders gold turns that share a date.
    index: int


def gold_turns_from_raw(raw: dict[str, Any]) -> list[GoldTurn]:
    """
    Extract the ``has_answer`` turns from one raw LongMemEval row.

    ``has_answer`` is present only on turns inside the answer sessions, so a
    missing key is a plain "not gold" rather than an error. Returns
    chronological order (date, then position in session).
    """
    sids = raw.get("haystack_session_ids") or []
    sessions = raw.get("haystack_sessions") or []
    dates = raw.get("haystack_dates") or []
    answer_sids = raw.get("answer_session_ids") or []
    if isinstance(answer_sids, str):
        answer_sids = [answer_sids]

    out: list[GoldTurn] = []
    for sid in answer_sids:
        try:
            i = sids.index(sid)
        except ValueError:
            # Gold session absent from the haystack — a dataset defect, not
            # something this module should raise on.
            continue
        session = sessions[i] if i < len(sessions) else []
        date = str(dates[i]) if i < len(dates) else ""
        for j, turn in enumerate(session):
            if not turn.get("has_answer"):
                continue
            out.append(GoldTurn(
                session_id=str(sid),
                date=date,
                role=str(turn.get("role", "")),
                content=str(turn.get("content", "")),
                index=j,
            ))
    out.sort(key=lambda g: (g.date, g.index))
    return out


# ── gold-turn recall ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Whitespace- and case-insensitive form for containment tests."""
    return " ".join(str(text or "").lower().split())


#: Prefix length used to match a gold turn against retrieved content. Whole-
#: string equality is too strict — the ingest path may trim, and a retrieved
#: item can be a span *within* the turn — while a short prefix collides
#: between turns that open with the same boilerplate. 80 characters clears
#: LongMemEval's openers ("I'm looking for some tips on …") without reaching
#: the end of the shortest gold turns.
_MATCH_PREFIX = 80


def gold_turn_recall(
    gold: Sequence[GoldTurn], retrieved_contents: Iterable[str]
) -> dict[str, Any]:
    """
    Fraction of gold turns whose text reached the reader.

    A gold turn counts as recalled when its normalised prefix appears in some
    retrieved item, or a retrieved item's prefix appears in it — the second
    direction catches the case where retrieval surfaced a *span* of the turn
    rather than the whole turn.

    Returns zeros with ``gold_turn_count == 0`` when the row has no gold
    turns (21 of the 500 rows), so callers can filter rather than divide.
    """
    golds = list(gold)
    haystack = [_normalise(c) for c in retrieved_contents if c]
    hits = 0
    missing: list[str] = []
    for g in golds:
        needle = _normalise(g.content)[:_MATCH_PREFIX]
        if needle and any(
            needle in h or (h and h[:_MATCH_PREFIX] in _normalise(g.content))
            for h in haystack
        ):
            hits += 1
        else:
            missing.append(f"{g.session_id}#{g.index}")
    return {
        "gold_turn_count": len(golds),
        "retrieved_gold_turn_count": hits,
        "gold_turn_recall": hits / len(golds) if golds else 0.0,
        "missing_gold_turns": missing,
    }


# ── supersession verdict ────────────────────────────────────────────────────

_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}
#: Grouped thousands ("400,000"), clock times ("7:30"), decimals, bare ints.
_NUMERIC = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?::\d{2})?(?:\.\d+)?")

#: Function words that carry no identity. Deliberately small: the lexical
#: path only ever compares *set differences* between two gold turns, so a
#: word shared by both is cancelled out before it can matter.
_STOPWORDS = frozenset({
    "that", "this", "with", "have", "been", "from", "they", "them", "there",
    "their", "would", "could", "should", "about", "into", "your", "mine",
    "what", "when", "where", "which", "will", "also", "just", "really",
    "very", "much", "many", "thing", "things", "time", "times", "going",
    "know", "think", "like", "want", "need", "make", "made", "take", "took",
    "over", "back", "down", "here", "than", "then", "most", "some",
})


def _numeric_values(text: Any) -> set[str]:
    """Numbers in *text*, with number words folded onto their digits."""
    t = str(text or "").lower()
    out: set[str] = set()
    for m in _NUMERIC.finditer(t):
        v = m.group(0).replace(",", "")
        out.add(v)
        # "4.0" and "4" are the same count.
        if re.fullmatch(r"\d+\.0+", v):
            out.add(v.split(".")[0])
    for word, digit in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            out.add(digit)
    return out


def _lexical_values(text: Any) -> set[str]:
    """Content words of four or more letters, minus function words."""
    return {
        w for w in re.findall(r"[a-z]{4,}", str(text or "").lower())
        if w not in _STOPWORDS
    }


def _verdict_for(
    gold: Sequence[GoldTurn],
    answer: str,
    expected_answer: Any,
    extract: Callable[[Any], set[str]],
) -> str | None:
    """One pass of the verdict rule under a given value extractor."""
    if len(gold) < 2:
        return None
    dates = {g.date for g in gold}
    if len(dates) < 2:
        # Same-date gold turns cannot establish which value supersedes.
        return None
    latest = max(dates)
    late: set[str] = set()
    early: set[str] = set()
    for g in gold:
        (late if g.date == latest else early).update(extract(g.content))
    distinct_late = late - early
    distinct_early = early - late
    if not distinct_late or not distinct_early:
        # The turns do not actually disagree on a value.
        return None
    if not (extract(expected_answer) & distinct_late):
        # The gold pair is not a supersession, or the later turn is not what
        # the expected answer is drawn from. Abstain rather than guess.
        return None
    found = extract(answer)
    if found & distinct_early and not (found & distinct_late):
        return "stale"
    if found & distinct_late:
        return "fresh"
    return None


def supersession_verdict(
    gold: Sequence[GoldTurn], answer: str, expected_answer: Any
) -> str | None:
    """
    ``"stale"``, ``"fresh"``, or ``None`` when undecidable.

    ``"stale"`` means the answer carries a value that appears **only** in the
    earlier gold turn, and none of the value the later turn supersedes it
    with — the model was handed both and took the retired one.

    The numeric extractor runs first because it is the sharper instrument
    (11 stale verdicts on the control, no false positives); the lexical one
    runs only when numbers cannot decide, and adds 4 more. Neither has ever
    returned ``"stale"`` for an answer that scored correct.
    """
    return (
        _verdict_for(gold, answer, expected_answer, _numeric_values)
        or _verdict_for(gold, answer, expected_answer, _lexical_values)
    )
