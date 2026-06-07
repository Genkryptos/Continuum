"""
continuum.promotion.synthesis
=============================
The v3 **synthesis (aggregation) layer** — derive *aggregate* facts so the
reader *reads* a count instead of *computing* one.

Why this exists (measured, not assumed): on LongMemEval-S v2.0, ~49% of the
residual failures are counting/aggregation questions ("how many X", "how long"),
and **the reader miscounts over both raw turns AND atomic facts** — fact
extraction surfaces individual items but does not aggregate, so gpt-oss-120b
still produces the same wrong count (e.g. "3 tops" when the answer is 5). See
``findings/roadmap_v3.md``.

The fix: at promotion time, group atomic facts by (subject, predicate) and count
**distinct** members **in code**. A code-computed count is correct by
construction — it removes both the reader's and an LLM-synthesizer's
miscount. The LLM's only job upstream is to emit structured
``(subject, predicate, object)`` triples; the counting is deterministic here.

This module is pure (no LLM, no DB) so the aggregation logic is unit-testable in
isolation. Converting a ``DerivedFact`` into an LTM ``entity_summary`` item and
wiring it into the promoter happen in separate steps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class StructuredFact:
    """
    One atomic membership fact, e.g. ``(user, tank, "goldfish tank")``.

    ``predicate`` is the *collection* being counted (a singular count-noun:
    "tank", "postcard", "top from H&M"); ``obj`` is the member. ``occurred_at``
    enables time-scoped aggregates ("since I started", "in the first 3 months").
    """

    subject: str
    predicate: str
    obj: str
    occurred_at: datetime | None = None
    source_id: str = ""


@dataclass
class DerivedFact:
    """A code-computed aggregate over a (subject, predicate) group."""

    subject: str
    predicate: str
    count: int
    members: list[str] = field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None

    @property
    def text(self) -> str:
        """Natural-language rendering for the reader, e.g. 'User has 3 tanks'."""
        noun = _pluralize(self.predicate, self.count)
        scope = ""
        if self.since is not None:
            scope = f" since {self.since.date().isoformat()}"
        elif self.until is not None:
            scope = f" as of {self.until.date().isoformat()}"
        subj = self.subject[:1].upper() + self.subject[1:] if self.subject else "User"
        return f"{subj} has {self.count} {noun}{scope}."


# ── normalization / pluralization ─────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace — for distinct-member dedup."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _pluralize(noun: str, count: int) -> str:
    """Minimal English pluralization sufficient for count-noun predicates."""
    noun = (noun or "item").strip()
    if count == 1:
        return noun
    # phrase like "top from H&M" → pluralize the head noun ("tops from H&M")
    head, sep, tail = noun.partition(" ")
    plural_head = _plural_word(head)
    return plural_head + sep + tail if sep else plural_head


def _plural_word(w: str) -> str:
    lw = w.lower()
    if re.search(r"(s|x|z|ch|sh)$", lw):
        return w + "es"
    if re.search(r"[^aeiou]y$", lw):
        return w[:-1] + "ies"
    return w + "s"


# ── the aggregator (deterministic count) ──────────────────────────────────────


def aggregate(
    facts: list[StructuredFact],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    dedup: bool = True,
    min_count: int = 1,
) -> list[DerivedFact]:
    """
    Group *facts* by (subject, predicate) and count distinct members in code.

    ``since`` / ``until`` scope by ``occurred_at`` (facts with no timestamp are
    kept only when no window is given — an unscoped count). ``dedup`` counts
    distinct normalized members (so the same item mentioned twice counts once).
    Returns one :class:`DerivedFact` per group with ``count >= min_count``,
    sorted by (subject, predicate) for stable output.
    """
    groups: dict[tuple[str, str], list[StructuredFact]] = {}
    for f in facts:
        if since is not None or until is not None:
            if f.occurred_at is None:
                continue  # can't place it in the window → exclude from scoped count
            if since is not None and f.occurred_at < since:
                continue
            if until is not None and f.occurred_at > until:
                continue
        groups.setdefault((f.subject, f.predicate), []).append(f)

    out: list[DerivedFact] = []
    for (subject, predicate), members in sorted(groups.items()):
        if dedup:
            seen: dict[str, str] = {}
            for m in members:
                seen.setdefault(_normalize(m.obj), m.obj)
            member_objs = list(seen.values())
        else:
            member_objs = [m.obj for m in members]
        if len(member_objs) < min_count:
            continue
        out.append(
            DerivedFact(
                subject=subject,
                predicate=predicate,
                count=len(member_objs),
                members=member_objs,
                since=since,
                until=until,
            )
        )
    return out


__all__ = ["StructuredFact", "DerivedFact", "aggregate"]
