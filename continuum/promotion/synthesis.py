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

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continuum.core.types import MemoryItem

log = logging.getLogger(__name__)

#: ``async (prompt) -> json_text`` — injected so extraction is testable w/o an LLM.
CompletionFn = Callable[[str], Awaitable[str]]


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


# ── structured-triple extraction (LLM → StructuredFact) ───────────────────────

_EXTRACT_PROMPT = """\
Extract COUNTABLE membership facts about the user from the text below. A fact is
one specific item that belongs to a collection the user counts, collects, owns,
buys, visits, or completes.

Output ONLY JSON: {"facts": [{"predicate": "<collection>", "object": "<item>"}]}

Rules:
- "predicate" is the SINGULAR collection noun, normalized and CONSISTENT: use the
  same predicate for the same kind of thing across all mentions (e.g. always
  "tank" for any aquarium/fish tank; "postcard" for postcards; "top" for
  tops/shirts/blouses; "coin" for coins). Generic over specific.
- "object" is the specific item (e.g. "goldfish tank", "vintage NYC postcard").
- One entry per distinct item. Do NOT count or aggregate — just list items.
- Ignore preferences, opinions, and non-countable statements.
- If there are no countable items, output {"facts": []}.

Text:
\"\"\"
%s
\"\"\"
"""


def _strip_json(raw: str) -> str:
    """Strip markdown fences / prose around a JSON object."""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    # keep from the first { to the last } if there's surrounding prose
    i, j = s.find("{"), s.rfind("}")
    return s[i : j + 1] if i != -1 and j != -1 and j > i else s


async def extract_structured_facts(
    text: str,
    *,
    completion_fn: CompletionFn,
    subject: str = "user",
    occurred_at: datetime | None = None,
) -> list[StructuredFact]:
    """
    LLM-extract countable membership triples from *text*.

    The model lists items with a consistent collection ``predicate``; the
    counting happens later in :func:`aggregate` (deterministic). Graceful: any
    LLM/parse failure returns ``[]`` (never raises into the caller). Predicate
    *consistency* is the quality risk — validate on real data via the eval A/B,
    not just these unit tests.
    """
    if not (text or "").strip():
        return []
    try:
        raw = await completion_fn(_EXTRACT_PROMPT % text)
        data = json.loads(_strip_json(raw))
    except Exception:
        log.exception("structured-fact extraction failed — returning none")
        return []

    facts: list[StructuredFact] = []
    for item in (data or {}).get("facts", []):
        if not isinstance(item, dict):
            continue
        pred = _normalize(str(item.get("predicate", "")))
        obj = str(item.get("object", "")).strip()
        if pred and obj:
            facts.append(
                StructuredFact(
                    subject=subject, predicate=pred, obj=obj, occurred_at=occurred_at
                )
            )
    return facts


# ── LTM representation + query-side detection ─────────────────────────────────

#: ``metadata["kind"]`` marker for synthesized aggregate facts in LTM.
ENTITY_SUMMARY_KIND = "entity_summary"

# Counting / aggregation questions — where a precomputed entity_summary should be
# surfaced to the reader. Deliberately tight (the v3 failure-set shape).
_COUNTING_RE = re.compile(
    r"\bhow many\b|\bhow much\b|\bhow long\b|\bhow often\b"
    r"|\b(?:total|number)\s+(?:of|number)\b|\bnumber of\b",
    re.IGNORECASE,
)


def is_counting_question(text: str) -> bool:
    """True for 'how many / how much / how long / number of …' questions."""
    return bool(text) and _COUNTING_RE.search(text) is not None


def to_memory_item(
    derived: DerivedFact,
    *,
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> MemoryItem:
    """
    Render a :class:`DerivedFact` as an LTM ``entity_summary`` item.

    Tagged ``kind=entity_summary`` (so retrieval can prioritize it on counting
    questions) and ``source=synthesis``; high importance because an aggregate is
    a durable, derived answer. The count/members ride in metadata for audit.
    """
    from continuum.core.types import MemoryItem, MemoryTier

    return MemoryItem(
        content=derived.text,
        tier=MemoryTier.LTM,
        embedding=embedding,
        session_id=session_id,
        importance=0.7,
        metadata={
            "kind": ENTITY_SUMMARY_KIND,
            "role": "memory",
            "source": "synthesis",
            "predicate": derived.predicate,
            "subject": derived.subject,
            "count": derived.count,
            "members": list(derived.members),
        },
    )


__all__ = [
    "StructuredFact",
    "DerivedFact",
    "aggregate",
    "extract_structured_facts",
    "to_memory_item",
    "is_counting_question",
    "ENTITY_SUMMARY_KIND",
    "CompletionFn",
]
