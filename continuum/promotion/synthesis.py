"""
continuum.promotion.synthesis
=============================
The v3 **synthesis (aggregation) layer** — derive *aggregate* facts so the
reader *reads* a count/total instead of *computing* one.

Why (measured): ~49% of LongMemEval-S v2.0 residual failures are
counting/aggregation questions, and the reader miscounts over both raw turns and
atomic facts. So we aggregate in CODE — correct by construction.

v3.1 (after the first A/B — 14% recovery, diagnosed):
- **SUM**, not just COUNT — "how many hours/days total" needs summing numeric
  quantities, not counting members.
- **Relevance filtering** — the first cut injected *all* ~40 aggregates per
  question, burying the relevant one. ``relevant_summaries()`` returns only the
  aggregate(s) whose predicate matches the question.
- Tighter extraction prompt (consistent predicates + quantities).

Pure (no LLM/DB) so the aggregation logic is unit-testable in isolation; the LLM
only lists items (and their quantities); the counting/summing happens here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continuum.core.types import MemoryItem

log = logging.getLogger(__name__)

#: ``async (prompt) -> json_text`` — injected so extraction is testable w/o an LLM.
CompletionFn = Callable[[str], Awaitable[str]]

#: ``metadata["kind"]`` marker for synthesized aggregate facts in LTM.
ENTITY_SUMMARY_KIND = "entity_summary"


@dataclass(frozen=True)
class StructuredFact:
    """
    One atomic membership fact, e.g. ``(user, tank, "goldfish tank")`` — or a
    measurement, ``(user, "gaming session", "Mon", quantity=2, unit="hours")``.

    ``predicate`` is the *collection* (singular count-noun, normalized +
    CONSISTENT); ``obj`` is the member. ``quantity``/``unit`` carry a number to
    SUM (hours, days, dollars). ``occurred_at`` enables time-scoped aggregates.
    """

    subject: str
    predicate: str
    obj: str
    occurred_at: datetime | None = None
    quantity: float | None = None
    unit: str = ""
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
    total: float | None = None  # sum of member quantities (SUM questions)
    unit: str = ""

    @property
    def text(self) -> str:
        """E.g. 'User has 3 tanks' / 'User has 4 sessions totaling 140 hours'."""
        noun = _pluralize(self.predicate, self.count)
        scope = ""
        if self.since is not None:
            scope = f" since {self.since.date().isoformat()}"
        elif self.until is not None:
            scope = f" as of {self.until.date().isoformat()}"
        subj = self.subject[:1].upper() + self.subject[1:] if self.subject else "User"
        out = f"{subj} has {self.count} {noun}{scope}"
        if self.total is not None and self.unit:
            out += f" totaling {_fmt_num(self.total)} {self.unit}"
        return out + "."


# ── normalization / formatting ────────────────────────────────────────────────


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


_PREPS = {"from", "of", "in", "on", "with", "for", "at", "to", "by"}


def _pluralize(noun: str, count: int) -> str:
    noun = (noun or "item").strip()
    if count == 1:
        return noun
    words = noun.split()
    if len(words) == 1:
        return _plural_word(words[0])
    # "top from H&M" → pluralize the word before the preposition ("tops from …");
    # "gaming session" (no prep) → pluralize the LAST word, the compound head.
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _PREPS:
            words[i - 1] = _plural_word(words[i - 1])
            return " ".join(words)
    words[-1] = _plural_word(words[-1])
    return " ".join(words)


def _plural_word(w: str) -> str:
    lw = w.lower()
    if re.search(r"(s|x|z|ch|sh)$", lw):
        return w + "es"
    if re.search(r"[^aeiou]y$", lw):
        return w[:-1] + "ies"
    return w + "s"


# ── the aggregator (deterministic count + sum) ────────────────────────────────


def aggregate(
    facts: list[StructuredFact],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    dedup: bool = True,
    min_count: int = 1,
) -> list[DerivedFact]:
    """
    Group *facts* by (subject, predicate); count distinct members, and SUM their
    quantities where present — both in code. Time-scoped via ``occurred_at``.
    """
    groups: dict[tuple[str, str], list[StructuredFact]] = {}
    for f in facts:
        if since is not None or until is not None:
            if f.occurred_at is None:
                continue
            if since is not None and f.occurred_at < since:
                continue
            if until is not None and f.occurred_at > until:
                continue
        groups.setdefault((f.subject, f.predicate), []).append(f)

    out: list[DerivedFact] = []
    for (subject, predicate), members in sorted(groups.items()):
        # distinct members → (object, quantity, unit), first occurrence wins.
        entries: list[tuple[str, float | None, str]] = []
        if dedup:
            seen: dict[str, tuple[str, float | None, str]] = {}
            for m in members:
                seen.setdefault(_normalize(m.obj), (m.obj, m.quantity, m.unit))
            entries = list(seen.values())
        else:
            entries = [(m.obj, m.quantity, m.unit) for m in members]
        if len(entries) < min_count:
            continue

        qs = [(q, u) for (_, q, u) in entries if q is not None]
        total = sum(q for q, _ in qs) if qs else None
        unit = next((u for _, u in qs if u), "") if qs else ""
        out.append(
            DerivedFact(
                subject=subject,
                predicate=predicate,
                count=len(entries),
                members=[o for o, _, _ in entries],
                since=since,
                until=until,
                total=total,
                unit=unit,
            )
        )
    return out


# ── structured-triple extraction (LLM → StructuredFact) ───────────────────────

_EXTRACT_PROMPT = """\
Extract COUNTABLE membership facts about the user from the text below. A fact is
one specific item the user counts, collects, owns, buys, visits, or completes —
or a measurement (time/money the user spent).

Output ONLY JSON: {"facts": [{"predicate": "<collection>", "object": "<item>",
"quantity": <number or null>, "unit": "<unit or "">"}]}

Rules:
- "predicate": SINGULAR collection noun, normalized and CONSISTENT — use the
  SAME predicate for the same kind of thing everywhere (e.g. always "tank" for
  any aquarium/fish tank; "citrus fruit" for any lemon/lime/orange; "top" for
  tops/shirts/blouses; "road trip" for trips). Generic over specific.
- "object": the specific item (e.g. "goldfish tank", "lemon", "Tokyo trip").
- "quantity"/"unit": ONLY for measurements (e.g. {"predicate":"gaming session",
  "object":"Monday","quantity":2,"unit":"hours"}). Otherwise quantity=null.
- One entry per distinct item. Do NOT count, sum, or aggregate — just list.
- Ignore preferences/opinions and non-countable statements.
- If nothing countable, output {"facts": []}.

Text:
\"\"\"
%s
\"\"\"
"""


def _strip_json(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    i, j = s.find("{"), s.rfind("}")
    return s[i : j + 1] if i != -1 and j != -1 and j > i else s


def _coerce_qty(v: object) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        return float(m.group()) if m else None
    return None


async def extract_structured_facts(
    text: str,
    *,
    completion_fn: CompletionFn,
    subject: str = "user",
    occurred_at: datetime | None = None,
) -> list[StructuredFact]:
    """LLM-extract countable membership triples (+ optional quantity). Graceful → []."""
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
        if not (pred and obj):
            continue
        facts.append(
            StructuredFact(
                subject=subject,
                predicate=pred,
                obj=obj,
                occurred_at=occurred_at,
                quantity=_coerce_qty(item.get("quantity")),
                unit=str(item.get("unit", "") or "").strip(),
            )
        )
    return facts


# ── disk cache (extraction is deterministic → re-runs are free) ───────────────


def disk_cached(fn: CompletionFn, cache_dir: str | Path, *, namespace: str = "") -> CompletionFn:
    """
    Wrap a completion fn with a content-addressed disk cache.

    Synthesis extraction is deterministic (temperature 0), so identical
    ``(namespace, prompt)`` inputs always yield the same response. Caching them
    to disk makes re-runs **free** (no LLM credits) and instant — invaluable when
    iterating on the *aggregation/injection* logic downstream of extraction. A
    changed prompt or model (different ``namespace``) is a cache miss, so
    correctness is preserved. One file per call (sha256 key) → safe under
    concurrent async writes.
    """
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    async def wrapped(prompt: str) -> str:
        key = hashlib.sha256(f"{namespace}\x00{prompt}".encode()).hexdigest()
        path = root / f"{key}.txt"
        with contextlib.suppress(OSError):
            return path.read_text()
        resp = await fn(prompt)
        with contextlib.suppress(OSError):
            path.write_text(resp)
        return resp

    return wrapped


# ── query-side: relevance filtering + detection + LTM item ────────────────────

_COUNTING_RE = re.compile(
    r"\bhow many\b|\bhow much\b|\bhow long\b|\bhow often\b"
    r"|\b(?:total|number)\s+(?:of|number)\b|\bnumber of\b",
    re.IGNORECASE,
)


def is_counting_question(text: str) -> bool:
    """True for 'how many / how much / how long / number of …' questions."""
    return bool(text) and _COUNTING_RE.search(text) is not None


def _predicate_in_question(predicate: str, question_lower: str) -> bool:
    """Does any (3+ char) word of the predicate appear as a word in the question?"""
    for w in re.findall(r"[a-z]+", predicate.lower()):
        if len(w) < 3:
            continue
        stem = w[:-1] if w.endswith("s") else w
        if re.search(rf"\b{re.escape(stem)}s?\b", question_lower):
            return True
    return False


def relevant_summaries(
    facts: list[DerivedFact], question: str, *, fallback_max: int = 6
) -> list[DerivedFact]:
    """
    The aggregate(s) relevant to *question* — only those whose predicate matches
    the question's words. The v3.0 flaw was injecting *all* ~40 aggregates; this
    hands the reader just the count it needs. Falls back to the largest count>=2
    groups (capped) when nothing matches by name.
    """
    ql = (question or "").lower()
    matched = [f for f in facts if _predicate_in_question(f.predicate, ql)]
    if matched:
        return matched
    return sorted((f for f in facts if f.count >= 2), key=lambda f: -f.count)[:fallback_max]


def to_memory_item(
    derived: DerivedFact,
    *,
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> MemoryItem:
    """Render a :class:`DerivedFact` as an LTM ``entity_summary`` item."""
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
            "total": derived.total,
            "unit": derived.unit,
            "members": list(derived.members),
        },
    )


__all__ = [
    "StructuredFact",
    "DerivedFact",
    "aggregate",
    "extract_structured_facts",
    "disk_cached",
    "relevant_summaries",
    "to_memory_item",
    "is_counting_question",
    "ENTITY_SUMMARY_KIND",
    "CompletionFn",
]
