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


# One flat (non-nested) JSON object — enough to salvage the fact records the
# extractor emits, which are flat {"predicate":..,"object":..,"quantity":..}.
_OBJ_RE = re.compile(r"\{[^{}]*\}")
# A double-quoted JSON string value, tolerating escaped chars inside.
_STR = r'"((?:[^"\\]|\\.)*)"'
_PRED_RE = re.compile(r'"predicate"\s*:\s*' + _STR)
_OBJ_FIELD_RE = re.compile(r'"object"\s*:\s*' + _STR)
_QTY_RE = re.compile(r'"quantity"\s*:\s*("?-?\d+(?:\.\d+)?"?|null)')
_UNIT_RE = re.compile(r'"unit"\s*:\s*' + _STR)


def _salvage_fact_items(raw: str) -> list[dict[str, object]]:
    """Regex-recover the well-formed fact objects from malformed JSON.

    gpt-4o-mini sometimes emits a single bad string (an unescaped quote inside a
    value, e.g. a model name) that breaks ``json.loads`` for the WHOLE response —
    which would otherwise drop every member in that session and undercount. We
    scan each flat ``{...}`` block and pull predicate/object/quantity/unit by
    regex, skipping only the genuinely broken records instead of the whole set.
    """
    items: list[dict[str, object]] = []
    for blob in _OBJ_RE.findall(raw or ""):
        p = _PRED_RE.search(blob)
        o = _OBJ_FIELD_RE.search(blob)
        if not (p and o):
            continue
        item: dict[str, object] = {"predicate": p.group(1), "object": o.group(1)}
        q = _QTY_RE.search(blob)
        if q and q.group(1) != "null":
            item["quantity"] = q.group(1).strip('"')
        u = _UNIT_RE.search(blob)
        if u:
            item["unit"] = u.group(1)
        items.append(item)
    return items


def _parse_fact_items(raw: str) -> list[dict[str, object]]:
    """Parse the extractor response into fact dicts, tolerantly.

    Fast path: strict ``json.loads`` of the ``{"facts": [...]}`` envelope. On any
    JSON error, fall back to regex salvage so one malformed record no longer
    discards an entire session's facts (the H&M-3-vs-5 / citrus-2-vs-3 cause).
    """
    try:
        data = json.loads(_strip_json(raw))
        if isinstance(data, dict):
            facts = data.get("facts", [])
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    salvaged = _salvage_fact_items(raw)
    if salvaged:
        log.warning("structured-fact JSON malformed — salvaged %d records", len(salvaged))
    return salvaged


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
    except Exception:
        log.exception("structured-fact extraction call failed — returning none")
        return []

    items = _parse_fact_items(raw)

    facts: list[StructuredFact] = []
    for item in items:
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


# A counting question scoped to a BOUNDED time window — "in the past two weeks",
# "this year", "in March", "since I started again", "3 weeks ago". For these the
# all-time synthesis total is WRONG (it counts members outside the window), so we
# must NOT inject it; the reader counts the in-window members from context.
# Unbounded totals ("so far", "in total", "currently", "do I have", "a typical
# week") deliberately do NOT match — synthesis is correct there.
_MONTHS = (
    "january|february|march|april|may|june|july|august"
    "|september|october|november|december"
)
_SCOPED_COUNT_RE = re.compile(
    r"\bin the (?:first|last|past)\s+(?:\w+\s+)?(?:day|week|month|year)s?\b"
    r"|\b(?:this|last|past|next)\s+(?:\w+\s+)?(?:day|week|month|year)s?\b"
    r"|\bsince I (?:started|began|moved|got|joined|switched|changed)\b"
    r"|\bin (?:" + _MONTHS + r")\b"
    r"|\b\w+\s+(?:days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)


def is_scoped_count_question(text: str) -> bool:
    """True for counting questions bounded to a time window (the synthesis total
    would be wrong — see the postcards 25-vs-33 / weddings 3-vs-4 regressions)."""
    return bool(text) and _SCOPED_COUNT_RE.search(text) is not None


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


# Sum-intent: the question wants a TOTAL of a quantity (hours, money, days), not
# a count of distinct members.
_SUM_INTENT_RE = re.compile(
    r"\bhow (?:much|long)\b|\btotal\b|\bin total\b|\baltogether\b|\bcombined\b"
    r"|\b(?:hours?|minutes?|days?|weeks?|months?|years?|dollars?|\$)\b",
    re.IGNORECASE,
)

# STRICT count intent — the ONLY phrasings the router may answer deterministically.
# "how many <X>" / "number of <X>" = a distinct-member count.
_COUNT_INTENT_RE = re.compile(r"\bhow many\b|\bnumber of\b", re.IGNORECASE)
# Money-sum intent — "how much did I spend", "total amount", "$".
_MONEY_SUM_RE = re.compile(
    r"\bhow much\b.*\b(?:spen[dt]|cost|pay|paid|money|amount|total|\$)\b"
    r"|\btotal\b.*\b(?:spen[dt]|cost|amount|money)\b",
    re.IGNORECASE,
)


def route_count_answer(facts: list[DerivedFact], question: str) -> str | None:
    """Deterministic answer for a counting question — the BARE value, bypassing
    the reader entirely.

    STRICT by design (full-500 lesson): an earlier version fired on *any* single
    predicate match and was 82% WRONG — it answered "how long is my commute" and
    "what yoga studio" with a garbage count. So we now fire ONLY on:
      • explicit "how many" / "number of"  → distinct count (and only count ≥ 2), or
      • money/measurement "how much / total ..." with a real summed quantity.
    Anything else (durations, recall, "how often") returns ``None`` → the reader
    answers. Still requires exactly ONE strict predicate match (never the fuzzy
    fallback) so a coincidental match can't hijack the answer.
    """
    if not (facts and question):
        return None
    if is_scoped_count_question(question):
        return None  # all-time total is wrong for a bounded-window question
    ql = question.lower()
    want_count = _COUNT_INTENT_RE.search(ql) is not None
    want_money = _MONEY_SUM_RE.search(ql) is not None
    if not (want_count or want_money):
        return None  # not a clear count/sum question → let the reader handle it
    matched = [f for f in facts if _predicate_in_question(f.predicate, ql)]
    if len(matched) != 1:
        return None  # 0 = no signal; >1 = ambiguous → let the reader decide
    d = matched[0]
    # SUM path: a real measurement total + sum/money intent → the total.
    if d.total is not None and (want_money or _SUM_INTENT_RE.search(ql)):
        return f"{_fmt_num(d.total)} {d.unit}".strip()
    # COUNT path: explicit "how many / number of", and only when ≥ 2 distinct
    # members (a count of 1 is usually a spurious single match — defer to reader).
    if want_count and d.count >= 2:
        return str(d.count)
    return None


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
