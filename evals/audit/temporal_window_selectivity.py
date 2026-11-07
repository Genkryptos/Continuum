"""
Can a question's date window find its evidence on LongMemEval-S? No.

    python3.12 -m evals.audit.temporal_window_selectivity

Zero LLM calls, zero network. Hindsight retrieves through a rule-based
temporal channel that filters memories by the date interval a question
names ("last weekend", "in March", "the past two weeks"). The 2026-11-07
gap audit proposed the same channel here: the turns are already dated and
the question has a ``question_date``, so the window is computable.

This script resolves every question's window and measures how many of the
answer sessions fall inside it against how many of *all* haystack
sessions do. A channel is only worth fusing if the first number clearly
exceeds the second. On LongMemEval-S it does not, at any window width:
the haystack's median date span is 10 days, so the distractor sessions
sit in the same window as the evidence. A second reason it fails: the
session date is when the user *talked about* an event, and many answer
sessions are dated well after the window the question names (December
activities discussed in a session dated 10 January).

The channel was therefore not built into the retriever.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import re
from pathlib import Path

DATA = Path("evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json")

_MONTHS = {
    name.lower(): idx for idx, name in enumerate(calendar.month_name) if name
}
_WEEKDAYS = {
    name.lower(): idx for idx, name in enumerate(calendar.day_name)
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "a couple of": 2, "couple of": 2,
    "few": 3, "a few": 3, "several": 3,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

# Sessions are dated on the day the user talked about the event, which is
# the day of it or a little after — so the window extends forward more
# than back.
_BEFORE = dt.timedelta(days=1)
_AFTER = dt.timedelta(days=3)

_NUM = r"(?P<n>\d+|a couple of|couple of|a few|few|several|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT = r"(?P<unit>day|week|month|year)s?"

_SPAN_RE = re.compile(
    rf"\b(?:in |over |during )?(?:the )?(?:past|last|previous) {_NUM} {_UNIT}\b",
    re.IGNORECASE,
)
_SPAN_ONE_RE = re.compile(
    r"\bin the (?:past|last) (?P<unit>day|week|month|year)\b", re.IGNORECASE,
)
_AGO_RE = re.compile(rf"\b(?P<since>since [^?.]*?)?{_NUM} {_UNIT} ago\b", re.IGNORECASE)
_LAST_UNIT_RE = re.compile(r"\blast (?P<unit>week|weekend|month|year)\b", re.IGNORECASE)
_THIS_UNIT_RE = re.compile(
    r"\b(?:this|so far this|in this|since the start of the|since the beginning of the) "
    r"(?P<unit>week|month|year)\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    r"\blast (?P<wd>" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(?P<since>since |from )?(?P<m>" + "|".join(_MONTHS) + r")(?: (?P<y>\d{4}))?\b",
    re.IGNORECASE,
)


def _n(raw: str) -> int:
    raw = raw.lower()
    return int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 1)


def _month_window(year: int, month: int) -> tuple[dt.date, dt.date]:
    last = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, last)


def _widen(start: dt.date, end: dt.date, today: dt.date) -> tuple[dt.date, dt.date]:
    return start - _BEFORE, min(end + _AFTER, today)


def resolve_windows(question: str, today: dt.date | None) -> list[tuple[dt.date, dt.date]]:
    """Absolute ``(start, end)`` windows named by ``question``, relative to ``today``."""
    if today is None or not question:
        return []
    q = question.lower()
    raw: list[tuple[dt.date, dt.date]] = []

    for m in _SPAN_RE.finditer(q):
        days = _n(m.group("n")) * _UNIT_DAYS[m.group("unit").lower()]
        raw.append((today - dt.timedelta(days=days), today))
    for m in _SPAN_ONE_RE.finditer(q):
        raw.append((today - dt.timedelta(days=_UNIT_DAYS[m.group("unit").lower()]), today))

    for m in _AGO_RE.finditer(q):
        unit = m.group("unit").lower()
        n = _n(m.group("n"))
        centre = today - dt.timedelta(days=n * _UNIT_DAYS[unit])
        if m.group("since"):
            # "since I started writing again three weeks ago" — open-ended.
            raw.append((centre, today))
            continue
        slack = {"day": 1, "week": 3, "month": 10, "year": 45}[unit]
        if m.group("n").lower() in ("few", "a few", "several", "couple of", "a couple of"):
            slack *= 2
        raw.append((centre - dt.timedelta(days=slack), centre + dt.timedelta(days=slack)))

    for m in _LAST_UNIT_RE.finditer(q):
        unit = m.group("unit").lower()
        if unit == "weekend":
            # The Saturday/Sunday before today (the previous one if today is
            # itself a weekend day).
            back = (today.weekday() - 5) % 7 or 7
            sat = today - dt.timedelta(days=back)
            raw.append((sat, sat + dt.timedelta(days=1)))
        elif unit == "week":
            monday = today - dt.timedelta(days=today.weekday() + 7)
            raw.append((monday, monday + dt.timedelta(days=6)))
        elif unit == "month":
            first = today.replace(day=1) - dt.timedelta(days=1)
            raw.append(_month_window(first.year, first.month))
        else:
            raw.append((dt.date(today.year - 1, 1, 1), dt.date(today.year - 1, 12, 31)))

    for m in _THIS_UNIT_RE.finditer(q):
        unit = m.group("unit").lower()
        if unit == "week":
            raw.append((today - dt.timedelta(days=today.weekday()), today))
        elif unit == "month":
            raw.append((today.replace(day=1), today))
        else:
            raw.append((dt.date(today.year, 1, 1), today))

    for m in _WEEKDAY_RE.finditer(q):
        back = (today.weekday() - _WEEKDAYS[m.group("wd").lower()]) % 7 or 7
        day = today - dt.timedelta(days=back)
        raw.append((day, day))

    for m in _MONTH_RE.finditer(q):
        month = _MONTHS[m.group("m").lower()]
        if m.group("m").lower() == "may" and not m.group("y") and not m.group("since"):
            # "may" is far more often the modal verb ("may I ask") than the month.
            before = q[: m.start()].rstrip()
            if not re.search(r"\b(in|during|of|since|from|until|by)$", before):
                continue
        if m.group("y"):
            year = int(m.group("y"))
        else:
            year = today.year if month <= today.month else today.year - 1
        start, end = _month_window(year, month)
        raw.append((start, today) if m.group("since") else (start, end))

    return [_widen(s, e, today) for s, e in raw if s <= today]


def in_windows(date: dt.date | None, windows: list[tuple[dt.date, dt.date]]) -> bool:
    """True when ``date`` falls inside any window."""
    if date is None:
        return False
    return any(start <= date <= end for start, end in windows)


def _qdate(raw: str) -> dt.date:
    cleaned = re.sub(r"\([A-Za-z]{3}\)", "", raw).strip()
    return dt.datetime.strptime(cleaned, "%Y/%m/%d %H:%M").date()


def main() -> int:
    rows = json.loads(DATA.read_text())
    measured: list[tuple[int, float, float]] = []
    for row in rows:
        windows = resolve_windows(row["question"], _qdate(row["question_date"]))
        if not windows:
            continue
        dates = {
            sid: _qdate(raw)
            for sid, raw in zip(row["haystack_session_ids"], row["haystack_dates"], strict=False)
        }
        gold = [dates[sid] for sid in row["answer_session_ids"] if sid in dates]
        if not gold:
            continue
        width = sum((end - start).days + 1 for start, end in windows)
        gold_in = sum(in_windows(d, windows) for d in gold) / len(gold)
        hay_in = sum(in_windows(d, windows) for d in dates.values()) / len(dates)
        measured.append((width, gold_in, hay_in))

    print(f"questions with a resolvable window: {len(measured)} / {len(rows)}")
    print(f"{'window width':>14} {'n':>4} {'gold in window':>15} {'haystack in window':>19}")
    for lo, hi in ((0, 8), (8, 20), (20, 45), (45, 100_000)):
        band = [m for m in measured if lo <= m[0] < hi]
        if not band:
            continue
        label = f"{lo}-{hi}d" if hi < 100_000 else f"{lo}d+"
        gold_in = sum(m[1] for m in band) / len(band)
        hay_in = sum(m[2] for m in band) / len(band)
        print(f"{label:>14} {len(band):>4} {gold_in:>15.2f} {hay_in:>19.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
