"""
bench/synth.py
==============
Deterministic synthetic chat-session generator for ingest benchmarks.

The output is intentionally *boring*: same template, varied details
(brand names, numbers, dates) seeded from a single int. The point is
not to challenge the LLM extractor but to give every comparison
system identical input bytes so timing differences are real.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    role: str           # "user" | "assistant"
    content: str
    session_id: str
    date: str           # ISO date


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]


# A small bank of plausible user-fact sentences; the generator
# composes turns from these so per-session content varies but the
# *distribution* (length, fact density) is stable.
_USER_TEMPLATES = [
    "I just moved to {city} last {month}.",
    "I bought a {brand} at {store} for ${amount}.",
    "My doctor is Dr. {lastname} at {clinic}.",
    "I started a new job at {company} as a {role} on {date}.",
    "I'm planning a {duration}-day trip to {destination} in {month}.",
    "My commute takes about {commute_min} minutes each way.",
    "I'm reading {book_count} books this month: {books}.",
    "I exercise {workouts_per_week} times a week at {gym}.",
]
_ASSISTANT_TEMPLATES = [
    "That sounds great! How are you finding {topic}?",
    "Got it — anything specific you'd like help planning?",
    "Noted. Anything else worth tracking on this?",
]
_CITIES   = ["Boston", "Austin", "Denver", "Seattle", "Portland", "Nashville"]
_BRANDS   = ["Sony WH-1000XM5", "iPhone 17 Pro", "Bose QC45", "Kindle Oasis"]
_STORES   = ["Target", "Best Buy", "Costco", "Apple Store"]
_LASTNAMES = ["Patel", "Nguyen", "Garcia", "Kim", "Rodriguez", "Singh"]
_CLINICS  = ["Northwell Health", "Mass General", "Mayo Clinic"]
_COMPANIES = ["Acme Co.", "Globex", "Initech", "Hooli", "Soylent Corp."]
_ROLES     = ["software engineer", "product manager", "designer", "data analyst"]
_DESTINATIONS = ["Iceland", "Japan", "Portugal", "New Zealand", "Croatia"]
_BOOKS = ["Project Hail Mary", "Klara and the Sun", "The Overstory"]
_GYMS  = ["Equinox", "Planet Fitness", "Crunch", "the YMCA"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def make_session(
    seed: int,
    *,
    user_id: str = "u-bench",
    n_turns: int = 6,
    base_date: dt.date | None = None,
) -> Session:
    """
    Generate one deterministic Session.

    Parameters
    ----------
    seed:
        Picks the templates + variable substitutions; same seed →
        identical session bytes across runs.
    n_turns:
        Total turns including assistant replies (so 6 → 3 user + 3 asst).
    base_date:
        ISO date stamped on every turn. Defaults to 2026-01-01 +
        ``seed`` days, so sessions span a synthetic timeline.
    """
    rng = random.Random(seed)
    base = (base_date or dt.date(2026, 1, 1)) + dt.timedelta(days=seed)
    sid = f"sess-{seed:06d}"

    def _user_text() -> str:
        tpl = rng.choice(_USER_TEMPLATES)
        return tpl.format(
            city=rng.choice(_CITIES),
            month=rng.choice(_MONTHS),
            brand=rng.choice(_BRANDS),
            store=rng.choice(_STORES),
            amount=rng.randint(20, 800),
            lastname=rng.choice(_LASTNAMES),
            clinic=rng.choice(_CLINICS),
            company=rng.choice(_COMPANIES),
            role=rng.choice(_ROLES),
            date=base.isoformat(),
            duration=rng.randint(3, 21),
            destination=rng.choice(_DESTINATIONS),
            commute_min=rng.choice([15, 25, 35, 45, 60]),
            book_count=rng.randint(1, 5),
            books=", ".join(rng.sample(_BOOKS, k=min(2, len(_BOOKS)))),
            workouts_per_week=rng.randint(1, 6),
            gym=rng.choice(_GYMS),
        )

    def _asst_text() -> str:
        return rng.choice(_ASSISTANT_TEMPLATES).format(topic=rng.choice(_CITIES))

    turns: list[Turn] = []
    date_iso = base.isoformat()
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        content = _user_text() if role == "user" else _asst_text()
        turns.append(Turn(role=role, content=content, session_id=sid, date=date_iso))
    _ = user_id  # tagged onto MemoryItem.metadata by the harness, not by the generator
    return Session(session_id=sid, date=date_iso, turns=tuple(turns))


def make_sessions(n: int, *, n_turns: int = 6) -> list[Session]:
    """Generate ``n`` sessions with deterministic per-seed content."""
    return [make_session(seed=i, n_turns=n_turns) for i in range(n)]


__all__ = ["Session", "Turn", "make_session", "make_sessions"]
