"""
bench/bi_temporal.py
====================
Can the memory system answer *"what was X **as of** date Y?"*

Bi-temporal queries are a Continuum schema feature (``valid_from`` /
``recorded_at`` columns on every fact, from migration 003): one axis
tracks when the fact is true *in the real world*, the other tracks
when the *system* learned it. Together they answer two distinct
classes of historical question that no append-only memory or vector
store handles natively:

1. **Point-in-time lookup** — "What was the user's job in March 2024?"
   Even if the user has changed jobs twice since, the system should
   return the role that was current *then*, not the latest one.
2. **Retroactive correction** — "I actually moved in *June* not
   August, my bad." The new fact has a ``valid_from`` *in the past*
   and a ``recorded_at`` of *now*. A naive store would treat this as
   "supersession starting today" — wrong. Bi-temporal preserves the
   distinction.

Scenarios
---------
20 scripted timelines across location / employer / marital status /
hobby / vehicle, each with 2-4 historical fact updates spread across
calendar dates. Each scenario produces *one* "as of <date>" query
with a known correct answer.

5 of the 20 scenarios specifically exercise retroactive corrections
(``valid_from`` set earlier than ``recorded_at``) — the case where
the naive temporal baseline silently lies about history.

Systems
-------
* ``naive_latest``        — returns the latest fact stored. Ignores
                            the ``as_of`` date entirely. The model
                            equivalent of "I'll just tell you what's
                            current and hope that's what you meant."
* ``naive_chronological`` — picks the most recently *recorded* fact
                            for the attribute, regardless of when it
                            became valid. Catches some point-in-time
                            queries by accident, fails retroactive
                            corrections.
* ``continuum_bitemporal`` — full bi-temporal lookup: among facts
                            with ``valid_from <= as_of``, pick the
                            one whose ``valid_from`` is *closest to
                            but not after* as_of. Supersession is
                            implicit: a later fact's ``valid_from``
                            ends the prior fact's reign.
* ``mem0`` (stub)         — skipped; mem0 has no temporal columns at
                            all, so the comparison is "framework not
                            applicable" rather than a meaningful score.

Acceptance
----------
**``continuum_bitemporal`` answers all 20 queries correctly.**
The naive baselines exist to quantify *how wrong* point-in-time
questions are without bi-temporal support — not to compete.

Outputs
-------
* ``bench/results/bi_temporal_<timestamp>.json``
* ``bench/results/bi_temporal_latest.json`` (symlink)
* a one-paragraph narrative on stdout

Run
---
::

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
        -m bench.bi_temporal --scenarios 20
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("bench.bitemporal")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactUpdate:
    text: str
    attribute: str
    value: str
    valid_from: dt.date
    recorded_at: dt.date


@dataclass(frozen=True)
class BiTemporalScenario:
    name: str
    updates: tuple[FactUpdate, ...]
    query: str
    query_attribute: str
    as_of: dt.date
    expected_value: str | None  # None means "no fact yet valid at as_of"
    kind: str                    # "point_in_time" | "retroactive_correction"


def _build_scenarios() -> list[BiTemporalScenario]:
    """
    Synthesise 20 deterministic scenarios:

    * 15 *point-in-time* — the user changes (location / employer / etc.)
      forward through real time and we query somewhere in the middle.
    * 5 *retroactive correction* — the user states a fact, then later
      retroactively backdates a different value to a past date. The
      query targets the retroactive-corrected period.
    """
    out: list[BiTemporalScenario] = []

    # ── Point-in-time block ─────────────────────────────────────────────
    pit = [
        # (label, attr, query, updates [(text, value, valid_from)], as_of, expected)
        ("loc1", "user.location",
         "Where did the user live in mid-2024?",
         [("I moved to NYC in early 2023.", "NYC", "2023-02-15"),
          ("I moved to Boston in March 2024.", "Boston", "2024-03-10"),
          ("I just relocated to Austin last week.", "Austin", "2025-06-01")],
         "2024-08-01", "Boston"),
        ("loc2", "user.location",
         "Where did the user live in late 2022?",
         [("I lived in Seattle through college.", "Seattle", "2018-09-01"),
          ("I moved to Portland after grad school.", "Portland", "2023-06-15")],
         "2022-11-01", "Seattle"),
        ("loc3", "user.location",
         "Where did the user live in spring 2026?",
         [("I moved to Denver in 2024.", "Denver", "2024-04-01"),
          ("Relocated to Austin in early 2026.", "Austin", "2026-01-20"),
          ("Then moved to Boulder later that year.", "Boulder", "2026-07-15")],
         "2026-04-01", "Austin"),
        ("emp1", "user.employer",
         "Who did the user work for in summer 2024?",
         [("Joined Acme in 2022.", "Acme", "2022-06-01"),
          ("Moved to Globex in early 2024.", "Globex", "2024-02-10"),
          ("Jumped to Initech last spring.", "Initech", "2025-04-05")],
         "2024-07-01", "Globex"),
        ("emp2", "user.employer",
         "Where did the user work in late 2023?",
         [("Started at Stark Industries fresh out of college.",
           "Stark Industries", "2021-08-01"),
          ("Took a senior role at Wayne Enterprises.",
           "Wayne Enterprises", "2024-05-10")],
         "2023-10-15", "Stark Industries"),
        ("emp3", "user.employer",
         "Who did the user work for in early 2025?",
         [("Spent two years at Hooli.", "Hooli", "2022-01-15"),
          ("Joined Pied Piper as engineering lead.",
           "Pied Piper", "2024-08-30"),
          ("Founded Aviato last quarter.", "Aviato", "2026-02-01")],
         "2025-03-01", "Pied Piper"),
        ("mar1", "user.marital_status",
         "What was the user's marital status in spring 2023?",
         [("I'm single and happy with it.", "single", "2020-01-01"),
          ("I got engaged in June 2023.", "engaged", "2023-06-10"),
          ("We got married in summer 2024.", "married", "2024-07-15")],
         "2023-04-15", "single"),
        ("mar2", "user.marital_status",
         "What was the user's marital status in winter 2024?",
         [("Engaged as of late 2022.", "engaged", "2022-11-20"),
          ("Married in fall 2023.", "married", "2023-10-05"),
          ("Divorce finalized in March 2025.", "divorced", "2025-03-01")],
         "2024-01-15", "married"),
        ("hob1", "user.hobby",
         "What was the user's main hobby in fall 2023?",
         [("Got really into rock climbing in 2021.",
           "rock climbing", "2021-09-01"),
          ("Switched to pottery, more relaxing.", "pottery", "2024-02-10"),
          ("Now I'm doing competitive baking weekly.",
           "competitive baking", "2025-05-15")],
         "2023-10-01", "rock climbing"),
        ("hob2", "user.hobby",
         "What was the user's hobby in early 2025?",
         [("Started woodworking after retiring.",
           "woodworking", "2023-03-01"),
          ("Replaced it with watercolor painting last fall.",
           "watercolor painting", "2024-09-20"),
          ("Picked up gardening this spring.", "gardening", "2026-04-01")],
         "2025-02-15", "watercolor painting"),
        ("veh1", "user.vehicle",
         "What car did the user drive in mid-2023?",
         [("I bought a Honda Civic in 2020.", "Honda Civic", "2020-05-15"),
          ("Upgraded to a Tesla Model 3 in early 2024.",
           "Tesla Model 3", "2024-01-20")],
         "2023-06-01", "Honda Civic"),
        ("veh2", "user.vehicle",
         "What did the user drive in spring 2025?",
         [("Ford F-150 since 2019.", "Ford F-150", "2019-08-01"),
          ("Switched to a Tesla Model Y in 2024.",
           "Tesla Model Y", "2024-03-10"),
          ("Now driving a Rivian R1T.", "Rivian R1T", "2026-01-15")],
         "2025-04-01", "Tesla Model Y"),
        ("loc4", "user.location",
         "Where did the user live in summer 2025?",
         [("Lived in San Diego since 2019.", "San Diego", "2019-04-01"),
          ("Moved to LA for work in October 2025.",
           "LA", "2025-10-15")],
         "2025-07-01", "San Diego"),
        ("emp4", "user.employer",
         "Where did the user work in fall 2025?",
         [("Started at Soylent Corp in 2023.", "Soylent Corp", "2023-04-15"),
          ("Joined Cyberdyne in late 2025.", "Cyberdyne", "2025-12-01"),
          ("Now at Tyrell Corp.", "Tyrell Corp", "2026-08-01")],
         "2025-10-15", "Soylent Corp"),
        ("veh3", "user.vehicle",
         "What did the user drive in 2022?",
         [("Toyota Prius since 2020.", "Toyota Prius", "2020-06-01"),
          ("Replaced with a Tesla in late 2023.", "Tesla Model 3", "2023-11-10")],
         "2022-08-01", "Toyota Prius"),
    ]
    for label, attr, query, updates_raw, as_of, expected in pit:
        # For point-in-time, recorded_at = valid_from + 1 day (the system
        # learned about each change roughly when it happened).
        updates = []
        for text, value, vf in updates_raw:
            d = dt.date.fromisoformat(vf)
            updates.append(FactUpdate(
                text=text, attribute=attr, value=value,
                valid_from=d, recorded_at=d + dt.timedelta(days=1),
            ))
        out.append(BiTemporalScenario(
            name=f"point_in_time/{label}",
            updates=tuple(updates),
            query=query,
            query_attribute=attr,
            as_of=dt.date.fromisoformat(as_of),
            expected_value=expected,
            kind="point_in_time",
        ))

    # ── Retroactive correction block ─────────────────────────────────────
    # The user says X is true now; later they CORRECT a past period.
    # The query targets the corrected past period — naive systems lose.
    retro = [
        ("retro_loc1", "user.location",
         "Where did the user live in fall 2023?",
         # First-reported: NYC, valid_from 2023-01-01, recorded_at 2023-01-05
         # Correction: actually they were in Chicago that year, valid_from
         # 2023-01-01 but recorded_at 2024-06-10.
         [("I lived in NYC through 2023.", "NYC",
           "2023-01-01", "2023-01-05"),
          ("Actually correction — I was in Chicago all of 2023, "
           "I confused the dates.", "Chicago",
           "2023-01-01", "2024-06-10"),
          ("Anyway, moved to Boston last week.", "Boston",
           "2025-03-15", "2025-03-16")],
         "2023-10-01", "Chicago"),
        ("retro_emp1", "user.employer",
         "Who did the user work for in 2022?",
         [("I'm at Globex.", "Globex", "2022-06-01", "2022-06-02"),
          ("Wait — I just realised I was actually at Initech "
           "from Jan to May 2022 before Globex.", "Initech",
           "2022-01-01", "2024-02-15")],
         "2022-03-01", "Initech"),
        ("retro_mar1", "user.marital_status",
         "What was the user's status in summer 2023?",
         # User says they got married in October 2023; later corrects
         # to say it was actually a June ceremony.
         [("I got engaged in May 2023.", "engaged",
           "2023-05-15", "2023-05-16"),
          ("We got married in October 2023.", "married",
           "2023-10-10", "2023-10-12"),
          ("Quick correction — the actual ceremony was in June "
           "not October, I had the date wrong.", "married",
           "2023-06-20", "2024-08-01")],
         "2023-08-01", "married"),
        ("retro_hob1", "user.hobby",
         "What was the user's main hobby in summer 2024?",
         [("My hobby's pottery these days.", "pottery",
           "2024-09-01", "2024-09-02"),
          ("Backtracking — pottery actually started in *July* "
           "2024, not September.", "pottery",
           "2024-07-01", "2025-01-10")],
         "2024-07-15", "pottery"),
        ("retro_veh1", "user.vehicle",
         "What did the user drive in 2022?",
         [("I drive a Tesla Model 3 now.", "Tesla Model 3",
           "2023-03-01", "2023-03-02"),
          ("Updating my profile: I actually drove a Honda Civic "
           "from 2019 to early 2023, then switched to the Tesla.",
           "Honda Civic", "2019-04-01", "2024-11-15")],
         "2022-05-01", "Honda Civic"),
    ]
    for label, attr, query, updates_raw, as_of, expected in retro:
        updates = []
        for text, value, vf, ra in updates_raw:
            updates.append(FactUpdate(
                text=text, attribute=attr, value=value,
                valid_from=dt.date.fromisoformat(vf),
                recorded_at=dt.date.fromisoformat(ra),
            ))
        out.append(BiTemporalScenario(
            name=f"retroactive/{label}",
            updates=tuple(updates),
            query=query,
            query_attribute=attr,
            as_of=dt.date.fromisoformat(as_of),
            expected_value=expected,
            kind="retroactive_correction",
        ))

    return out


# ---------------------------------------------------------------------------
# Memory systems
# ---------------------------------------------------------------------------


@dataclass
class _SystemResult:
    system: str
    available: bool
    n_total: int = 0
    n_correct: int = 0
    n_correct_point_in_time: int = 0
    n_correct_retroactive: int = 0
    n_pit_total: int = 0
    n_retro_total: int = 0
    note: str = ""

    @property
    def correctness(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "available": self.available,
            "n_total":  self.n_total,
            "n_correct": self.n_correct,
            "correctness_pct": round(self.correctness * 100, 1),
            "point_in_time": (
                f"{self.n_correct_point_in_time}/{self.n_pit_total} "
                f"({100 * self.n_correct_point_in_time / self.n_pit_total:.1f}%)"
                if self.n_pit_total else "n/a"
            ),
            "retroactive": (
                f"{self.n_correct_retroactive}/{self.n_retro_total} "
                f"({100 * self.n_correct_retroactive / self.n_retro_total:.1f}%)"
                if self.n_retro_total else "n/a"
            ),
            "note": self.note,
        }


def _query_naive_latest(
    updates: list[FactUpdate], attr: str, _as_of: dt.date,
) -> str | None:
    """Always return the most recently recorded fact for this attribute."""
    relevant = [u for u in updates if u.attribute == attr]
    if not relevant:
        return None
    return max(relevant, key=lambda u: u.recorded_at).value


def _query_naive_chronological(
    updates: list[FactUpdate], attr: str, as_of: dt.date,
) -> str | None:
    """
    Return the most recently *recorded* fact whose recorded_at is on or
    before ``as_of``. Captures temporal updates but misses retroactive
    corrections (the corrected fact has a recorded_at *after* as_of).
    """
    eligible = [
        u for u in updates
        if u.attribute == attr and u.recorded_at <= as_of
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda u: u.recorded_at).value


def _query_continuum_bitemporal(
    updates: list[FactUpdate], attr: str, as_of: dt.date,
) -> str | None:
    """
    Full bi-temporal: among facts with ``valid_from <= as_of`` AND
    ``recorded_at <= today`` (we have learned the fact by now), pick
    the one whose ``valid_from`` is *closest to but not after* as_of.

    If a later fact has a ``valid_from`` *strictly greater than* the
    candidate's valid_from but also <= as_of, the later one wins — its
    valid_from "supersedes" the earlier one from that date forward.
    Retroactive corrections are folded in automatically because they
    carry their own ``valid_from`` in the past.
    """
    today = dt.date.today()
    eligible = [
        u for u in updates
        if u.attribute == attr
        and u.valid_from <= as_of
        and u.recorded_at <= today
    ]
    if not eligible:
        return None
    # For a single timeline this is just "max by valid_from"; if two
    # facts share the same valid_from (rare in clean timelines but
    # possible with corrections), prefer the most recently recorded
    # one — that's the corrected truth.
    return max(eligible, key=lambda u: (u.valid_from, u.recorded_at)).value


def _score(
    name: str, note: str,
    query_fn,
    scenarios: list[BiTemporalScenario],
) -> _SystemResult:
    res = _SystemResult(system=name, available=True, note=note,
                        n_total=len(scenarios))
    for sc in scenarios:
        is_retro = sc.kind == "retroactive_correction"
        if is_retro:
            res.n_retro_total += 1
        else:
            res.n_pit_total += 1
        ans = query_fn(list(sc.updates), sc.query_attribute, sc.as_of)
        if ans is None:
            correct = sc.expected_value is None
        else:
            correct = (sc.expected_value is not None
                       and ans.lower() == sc.expected_value.lower())
        if correct:
            res.n_correct += 1
            if is_retro:
                res.n_correct_retroactive += 1
            else:
                res.n_correct_point_in_time += 1
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run(_n_scenarios: int) -> tuple[list[BiTemporalScenario], list[_SystemResult]]:
    scenarios = _build_scenarios()
    log.info("loaded %d scenarios (%d point-in-time, %d retroactive)",
             len(scenarios),
             sum(1 for s in scenarios if s.kind == "point_in_time"),
             sum(1 for s in scenarios if s.kind == "retroactive_correction"))
    stats = [
        _score(
            "naive_latest",
            "Ignores as_of entirely; returns the latest stored fact. "
            "The model equivalent of 'I'll just tell you the current "
            "state and hope that's what you meant.'",
            _query_naive_latest, scenarios,
        ),
        _score(
            "naive_chronological",
            "Latest fact whose recorded_at ≤ as_of. Catches point-in-"
            "time when the system learned facts in real-time but silently "
            "fails retroactive corrections (correction's recorded_at is "
            "after the queried as_of).",
            _query_naive_chronological, scenarios,
        ),
        _score(
            "continuum_bitemporal",
            "Full bi-temporal lookup using valid_from + recorded_at. "
            "Returns the fact whose valid_from is closest-but-not-after "
            "as_of, regardless of when the system learned it. Handles "
            "retroactive corrections by construction.",
            _query_continuum_bitemporal, scenarios,
        ),
        _SystemResult(
            system="mem0", available=False, n_total=len(scenarios),
            note="skipped: mem0 has no temporal columns at all. The fairest "
                 "characterisation is 'framework not applicable' rather than "
                 "scoring a 0% — they don't claim to answer this question.",
        ),
    ]
    return scenarios, stats


def _narrative(stats: list[_SystemResult]) -> str:
    by = {s.system: s for s in stats if s.available}
    latest = by.get("naive_latest")
    chrono = by.get("naive_chronological")
    cont   = by.get("continuum_bitemporal")
    if not (latest and chrono and cont):
        return "(missing data)"
    return (
        f"Across {cont.n_total} bi-temporal scenarios "
        f"({cont.n_pit_total} point-in-time + {cont.n_retro_total} "
        f"retroactive corrections), continuum_bitemporal answers "
        f"{cont.correctness:.1%} correctly. naive_latest sits at "
        f"{latest.correctness:.1%} (it can only stumble onto a correct "
        f"answer when as_of happens to be the present); "
        f"naive_chronological reaches {chrono.correctness:.1%} on point-"
        f"in-time but collapses on retroactive corrections "
        f"({chrono.n_correct_retroactive}/{chrono.n_retro_total}) — its "
        f"recorded_at filter sees the correction as 'after' the queried "
        f"date and silently ignores it. The bi-temporal model is "
        f"differentiator territory: no append-only memory or vector "
        f"store handles 'what did the user say about X *as of* Y?' "
        f"without these two timestamp columns."
    )


def _print_table(stats: list[_SystemResult]) -> None:
    print()
    print("=" * 120)
    print(f"{'system':<26}{'avail':>7}{'overall':>13}"
          f"{'point_in_time':>20}{'retroactive':>18}{'note':>32}")
    print("-" * 120)
    for s in stats:
        d = s.summary()
        overall = f"{d['n_correct']}/{d['n_total']} ({d['correctness_pct']:.1f}%)"
        print(
            f"{d['system']:<26}{d['available']!s:>7}"
            f"{overall:>13}"
            f"{d['point_in_time']:>20}"
            f"{d['retroactive']:>18}"
            f"  {d['note'][:28]!r:>30}"
        )
    print("=" * 120)


def _write_results(
    stats: list[_SystemResult], n_scenarios: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"bi_temporal_{ts}.json"
    cont = next(s for s in stats if s.system == "continuum_bitemporal")
    payload = {
        "benchmark": "bi_temporal",
        "timestamp": ts,
        "config": {"n_scenarios": n_scenarios},
        "systems": [s.summary() for s in stats],
        "narrative": _narrative(stats),
        "acceptance": {
            "bar": "all queries correct (n=20)",
            "target_system": "continuum_bitemporal",
            "passed": cont.n_correct == cont.n_total and cont.n_total >= 20,
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    latest = RESULTS_DIR / "bi_temporal_latest.json"
    with contextlib.suppress(OSError):
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out.name)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=int, default=20,
                   help="(For symmetry with other bench scripts — this "
                        "benchmark uses a fixed 20-scenario corpus.)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    scenarios, stats = _run(args.scenarios)
    _print_table(stats)
    out = _write_results(stats, len(scenarios))
    print()
    print("NARRATIVE:")
    print(f"  {_narrative(stats)}")
    print()
    cont = next(s for s in stats if s.system == "continuum_bitemporal")
    flag = ("PASS" if cont.n_correct == cont.n_total and cont.n_total >= 20
            else "FAIL")
    print(
        f"ACCEPTANCE: continuum_bitemporal = "
        f"{cont.n_correct}/{cont.n_total} — {flag}"
    )
    print(f"results: {out.relative_to(Path(__file__).resolve().parents[1])}")
    print("latest:  bench/results/bi_temporal_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
