"""
bench/bi_temporal_banking.py
============================
Banking-domain instantiation of the bi-temporal "as of date Y" benchmark.

This is the *same evaluation* as :mod:`bench.bi_temporal` — the three query
strategies (``naive_latest``, ``naive_chronological``,
``continuum_bitemporal``) and the scorer are imported unchanged. **Only the
scenario corpus is swapped** for banking facts.

Banking is the canonical bi-temporal domain: this is exactly why bi-temporal
databases were invented for finance and insurance. Two questions recur and
neither an append-only store nor a single-time-axis store answers correctly:

1. **Point-in-time / regulatory "as of"** — "What was the credit limit *on
   the date the disputed charge posted*?", "What rate applied *on the
   statement date*?", "What was the customer's KYC risk rating *at the time
   of the audit*?" The current value is irrelevant; you need the value that
   was in force *then*.
2. **Retroactive correction** — a backdated fee reversal, a promo rate that
   should have applied from an earlier date, a beneficiary-change form
   processed late but effective from signing, an employment start date
   corrected on a loan file. These carry a ``valid_from`` in the *past* and a
   ``recorded_at`` of *now*. A single-axis (chronological) store treats the
   correction as "effective today" and silently reports the wrong history —
   an audit failure.

The 20-scenario corpus is 15 point-in-time + 5 retroactive corrections. The
retroactive block is the whole point: it is the only place valid-time and
transaction-time must be kept separate, and it is precisely what a bank's
audit / dispute / regulatory-reporting flow relies on.

Run
---
::

    python -m bench.bi_temporal_banking --scenarios 20
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import random
import sys
import time
from pathlib import Path

# Reuse the *identical* query strategies + scorer from the general benchmark.
from bench.bi_temporal import (
    BiTemporalScenario,
    FactUpdate,
    _print_table,
    _query_continuum_bitemporal,
    _query_naive_chronological,
    _query_naive_latest,
    _score,
    _SystemResult,
)

log = logging.getLogger("bench.bitemporal_banking")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Procedural generator (scales the corpus to N questions)
# ---------------------------------------------------------------------------
# The 20 hand-authored scenarios below are readable and realistic; to run a
# large "tense-level" test (e.g. --scenarios 500, matching LongMemEval-S) we
# top the curated set up with procedurally-generated timelines.
#
# Ground truth is defined by construction, NOT by any query algorithm: each
# scenario lays down a sequence of validity intervals (value, valid_from), and
# the value "in force as of D" is simply the interval D falls in — the bare
# definition of a bi-temporal as-of. continuum_bitemporal implements exactly
# that definition, so scoring it against the constructed truth is a correctness
# proof; the naive baselines implement *different* (wrong) definitions and are
# expected to miss. Nothing here is tuned to make Continuum win.

_TODAY = dt.date(2026, 6, 1)  # keep every recorded_at safely <= real "now"

_STREETS = [
    "Oak", "Maple", "Pine", "Cedar", "Elm", "Aspen", "Birch", "Willow",
    "Juniper", "Harbor", "Lakeshore", "Riverside", "Sunset", "Highland",
    "Meadow", "Kingston", "Devon", "Sherwood", "Camden", "Baker", "Marlow",
    "Orchard", "Chestnut", "Poplar", "Bayfront", "Coral", "Seabreeze",
]
_SUFFIX = ["Street", "Avenue", "Lane", "Road", "Court", "Drive", "Way",
           "Place", "Terrace", "Boulevard"]
_RISK = ["Standard", "Low", "Medium", "High", "Enhanced Due Diligence",
         "Restricted"]
_KYC = ["Pending Review", "Verified", "Enhanced Due Diligence", "Refresh Due",
        "Lapsed"]
_PLANS = ["Basic Checking", "Premium Checking", "Private Client",
          "Student Saver", "Everyday Saver", "Wealth Saver", "Silver Card",
          "Gold Card", "Platinum Card", "Signature Plan", "Elite Current",
          "Reserve Checking", "Prestige Account", "Priority Banking",
          "Ultimate Plan"]
_EMPLOYERS = [
    "Acme Logistics", "Globex Retail", "Initech Systems", "Northwind Traders",
    "Contoso Ltd", "Fabrikam Inc", "Stark Foods", "Wayne Freight",
    "Oscorp Media", "Soylent Foods", "Cyberdyne Labs", "Tyrell Design",
    "Massive Dynamic", "Umbrella Retail", "InGen Bio", "Vandelay Imports",
    "Kramerica Corp", "Sterling Media", "Dunder Mifflin", "Vance Cooling",
    "Pied Piper", "Aviato", "Hooli Cloud", "Wonka Industries", "Gekko Capital",
]
_FIRST = ["Robert", "Susan", "Daniel", "Maria", "Elena", "Miguel", "James",
          "Karen", "Grace", "Ahmed", "Sana", "Bilal", "Wei", "Ling", "Hao",
          "Olivia", "Nathan", "Ruby", "Priya", "Anil", "Kavya", "Tomas",
          "Lena", "Erik", "Kojo", "Ama"]
_LAST = ["Hale", "Cruz", "Poole", "Khan", "Chen", "Ford", "Rao", "Berg",
         "Owusu", "Nguyen", "Patel", "Garcia", "Kim", "Rossi", "Silva"]
_ACCT_TYPES = ["Checking", "Savings", "Money Market", "Joint", "Card"]


def _gen_address(rng: random.Random) -> str:
    return f"{rng.randint(1, 999)} {rng.choice(_STREETS)} {rng.choice(_SUFFIX)}"


def _gen_rate(rng: random.Random) -> str:
    return f"{rng.uniform(0.10, 9.90):.2f}%"


def _gen_limit(rng: random.Random) -> str:
    return f"${rng.choice([1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50])},000"


def _gen_overdraft(rng: random.Random) -> str:
    return f"${rng.choice([200, 350, 500, 750, 1000, 1500, 2000, 3000])}"


def _gen_phone(rng: random.Random) -> str:
    return f"555-{rng.randint(100, 9999):04d}"


def _gen_nominee(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def _gen_account(rng: random.Random) -> str:
    return f"{rng.choice(_ACCT_TYPES)} x{rng.randint(1000, 9999)}"


# (attribute name, value generator) — the pool the generator samples from.
_ATTRS = [
    ("account.mailing_address", _gen_address),
    ("loan.interest_rate", _gen_rate),
    ("savings.apy", _gen_rate),
    ("card.credit_limit", _gen_limit),
    ("account.overdraft_limit", _gen_overdraft),
    ("customer.risk_rating", lambda r: r.choice(_RISK)),
    ("customer.kyc_status", lambda r: r.choice(_KYC)),
    ("account.registered_phone", _gen_phone),
    ("account.product_plan", lambda r: r.choice(_PLANS)),
    ("customer.employer", lambda r: r.choice(_EMPLOYERS)),
    ("account.nominee", _gen_nominee),
    ("autopay.default_account", _gen_account),
]


def _distinct(rng: random.Random, gen, k: int) -> list[str]:
    """k distinct values from a (possibly small, categorical) generator."""
    seen: list[str] = []
    tries = 0
    while len(seen) < k and tries < 2000:
        v = gen(rng)
        tries += 1
        if v not in seen:
            seen.append(v)
    return seen


def _gen_point_in_time(rng: random.Random, idx: int) -> BiTemporalScenario:
    """Forward timeline; query lands in a *middle* interval (never the last),
    so naive_latest — which returns the current value — is wrong by design."""
    attr, gen = rng.choice(_ATTRS)
    k = rng.choice([2, 3, 4])
    values = _distinct(rng, gen, k)
    k = len(values)
    dates = [_TODAY - dt.timedelta(days=rng.randint(2000, 2900))]
    for _ in range(1, k):
        dates.append(dates[-1] + dt.timedelta(days=rng.randint(150, 500)))
    updates = [
        FactUpdate(text=f"{attr} set to {v}.", attribute=attr, value=v,
                   valid_from=d, recorded_at=d + dt.timedelta(days=1))
        for v, d in zip(values, dates, strict=True)
    ]
    t = rng.randint(0, k - 2)                      # target a non-final interval
    gap = (dates[t + 1] - dates[t]).days
    as_of = dates[t] + dt.timedelta(days=rng.randint(30, max(31, gap - 30)))
    return BiTemporalScenario(
        name=f"point_in_time/gen{idx:04d}",
        updates=tuple(updates),
        query=f"What was {attr} as of {as_of.isoformat()}?",
        query_attribute=attr,
        as_of=as_of,
        expected_value=values[t],
        kind="point_in_time",
    )


def _gen_retroactive(rng: random.Random, idx: int) -> BiTemporalScenario:
    """Original value A, a backdated correction B (valid_from in the past,
    recorded_at after as_of), and a trailing current value C. as_of falls in
    B's interval. naive_chronological can't see B (recorded after as_of);
    naive_latest returns C; only bi-temporal recovers B."""
    attr, gen = rng.choice(_ATTRS)
    a_val, b_val, c_val = _distinct(rng, gen, 3)[:3]
    v0 = _TODAY - dt.timedelta(days=rng.randint(2200, 2700))
    v1 = v0 + dt.timedelta(days=rng.randint(160, 400))     # correction's true start
    v2 = v1 + dt.timedelta(days=rng.randint(260, 520))     # trailing change
    as_of = v1 + dt.timedelta(days=rng.randint(30, (v2 - v1).days - 30))
    r0 = v0 + dt.timedelta(days=1)
    r1 = min(as_of + dt.timedelta(days=rng.randint(60, 300)), _TODAY)  # after as_of
    r2 = min(max(r1 + dt.timedelta(days=rng.randint(30, 200)),
                 v2 + dt.timedelta(days=1)), _TODAY)        # latest recorded
    updates = (
        FactUpdate(text=f"{attr} recorded as {a_val}.", attribute=attr,
                   value=a_val, valid_from=v0, recorded_at=r0),
        FactUpdate(text=f"Backdated correction: {attr} was {b_val} effective "
                        f"{v1.isoformat()}.", attribute=attr,
                   value=b_val, valid_from=v1, recorded_at=r1),
        FactUpdate(text=f"{attr} later changed to {c_val}.", attribute=attr,
                   value=c_val, valid_from=v2, recorded_at=r2),
    )
    return BiTemporalScenario(
        name=f"retroactive/gen{idx:04d}",
        updates=updates,
        query=f"What was {attr} as of {as_of.isoformat()} "
              f"(after the backdated correction)?",
        query_attribute=attr,
        as_of=as_of,
        expected_value=b_val,
        kind="retroactive_correction",
    )


# ---------------------------------------------------------------------------
# Banking scenario corpus
# ---------------------------------------------------------------------------


def _build_scenarios(n: int = 20) -> list[BiTemporalScenario]:
    out: list[BiTemporalScenario] = []

    # ── Point-in-time block (15) — regulatory "as of" ───────────────────────
    # (label, attr, query, updates [(text, value, valid_from)], as_of, expected)
    pit = [
        ("addr1", "account.mailing_address",
         "What mailing address was on file when the June 2024 statement was issued?",
         [("Address set to 12 Oak Street.", "12 Oak Street", "2022-01-10"),
          ("Moved to 88 Maple Avenue.", "88 Maple Avenue", "2024-09-01")],
         "2024-06-15", "12 Oak Street"),
        ("addr2", "account.mailing_address",
         "What was the mailing address on record in early 2023?",
         [("Address on file: 400 Pine Road.", "400 Pine Road", "2021-03-01"),
          ("Updated to 27 Cedar Court.", "27 Cedar Court", "2023-11-01")],
         "2023-01-15", "400 Pine Road"),
        ("rate1", "loan.interest_rate",
         "What was the mortgage rate as of the March 2024 payment?",
         [("Mortgage opened at 4.25%.", "4.25%", "2021-05-01"),
          ("Refinanced to 5.75%.", "5.75%", "2024-07-10")],
         "2024-03-01", "4.25%"),
        ("rate2", "savings.apy",
         "What savings APY applied in late 2022?",
         [("Savings APY set at 0.50%.", "0.50%", "2020-01-01"),
          ("APY raised to 1.25%.", "1.25%", "2023-04-01")],
         "2022-10-01", "0.50%"),
        ("limit1", "card.credit_limit",
         "What was the card credit limit when the disputed charge posted in late 2023?",
         [("Card limit set at $5,000.", "$5,000", "2022-02-01"),
          ("Limit increased to $10,000.", "$10,000", "2024-01-15")],
         "2023-11-20", "$5,000"),
        ("limit2", "card.credit_limit",
         "What was the credit limit in spring 2025?",
         [("Limit set at $8,000.", "$8,000", "2023-06-01"),
          ("Raised to $12,000.", "$12,000", "2024-02-01"),
          ("Raised again to $20,000.", "$20,000", "2025-09-01")],
         "2025-04-01", "$12,000"),
        ("kyc1", "customer.risk_rating",
         "What was the customer's risk rating at the time of the 2024 audit?",
         [("Risk rating: Standard.", "Standard", "2021-03-01"),
          ("Escalated to Enhanced Due Diligence.",
           "Enhanced Due Diligence", "2025-02-01")],
         "2024-05-01", "Standard"),
        ("kyc2", "customer.kyc_status",
         "What was the customer's KYC status in mid-2023?",
         [("KYC status: Pending Review.", "Pending Review", "2022-01-01"),
          ("KYC status: Verified.", "Verified", "2023-01-10"),
          ("Escalated to Enhanced Due Diligence.",
           "Enhanced Due Diligence", "2024-08-01")],
         "2023-06-01", "Verified"),
        ("phone1", "account.registered_phone",
         "Which mobile number was registered when the OTP was sent in early 2023?",
         [("Registered mobile 555-0101.", "555-0101", "2020-06-01"),
          ("Updated mobile to 555-0188.", "555-0188", "2023-08-10")],
         "2023-02-01", "555-0101"),
        ("plan1", "account.product_plan",
         "What account plan was active in fall 2023?",
         [("Plan: Basic Checking.", "Basic Checking", "2021-01-01"),
          ("Upgraded to Premium Checking.", "Premium Checking", "2024-03-01")],
         "2023-10-01", "Basic Checking"),
        ("plan2", "card.product_plan",
         "What card plan applied in summer 2025?",
         [("Card: Silver Card.", "Silver Card", "2022-05-01"),
          ("Upgraded to Gold Card.", "Gold Card", "2024-06-01"),
          ("Upgraded to Platinum Card.", "Platinum Card", "2026-01-01")],
         "2025-07-01", "Gold Card"),
        ("emp1", "customer.employer",
         "Who was the customer's employer on record in summer 2024?",
         [("Employer: Acme Logistics.", "Acme Logistics", "2022-06-01"),
          ("Moved to Globex Retail.", "Globex Retail", "2024-02-10"),
          ("Moved to Initech Systems.", "Initech Systems", "2025-04-05")],
         "2024-07-01", "Globex Retail"),
        ("over1", "account.overdraft_limit",
         "What was the overdraft limit as of the January 2024 overdraft?",
         [("Overdraft limit: $500.", "$500", "2021-09-01"),
          ("Overdraft limit raised to $1,500.", "$1,500", "2024-05-01")],
         "2024-01-05", "$500"),
        ("nom1", "account.nominee",
         "Who was the account nominee in early 2022?",
         [("Nominee: Robert Hale.", "Robert Hale", "2019-01-01"),
          ("Nominee changed to Susan Hale.", "Susan Hale", "2023-03-01")],
         "2022-02-01", "Robert Hale"),
        ("auto1", "autopay.default_account",
         "Which account was the autopay default in mid-2023?",
         [("Autopay default: Checking x1234.", "Checking x1234", "2021-04-01"),
          ("Autopay default: Savings x5678.", "Savings x5678", "2024-01-01")],
         "2023-06-01", "Checking x1234"),
    ]
    for label, attr, query, updates_raw, as_of, expected in pit:
        # Point-in-time: recorded_at = valid_from + 1 day (the bank learned of
        # each change roughly when it happened).
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

    # ── Retroactive-correction block (5) — backdated banking corrections ─────
    # Each has: an originally-recorded value, a later backdated correction
    # (valid_from in the past, recorded_at now), and a trailing current value.
    # The query targets the corrected past period. naive_chronological can't
    # see the correction (its recorded_at is after as_of); naive_latest returns
    # the trailing current value; only bi-temporal recovers the truth.
    # (label, attr, query, updates [(text, value, valid_from, recorded_at)], as_of, expected)
    retro = [
        ("retro_rate1", "savings.apy",
         "What savings APY applied to the account in Q1 2024 (after the rate review)?",
         [("Savings APY recorded at 1.50%.", "1.50%",
           "2024-01-01", "2024-01-02"),
          ("Rate review: the 3.00% promotional APY should have applied from "
           "Jan 1, 2024 (bank error).", "3.00%",
           "2024-01-01", "2024-08-15"),
          ("APY later moved to 4.50%.", "4.50%",
           "2025-06-01", "2025-06-02")],
         "2024-02-01", "3.00%"),
        ("retro_addr1", "account.mailing_address",
         "What mailing address applied in fall 2023 per the fraud-dispute review?",
         [("Address on file: 88 Maple Avenue.", "88 Maple Avenue",
           "2023-01-01", "2023-01-03"),
          ("Dispute finding: customer had moved to 5 Birchwood Lane effective "
           "Aug 1, 2023.", "5 Birchwood Lane",
           "2023-08-01", "2024-03-20"),
          ("Customer later moved to 260 Seabreeze Court.", "260 Seabreeze Court",
           "2025-05-01", "2025-05-02")],
         "2023-10-01", "5 Birchwood Lane"),
        ("retro_emp1", "customer.employer",
         "Who was the customer's employer in early 2022 per the corrected loan file?",
         [("Employer: Globex Retail.", "Globex Retail",
           "2022-06-01", "2022-06-02"),
          ("Loan-file correction: customer was at Initech Systems from Jan to "
           "May 2022 before Globex.", "Initech Systems",
           "2022-01-01", "2024-02-15"),
          ("Now employed at Fabrikam Inc.", "Fabrikam Inc",
           "2025-01-01", "2025-01-02")],
         "2022-03-01", "Initech Systems"),
        ("retro_fee1", "account.product_plan",
         "What account plan applied in summer 2024 after the fee-reversal review?",
         [("Plan on file: Premium Checking.", "Premium Checking",
           "2024-01-01", "2024-01-02"),
          ("Fee-reversal review: customer qualified for fee-free Basic "
           "Checking from Jun 1, 2024.", "Basic Checking",
           "2024-06-01", "2025-01-10"),
          ("Later upgraded to Private Client.", "Private Client",
           "2025-03-01", "2025-03-02")],
         "2024-07-15", "Basic Checking"),
        ("retro_nom1", "account.nominee",
         "Who was the account nominee in spring 2023 per the corrected records?",
         [("Nominee: Robert Hale.", "Robert Hale",
           "2020-01-01", "2020-01-02"),
          ("Nominee-change form signed Feb 2023 was processed late; effective "
           "date backdated to Feb 15, 2023.", "Susan Hale",
           "2023-02-15", "2023-11-05"),
          ("Nominee later updated to Daniel Hale.", "Daniel Hale",
           "2024-06-01", "2024-06-02")],
         "2023-04-01", "Susan Hale"),
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

    # Curated set is 20 (15 point-in-time + 5 retroactive). For a larger run,
    # top up procedurally while preserving the same 75/25 pit:retro split.
    if n > len(out):
        rng = random.Random(20260801)
        pit_target = round(n * 0.75)
        retro_target = n - pit_target
        cur_pit = sum(1 for s in out if s.kind == "point_in_time")
        cur_retro = len(out) - cur_pit
        for i in range(max(0, pit_target - cur_pit)):
            out.append(_gen_point_in_time(rng, i))
        for i in range(max(0, retro_target - cur_retro)):
            out.append(_gen_retroactive(rng, i))

    return out[:n] if n < len(out) else out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run(n_scenarios: int) -> tuple[list[BiTemporalScenario], list[_SystemResult]]:
    scenarios = _build_scenarios(n_scenarios)
    log.info("loaded %d banking scenarios (%d point-in-time, %d retroactive)",
             len(scenarios),
             sum(1 for s in scenarios if s.kind == "point_in_time"),
             sum(1 for s in scenarios if s.kind == "retroactive_correction"))
    stats = [
        _score(
            "naive_latest",
            "Ignores as_of; returns the latest recorded value. Answers 'what "
            "is it now', not 'what was it on the statement/audit date'.",
            _query_naive_latest, scenarios,
        ),
        _score(
            "naive_chronological",
            "Latest value whose recorded_at ≤ as_of. Handles forward "
            "point-in-time changes but silently fails backdated corrections — "
            "an audit-trail failure for a bank.",
            _query_naive_chronological, scenarios,
        ),
        _score(
            "continuum_bitemporal",
            "Full bi-temporal lookup on valid_from + recorded_at. Returns the "
            "value in force at as_of regardless of when it was recorded, so "
            "backdated fee reversals / rate corrections / late-processed forms "
            "resolve correctly.",
            _query_continuum_bitemporal, scenarios,
        ),
        _SystemResult(
            system="mem0", available=False, n_total=len(scenarios),
            note="skipped: mem0 has no temporal columns — 'framework not "
                 "applicable' rather than a meaningful score.",
        ),
    ]
    return scenarios, stats


def _narrative(stats: list[_SystemResult]) -> str:
    by = {s.system: s for s in stats if s.available}
    latest = by.get("naive_latest")
    chrono = by.get("naive_chronological")
    cont = by.get("continuum_bitemporal")
    if not (latest and chrono and cont):
        return "(missing data)"
    return (
        f"Across {cont.n_total} banking bi-temporal scenarios "
        f"({cont.n_pit_total} regulatory point-in-time + {cont.n_retro_total} "
        f"backdated corrections), continuum_bitemporal answers "
        f"{cont.correctness:.1%} correctly. naive_latest sits at "
        f"{latest.correctness:.1%} — it reports today's value for questions "
        f"about a statement, audit, or dispute date. naive_chronological "
        f"reaches {chrono.correctness:.1%} on point-in-time but collapses on "
        f"the backdated corrections "
        f"({chrono.n_correct_retroactive}/{chrono.n_retro_total}): a fee "
        f"reversal, promo-rate correction, or late-processed beneficiary form "
        f"carries a recorded_at *after* the queried date, so a single-axis "
        f"store cannot see it and silently reports the wrong history. "
        f"Splitting valid-time from transaction-time is exactly what a bank's "
        f"audit, dispute, and regulatory-reporting flows require."
    )


def _write_results(stats: list[_SystemResult], n_scenarios: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"bi_temporal_banking_{ts}.json"
    cont = next(s for s in stats if s.system == "continuum_bitemporal")
    payload = {
        "benchmark": "bi_temporal_banking",
        "domain": "retail_banking_chatbot",
        "timestamp": ts,
        "config": {"n_scenarios": n_scenarios},
        "systems": [s.summary() for s in stats],
        "narrative": _narrative(stats),
        "acceptance": {
            "bar": f"all queries correct (n={cont.n_total})",
            "target_system": "continuum_bitemporal",
            "passed": cont.n_correct == cont.n_total and cont.n_total >= 20,
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    latest = RESULTS_DIR / "bi_temporal_banking_latest.json"
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
    print("latest:  bench/results/bi_temporal_banking_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
