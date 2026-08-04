"""
bench/supersession_banking.py
=============================
Banking-domain instantiation of the supersession-correctness benchmark.

This is the *same evaluation* as :mod:`bench.supersession_correctness` —
the two memory systems (``NaiveAppendSystem``, ``ContinuumSupersessionSystem``),
the embedder, and the scorer are imported unchanged from that module. **Only
the scenario corpus is swapped**: instead of consumer life-facts (location,
pet, hobby …) the scenarios plant *banking* facts that a bank's customer-
service chatbot must keep current:

* mailing address on file           (statements, cards → mailed to the wrong
                                      place if stale)
* registered mobile for OTP / 2FA   (a stale number sends the one-time code
                                      to someone who no longer owns the line)
* employer on record (KYC / income) (drives lending & AML review)
* account nominee / beneficiary     (a stale nominee is paid out to the wrong
                                      person)
* account product plan / tier       (fee schedule, entitlements)
* autopay default account           (money moves from the wrong account)

Each of these is a place where "surface the *current* fact, never the
superseded one" is not a nicety but a compliance / correctness requirement.
Because the machinery is identical, the banking number is directly
comparable to the general-domain number reported in the paper.

Run
---
::

    python -m bench.supersession_banking --scenarios 50
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import sys
import time
from pathlib import Path

# Reuse the *identical* systems + scorer from the general benchmark. Only the
# corpus below is banking-specific; the evaluation is unchanged.
from bench.supersession_correctness import (
    ContinuumSupersessionSystem,
    NaiveAppendSystem,
    Scenario,
    Turn,
    _print_table,
    _score,
    _SystemResult,
)

log = logging.getLogger("bench.supersession_banking")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Banking scenario corpus
# ---------------------------------------------------------------------------
# Six attribute templates, each with nine (old → mid → new) instantiations.
# Every template plants two contradicting fact turns (old→mid, then mid→new)
# so the *current* answer is the last value and the immediately-superseded
# value is the stale one — mirroring the general benchmark's structure exactly.

_TEMPLATES = [
    {
        "attr": "account.mailing_address",
        "query": "What is the customer's current mailing address on file?",
        "fact_tpl": "Please update my mailing address from {old} to {new}.",
        "transitions": [
            ("12 Oak Street", "88 Maple Avenue", "5 Birchwood Lane"),
            ("400 Pine Road", "27 Cedar Court", "913 Willow Drive"),
            ("15 Elm Street", "62 Aspen Way", "301 Juniper Place"),
            ("7 Harbor View", "44 Lakeshore Blvd", "120 Riverside Terrace"),
            ("9 Sunset Blvd", "76 Highland Ave", "233 Meadow Lane"),
            ("3 Kingston Row", "58 Devon Street", "410 Sherwood Drive"),
            ("21 Camden Close", "99 Baker Street", "17 Marlow Gardens"),
            ("6 Orchard Lane", "82 Chestnut Ave", "150 Poplar Street"),
            ("11 Bayfront Road", "35 Coral Way", "260 Seabreeze Court"),
        ],
    },
    {
        "attr": "account.registered_phone",
        "query": "What is the customer's current registered mobile number for OTP?",
        "fact_tpl": "Change my registered mobile for one-time codes from {old} to {new}.",
        "transitions": [
            ("555-0101", "555-0142", "555-0188"),
            ("555-0203", "555-0247", "555-0299"),
            ("555-0311", "555-0356", "555-0390"),
            ("555-0412", "555-0455", "555-0491"),
            ("555-0513", "555-0558", "555-0597"),
            ("555-0614", "555-0659", "555-0688"),
            ("555-0715", "555-0760", "555-0799"),
            ("555-0816", "555-0861", "555-0895"),
            ("555-0917", "555-0962", "555-0990"),
        ],
    },
    {
        "attr": "customer.employer",
        "query": "Who is the customer's current employer on record?",
        "fact_tpl": "For my KYC update: I left {old} and now work at {new}.",
        "transitions": [
            ("Acme Logistics", "Globex Retail", "Initech Systems"),
            ("Northwind Traders", "Contoso Ltd", "Fabrikam Inc"),
            ("Stark Foods", "Wayne Freight", "Oscorp Media"),
            ("Soylent Foods", "Cyberdyne Labs", "Tyrell Design"),
            ("Massive Dynamic", "Umbrella Retail", "InGen Bio"),
            ("Vandelay Imports", "Kramerica Corp", "Sterling Media"),
            ("Dunder Mifflin", "Vance Cooling", "Schrute Farms"),
            ("Pied Piper", "Aviato", "Hooli Cloud"),
            ("Wonka Industries", "Gekko Capital", "Clampett Oil"),
        ],
    },
    {
        "attr": "account.nominee",
        "query": "Who is the current nominee on the customer's account?",
        "fact_tpl": "I want to change my account nominee from {old} to {new}.",
        "transitions": [
            ("Robert Hale", "Susan Hale", "Daniel Hale"),
            ("Maria Cruz", "Elena Cruz", "Miguel Cruz"),
            ("James Poole", "Karen Poole", "Grace Poole"),
            ("Ahmed Khan", "Sana Khan", "Bilal Khan"),
            ("Wei Chen", "Ling Chen", "Hao Chen"),
            ("Olivia Ford", "Nathan Ford", "Ruby Ford"),
            ("Priya Rao", "Anil Rao", "Kavya Rao"),
            ("Tomas Berg", "Lena Berg", "Erik Berg"),
            ("Grace Owusu", "Kojo Owusu", "Ama Owusu"),
        ],
    },
    {
        "attr": "account.product_plan",
        "query": "What is the customer's current account plan?",
        "fact_tpl": "I'd like to switch my account from {old} to {new}.",
        "transitions": [
            ("Basic Checking", "Premium Checking", "Private Client"),
            ("Student Saver", "Everyday Saver", "Wealth Saver"),
            ("Silver Card", "Gold Card", "Platinum Card"),
            ("Lite Plan", "Plus Plan", "Signature Plan"),
            ("Standard Current", "Advantage Current", "Elite Current"),
            ("Blue Checking", "Preferred Checking", "Reserve Checking"),
            ("Starter Account", "Growth Account", "Prestige Account"),
            ("Core Banking", "Select Banking", "Priority Banking"),
            ("Essential Plan", "Complete Plan", "Ultimate Plan"),
        ],
    },
    {
        "attr": "autopay.default_account",
        "query": "Which account is the customer's current default for autopay?",
        "fact_tpl": "Change my autopay default account from {old} to {new}.",
        "transitions": [
            ("Checking x1234", "Savings x5678", "Checking x9012"),
            ("Card x4111", "Checking x4222", "Savings x4333"),
            ("Savings x7001", "Checking x7002", "Money Market x7003"),
            ("Checking x8100", "Card x8200", "Checking x8300"),
            ("Joint x9001", "Checking x9002", "Savings x9003"),
            ("Card x1500", "Checking x1600", "Card x1700"),
            ("Savings x2100", "Money Market x2200", "Checking x2300"),
            ("Checking x3100", "Savings x3200", "Joint x3300"),
            ("Card x5100", "Checking x5200", "Savings x5300"),
        ],
    },
]

_NOISE_TURNS = [
    "By the way, what are your branch hours this weekend?",
    "Thanks, that's really helpful.",
    "Also — is the mobile app running slow for anyone else today?",
    "No rush on any of this, whenever you get to it.",
    "Quick one: how long do wire transfers usually take?",
    "Appreciate you sorting this out for me.",
]


def _build_scenarios(n: int) -> list[Scenario]:
    """Synthesise ``n`` banking update-then-query scenarios.

    Structure is identical to the general benchmark: pick a template, pick an
    (old, mid, new) instantiation, plant two contradicting fact turns with
    1–3 noise turns interleaved, then ask for the *current* value.
    """
    rng = random.Random(7)
    out: list[Scenario] = []
    idx = 0
    while len(out) < n:
        tpl = _TEMPLATES[idx % len(_TEMPLATES)]
        instances = tpl["transitions"]
        inst = instances[(idx // len(_TEMPLATES)) % len(instances)]

        facts = [
            Turn(text=tpl["fact_tpl"].format(old=a, new=b),
                 attribute_tag=tpl["attr"], value=b)
            for a, b in [(inst[0], inst[1]), (inst[1], inst[2])]
        ]

        turns: list[Turn] = []
        for f in facts:
            for _ in range(rng.randint(1, 3)):
                turns.append(Turn(text=rng.choice(_NOISE_TURNS),
                                  attribute_tag=None, value=None))
            turns.append(f)

        current = facts[-1].value
        stale = tuple(f.value for f in facts[:-1] if f.value)
        out.append(Scenario(
            name=f"{tpl['attr']}#{idx:03d}",
            turns=tuple(turns),
            query=tpl["query"],
            current_answer=current or "",
            stale_answers=stale,
        ))
        idx += 1
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run(n_scenarios: int) -> list[_SystemResult]:
    from sentence_transformers import SentenceTransformer

    log.info("loading embedder all-MiniLM-L6-v2 on CPU …")
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu",
    )
    scenarios = _build_scenarios(n_scenarios)
    log.info("synthesised %d banking scenarios", len(scenarios))

    naive = NaiveAppendSystem(
        name="naive_append",
        note="Append-only fact store + cosine top-1. The 'everyone else' "
             "default: an updated address / phone / nominee never retires the "
             "old one, so cosine can surface a stale value.",
        embedder=embedder,
    )
    continuum = ContinuumSupersessionSystem(
        name="continuum_supersession",
        note="Continuum's LTM schema: superseded_by edges retire the prior "
             "value on the same attribute; retrieval filters to current rows. "
             "Old values stay in history (audit) but never surface as current.",
        embedder=embedder,
    )

    out = [_score(s, scenarios) for s in (naive, continuum)]
    out.append(_SystemResult(
        system="mem0", available=False, n_scenarios=len(scenarios),
        note="skipped: mem0 v1 exposes no supersession primitive (it "
             "overwrites in place, losing the audit trail a bank needs).",
    ))
    return out


def _narrative(stats: list[_SystemResult]) -> str:
    by = {s.system: s for s in stats}
    naive = by.get("naive_append")
    cont = by.get("continuum_supersession")
    if not (naive and cont):
        return "(missing data)"
    delta = (cont.correctness - naive.correctness) * 100
    return (
        f"Across {cont.n_scenarios} scripted banking update-then-query "
        f"scenarios (mailing address, registered OTP number, employer/KYC, "
        f"account nominee, product plan, autopay default), "
        f"continuum_supersession surfaces the *current* value "
        f"{cont.correctness:.1%} of the time vs naive_append at "
        f"{naive.correctness:.1%} — a {delta:+.1f}pp delta. naive_append "
        f"returned a stale (now-superseded) value in "
        f"{naive.n_stale_returned}/{naive.n_scenarios} cases; in a banking "
        f"chatbot each of those is a statement mailed to the old address, an "
        f"OTP sent to a relinquished phone, or a payout to the wrong nominee. "
        f"The supersession filter is a schema-level WHERE clause, not a "
        f"re-rank, so it adds no retrieval cost — and it keeps the superseded "
        f"rows in history for audit rather than deleting them."
    )


def _write_results(stats: list[_SystemResult], n_scenarios: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"supersession_banking_{ts}.json"
    payload = {
        "benchmark": "supersession_banking",
        "domain": "retail_banking_chatbot",
        "timestamp": ts,
        "config": {"n_scenarios": n_scenarios},
        "systems": [s.summary() for s in stats],
        "narrative": _narrative(stats),
        "acceptance": {
            "bar_pct": 95.0,
            "target_system": "continuum_supersession",
            "passed": next(
                (s.correctness * 100 >= 95.0
                 for s in stats if s.system == "continuum_supersession"),
                False,
            ),
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    latest = RESULTS_DIR / "supersession_banking_latest.json"
    with contextlib.suppress(OSError):
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out.name)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=int, default=50,
                   help="Number of update-then-query scenarios (default 50).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    stats = _run(args.scenarios)
    _print_table(stats)
    out = _write_results(stats, args.scenarios)
    print()
    print("NARRATIVE:")
    print(f"  {_narrative(stats)}")
    print()
    cont = next((s for s in stats if s.system == "continuum_supersession"), None)
    if cont:
        bar = 95.0
        flag = "PASS" if cont.correctness * 100 >= bar else "FAIL"
        print(f"ACCEPTANCE: continuum_supersession = {cont.correctness*100:.1f}% "
              f"(bar {bar}%) — {flag}")
    print(f"results: {out.relative_to(Path(__file__).resolve().parents[1])}")
    print("latest:  bench/results/supersession_banking_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
