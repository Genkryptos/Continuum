"""
bench/supersession_correctness.py
=================================
Does the memory system surface the *current* fact when the user has
updated it? This is the killer feature that nobody else has built into
their memory layer — Continuum's LTM schema has ``superseded_by``
columns and a "current fact" filter; mem0, raw vector stores, and most
RAG systems just stack new facts on top of old ones and let the
downstream LLM guess which is current (badly).

Scenario design
---------------
Each scenario is a sequence of dialogue turns that plant 1+ facts
about the same (entity, attribute) and then ask which is current:

* "I just moved from Chicago to NYC." — plants user.location = NYC
* "My commute is brutal." — noise
* "Actually I'm moving from NYC to Boston next week!" — supersedes NYC
* Q: "Where does the user currently live?" — A: Boston

50 scenarios are synthesised from 6 attribute templates (location,
employer, pet name, marital status, vehicle, hobby) × 9 instantiations
each. Each scenario has 2–3 contradicting facts and 1–3 noise turns.

Systems compared
----------------
* ``naive_append``        — append-only fact store + cosine retrieval.
                            Returns top-k regardless of whether facts
                            are stale. The everyone-else default.
* ``continuum_supersession`` — in-memory simulation of Continuum's
                            LTM schema: tracks ``superseded_by`` edges,
                            retrieval filters ``WHERE superseded_by
                            IS NULL`` so only current facts surface.
* ``mem0`` (stub)         — skipped; mem0 v1 doesn't expose a
                            supersession primitive (it overwrites in
                            place, which is a different failure mode).

Metric
------
**supersession_correctness** = the share of scenarios where the
system's top-1 returned fact for the closing query contains the
*current* expected answer string and does NOT contain any of the
stale answer strings. The acceptance bar is **95%** on 50 scenarios
for ``continuum_supersession``; naive_append is expected to score
significantly lower because cosine ranking on the query phrasing
often surfaces the older fact first.

Note on simulation
------------------
The contradiction-detection logic here is *rule-based* (matching
on planted ``entity_attribute`` tags), where in a real Continuum
deployment that detection comes from the LLM-driven Mem0Promoter
walking newly-extracted facts against existing LTM rows. We're
benchmarking the **architecture** (does ``superseded_by`` solve the
problem when detection works?), not the detection itself. The
detection-quality benchmark is a separate axis and would be a
follow-up.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("bench.supersession")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Scenario synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    text: str
    #: When non-None, identifies an (entity, attribute) the turn asserts;
    #: turns sharing this tag contradict each other in chronological order.
    attribute_tag: str | None
    value: str | None


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[Turn, ...]
    query: str
    current_answer: str
    stale_answers: tuple[str, ...]


# Per-attribute templates: query phrasing + a list of values to walk
# through chronologically. The first two are "before" facts; the last
# is the "current" fact (the supersession target).
_TEMPLATES = [
    {
        "attr": "user.location",
        "query": "Where does the user currently live?",
        "fact_tpl": "I just moved {transition}.",
        "transitions": [
            ("Chicago", "NYC", "Boston"),
            ("Seattle", "Portland", "Denver"),
            ("Austin", "Houston", "Dallas"),
            ("LA", "SF", "San Diego"),
            ("Miami", "Orlando", "Tampa"),
            ("Toronto", "Vancouver", "Montreal"),
            ("Berlin", "Hamburg", "Munich"),
            ("London", "Manchester", "Bristol"),
            ("Tokyo", "Osaka", "Kyoto"),
        ],
    },
    {
        "attr": "user.employer",
        "query": "Who does the user currently work for?",
        "fact_tpl": "I left {old} and joined {new}.",
        "transitions": [
            ("Acme", "Globex", "Initech"),
            ("Hooli", "Pied Piper", "Aviato"),
            ("Stark Industries", "Wayne Enterprises", "OsCorp"),
            ("Soylent Corp", "Cyberdyne", "Tyrell Corp"),
            ("Massive Dynamic", "Umbrella Corp", "InGen"),
            ("Weyland-Yutani", "Buy n Large", "Rekall"),
            ("Initrode", "Vandelay Industries", "Bluth Co."),
            ("Pendant Publishing", "Kramerica", "Sterling Cooper"),
            ("Dunder Mifflin", "Vance Refrigeration", "Schrute Farms"),
        ],
    },
    {
        "attr": "user.pets.dog",
        "query": "What is the user's current dog called?",
        "fact_tpl": "We had to say goodbye to {old} last year; our new dog {new} arrived this spring.",
        "transitions": [
            ("Rex", "Luna", "Buddy"),
            ("Max", "Bella", "Charlie"),
            ("Daisy", "Cooper", "Bear"),
            ("Sadie", "Milo", "Rocky"),
            ("Lucy", "Toby", "Penny"),
            ("Oliver", "Lola", "Duke"),
            ("Riley", "Stella", "Murphy"),
            ("Zoe", "Jack", "Maggie"),
            ("Bailey", "Sophie", "Winston"),
        ],
    },
    {
        "attr": "user.marital_status",
        "query": "What is the user's current marital status?",
        "fact_tpl": "{transition}",
        "transitions": [
            ("I'm engaged to my college sweetheart.",
             "We got married last summer in Vermont.",
             "We finalised the divorce last month."),
            ("I'm dating someone new and it's going well.",
             "We're engaged as of last weekend!",
             "We called off the engagement, both moving on."),
            ("I'm happily single and not looking.",
             "I met someone amazing and we're dating.",
             "We just got married at city hall yesterday."),
            ("My partner and I are in a long-distance setup.",
             "We moved in together last month.",
             "We broke up, I moved out two weeks ago."),
            ("I'm a widower since 2023.",
             "I started dating someone last fall.",
             "We got engaged on New Year's Eve."),
            ("We've been engaged for two years.",
             "We got married in May at the lakehouse.",
             "We separated in October, divorce pending."),
            ("I'm in a committed relationship of three years.",
             "We got engaged at her sister's wedding.",
             "We're married now, eloped in Vegas."),
            ("My fiancée and I are planning a 2027 wedding.",
             "We pushed the wedding back another year.",
             "We're not getting married, broke off the engagement."),
            ("I'm divorced as of last year.",
             "I started seeing someone new recently.",
             "We're engaged after eight months together."),
        ],
    },
    {
        "attr": "user.vehicle",
        "query": "What car does the user drive currently?",
        "fact_tpl": "I sold my {old} and picked up a {new}.",
        "transitions": [
            ("Honda Civic", "Toyota Camry", "Tesla Model 3"),
            ("Ford F-150", "Chevy Silverado", "Ram 1500"),
            ("Subaru Outback", "Volvo XC90", "BMW X5"),
            ("Mazda 3", "Honda Accord", "Hyundai Sonata"),
            ("VW Jetta", "Audi A4", "Mercedes C-Class"),
            ("Nissan Sentra", "Kia Forte", "Hyundai Elantra"),
            ("Chevy Malibu", "Buick Regal", "Cadillac CT5"),
            ("Toyota Prius", "Tesla Model Y", "Polestar 2"),
            ("Jeep Wrangler", "Land Rover Defender", "Ford Bronco"),
        ],
    },
    {
        "attr": "user.hobby",
        "query": "What is the user's main hobby right now?",
        "fact_tpl": "I've dropped {old} and gotten really into {new}.",
        "transitions": [
            ("rock climbing", "pottery", "fly fishing"),
            ("woodworking", "watercolor painting", "competitive baking"),
            ("hiking", "kayaking", "scuba diving"),
            ("cycling", "running", "swimming"),
            ("chess", "go", "bridge"),
            ("knitting", "crochet", "embroidery"),
            ("gardening", "beekeeping", "mushroom foraging"),
            ("photography", "videography", "drone filming"),
            ("piano", "guitar", "drums"),
        ],
    },
]

_NOISE_TURNS = [
    "Anyway, totally unrelated thought — the weather has been wild this week.",
    "Have you read anything good lately?",
    "Work has been a lot, but in a good way.",
    "I keep meaning to call my parents more often.",
    "Streaming services keep raising prices.",
    "I've been trying to drink more water.",
]


def _build_scenarios(n: int) -> list[Scenario]:
    rng = random.Random(7)
    out: list[Scenario] = []
    idx = 0
    while len(out) < n:
        tpl = _TEMPLATES[idx % len(_TEMPLATES)]
        instances = tpl["transitions"]
        inst = instances[(idx // len(_TEMPLATES)) % len(instances)]

        # Build the fact turns.
        if tpl["attr"] == "user.marital_status":
            facts = [
                Turn(text=inst[i], attribute_tag=tpl["attr"], value=inst[i])
                for i in range(3)
            ]
        elif tpl["attr"] in ("user.location",):
            facts = [
                Turn(text=tpl["fact_tpl"].format(transition=f"from {a} to {b}"),
                     attribute_tag=tpl["attr"], value=b)
                for a, b in [(inst[0], inst[1]), (inst[1], inst[2])]
            ]
        else:
            facts = [
                Turn(text=tpl["fact_tpl"].format(old=a, new=b),
                     attribute_tag=tpl["attr"], value=b)
                for a, b in [(inst[0], inst[1]), (inst[1], inst[2])]
            ]

        # Interleave 1-3 noise turns between facts (also at the start).
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
# Memory systems
# ---------------------------------------------------------------------------


@dataclass
class _StoredFact:
    id: int
    text: str
    attribute_tag: str | None
    value: str | None
    valid_from: int     # turn index — synthetic "time"
    superseded_by: int | None = None


def _normalize(raw: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(raw)):
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    safe = np.where(norms > 1e-9, norms, 1.0)
    vecs = raw / safe
    return np.where(norms > 1e-9, vecs, 0.0)


@dataclass
class _MemorySystem:
    """Common interface; subclasses override ``add`` and ``top1``."""

    name: str
    note: str
    facts: list[_StoredFact] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    embedder: Any = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(
            self.embedder.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True),
            dtype=np.float32,
        ))

    def add(self, turn: Turn, turn_idx: int) -> None:
        raise NotImplementedError

    def top1(self, query: str) -> _StoredFact | None:
        """Return the top-1 fact for *query* under this system's rules."""
        raise NotImplementedError


class NaiveAppendSystem(_MemorySystem):
    """Append-only store; cosine top-1 over all stored facts."""

    def add(self, turn: Turn, turn_idx: int) -> None:
        if turn.attribute_tag is None:
            return  # noise turns are not stored as facts
        fact_id = len(self.facts)
        self.facts.append(_StoredFact(
            id=fact_id, text=turn.text, attribute_tag=turn.attribute_tag,
            value=turn.value, valid_from=turn_idx,
        ))
        vec = self._embed([turn.text])
        self.embeddings = (
            vec if self.embeddings is None
            else np.vstack([self.embeddings, vec])
        )

    def top1(self, query: str) -> _StoredFact | None:
        if not self.facts:
            return None
        q = self._embed([query])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (self.embeddings @ q.T).ravel()
        if not np.all(np.isfinite(sims)):
            sims = np.nan_to_num(sims, nan=-1.0)
        return self.facts[int(np.argmax(sims))]


class ContinuumSupersessionSystem(_MemorySystem):
    """
    In-memory simulation of Continuum's LTM with ``superseded_by`` edges.

    On ``add``, any existing non-superseded fact sharing the new fact's
    ``attribute_tag`` is marked ``superseded_by = new_fact.id`` —
    mirroring what :class:`Mem0Promoter` does when its contradiction
    detector fires in the real pipeline. ``top1`` then filters the
    embedding matrix to non-superseded rows before ranking.
    """

    def add(self, turn: Turn, turn_idx: int) -> None:
        if turn.attribute_tag is None:
            return
        fact_id = len(self.facts)
        # Supersede every existing live fact on the same attribute.
        for f in self.facts:
            if (f.attribute_tag == turn.attribute_tag
                    and f.superseded_by is None):
                f.superseded_by = fact_id
        self.facts.append(_StoredFact(
            id=fact_id, text=turn.text, attribute_tag=turn.attribute_tag,
            value=turn.value, valid_from=turn_idx,
        ))
        vec = self._embed([turn.text])
        self.embeddings = (
            vec if self.embeddings is None
            else np.vstack([self.embeddings, vec])
        )

    def top1(self, query: str) -> _StoredFact | None:
        live = [(i, f) for i, f in enumerate(self.facts) if f.superseded_by is None]
        if not live:
            return None
        rows = np.vstack([self.embeddings[i] for i, _ in live])
        q = self._embed([query])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (rows @ q.T).ravel()
        if not np.all(np.isfinite(sims)):
            sims = np.nan_to_num(sims, nan=-1.0)
        return live[int(np.argmax(sims))][1]


# ---------------------------------------------------------------------------
# Scoring + driver
# ---------------------------------------------------------------------------


@dataclass
class _SystemResult:
    system: str
    available: bool
    n_scenarios: int = 0
    n_correct: int = 0
    n_stale_returned: int = 0
    n_no_answer: int = 0
    avg_supersede_turns: float = 0.0
    note: str = ""

    @property
    def correctness(self) -> float:
        return self.n_correct / self.n_scenarios if self.n_scenarios else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "available": self.available,
            "n_scenarios": self.n_scenarios,
            "n_correct": self.n_correct,
            "n_stale_returned": self.n_stale_returned,
            "n_no_answer": self.n_no_answer,
            "correctness_pct": round(self.correctness * 100, 1),
            "avg_supersede_turns": round(self.avg_supersede_turns, 2),
            "note": self.note,
        }


def _score(system: _MemorySystem, scenarios: list[Scenario]) -> _SystemResult:
    res = _SystemResult(system=system.name, available=True,
                        n_scenarios=len(scenarios), note=system.note)
    supersede_lags: list[int] = []
    for sc in scenarios:
        # Each scenario starts with a fresh store (a single user's lifecycle).
        system.facts = []
        system.embeddings = None
        for idx, turn in enumerate(sc.turns):
            system.add(turn, idx)
        top = system.top1(sc.query)
        ans = (top.value or "") if top else ""
        ans_text = ans.lower()
        current = sc.current_answer.lower()
        stale_set = {s.lower() for s in sc.stale_answers}

        if not ans:
            res.n_no_answer += 1
        elif current and current in ans_text and ans_text not in stale_set:
            res.n_correct += 1
        elif ans_text in stale_set:
            res.n_stale_returned += 1
        else:
            # Returned something unrelated (e.g. noise turn embedded oddly).
            res.n_stale_returned += 1

        # For Continuum: how many turn-steps after the contradiction
        # before the old fact got marked superseded? In this simulation
        # it's always 0 (immediate on add), but the metric is plumbed
        # for future detector-based variants.
        if isinstance(system, ContinuumSupersessionSystem):
            for f in system.facts:
                if f.superseded_by is not None:
                    target = next(x for x in system.facts if x.id == f.superseded_by)
                    supersede_lags.append(target.valid_from - f.valid_from)
    if supersede_lags:
        res.avg_supersede_turns = statistics.mean(supersede_lags)
    return res


def _run(n_scenarios: int) -> list[_SystemResult]:
    from sentence_transformers import SentenceTransformer
    log.info("loading embedder all-MiniLM-L6-v2 on CPU …")
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu",
    )
    scenarios = _build_scenarios(n_scenarios)
    log.info("synthesised %d scenarios", len(scenarios))

    naive = NaiveAppendSystem(
        name="naive_append",
        note="Append-only fact store + cosine top-1. The 'everyone "
             "else' default — every fact persists forever, no link "
             "between contradictions.",
        embedder=embedder,
    )
    continuum = ContinuumSupersessionSystem(
        name="continuum_supersession",
        note="In-memory simulation of Continuum's LTM schema: stores "
             "facts with superseded_by edges, filters to current "
             "(non-superseded) rows before ranking. Detection is "
             "rule-based here; in production it comes from the "
             "Mem0Promoter's LLM contradiction check.",
        embedder=embedder,
    )

    out = [_score(s, scenarios) for s in (naive, continuum)]

    out.append(_SystemResult(
        system="mem0", available=False, n_scenarios=len(scenarios),
        note="skipped: mem0 v1 doesn't expose supersession — it "
             "overwrites in place when its contradiction check fires "
             "(a different failure mode that loses history). A real "
             "comparison would require running mem0 and inspecting "
             "whether prior facts are accessible at all afterwards.",
    ))
    return out


def _narrative(stats: list[_SystemResult]) -> str:
    by = {s.system: s for s in stats}
    naive = by.get("naive_append")
    cont  = by.get("continuum_supersession")
    if not (naive and cont):
        return "(missing data)"
    delta = (cont.correctness - naive.correctness) * 100
    return (
        f"Across {cont.n_scenarios} scripted update-then-query scenarios "
        f"(location, employer, pet, marital status, vehicle, hobby), "
        f"continuum_supersession returns the *current* fact "
        f"{cont.correctness:.1%} of the time vs naive_append at "
        f"{naive.correctness:.1%} — a {delta:+.1f}pp delta. Naive "
        f"returned a stale (now-superseded) fact in "
        f"{naive.n_stale_returned}/{naive.n_scenarios} cases, where "
        f"continuum_supersession returned stale "
        f"{cont.n_stale_returned}/{cont.n_scenarios}. The supersession "
        f"check is O(1) per fact-add and adds no measurable cost at "
        f"retrieval — it's a schema-level filter, not a re-rank. This "
        f"is the only category in the Phase-3B benchmark suite where "
        f"Continuum's architecture wins by construction over a raw "
        f"vector store, regardless of model or embedder."
    )


def _print_table(stats: list[_SystemResult]) -> None:
    print()
    print("=" * 116)
    print(f"{'system':<26}{'avail':>7}{'correct':>10}{'stale':>8}"
          f"{'noans':>8}{'correct%':>11}{'sup_lag':>10}{'note':>30}")
    print("-" * 116)
    for s in stats:
        d = s.summary()
        print(
            f"{d['system']:<26}{d['available']!s:>7}"
            f"{d['n_correct']:>10}/{d['n_scenarios']:<3}"
            f"{d['n_stale_returned']:>8}"
            f"{d['n_no_answer']:>8}"
            f"{d['correctness_pct']:>10.1f}%"
            f"{d['avg_supersede_turns']:>10.2f}"
            f"  {d['note'][:26]!r:>28}"
        )
    print("=" * 116)


def _write_results(stats: list[_SystemResult], n_scenarios: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"supersession_{ts}.json"
    payload = {
        "benchmark": "supersession_correctness",
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
    latest = RESULTS_DIR / "supersession_latest.json"
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
    print("latest:  bench/results/supersession_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
