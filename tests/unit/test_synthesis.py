"""
tests/unit/test_synthesis.py
============================
The v3 synthesis aggregator (continuum.promotion.synthesis) — deterministic
counting that the reader (and an LLM synthesizer) get wrong. Cases mirror the
real LongMemEval-S v2.0 failures: H&M tops (reader said 3, answer 5), postcards
(reader said 50, answer 25 — a scoping error), tanks, etc.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from continuum.promotion.synthesis import DerivedFact, StructuredFact, aggregate

pytestmark = pytest.mark.unit


def _f(pred: str, obj: str, when: datetime | None = None) -> StructuredFact:
    return StructuredFact(subject="user", predicate=pred, obj=obj, occurred_at=when)


# ── basic counting (the core fix) ─────────────────────────────────────────────


def test_counts_distinct_members() -> None:
    facts = [_f("tank", "goldfish tank"), _f("tank", "shrimp tank"), _f("tank", "betta tank")]
    (d,) = aggregate(facts)
    assert d.count == 3
    assert d.text == "User has 3 tanks."


def test_dedup_same_member_counted_once() -> None:
    # "goldfish tank" mentioned twice across sessions → still 2 distinct tanks.
    facts = [_f("tank", "Goldfish tank"), _f("tank", "goldfish  tank"), _f("tank", "shrimp tank")]
    (d,) = aggregate(facts)
    assert d.count == 2
    assert sorted(m.lower() for m in d.members) == ["goldfish tank", "shrimp tank"]


def test_hm_tops_real_failure_case() -> None:
    # v2.0: reader answered "three", true answer was five. Code counts five.
    tops = [_f("top from H&M", f"top #{i}") for i in range(1, 6)]
    (d,) = aggregate(tops)
    assert d.count == 5
    assert d.text == "User has 5 tops from H&M."  # head-noun pluralized


def test_singular_render() -> None:
    (d,) = aggregate([_f("tank", "the only tank")])
    assert d.count == 1
    assert d.text == "User has 1 tank."


# ── time scoping (the postcards "since I started" failure) ────────────────────


def test_scoped_count_since_cutoff() -> None:
    # 50 postcards total, but only 25 added since the user "started collecting".
    start = datetime(2023, 1, 1)
    old = [_f("postcard", f"old #{i}", datetime(2022, 6, 1)) for i in range(25)]
    new = [_f("postcard", f"new #{i}", datetime(2023, 3, 1)) for i in range(25)]
    facts = old + new

    (total,) = aggregate(facts)
    assert total.count == 50  # unscoped = everything

    (scoped,) = aggregate(facts, since=start)
    assert scoped.count == 25  # only the ones since the start point
    assert "since 2023-01-01" in scoped.text


def test_scoping_excludes_undated_facts() -> None:
    facts = [_f("trip", "Paris", datetime(2023, 5, 1)), _f("trip", "no-date trip", None)]
    (d,) = aggregate(facts, since=datetime(2023, 1, 1))
    assert d.count == 1  # the undated one can't be placed in the window


# ── grouping + thresholds ─────────────────────────────────────────────────────


def test_separate_predicates_grouped_independently() -> None:
    facts = [_f("tank", "a"), _f("tank", "b"), _f("coin", "x")]
    out = {d.predicate: d.count for d in aggregate(facts)}
    assert out == {"tank": 2, "coin": 1}


def test_min_count_filters_small_groups() -> None:
    facts = [_f("tank", "a"), _f("tank", "b"), _f("coin", "x")]
    out = aggregate(facts, min_count=2)
    assert [d.predicate for d in out] == ["tank"]  # the singleton 'coin' dropped


def test_no_dedup_option_counts_raw() -> None:
    facts = [_f("tank", "goldfish tank"), _f("tank", "goldfish tank")]
    assert aggregate(facts)[0].count == 1            # dedup default
    assert aggregate(facts, dedup=False)[0].count == 2


def test_empty_input() -> None:
    assert aggregate([]) == []


def test_derived_fact_is_constructable() -> None:
    d = DerivedFact(subject="user", predicate="book", count=3, members=["a", "b", "c"])
    assert d.text == "User has 3 books."
