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
from pathlib import Path

import pytest

from continuum.core.types import MemoryTier
from continuum.promotion.synthesis import (
    ENTITY_SUMMARY_KIND,
    DerivedFact,
    StructuredFact,
    aggregate,
    disk_cached,
    extract_structured_facts,
    is_counting_question,
    is_scoped_count_question,
    relevant_summaries,
    route_count_answer,
    to_memory_item,
)

pytestmark = pytest.mark.unit


def _fake_llm(reply: str):
    async def fn(prompt: str) -> str:
        return reply

    return fn


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
    assert aggregate(facts)[0].count == 1  # dedup default
    assert aggregate(facts, dedup=False)[0].count == 2


def test_empty_input() -> None:
    assert aggregate([]) == []


# ── v3.1: SUM aggregation (the "how many hours total" failures) ────────────────


def _q(pred: str, obj: str, qty: float, unit: str) -> StructuredFact:
    return StructuredFact(subject="user", predicate=pred, obj=obj, quantity=qty, unit=unit)


def test_sums_quantities() -> None:
    # "how many hours total playing games" → 140, not a count of sessions.
    facts = [
        _q("gaming session", "Mon", 40, "hours"),
        _q("gaming session", "Tue", 60, "hours"),
        _q("gaming session", "Wed", 40, "hours"),
    ]
    (d,) = aggregate(facts)
    assert d.count == 3
    assert d.total == 140
    assert d.text == "User has 3 gaming sessions totaling 140 hours."


def test_no_quantity_no_total() -> None:
    (d,) = aggregate([_f("tank", "a"), _f("tank", "b")])
    assert d.total is None
    assert "totaling" not in d.text


# ── v3.1: relevance filtering (the "40-summary flood" fix) ─────────────────────


def test_relevant_summaries_matches_predicate() -> None:
    facts = aggregate(
        [_f("tank", "a"), _f("tank", "b"), _f("coin", "x"), _f("coin", "y"), _f("book", "z")]
    )
    chosen = relevant_summaries(facts, "How many tanks do I have?")
    assert [f.predicate for f in chosen] == ["tank"]  # only the relevant one


def test_relevant_summaries_multiword_predicate() -> None:
    facts = aggregate([_f("citrus fruit", "lemon"), _f("citrus fruit", "lime")])
    chosen = relevant_summaries(facts, "How many citrus fruits have I used?")
    assert len(chosen) == 1 and chosen[0].predicate == "citrus fruit"


def test_relevant_summaries_fallback_when_no_match() -> None:
    # nothing matches by name → fall back to the largest count>=2 groups.
    facts = aggregate([_f("tank", "a"), _f("tank", "b"), _f("coin", "x")])
    chosen = relevant_summaries(facts, "How many widgets do I own?")
    assert [f.predicate for f in chosen] == ["tank"]  # count 2; 'coin' (1) dropped


async def test_extract_carries_quantity() -> None:
    reply = (
        '{"facts": [{"predicate": "session", "object": "Mon", "quantity": 2.5, "unit": "hours"}]}'
    )
    facts = await extract_structured_facts("played 2.5h", completion_fn=_fake_llm(reply))
    assert facts[0].quantity == 2.5 and facts[0].unit == "hours"


# ── disk cache (save extraction → no repeat credits) ──────────────────────────


async def test_disk_cache_serves_repeats_without_calling_llm(tmp_path: Path) -> None:
    calls = {"n": 0}

    async def counting_fn(prompt: str) -> str:
        calls["n"] += 1
        return f"resp:{prompt}"

    cached = disk_cached(counting_fn, tmp_path, namespace="gpt-4o-mini")
    a = await cached("hello")
    b = await cached("hello")  # served from disk → no second call
    assert a == b == "resp:hello"
    assert calls["n"] == 1  # the LLM was hit exactly once
    assert len(list(tmp_path.glob("*.txt"))) == 1  # one cached file written

    # a fresh wrapper over the SAME dir still hits the cache (persisted to disk)
    calls["n"] = 0
    cached2 = disk_cached(counting_fn, tmp_path, namespace="gpt-4o-mini")
    assert await cached2("hello") == "resp:hello"
    assert calls["n"] == 0  # zero credits on re-run


async def test_disk_cache_namespace_separates(tmp_path: Path) -> None:
    calls = {"n": 0}

    async def counting_fn(prompt: str) -> str:
        calls["n"] += 1
        return "x"

    await disk_cached(counting_fn, tmp_path, namespace="model-a")("p")
    await disk_cached(counting_fn, tmp_path, namespace="model-b")("p")  # different model → miss
    assert calls["n"] == 2


def test_derived_fact_is_constructable() -> None:
    d = DerivedFact(subject="user", predicate="book", count=3, members=["a", "b", "c"])
    assert d.text == "User has 3 books."


# ── LTM item + counting-question detection ────────────────────────────────────


@pytest.mark.parametrize(
    "q",
    [
        "How many tanks do I have?",
        "How much time do I dedicate to coding each day?",
        "How long have I lived in Harajuku?",
        "What is the total number of coins in my collection?",
    ],
)
def test_is_counting_question_positive(q: str) -> None:
    assert is_counting_question(q) is True


@pytest.mark.parametrize(
    "q",
    ["What is my favorite rice?", "Where do I live now?", "Did I move to Boston?", ""],
)
def test_is_counting_question_negative(q: str) -> None:
    assert is_counting_question(q) is False


@pytest.mark.parametrize(
    "q",
    [
        "How many new postcards have I added since I started collecting again?",
        "How many weddings have I attended this year?",
        "How many doctor's appointments did I go to in March?",
        "How many times did I bake in the past two weeks?",
        "How many art events did I attend in the past month?",
        "How many weeks ago did I attend the festival?",
    ],
)
def test_scoped_count_question_positive(q: str) -> None:
    # bounded time window → synthesis all-time total must NOT be injected.
    assert is_scoped_count_question(q) is True


@pytest.mark.parametrize(
    "q",
    [
        "How many tops have I bought from H&M so far?",  # unbounded total
        "How many tanks do I currently have?",  # current total
        "How many model kits have I worked on or bought?",  # lifetime total
        "How many fitness classes do I attend in a typical week?",  # rate, not window
        "How many pre-1920 American coins do I have?",  # attribute filter, not time
        "How many sports have I played competitively in the past?",  # 'in the past' w/o unit = ever
    ],
)
def test_scoped_count_question_negative(q: str) -> None:
    # unbounded totals → synthesis IS correct, keep injecting.
    assert is_scoped_count_question(q) is False


# ── structured-triple extraction (fake LLM) ───────────────────────────────────


async def test_extract_parses_triples() -> None:
    reply = '{"facts": [{"predicate": "Tank", "object": "goldfish tank"}, {"predicate": "tank", "object": "shrimp tank"}]}'
    facts = await extract_structured_facts("I set up two tanks", completion_fn=_fake_llm(reply))
    assert len(facts) == 2
    assert all(f.predicate == "tank" for f in facts)  # normalized (lowercased)
    assert {f.obj for f in facts} == {"goldfish tank", "shrimp tank"}


async def test_extract_strips_markdown_fence() -> None:
    reply = '```json\n{"facts": [{"predicate": "coin", "object": "1909 penny"}]}\n```'
    facts = await extract_structured_facts("a coin", completion_fn=_fake_llm(reply))
    assert len(facts) == 1 and facts[0].predicate == "coin"


async def test_extract_graceful_on_garbage() -> None:
    facts = await extract_structured_facts("x", completion_fn=_fake_llm("not json at all"))
    assert facts == []


async def test_extract_salvages_malformed_json() -> None:
    # gpt-4o-mini emits an unescaped quote in ONE record (model name) that breaks
    # json.loads for the whole response. Salvage must recover the other records
    # instead of dropping the entire session (the H&M-3-vs-5 undercount cause).
    reply = (
        '{"facts": ['
        '{"predicate": "model kit", "object": "Revell F-15"},'
        '{"predicate": "model kit", "object": "Tamiya "Spitfire" 1/48"},'  # bad inner quotes
        '{"predicate": "model kit", "object": "Airfix Mustang"},'
        '{"predicate": "model kit", "object": "Bandai Gundam"},'
        '{"predicate": "model kit", "object": "Italeri Tank"}]}'
    )
    facts = await extract_structured_facts("model kits", completion_fn=_fake_llm(reply))
    (d,) = aggregate(facts)
    # All 5 members recovered (the malformed one as a truncated-but-distinct
    # member) — for COUNTING, a real kit is one kit regardless of name mangling.
    # The point: we don't drop the whole session to 0 over one bad record.
    assert d.count == 5
    assert d.predicate == "model kit"


async def test_extract_salvage_carries_quantity() -> None:
    reply = (
        '{"facts": ['
        '{"predicate": "session", "object": "Mon", "quantity": 40, "unit": "hours"},'
        '{"predicate": "session", "object": "bad "x" record"},'  # malformed → triggers salvage
        '{"predicate": "session", "object": "Tue", "quantity": 60, "unit": "hours"}]}'
    )
    facts = await extract_structured_facts("sessions", completion_fn=_fake_llm(reply))
    (d,) = aggregate(facts)
    assert d.count == 3  # all three members survive salvage
    assert d.total == 100  # 40 + 60 quantities preserved through salvage


async def test_salvage_skips_record_with_no_object_field() -> None:
    # a record with no parseable object is the only one genuinely dropped.
    reply = (
        '{"facts": ['
        '{"predicate": "tank", "object": "goldfish"},'
        '{"predicate": "tank"},'  # no object → skipped
        '{"predicate": "tank", "object": "shrimp"} garbage trailing'
    )
    facts = await extract_structured_facts("tanks", completion_fn=_fake_llm(reply))
    (d,) = aggregate(facts)
    assert d.count == 2  # goldfish + shrimp; the object-less record dropped


async def test_extract_empty_text() -> None:
    facts = await extract_structured_facts("   ", completion_fn=_fake_llm('{"facts":[]}'))
    assert facts == []


async def test_extract_then_aggregate_end_to_end() -> None:
    # the full v3 logic path: extract triples → count distinct → correct count.
    reply = (
        '{"facts": ['
        '{"predicate": "top", "object": "black top"},'
        '{"predicate": "top", "object": "white top"},'
        '{"predicate": "top", "object": "striped top"},'
        '{"predicate": "top", "object": "denim top"},'
        '{"predicate": "top", "object": "linen top"}]}'
    )
    facts = await extract_structured_facts("H&M haul", completion_fn=_fake_llm(reply))
    (derived,) = aggregate(facts)
    assert derived.count == 5
    assert derived.text == "User has 5 tops."


# ── deterministic answer router (bypass the reader) ───────────────────────────


def test_router_returns_bare_count() -> None:
    facts = aggregate([_f("tank", "a"), _f("tank", "b"), _f("tank", "c")])
    assert route_count_answer(facts, "How many tanks do I have?") == "3"


def test_router_returns_total_for_sum_intent() -> None:
    facts = aggregate(
        [_q("gaming session", "Mon", 40, "hours"), _q("gaming session", "Tue", 100, "hours")]
    )
    # "how many hours total" → the SUM with unit, not the session count.
    # (predicate words must appear in the question for the strict match.)
    assert (
        route_count_answer(facts, "How many hours total across my gaming sessions?") == "140 hours"
    )


def test_router_declines_on_predicate_mismatch() -> None:
    # Conservative by design: when the question vocabulary ("games") doesn't match
    # the extracted predicate ("gaming session"), the router returns None and the
    # reader takes over — never a confident wrong answer. (Better extraction with
    # consistent predicates is what closes this gap.)
    facts = aggregate([_q("gaming session", "Mon", 40, "hours")])
    assert route_count_answer(facts, "How many hours have I played games?") is None


def test_router_none_when_no_match() -> None:
    facts = aggregate([_f("tank", "a"), _f("tank", "b")])
    # nothing matches "widgets" → None, so the caller falls back to the reader.
    assert route_count_answer(facts, "How many widgets do I own?") is None


def test_router_none_when_ambiguous() -> None:
    facts = aggregate([_f("tank", "a"), _f("tank", "b"), _f("coin", "x"), _f("coin", "y")])
    # both 'tank' and 'coin' would need to match for ambiguity; here only one
    # matches, so confirm a CLEAN single match still routes…
    assert route_count_answer(facts, "How many tanks?") == "2"
    # …and a question naming two tracked predicates is ambiguous → None.
    assert route_count_answer(facts, "How many tanks and coins?") is None


def test_router_none_for_scoped_question() -> None:
    facts = aggregate([_f("postcard", f"#{i}") for i in range(33)])
    # "since I started again" is a bounded window → all-time total is wrong → None.
    q = "How many postcards have I added since I started collecting again?"
    assert route_count_answer(facts, q) is None


def test_router_none_on_empty() -> None:
    assert route_count_answer([], "How many tanks?") is None


def test_router_declines_non_count_questions() -> None:
    # THE full-500 bug: the router fired on duration/recall questions where a
    # predicate coincidentally matched, returning a garbage count. It must now
    # decline anything that isn't an explicit "how many / number of / how much $".
    facts = aggregate([_f("session", "a"), _f("session", "b")])
    assert route_count_answer(facts, "How long is my commute by session?") is None  # duration
    studio = aggregate([_f("studio", "a"), _f("studio", "b")])
    assert route_count_answer(studio, "What yoga studio do I attend?") is None  # recall
    assert route_count_answer(studio, "How often do I visit the studio?") is None  # frequency


def test_router_declines_count_of_one() -> None:
    # a single matched member is usually a spurious match → defer to the reader.
    facts = aggregate([_f("tank", "only one")])
    assert route_count_answer(facts, "How many tanks do I have?") is None


def test_router_money_sum() -> None:
    facts = aggregate([_q("workshop", "a", 300, "dollars"), _q("workshop", "b", 420, "dollars")])
    assert route_count_answer(facts, "How much did I spend on workshops?") == "720 dollars"


def test_to_memory_item_tags_entity_summary() -> None:
    d = DerivedFact(subject="user", predicate="tank", count=3, members=["a", "b", "c"])
    item = to_memory_item(d, session_id="s", embedding=[0.1, 0.2])
    assert item.tier == MemoryTier.LTM
    assert item.content == "User has 3 tanks."
    assert item.metadata["kind"] == ENTITY_SUMMARY_KIND
    assert item.metadata["count"] == 3
    assert item.metadata["source"] == "synthesis"
    assert item.embedding == [0.1, 0.2]
