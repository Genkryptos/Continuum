"""
tests/unit/test_local_extractor.py
==================================
The dependency-free ($0, no API) rule-based membership extractor. Precision-
biased: clean acquisition/possession patterns produce triples; noise is dropped
(the router defers to the reader on a miss, but a spurious fact would mis-count).
"""

from __future__ import annotations

import json

import pytest

from continuum.promotion.local_extractor import (
    extract_facts_local,
    local_completion_fn,
)
from continuum.promotion.synthesis import aggregate, extract_structured_facts

pytestmark = pytest.mark.unit


def _preds(text: str) -> list[tuple[str, str]]:
    return [(f["predicate"], f["object"]) for f in extract_facts_local(text)]  # type: ignore[misc]


# ── core membership extraction ────────────────────────────────────────────────


def test_extracts_acquired_items() -> None:
    text = "I set up a goldfish tank. Later I added a shrimp tank. I built a betta tank too."
    preds = _preds(text)
    assert all(p == "tank" for p, _ in preds)
    assert {o for _, o in preds} == {"goldfish tank", "shrimp tank", "betta tank"}


def test_head_before_preposition() -> None:
    # "top from H&M" → head is "top", not "h&m".
    facts = extract_facts_local("I bought a top from H&M yesterday.")
    assert facts and facts[0]["predicate"] == "top"


def test_singularizes_predicate() -> None:
    facts = extract_facts_local("I collected a 1909 penny.")
    assert facts[0]["predicate"] == "penny"


def test_rejects_stopword_heads() -> None:
    # "got a chance", "had a great time" — not collections.
    assert extract_facts_local("I got a chance to relax and had a great time.") == []


def test_trims_trailing_adverbs() -> None:
    facts = extract_facts_local("I bought a leather jacket yesterday.")
    assert facts and facts[0]["object"] == "leather jacket"  # 'yesterday' trimmed


def test_dedups_identical_mentions() -> None:
    facts = extract_facts_local("I have a goldfish tank. I still have the goldfish tank.")
    # same (predicate, object) once.
    assert len([f for f in facts if f["predicate"] == "tank"]) == 1


# ── measurement (for SUM) ─────────────────────────────────────────────────────


def test_attaches_measurement() -> None:
    facts = extract_facts_local("I attended a gaming session for 40 hours.")
    assert facts and facts[0].get("quantity") == "40"
    assert facts[0].get("unit") == "hours"


# ── end-to-end through the real pipeline (count comes out right) ───────────────


def test_end_to_end_counts_via_aggregate() -> None:
    text = "I bought a black top. I bought a white top. I got another striped top."
    facts = extract_facts_local(text)
    structured = []
    from continuum.promotion.synthesis import StructuredFact

    for f in facts:
        structured.append(StructuredFact(subject="user", predicate=str(f["predicate"]), obj=str(f["object"])))
    (d,) = aggregate(structured)
    assert d.count == 3
    assert d.predicate == "top"


async def test_local_completion_fn_plugs_into_extract() -> None:
    # the completion_fn wrapper recovers text from the synthesis prompt and
    # returns the same JSON shape — so extract_structured_facts consumes it.
    from continuum.promotion.synthesis import _EXTRACT_PROMPT

    text = "I adopted a tabby cat. I adopted a siamese cat."
    prompt = _EXTRACT_PROMPT % text
    raw = await local_completion_fn(prompt)
    assert json.loads(raw)["facts"]  # non-empty, valid JSON
    facts = await extract_structured_facts(text, completion_fn=local_completion_fn)
    (d,) = aggregate(facts)
    assert d.count == 2 and d.predicate == "cat"


def test_empty_text() -> None:
    assert extract_facts_local("") == []
    assert extract_facts_local("How are you today?") == []
