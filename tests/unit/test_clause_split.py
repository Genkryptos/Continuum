"""
tests/unit/test_clause_split.py
===============================
`split_facts` — atomic facts before embedding. The bias is deliberate: shredding
one fact into fragments is far worse than leaving a compound one intact, so a
missed split is acceptable and a false split is not.
"""

from __future__ import annotations

import pytest

from continuum.promotion.clause_split import split_facts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # measured case: compound scored 0.479 vs 0.614 split, losing to distractors
        (
            "We're going with Postgres 16, pgvector needs 0.8 or newer.",
            ["We're going with Postgres 16.", "pgvector needs 0.8 or newer."],
        ),
        (
            "Priya is handling the frontend, I'm on the API side.",
            ["Priya is handling the frontend.", "I'm on the API side."],
        ),
        (
            "Postgres runs on 5433. Redis was skipped.",
            ["Postgres runs on 5433.", "Redis was skipped."],
        ),
    ],
)
def test_splits_genuinely_compound_statements(text: str, expected: list[str]) -> None:
    assert split_facts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I live in Bhilai, India.",  # appositive
        "I know Python, Rust, and Go.",  # list
        "My laptop is a MacBook Pro, which I bought in 2024.",  # relative clause
        "We use Postgres, because Redis was too much ops.",  # subordinate
        "I moved to NYC, and Priya stayed in Boston.",  # 'and' continuation
        "The API caps us at 100 rpm, so we batch requests.",  # 'so' continuation
        "Deploy targets are staging, production, and dev.",  # noun list
        "I moved from Boston to New York City.",  # single fact
    ],
)
def test_never_shreds_a_single_fact(text: str) -> None:
    assert split_facts(text) == [text]


def test_degenerate_input_is_returned_unchanged() -> None:
    assert split_facts("") == [""]
    assert split_facts("   ") == ["   "]
    assert split_facts("hi") == ["hi"]
