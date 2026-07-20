"""
tests/unit/test_attribute_extract.py
====================================
`extract_attribute` — deterministic, conservative attribute inference. The
load-bearing property is precision: a WRONG tag makes `current` answer with a
fact about something else, so the false-tag rate on non-attributes must be 0.
"""

from __future__ import annotations

import pytest

from continuum.promotion.attribute_extract import extract_attribute

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "attribute"),
    [
        ("I live in Boston.", "residence"),
        ("I moved to New York City.", "residence"),
        ("I relocated to Berlin last year.", "residence"),
        ("I work at Stripe.", "employer"),
        ("I joined Infosys in 2023.", "employer"),
        ("I left Infosys and joined Nimbus.", "employer"),
        ("My name is Mayank.", "name"),
        ("Call me Sam.", "name"),
        ("I drive a red Tesla Model 3.", "vehicle"),
        ("I drive a Honda Civic.", "vehicle"),
        ("My favorite editor is Neovim.", "favorite editor"),
        ("My dog is a corgi named Pixel.", "dog"),
        ("My coffee order is a flat white.", "coffee order"),
    ],
)
def test_extracts_canonical_and_possessive_attributes(text: str, attribute: str) -> None:
    assert extract_attribute(text) == attribute


@pytest.mark.parametrize(
    "text",
    [
        # figurative / idiomatic uses of the trigger verbs
        "I drive a hard bargain.",
        "I live in fear of merge conflicts.",
        "I work out at the gym on Tuesdays.",
        "I moved the file to the archive folder.",
        "I joined the two tables with a LEFT JOIN.",
        # opinions phrased possessively
        "My understanding is that it ships in Q4.",
        "My take is we should skip Redis.",
        "My guess is two weeks.",
        # not about the user, or not an attribute at all
        "She lives in Paris.",
        "We decided to use Stripe.",
        "Postgres 16 is our primary database.",
        "I like Python a lot.",
        "The deploy runs migrations automatically.",
    ],
)
def test_never_tags_a_non_attribute(text: str) -> None:
    assert extract_attribute(text) is None


def test_empty_is_none() -> None:
    assert extract_attribute("") is None
    assert extract_attribute("   ") is None
