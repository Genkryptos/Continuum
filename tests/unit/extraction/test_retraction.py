"""
tests/unit/extraction/test_retraction.py
========================================
The retraction detector — explicit negation of a *prior* claim ("never been to
X", "that was wrong") vs a temporal change or a standing negative preference.
Precision matters: a false positive can wrongly retire a real memory, so the
matcher deliberately does NOT fire on "I never eat meat".
"""

from __future__ import annotations

import pytest

from continuum.extraction.retraction import is_retraction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        "Actually i never been to bengaluru now i am back to bhilai",  # the live bug
        "I was never in Bengaluru",
        "I never went to Paris",
        "I never visited that office",
        "That's wrong, I live in Boston",
        "That was incorrect",
        "that is not true",
        "Scratch that, my name is Bob",
        "Forget what I said earlier",
        "I didn't actually move to Delhi",
        "Actually I never said that",
        "ignore that",
    ],
)
def test_positive_retractions(text: str) -> None:
    assert is_retraction(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I never eat meat",  # standing preference — a fact, not a retraction
        "I moved to Boston",  # temporal change → supersession, not retraction
        "I live in New York",
        "I have a never-ending list",  # "never-ending" must not match
        "My flight was delayed",
        "I was in Bengaluru last year",  # a positive past claim
        "I prefer dark mode",
        "",  # empty
    ],
)
def test_negative_non_retractions(text: str) -> None:
    assert is_retraction(text) is False
