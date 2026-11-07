"""
Tests for :mod:`continuum.promotion.scalar_supersession`.

The rule's value is that it is *deterministic* and *biased toward abstention*:
a false supersession retires a fact the reader needs, which is strictly worse
than paying for one more LLM call. The abstention cases therefore carry at
least as much weight here as the firing ones.
"""

from __future__ import annotations

from continuum.promotion.scalar_supersession import (
    find_superseded,
    frame_overlap,
    scalar_values,
    supersedes,
)

KOREAN_OLD = (
    "have you tried any good Korean restaurants in your city "
    "lately? I've tried three different ones recently"
)
KOREAN_NEW = (
    "have you tried any good Korean restaurants in your city "
    "lately? I've tried four different ones so far"
)


# ── scalar_values ───────────────────────────────────────────────────────────


def test_number_words_fold_onto_digits():
    assert "3" in scalar_values("I've tried three of them")
    assert scalar_values("three") == scalar_values("3")


def test_grouped_thousands_and_clock_times():
    assert "400000" in scalar_values("pre-approved for $400,000")
    assert "7:30" in scalar_values("I wake up at 7:30 am")


def test_url_numbers_are_not_scalars():
    """The largest single source of false positives in the measurement."""
    vals = scalar_values("watch https://www.youtube.com/watch?v=l9TA9MpsQm0 now")
    assert vals == set()


def test_long_opaque_identifiers_are_stripped():
    assert scalar_values("order ABC123XYZ789 shipped") == set()


def test_real_numbers_survive_alongside_a_url():
    assert "5" in scalar_values("I watched 5 films, see https://example.com/a1b2c3d4")


# ── frame_overlap ───────────────────────────────────────────────────────────


def test_overlap_is_high_for_a_restatement():
    assert frame_overlap(KOREAN_OLD, KOREAN_NEW) > 0.5


def test_overlap_is_low_for_unrelated_text():
    assert frame_overlap(KOREAN_OLD, "my flight to Denver was delayed again") < 0.1


def test_overlap_of_empty_texts_is_zero_not_an_error():
    assert frame_overlap("", "") == 0.0


# ── supersedes: fires ───────────────────────────────────────────────────────


def test_restated_count_supersedes():
    assert supersedes(KOREAN_NEW, KOREAN_OLD)


def test_restated_amount_supersedes():
    assert supersedes(
        "I was pre-approved for $400,000 on the mortgage with Wells Fargo",
        "I was pre-approved for $350,000 on the mortgage with Wells Fargo",
    )


# ── supersedes: abstains (the precision guarantees) ─────────────────────────


def test_same_value_is_agreement_not_contradiction():
    assert not supersedes("I have three cats", "I have three cats and a dog")


def test_unrelated_topics_with_different_numbers_do_not_fire():
    assert not supersedes("I ran 5 miles this morning", "the meeting has 3 attendees confirmed")


def test_no_numbers_means_no_verdict():
    assert not supersedes("I moved to Boston recently", "I moved to New York City recently")


def test_one_side_without_a_number_does_not_fire():
    assert not supersedes("I've tried four different ones", "I've tried Korean restaurants before")


def test_url_only_difference_does_not_fire():
    """Two messages differing solely by a link must not supersede."""
    a = "sorry I meant this link https://www.youtube.com/watch?v=l9TA9MpsQm0"
    b = "sorry I meant this link https://www.youtube.com/watch?v=tL4tde6uZ_Q"
    assert not supersedes(a, b)


def test_raising_min_overlap_makes_it_stricter():
    assert supersedes(KOREAN_NEW, KOREAN_OLD, min_overlap=0.10)
    assert not supersedes(KOREAN_NEW, KOREAN_OLD, min_overlap=0.99)


# ── find_superseded ─────────────────────────────────────────────────────────


def test_picks_the_matching_neighbour():
    assert find_superseded(KOREAN_NEW, ["the dog is called Rex", KOREAN_OLD]) == 1


def test_returns_none_when_nothing_matches():
    assert find_superseded(KOREAN_NEW, ["the dog is called Rex"]) is None


def test_empty_neighbours_is_none_not_an_error():
    assert find_superseded(KOREAN_NEW, []) is None


def test_highest_overlap_neighbour_wins():
    weak = "I've tried two restaurants"
    assert find_superseded(KOREAN_NEW, [weak, KOREAN_OLD]) == 1


def test_symmetry_is_documented_not_accidental():
    """
    The rule detects disagreement, not direction — the caller owns ordering.
    Locking this in so a future change cannot silently start assuming it
    knows which fact is newer.
    """
    assert supersedes(KOREAN_NEW, KOREAN_OLD)
    assert supersedes(KOREAN_OLD, KOREAN_NEW)
