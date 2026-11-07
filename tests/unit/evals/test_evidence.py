"""
Tests for :mod:`evals.longmemeval.evidence`.

The stale detector is a measurement instrument whose value is its precision:
it exists so that "supersession failed here" can be asserted rather than
guessed, and a false positive would make it worse than nothing. The tests
therefore weight abstention cases at least as heavily as firing ones.
"""

from __future__ import annotations

import pytest

from evals.longmemeval.evidence import (
    GoldTurn,
    gold_turn_recall,
    gold_turns_from_raw,
    supersession_verdict,
)


def _turn(date: str, content: str, *, index: int = 0, sid: str = "s1") -> GoldTurn:
    return GoldTurn(session_id=sid, date=date, role="user", content=content, index=index)


# ── gold_turns_from_raw ─────────────────────────────────────────────────────


def _raw(**over):
    base = {
        "answer_session_ids": ["a1"],
        "haystack_session_ids": ["h0", "a1"],
        "haystack_dates": ["2023/01/01 (Sun) 00:00", "2023/02/02 (Thu) 00:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "noise"}],
            [
                {"role": "user", "content": "not gold", "has_answer": False},
                {"role": "user", "content": "the answer", "has_answer": True},
            ],
        ],
    }
    base.update(over)
    return base


def test_extracts_only_flagged_turns():
    gold = gold_turns_from_raw(_raw())
    assert [g.content for g in gold] == ["the answer"]
    assert gold[0].session_id == "a1"
    assert gold[0].index == 1


def test_returns_chronological_order():
    raw = _raw(
        answer_session_ids=["a2", "a1"],
        haystack_session_ids=["a1", "a2"],
        haystack_dates=["2023/02/02 (Thu) 00:00", "2023/01/01 (Sun) 00:00"],
        haystack_sessions=[
            [{"role": "user", "content": "later", "has_answer": True}],
            [{"role": "user", "content": "earlier", "has_answer": True}],
        ],
    )
    assert [g.content for g in gold_turns_from_raw(raw)] == ["earlier", "later"]


def test_missing_gold_session_is_skipped_not_raised():
    raw = _raw(answer_session_ids=["nope"])
    assert gold_turns_from_raw(raw) == []


def test_string_answer_session_id_is_accepted():
    # Some rows carry a bare string rather than a list.
    assert len(gold_turns_from_raw(_raw(answer_session_ids="a1"))) == 1


def test_row_with_no_flagged_turns():
    raw = _raw(
        haystack_sessions=[
            [{"role": "user", "content": "noise"}],
            [{"role": "user", "content": "nothing flagged"}],
        ]
    )
    assert gold_turns_from_raw(raw) == []


# ── gold_turn_recall ────────────────────────────────────────────────────────


def test_recall_counts_turns_not_sessions():
    """The failure session recall cannot see: two gold turns, one retrieved."""
    gold = [
        _turn("2023/01/01", "I have tried three Korean restaurants recently"),
        _turn("2023/02/02", "I have tried four Korean restaurants so far", index=1),
    ]
    out = gold_turn_recall(gold, [gold[0].content])
    assert out["gold_turn_count"] == 2
    assert out["retrieved_gold_turn_count"] == 1
    assert out["gold_turn_recall"] == pytest.approx(0.5)
    assert out["missing_gold_turns"] == ["s1#1"]


def test_recall_is_whitespace_and_case_insensitive():
    gold = [_turn("2023/01/01", "I Graduated With A Degree In Business Administration")]
    retrieved = ["  i graduated with a  degree in business   administration  "]
    assert gold_turn_recall(gold, retrieved)["gold_turn_recall"] == 1.0


def test_recall_matches_a_span_of_the_turn():
    turn = "I graduated with a degree in Business Administration which has helped"
    gold = [_turn("2023/01/01", turn)]
    # Retrieval surfaced a longer item that contains the turn.
    assert gold_turn_recall(gold, [f"prefix... {turn} ...suffix"])["gold_turn_recall"] == 1.0


def test_no_gold_turns_reports_zero_count_not_zero_recall_over_nothing():
    out = gold_turn_recall([], ["anything"])
    assert out["gold_turn_count"] == 0
    assert out["gold_turn_recall"] == 0.0


def test_empty_retrieval_misses_everything():
    gold = [_turn("2023/01/01", "some answer bearing turn of sufficient length")]
    assert gold_turn_recall(gold, [])["gold_turn_recall"] == 0.0


# ── supersession_verdict ────────────────────────────────────────────────────

KOREAN = [
    _turn(
        "2023/08/11",
        "have you tried any good Korean restaurants? I've tried three different ones recently",
    ),
    _turn(
        "2023/09/30",
        "have you tried any good Korean restaurants? I've tried four different ones so far",
        index=1,
    ),
]


def test_stale_when_answer_takes_the_superseded_number():
    assert (
        supersession_verdict(KOREAN, "You've tried **three** Korean restaurants.", "four")
        == "stale"
    )


def test_fresh_when_answer_takes_the_current_number():
    assert supersession_verdict(KOREAN, "You've tried four Korean restaurants.", "four") == "fresh"


def test_number_words_and_digits_are_the_same_value():
    assert supersession_verdict(KOREAN, "You've tried 3.", "4") == "stale"


def test_grouped_thousands_are_compared_as_values():
    gold = [
        _turn("2023/01/01", "I was pre-approved for $350,000 with Wells Fargo"),
        _turn("2023/06/01", "I was pre-approved for $400,000 instead", index=1),
    ]
    assert (
        supersession_verdict(gold, "You were pre-approved for **$350,000**.", "$400,000") == "stale"
    )


def test_clock_times_are_compared_as_values():
    gold = [
        _turn("2023/01/01", "I wake up around 8:30 am on Saturdays"),
        _turn("2023/06/01", "I now wake up around 7:30 am on Saturdays", index=1),
    ]
    assert supersession_verdict(gold, "You wake up around 8:30 am.", "7:30 am") == "stale"


def test_lexical_path_handles_non_numeric_supersession():
    gold = [
        _turn("2023/01/01", "the painting hangs above the living room sofa"),
        _turn("2023/06/01", "I moved the painting into the bedroom", index=1),
    ]
    assert (
        supersession_verdict(gold, "It is displayed above your living room sofa.", "in my bedroom")
        == "stale"
    )


# ── abstention: the cases that protect precision ────────────────────────────


def test_abstains_on_a_single_gold_turn():
    assert supersession_verdict([KOREAN[0]], "three", "three") is None


def test_abstains_when_gold_turns_share_a_date():
    same_day = [KOREAN[0], _turn("2023/08/11", "and four more later", index=1)]
    assert supersession_verdict(same_day, "three", "four") is None


def test_abstains_when_expected_answer_does_not_confirm_the_later_turn():
    """
    The guard that keeps precision at 100%: two gold turns can differ without
    the later one superseding the earlier. If the expected answer is not the
    later value, this is not a supersession and no verdict is issued.
    """
    assert supersession_verdict(KOREAN, "three", "three") is None


def test_abstains_when_the_turns_do_not_disagree():
    same = [
        _turn("2023/01/01", "I have three of them"),
        _turn("2023/06/01", "I still have three of them", index=1),
    ]
    assert supersession_verdict(same, "three", "three") is None


def test_abstains_when_answer_mentions_neither_value():
    assert supersession_verdict(KOREAN, "I don't have that information.", "four") is None


def test_answer_carrying_both_values_is_fresh_not_stale():
    """
    "You used to have three, now four" is a correct answer. Requiring the
    absence of the current value is what stops it being called stale.
    """
    assert supersession_verdict(KOREAN, "You had tried three, and now four.", "four") == "fresh"


def test_numeric_extractor_wins_over_lexical_when_both_could_fire():
    gold = [
        _turn("2023/01/01", "I own three vintage cameras stored in the attic"),
        _turn("2023/06/01", "I own five vintage cameras stored in the attic", index=1),
    ]
    assert supersession_verdict(gold, "You own three cameras.", "five") == "stale"


def test_non_string_expected_answer_is_coerced():
    # LongMemEval ground truth is sometimes an int.
    assert supersession_verdict(KOREAN, "three", 4) == "stale"
