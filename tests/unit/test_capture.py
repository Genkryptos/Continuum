"""
tests/unit/test_capture.py
==========================
``continuum.promotion.capture`` — deciding whether a turn contains a durable
fact about the user.

The interesting number here is the **false-capture** count, not the hit rate.
Automatic capture is the feature most able to make memory worse: a store that
swallows "I ran the tests" and the odd API key buries the facts that mattered,
and forgetting is the one operation memory cannot take back. So the sets below
are weighted toward things that must be refused, including the ones that wear
a perfectly stative disguise ("I have a meeting", "my build is failing").
"""

from __future__ import annotations

import pytest

from continuum.promotion.capture import extract_durable_facts, looks_like_secret

pytestmark = pytest.mark.unit


DURABLE = [
    "I live in Boston.",
    "My daughter is named Mira.",
    "I work at Stripe as a staff engineer.",
    "I use Neovim as my main editor.",
    "I am allergic to penicillin.",
    "I prefer PostgreSQL over MySQL.",
    "My favorite cuisine is Japanese.",
    "I drive a red Tesla Model 3.",
    "I speak Hindi and English fluently.",
    "I studied computer science at IIT.",
    "I own a corgi named Pixel.",
    "My laptop is a 14-inch MacBook Pro.",
    "I grew up in Bhilai.",
    "I graduated in 2019.",
    "I have two younger brothers.",
    "I have a cat named Sopa.",
    "I have a peanut allergy.",
    "I am vegetarian.",
    "I keep my venvs under ~/.venvs.",
    "I usually work late at night.",
    # Changes of state. Longitudinal use exposed these: the stative frames accept
    # "I live in Porto" but refused "I moved to Berlin", so a month of daily use
    # locked in day-one facts and dropped every correction — memory drifting
    # further from the truth the longer it ran.
    "I moved to Berlin.",
    "I moved from Porto to Berlin.",
    "I joined Stripe.",
    "I left Nimbus Data and joined Stripe.",
    "I switched from Neovim to Zed.",
]

EPHEMERAL = [
    "I ran the tests and they passed.",
    "I fixed the bug in the retriever.",
    "I just deployed to staging.",
    "I pushed the commit.",
    "I checked the logs again.",
    # …and the work chatter that wears the same shape as those changes.
    "I switched branches.",
    "I moved the file to src/.",
    "I joined the tables on user_id.",
    "I switched to the other terminal.",
    "I changed the config value.",
    "I left the meeting early.",
    "I moved on to the next ticket.",
    "I am running the tests now.",
    "I am getting an import error.",
    "I am debugging the retry logic.",
    "I am waiting for CI.",
    "I am using this file for notes.",
    # …and habitual forms of the same actions, which is why _EPISODIC has to see
    # the adverb too.
    "I always run the tests before pushing.",
    "I usually push to main.",
    "I always get this error.",
    "My build runs on CI.",
    "My test runs slowly.",
    "I have a meeting at 3pm tomorrow.",
    # A modifier between the determiner and the head used to walk straight past
    # the filter — found by using the thing, not by the curated set below.
    "I have a dentist appointment on Thursday.",
    "I have a team meeting at 3pm.",
    "I have a really bad headache today.",
    "I have an interview on Monday.",
    "I have a headache today.",
    "I have a question about the API.",
    "I am done with this task.",
    "I am on the release-3.0 branch.",
    "My build is failing on CI.",
    "My test is red again.",
    "My branch is out of date.",
]

NOT_A_STATEMENT = [
    "How do I configure the reranker?",
    "Can you add a test for this?",
    "Please update the changelog.",
    "Show me the timeline.",
    "Let's run the eval again.",
    "ok do that",
    "yes",
]

NOT_DURABLE_YET = [
    "If I move to Berlin I will need a new visa.",
    "I might switch to Postgres later.",
    "I was thinking about buying a car.",
    "I am planning to learn Rust.",
    "I want to refactor this module.",
]

NOT_ABOUT_THE_USER = [
    "You are running Opus 4.8.",
    "The tests are failing on main.",
    "It is a bi-temporal store.",
    "We should ship the release.",
    "I love this bug.",
    "I use this file for scratch notes.",
]

RETRACTIONS = [
    "I don't live in Boston anymore.",
    "I no longer work at Infosys.",
    "I never run the tests locally.",
]

SECRETS = [
    "My API key is sk-proj-abc123def456ghi789jkl.",
    "I use password hunter2 for the database.",
    "My token is ghp_16C7e42F292c6912E7710c838347Ae178B4a.",
    "My card number is 4111 1111 1111 1111.",
    "My DSN is postgresql://admin:s3cret@10.0.0.1:5432/prod.",
    "My SSN is 123-45-6789.",
]

REFUSE = EPHEMERAL + NOT_A_STATEMENT + NOT_DURABLE_YET + NOT_ABOUT_THE_USER + RETRACTIONS + SECRETS


@pytest.mark.parametrize("sentence", DURABLE)
def test_durable_facts_are_captured(sentence: str) -> None:
    assert extract_durable_facts(sentence), f"missed a durable fact: {sentence!r}"


@pytest.mark.parametrize("sentence", REFUSE)
def test_everything_else_is_refused(sentence: str) -> None:
    got = extract_durable_facts(sentence)
    assert not got, f"wrongly captured {sentence!r} -> {[f.text for f in got]}"


def test_no_false_captures_across_the_whole_adversarial_set() -> None:
    # The headline number, asserted as one fact so a regression is obvious.
    wrong = [s for s in REFUSE if extract_durable_facts(s)]
    assert wrong == [], f"{len(wrong)}/{len(REFUSE)} false captures: {wrong}"


def test_a_mixed_turn_yields_only_the_fact() -> None:
    turn = "I ran the tests and they passed. I live in Boston. Can you fix the bug?"
    assert [f.text for f in extract_durable_facts(turn)] == ["I live in Boston."]


def test_a_secret_drops_its_whole_sentence_not_just_the_token() -> None:
    # Redacting and keeping the rest would store "my api key is" — useless, and
    # it teaches the store that key-shaped sentences are worth remembering.
    turn = "I live in Boston. My API key is sk-proj-abc123def456ghi789jkl."
    assert [f.text for f in extract_durable_facts(turn)] == ["I live in Boston."]


def test_secrets_are_detected_on_their_own() -> None:
    assert looks_like_secret("AKIAIOSFODNN7EXAMPLE")
    assert looks_like_secret("my password is hunter2")
    assert not looks_like_secret("I live in Boston")


def test_duplicates_within_a_turn_collapse() -> None:
    turn = "I live in Boston. I live in Boston."
    assert len(extract_durable_facts(turn)) == 1


def test_empty_and_junk_input_is_safe() -> None:
    for junk in ("", "   ", "\n\n", "?!", "a"):
        assert extract_durable_facts(junk) == []


def test_a_very_long_sentence_is_refused() -> None:
    # A wall of text is a paste, not a stated fact — and one embedding over it
    # would be a meaningless vector anyway.
    assert extract_durable_facts("I live in " + "x" * 400) == []


def test_the_rule_that_fired_is_reported() -> None:
    # Provenance: when a capture looks wrong, you need to know which rule to fix.
    assert extract_durable_facts("I live in Boston.")[0].rule == "stative-i"
    assert extract_durable_facts("My laptop is a MacBook.")[0].rule == "stative-my"
    assert extract_durable_facts("I moved to Berlin.")[0].rule == "changed-attribute"


def test_a_correction_is_captured_alongside_the_fact_it_replaces() -> None:
    """The longitudinal failure, in one assertion.

    Without this the store keeps "I live in Porto" forever and silently ignores
    "I moved to Berlin" — it drifts further from the truth the longer it runs,
    which is the worst thing a memory system can do.
    """
    assert extract_durable_facts("I live in Porto.")
    assert extract_durable_facts("I moved from Porto to Berlin.")
