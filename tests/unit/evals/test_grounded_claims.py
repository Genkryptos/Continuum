"""
tests/unit/evals/test_grounded_claims.py
========================================
Tests for the grounded-claim candidate shape and the claim-aware ranker.

Three acceptance scenarios from the spec:

1. **IKEA durations** — same word ("IKEA") in two clauses, but the
   subject of the duration is different ("bookshelf" vs "table"). The
   ranker must surface the 4-hour bookshelf claim for a bookshelf
   question, NOT the 2-hour table claim, by reading
   ``claim.subject`` instead of treating both as bare ``duration``
   candidates.
2. **Stanford vs UCLA** — same answer type ("location") but different
   relations. UCLA is the object of a completed-event verb
   ("got my Bachelor's AT UCLA"); Stanford is the object of a
   hypothetical verb ("looking at Stanford for Master's"). The
   ranker must prefer the factual relation.
3. **Painting value** — the spec wants ``"triple what I paid"`` to
   survive as the answer to "what is the painting worth?", NOT
   the numeric amount the user originally paid. This requires the
   new ``extract_value_claims`` path to emit a ``"is_worth"``
   relation with a qualitative object, distinct from the
   ``"paid"`` relation that holds the numeric amount.
"""
from __future__ import annotations

from evals.longmemeval.candidates import (
    FACTUAL_RELATIONS,
    HYPOTHETICAL_RELATIONS,
    Candidate,
    extract_durations,
    extract_locations,
    extract_value_claims,
    rank_claims_for_question,
)
from evals.longmemeval.question_type import QuestionType

# ─── Candidate dataclass: new fields exist + defaults safe ─────────────────


def test_candidate_dataclass_back_compat():
    """Legacy positional/keyword construction still works."""
    c = Candidate(
        value="7", normalized_value="7", candidate_type="count", unit="shirts",
    )
    assert c.value == "7"
    assert c.candidate_type == "count"
    # New fields default cleanly
    assert c.claim == ""
    assert c.subject == ""
    assert c.relation == ""
    assert c.object_ == ""
    assert c.source_span == ""


def test_candidate_to_dict_renders_object_key_without_underscore():
    """JSON-shape uses ``object`` (no underscore) per the spec."""
    c = Candidate(
        value="UCLA", normalized_value="ucla", candidate_type="location",
        subject="Bachelor's degree", relation="got_at", object_="UCLA",
        claim="I got my Bachelor's degree at UCLA in 2018.",
        source_span="at UCLA",
        answer_type="location",
    )
    d = c.to_dict()
    assert d["object"] == "UCLA"
    assert d["subject"] == "Bachelor's degree"
    assert d["relation"] == "got_at"
    assert d["answer_type"] == "location"
    assert "object_" not in d  # internal name must not leak


# ─── IKEA durations: subject disambiguates 2h vs 4h ────────────────────────


def test_ikea_durations_carry_subject_per_clause():
    text = (
        "Assembling the IKEA bookshelf took 4 hours from start to finish. "
        "Last week I put together a small IKEA table — that one only took 2 hours."
    )
    cs = extract_durations(text, source_session_id="s_ikea")
    # Both durations should be extracted
    objects = {c.object_ for c in cs}
    assert "4 hours" in objects
    assert "2 hours" in objects
    # Each must carry the subject of its OWN clause
    by_object = {c.object_: c for c in cs}
    assert "bookshelf" in by_object["4 hours"].subject.lower()
    assert "table" in by_object["2 hours"].subject.lower()
    # Source attribution intact
    assert all(c.source_session_id == "s_ikea" for c in cs)
    # claim/source_span populated, not empty
    for c in cs:
        assert c.claim, "claim must be populated"
        assert c.source_span, "source_span must be populated"
        assert c.answer_type == "duration"
        assert c.relation == "duration_of"


def test_ikea_bookshelf_ranks_above_table():
    """Ranker picks the 4-hour bookshelf claim, not the 2-hour table claim."""
    text = (
        "Assembling the IKEA bookshelf took 4 hours from start to finish. "
        "Last week I put together a small IKEA table — that one only took 2 hours."
    )
    cs = extract_durations(text, source_session_id="s_ikea")
    ranked = rank_claims_for_question(
        cs,
        "How long did the IKEA bookshelf take?",
        question_type=QuestionType.DURATION,
    )
    assert ranked, "ranker returned nothing for a clearly answerable question"
    top = ranked[0]
    assert top.object_ == "4 hours"
    assert "bookshelf" in top.subject.lower()
    # The 2-hour table claim must be strictly ranked below
    other = [c for c in ranked if c.object_ == "2 hours"]
    assert other, "table claim should still be present, just lower"
    assert ranked.index(top) < ranked.index(other[0])


def test_ikea_table_question_picks_table_claim():
    """Symmetric: asking about the table picks 2 hours, not 4."""
    text = (
        "Assembling the IKEA bookshelf took 4 hours from start to finish. "
        "Last week I put together a small IKEA table — that one only took 2 hours."
    )
    cs = extract_durations(text, source_session_id="s_ikea")
    ranked = rank_claims_for_question(
        cs,
        "How long did the IKEA table take?",
        question_type=QuestionType.DURATION,
    )
    assert ranked
    assert ranked[0].object_ == "2 hours"
    assert "table" in ranked[0].subject.lower()


# ─── Stanford vs UCLA — factual relation beats hypothetical ────────────────


def test_ucla_bachelors_outranks_stanford_masters():
    """
    Completed-event location (UCLA / Bachelor's) ranks above the
    hypothetical option (Stanford / Master's) for a Bachelor's question.
    """
    text = (
        "I got my Bachelor's degree at UCLA in 2018. "
        "I'm currently looking at Stanford for Master's programs."
    )
    cs = extract_locations(text, source_session_id="s_edu")
    # Both UCLA and Stanford should be present
    objects = {c.object_ for c in cs}
    assert "UCLA" in objects or "Ucla" in objects or any(
        c.object_.lower() == "ucla" for c in cs
    )
    assert any(c.object_.lower() == "stanford" for c in cs)

    # Relations distinguish them
    ucla = next(c for c in cs if c.object_.lower() == "ucla")
    stanford = next(c for c in cs if c.object_.lower() == "stanford")
    assert ucla.relation in FACTUAL_RELATIONS
    assert stanford.relation in HYPOTHETICAL_RELATIONS or \
           stanford.relation == "looking_at"

    # Rank for the Bachelor's question — UCLA must win
    ranked = rank_claims_for_question(
        cs,
        "Where did I get my Bachelor's degree?",
        question_type=QuestionType.LOCATION,
    )
    assert ranked
    assert ranked[0].object_.lower() == "ucla"
    # And Stanford must NOT outrank UCLA
    ucla_idx = next(
        i for i, c in enumerate(ranked) if c.object_.lower() == "ucla"
    )
    stanford_idx = next(
        (i for i, c in enumerate(ranked) if c.object_.lower() == "stanford"),
        None,
    )
    if stanford_idx is not None:
        assert ucla_idx < stanford_idx, (
            "factual UCLA claim must outrank hypothetical Stanford claim"
        )


def test_stanford_masters_question_can_still_surface_stanford():
    """
    When the question explicitly asks about the Master's, Stanford
    still wins — the hypothetical penalty is relative, not absolute.
    """
    text = (
        "I got my Bachelor's degree at UCLA in 2018. "
        "I'm currently looking at Stanford for Master's programs."
    )
    cs = extract_locations(text, source_session_id="s_edu")
    ranked = rank_claims_for_question(
        cs,
        "Where am I looking at for my Master's?",
        question_type=QuestionType.LOCATION,
    )
    assert ranked
    # Stanford should at least appear in the result set
    assert any(c.object_.lower() == "stanford" for c in ranked)


# ─── Painting value — qualitative object survives ──────────────────────────


def test_painting_worth_emits_qualitative_object():
    text = (
        "I paid 200 dollars for the painting in 2015. "
        "Today the painting is worth triple what I paid."
    )
    cs = extract_value_claims(text, source_session_id="s_painting")
    # Two claims emitted: paid + worth.
    relations = {c.relation for c in cs}
    assert "is_worth" in relations
    assert "paid" in relations
    worth = next(c for c in cs if c.relation == "is_worth")
    paid = next(c for c in cs if c.relation == "paid")

    # The qualitative answer must NOT be reduced to a digit count
    assert "triple" in worth.object_.lower()
    assert worth.object_.lower() != "200"
    assert worth.candidate_type == "value"
    assert worth.answer_type == "value"
    assert worth.subject.lower().endswith("painting")

    # And the paid claim carries the numeric amount, NOT the qualitative phrase
    assert "200" in paid.object_
    assert "triple" not in paid.object_.lower()


def test_painting_worth_question_picks_qualitative_claim():
    """
    The ranker prefers the ``is_worth`` claim with the qualitative
    object over the ``paid`` claim with the numeric amount when the
    question asks about *worth*.
    """
    text = (
        "I paid 200 dollars for the painting in 2015. "
        "Today the painting is worth triple what I paid."
    )
    cs = extract_value_claims(text, source_session_id="s_painting")
    ranked = rank_claims_for_question(
        cs,
        "What is the painting worth today?",
        question_type=QuestionType.GENERIC_FACT,
    )
    assert ranked
    top = ranked[0]
    assert top.relation == "is_worth"
    assert "triple" in top.object_.lower()
    # The numeric "200 dollars" must NOT be the top answer.
    assert "200" not in top.object_


def test_painting_paid_question_picks_numeric_claim():
    """Symmetric: asking what was paid surfaces the numeric amount."""
    text = (
        "I paid 200 dollars for the painting in 2015. "
        "Today the painting is worth triple what I paid."
    )
    cs = extract_value_claims(text, source_session_id="s_painting")
    ranked = rank_claims_for_question(
        cs,
        "How much did I pay for the painting?",
        question_type=QuestionType.GENERIC_FACT,
    )
    assert ranked
    top = ranked[0]
    assert top.relation == "paid"
    assert "200" in top.object_


# ─── Telemetry / observability ─────────────────────────────────────────────


def test_grounded_claim_fields_survive_serialization():
    text = "Assembling the IKEA bookshelf took 4 hours."
    cs = extract_durations(text, source_session_id="s_x")
    assert cs
    d = cs[0].to_dict()
    assert d["claim"]
    assert d["subject"].lower().endswith("bookshelf")
    assert d["object"] == "4 hours"
    assert d["relation"] == "duration_of"
    assert d["answer_type"] == "duration"
    assert d["source_span"]
    # candidate_type preserved as metadata (per the spec)
    assert d["candidate_type"] == "duration"
