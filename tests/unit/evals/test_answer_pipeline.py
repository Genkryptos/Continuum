"""
tests/unit/evals/test_answer_pipeline.py
========================================
Unit coverage for the Phase-A answer pipeline helpers:

* :mod:`evals.longmemeval.question_type`  — deterministic classifier.
* :mod:`evals.longmemeval.candidates`     — typed extractors + filter +
  count override.
* :mod:`evals.longmemeval.answer_post`    — cleanup, shape validator,
  IDK guard, repair-prompt builder.

The pipeline is the bottleneck identified in the 500-row analysis —
these tests pin the deterministic contracts that every adapter run
relies on so regressions surface in CI, not on the next eval spend.
"""

from __future__ import annotations

from evals.longmemeval.answer_post import (
    build_repair_prompt,
    clean_final_answer,
    is_idk,
    should_block_idk,
    validate_answer_shape,
)
from evals.longmemeval.candidates import (
    Candidate,
    best_count_for_object,
    extract_candidates_from_text,
    extract_counts,
    extract_durations,
    extract_preferences,
    filter_candidates_for_question,
)
from evals.longmemeval.question_type import (
    QuestionType,
    classify,
    is_multi_session_hint,
)

# ─── Question-type classifier ──────────────────────────────────────────────


def test_classify_count():
    assert classify("How many shirts did I buy?") == QuestionType.COUNT
    assert classify("How much coffee do I drink?") == QuestionType.COUNT


def test_classify_duration():
    assert classify("How long is my commute?") == QuestionType.DURATION
    assert classify("How long does it take to bake?") == QuestionType.DURATION


def test_classify_date_time():
    assert classify("When did I start the new job?") == QuestionType.DATE_TIME
    assert classify("What day did we meet?") == QuestionType.DATE_TIME


def test_classify_knowledge_update_beats_temporal():
    # The KU rules run BEFORE the temporal rules; "changed" wins over
    # "before/after".
    assert (
        classify(
            "What did I use to drink before I changed my preference?",
        )
        == QuestionType.KNOWLEDGE_UPDATE
    )


def test_classify_preference():
    assert classify("What's my favorite movie?") == QuestionType.PREFERENCE
    assert classify("Do I prefer coffee or tea?") == QuestionType.PREFERENCE


def test_classify_location_and_person():
    assert classify("Where did I leave my keys?") == QuestionType.LOCATION
    assert classify("Who is my dentist?") == QuestionType.PERSON


def test_classify_falls_back_to_generic():
    assert classify("This is not really a question.") == QuestionType.GENERIC_FACT
    assert classify("") == QuestionType.GENERIC_FACT


def test_multi_session_hint():
    assert is_multi_session_hint("How many across all sessions?")
    assert not is_multi_session_hint("How many shirts did I buy?")


# ─── Candidate extractors ─────────────────────────────────────────────────


def test_extract_counts_anchored_frame():
    cs = extract_counts("I have 7 shirts in my closet.", source_session_id="s1")
    assert any(c.normalized_value == "7" and "shirt" in c.unit.lower() for c in cs)
    # Source id propagates so the trace can attribute back.
    assert all(c.source_session_id == "s1" for c in cs)


def test_extract_counts_drops_noise_units():
    """Noise units like 'thing'/'rule'/'time' must NOT become count facts."""
    cs = extract_counts("There are 20 rules and 5 things to remember.")
    units = {c.unit.lower().rstrip("s") for c in cs}
    assert "rule" not in units
    assert "thing" not in units


def test_extract_durations_picks_up_units():
    cs = extract_durations("My commute is 45 minutes each way.")
    assert any("45" in c.normalized_value for c in cs)


def test_extract_preferences_caught():
    cs = extract_preferences("I prefer black coffee in the morning.")
    assert cs, "preference extractor returned nothing"
    assert all(c.candidate_type == "preference" for c in cs)


def test_extract_candidates_from_text_mixes_types():
    cs = extract_candidates_from_text(
        "I have 7 shirts. My favorite color is blue. I prefer iced tea.",
    )
    types = {c.candidate_type for c in cs}
    assert "count" in types
    assert "preference" in types


# ─── Filter by question type ───────────────────────────────────────────────


def test_filter_candidates_returns_only_relevant_types():
    cands = [
        Candidate("7", "7", "count", unit="shirts", confidence=0.9),
        Candidate("blue", "blue", "preference", confidence=0.7),
        Candidate("Mayo Clinic", "mayo clinic", "location", confidence=0.6),
    ]
    out = filter_candidates_for_question(cands, QuestionType.COUNT)
    assert len(out) == 1 and out[0].candidate_type == "count"
    out = filter_candidates_for_question(cands, QuestionType.PREFERENCE)
    assert len(out) == 1 and out[0].candidate_type == "preference"


def test_filter_candidates_passthrough_for_generic():
    cands = [
        Candidate("7", "7", "count", confidence=0.9),
        Candidate("blue", "blue", "preference", confidence=0.7),
    ]
    out = filter_candidates_for_question(cands, QuestionType.GENERIC_FACT)
    assert len(out) == 2  # unfiltered


# ─── best_count_for_object — the override gate ─────────────────────────────


def test_best_count_strict_match_required():
    cands = [
        Candidate("22", "22", "count", unit="Marvel", confidence=0.7),
        Candidate("7", "7", "count", unit="shirts", confidence=0.9),
    ]
    # Question doesn't mention "marvel" or "shirts" — returns None to avoid
    # hijacking a correct synthesis answer with arbitrary numbers.
    assert best_count_for_object(cands, "How many things did I buy?") is None
    # Question mentions "shirts" — that candidate wins.
    chosen = best_count_for_object(cands, "How many shirts do I own?")
    assert chosen is not None and chosen.normalized_value == "7"


def test_best_count_respects_confidence_floor():
    cands = [Candidate("99", "99", "count", unit="shirts", confidence=0.4)]
    # Below 0.6 confidence floor → None even with unit match.
    assert best_count_for_object(cands, "how many shirts") is None


# ─── clean_final_answer ────────────────────────────────────────────────────


def test_clean_final_answer_strips_subanswer_prefix():
    out = clean_final_answer("Matching facts: x, y, z\nSub-answer: 7 shirts")
    assert out == "7 shirts"


def test_clean_final_answer_strips_json_wrapper():
    out = clean_final_answer('{"final_answer": "Mayo Clinic"}')
    assert out == "Mayo Clinic"


def test_clean_final_answer_keeps_plain_answer():
    assert clean_final_answer("7 shirts") == "7 shirts"
    # Short answers shed trailing periods.
    assert clean_final_answer("7 shirts.") == "7 shirts"


def test_clean_final_answer_handles_bullet_evidence():
    txt = "- evidence one\n- evidence two\nThe answer is 45 minutes."
    out = clean_final_answer(txt)
    assert out.endswith("45 minutes")


# ─── validate_answer_shape ────────────────────────────────────────────────


def test_validate_count_requires_number():
    ok, _ = validate_answer_shape("7 shirts", QuestionType.COUNT)
    assert ok
    ok, reason = validate_answer_shape("a lot", QuestionType.COUNT)
    assert not ok and "count_missing_number" in reason


def test_validate_duration_requires_unit_and_number():
    ok, _ = validate_answer_shape("45 minutes", QuestionType.DURATION)
    assert ok
    ok, reason = validate_answer_shape("a long time", QuestionType.DURATION)
    assert not ok and "duration_missing" in reason


def test_validate_date_time_accepts_relative_phrases():
    ok, _ = validate_answer_shape("yesterday", QuestionType.DATE_TIME)
    assert ok
    ok, _ = validate_answer_shape("February 14th, 2024", QuestionType.DATE_TIME)
    assert ok


def test_validate_location_rejects_event_verbs():
    ok, _ = validate_answer_shape("Mayo Clinic", QuestionType.LOCATION)
    assert ok
    ok, reason = validate_answer_shape("redeemed the coupon", QuestionType.LOCATION)
    assert not ok and "location_not_place_shaped" in reason


def test_validate_person_requires_name_shape():
    ok, _ = validate_answer_shape("Dr. Smith", QuestionType.PERSON)
    assert ok
    ok, reason = validate_answer_shape("my doctor", QuestionType.PERSON)
    # Lowercase noun phrase → rejected.
    assert not ok and "entity_or_person_not_named" in reason


def test_validate_preference_rejects_generic_advice():
    ok, reason = validate_answer_shape(
        "you should try iced tea",
        QuestionType.PREFERENCE,
    )
    assert not ok and "preference_is_generic_advice" in reason
    ok, _ = validate_answer_shape("iced tea", QuestionType.PREFERENCE)
    assert ok


def test_validate_rejects_idk():
    ok, reason = validate_answer_shape(
        "I don't know based on the conversation.",
        QuestionType.COUNT,
    )
    assert not ok and reason == "answer_is_idk"


def test_validate_rejects_empty():
    ok, reason = validate_answer_shape("", QuestionType.COUNT)
    assert not ok and reason == "empty_answer"


# ─── is_idk / should_block_idk ────────────────────────────────────────────


def test_is_idk_recognises_common_refusals():
    assert is_idk("I don't know.")
    assert is_idk("I'm not sure based on what was shared.")
    assert is_idk("No information was provided.")
    assert not is_idk("7 shirts")


def test_should_block_idk_when_candidates_exist():
    cands = [Candidate("7", "7", "count", unit="shirts", confidence=0.9)]
    assert should_block_idk("I don't know", QuestionType.COUNT, cands)


def test_should_block_idk_letthrough_when_no_candidates():
    assert not should_block_idk("I don't know", QuestionType.COUNT, [])


def test_should_block_idk_ignores_non_idk_answers():
    cands = [Candidate("7", "7", "count", unit="shirts", confidence=0.9)]
    assert not should_block_idk("7 shirts", QuestionType.COUNT, cands)


# ─── build_repair_prompt ──────────────────────────────────────────────────


def test_build_repair_prompt_includes_candidates_and_shape_hint():
    cands = [
        Candidate("7", "7", "count", unit="shirts", confidence=0.9, source_session_id="s1"),
        Candidate("3", "3", "count", unit="hats", confidence=0.7, source_session_id="s2"),
    ]
    prompt = build_repair_prompt(
        "How many shirts?",
        QuestionType.COUNT,
        cands,
        failed_answer="I don't know",
        reason="answer_is_idk",
    )
    # Question + question type + reason all surface
    assert "How many shirts?" in prompt
    assert "COUNT" in prompt
    assert "answer_is_idk" in prompt
    # Each candidate's value + source session id rendered
    assert "7" in prompt and "shirts" in prompt
    assert "s1" in prompt and "s2" in prompt
    # Shape hint specific to COUNT present
    assert "number" in prompt.lower()
    # Hard guard against IDK
    assert "I don't know" in prompt or "I don’t know" in prompt


def test_build_repair_prompt_handles_no_candidates():
    prompt = build_repair_prompt(
        "How many shirts?",
        QuestionType.COUNT,
        candidates=[],
        failed_answer="",
        reason="empty_answer",
    )
    assert "no structured candidates" in prompt
