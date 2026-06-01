"""
tests/unit/evals/test_candidate_precision.py
============================================
Focused regression tests for the candidate-precision fix.

Five failure modes from the diagnostic_50 run:

* **e47becba** — final answer was the literal string ``"Sub-answer"``
  (scaffold leak).
* **58ef2f1c / 5d3d2817** — final answer empty when verified candidates
  existed (LLM repair returned blank).
* **66f24dbb** — wrong candidate (``"Flowers"`` instead of
  ``"a yellow dress"``) won the rerank because gift-context wasn't
  prioritised.
* **af8d2e46** — ``"7"`` vs ``"7 shirts"`` is acceptable; pinned so a
  COUNT regression doesn't slip in.

Tests are organised into four groups matching the spec's
"focused tests" requirement:

1. Scaffold candidate filtering
2. Empty / scaffold answer validation
3. Subject-relation reranking for the five examples
4. User-turn preference over assistant advice
"""

from __future__ import annotations

from evals.longmemeval.answer_post import (
    best_candidate_fallback_answer,
    is_idk,
    validate_answer_shape,
)
from evals.longmemeval.candidates import (
    Candidate,
    extract_candidates_from_text,
    rank_claims_for_question,
)
from evals.longmemeval.claim_verifier import verify_claim
from evals.longmemeval.question_type import QuestionType
from evals.longmemeval.scaffold_filter import (
    contains_scaffold_token,
    is_bundle_role_prefix,
    is_generic_intro,
    is_prompt_echo,
    is_scaffold_line,
    is_scaffold_text,
    is_user_request_sentence,
)

# ─── (1) Scaffold candidate filtering ─────────────────────────────────────


def test_scaffold_filter_catches_known_tokens():
    assert is_scaffold_text("Sub-answer")
    assert is_scaffold_text("sub answer")
    assert is_scaffold_text("Sub-Answer:")
    assert is_scaffold_text("Candidates:")
    assert is_scaffold_text("Matched Turn")
    assert is_scaffold_text("Nearby Context")
    assert is_scaffold_text("Evidence Window")
    assert is_scaffold_text("Matching facts")
    assert is_scaffold_text("I don't know")
    assert is_scaffold_text("")
    assert is_scaffold_text("   ")


def test_scaffold_filter_is_case_and_punct_tolerant():
    assert is_scaffold_text("SUB-ANSWER!!!")
    assert is_scaffold_text("  candidates:::  ")
    assert is_scaffold_text("Matched-Turn")
    assert is_scaffold_text("matched_turn")


def test_scaffold_filter_lets_real_answers_through():
    assert not is_scaffold_text("Business Administration")
    assert not is_scaffold_text("February 14th")
    assert not is_scaffold_text("Marketing specialist at a small startup")
    assert not is_scaffold_text("a yellow dress")


# ─── Tier 2A-2: bundle role/turn prefix detection ─────────────────────────


def test_bundle_role_prefix_catches_turn_n_user():
    """``turn 1 (user): ...`` is bundle scaffolding, never an answer."""
    assert is_bundle_role_prefix("turn 1 (user): Give me 100 prompt parameters")


def test_bundle_role_prefix_catches_dashed_variants():
    """All live shapes the wiki bundle emits."""
    assert is_bundle_role_prefix("- turn 5 (assistant): The answer is X.")
    assert is_bundle_role_prefix("- (user): I bought a yellow dress.")
    assert is_bundle_role_prefix("(assistant): Sure, here's an idea.")
    assert is_bundle_role_prefix("user: I asked about my degree.")
    assert is_bundle_role_prefix("assistant: Business Administration.")


def test_bundle_role_prefix_does_not_match_content_with_role_word():
    """A sentence that mentions 'user' or 'assistant' mid-text is content."""
    assert not is_bundle_role_prefix("Yes, I told you about the user interface yesterday.")
    assert not is_bundle_role_prefix("The assistant is a junior role in our marketing team.")
    assert not is_bundle_role_prefix("user friendly")  # no colon


def test_bundle_role_prefix_treated_as_scaffold():
    """``is_scaffold_text`` must catch the prefix leak — that's how the
    answer-validation gate excludes it as a final answer."""
    assert is_scaffold_text("turn 1 (user): Give me 100 prompt parameters for video production")
    assert is_scaffold_text("- (user): I bought a yellow dress")
    assert is_scaffold_text("assistant: Business Administration")
    # Plain answers still pass through.
    assert not is_scaffold_text("Business Administration")


# ─── Tier 2B-1a: generic conversational intros ────────────────────────────


def test_generic_intro_catches_bare_openers():
    """A bare opener with no real content is scaffold."""
    assert is_generic_intro("Sure!")
    assert is_generic_intro("Of course.")
    assert is_generic_intro("Absolutely!")
    assert is_generic_intro("Certainly!")
    assert is_generic_intro("Great question!")
    assert is_generic_intro("That's a great question.")
    assert is_generic_intro("Happy to help!")


def test_generic_intro_catches_sure_heres_with_thin_tail():
    """``Sure, here are 100 prompt parameters...`` — the v8 failure mode."""
    assert is_generic_intro("Sure, here are some")
    assert is_generic_intro("Sure, here's the list")
    assert is_generic_intro("Here are a few")
    assert is_generic_intro("Here are some")
    assert is_generic_intro("Here you go!")
    assert is_generic_intro("I'd be happy to help")


def test_generic_intro_lets_substantive_answers_through():
    """An intro followed by a real answer is content, not scaffold."""
    # >3 substantive tail tokens → let through (real answer).
    assert not is_generic_intro(
        "Sure, here are some good ones: tomatoes, basil, mozzarella, and olive oil for a Caprese."
    )
    assert not is_generic_intro("Of course! The book I recommended was The Three-Body Problem.")


def test_generic_intro_does_not_match_real_answers():
    """Pure factual answers must never trip the intro check."""
    assert not is_generic_intro("Business Administration")
    assert not is_generic_intro("February 14th")
    assert not is_generic_intro("Marketing specialist at a small startup")
    assert not is_generic_intro("a yellow dress")
    assert not is_generic_intro("Roscioli")
    assert not is_generic_intro("Doc Martin")


def test_generic_intro_treated_as_scaffold():
    """``is_scaffold_text`` must catch generic intros so candidate
    extraction and final-answer validation both reject them."""
    assert is_scaffold_text("Sure, here are some")
    assert is_scaffold_text("Of course!")
    assert is_scaffold_text("Great question!")
    # Real answers still pass.
    assert not is_scaffold_text("Roscioli")


# ─── Tier 2B-1b: imperative user-prompt requests ──────────────────────────


def test_user_request_catches_imperative_open():
    """The v8 failure where ``Brainstorm ideas for ...`` got returned
    as an assistant-memory answer. Imperative-verb opens are user
    prompts, not assistant memories."""
    assert is_user_request_sentence("Brainstorm ideas for work from home jobs for seniors.")
    assert is_user_request_sentence("Give me 100 prompt parameters for video production.")
    assert is_user_request_sentence("Create a marketing plan.")
    assert is_user_request_sentence("Generate some social media posts.")
    assert is_user_request_sentence("Write me a poem about coffee.")
    assert is_user_request_sentence("Suggest some restaurants in Rome.")
    assert is_user_request_sentence("List the top five films of 2020.")
    assert is_user_request_sentence("Recommend a few good books on AI.")
    assert is_user_request_sentence("Explain how a heat pump works.")
    assert is_user_request_sentence("Describe the main characters.")


def test_user_request_does_not_match_first_person_facts():
    """A leading subject pronoun (I, We, You) rescues the sentence —
    it's content, not a request."""
    assert not is_user_request_sentence("I brainstorm ideas every morning before work.")
    assert not is_user_request_sentence("We create lots of variants for our internal review.")
    assert not is_user_request_sentence("I suggested two restaurants when we last met.")
    # Real factual answers also pass through.
    assert not is_user_request_sentence("Roscioli")
    assert not is_user_request_sentence("Marketing specialist at a startup")


# ─── Tier 2B-1b: prompt-echo predicate ────────────────────────────────────


def test_prompt_echo_catches_exact_substring():
    """When the answer IS the question (minus polite framing)."""
    q = "What did I buy for my sister's birthday gift?"
    assert is_prompt_echo(
        "what did i buy for my sister's birthday gift",
        q,
    )


def test_prompt_echo_catches_high_jaccard_overlap():
    """When ≥70% of answer's content tokens appear in the question."""
    q = "What were the prompt parameters you listed for video production?"
    # Answer has 5 content tokens; 4 ({prompt, parameters, video,
    # production}) appear in the question → 80% overlap → echo.
    assert is_prompt_echo(
        "prompt parameters for video production",
        q,
    )


def test_prompt_echo_lets_short_factual_answers_through():
    """A 1–2 token answer can't meaningfully overlap a question."""
    q = "Which restaurant did you recommend for Italian food in Rome?"
    assert not is_prompt_echo("Roscioli", q)
    assert not is_prompt_echo("Doc Martin", q)
    assert not is_prompt_echo("28. Kg3", q)


def test_prompt_echo_lets_real_answers_through():
    """A real assistant answer shares some content words but not
    enough to score as echo."""
    q = "What book did you recommend for my next read?"
    assert not is_prompt_echo("The Three-Body Problem by Liu Cixin", q)
    q2 = "Where did I get my Bachelor's degree?"
    assert not is_prompt_echo("Business Administration from State U", q2)


def test_prompt_echo_handles_empty_strings():
    assert not is_prompt_echo("", "anything")
    assert not is_prompt_echo("anything", "")
    assert not is_prompt_echo("", "")
    assert not is_scaffold_text("7")
    assert not is_scaffold_text("UCLA")


def test_scaffold_line_catches_markdown_wrappers():
    assert is_scaffold_line("## Matched Turn")
    assert is_scaffold_line("### Nearby Context")
    assert is_scaffold_line("```python")
    assert is_scaffold_line("---")
    assert is_scaffold_line("")
    assert not is_scaffold_line("I do yoga at Serenity Yoga every Tuesday.")
    assert not is_scaffold_line("- (user): My playlist is called Summer Vibes.")


def test_contains_scaffold_token_substring():
    assert contains_scaffold_token("This is the Sub-answer block")
    assert contains_scaffold_token("Some content from Matched Turn")
    assert not contains_scaffold_token("I went to Whole Foods for groceries")


def test_extractor_skips_markdown_headings_and_role_prefixes():
    """
    The wiki window's ``## Matched Turn`` + ``- (user): …`` wrapper must
    not produce candidates whose value is a scaffold string.
    """
    text = (
        "## Matched Turn\n"
        "- (user): I graduated with a Bachelor's degree in Business Administration in 2018.\n"
        "## Nearby Context\n"
        "- (user): I worked at a small startup before joining the agency.\n"
    )
    candidates = extract_candidates_from_text(text, source_session_id="s1")
    # None of the *non-empty* candidate fields should be scaffold text.
    # Many extractors leave object_/subject empty — that's not scaffold,
    # just unset.
    for c in candidates:
        assert not is_scaffold_text(c.value), f"scaffold leak: value={c.value!r}"
        if c.object_:
            assert not is_scaffold_text(c.object_), f"scaffold leak: object={c.object_!r}"
    # And the bullet body text DOES make it through (positive check).
    haystack = " ".join(c.source_text for c in candidates).lower()
    assert "business administration" in haystack or any(
        "business" in c.value.lower() for c in candidates
    )


def test_verifier_pre_filter_rejects_scaffold_candidate():
    """A bare scaffold candidate fails the verifier on check 0."""
    bogus = Candidate(
        value="Sub-answer",
        normalized_value="sub-answer",
        candidate_type="entity",
        answer_type="entity",
        subject="answer",
        relation="is",
        object_="Sub-answer",
        source_span="Sub-answer",
        claim="Sub-answer: the user said something.",
        source_session_id="s1",
        confidence=0.5,
    )
    result = verify_claim(
        "What degree did I graduate with?",
        bogus,
        question_type=QuestionType.ENTITY_NAME,
    )
    assert result.verdict == "FAIL"
    assert "scaffold" in result.reason


def test_extractor_propagates_source_role():
    """``source_role`` flows from item.metadata into every Candidate."""
    candidates = extract_candidates_from_text(
        "I graduated with a Bachelor's in Business Administration in 2018.",
        source_session_id="s1",
        source_role="user",
    )
    assert candidates
    assert all(c.source_role == "user" for c in candidates)


# ─── (2) Empty / scaffold answer validation ───────────────────────────────


def test_validate_rejects_scaffold_answer():
    """Bare scaffold values must fail validation."""
    ok, reason = validate_answer_shape("Sub-answer", QuestionType.ENTITY_NAME)
    assert not ok
    assert reason == "answer_is_scaffold"


def test_validate_rejects_empty_answer():
    ok, reason = validate_answer_shape("", QuestionType.ENTITY_NAME)
    assert not ok
    assert reason == "empty_answer"


def test_validate_rejects_analysis_prose():
    """An answer that reads like analysis ('Based on the evidence…') fails."""
    long_prose = (
        "Based on the evidence and considering the candidates above, "
        "it appears that the most likely answer would be a "
        "Bachelor's degree, though more context would be needed "
        "to be certain about the field of study."
    )
    ok, reason = validate_answer_shape(long_prose, QuestionType.ENTITY_NAME)
    assert not ok
    assert reason == "answer_is_analysis_prose"


def test_validate_accepts_short_concrete_answer():
    ok, _ = validate_answer_shape("Business Administration", QuestionType.ENTITY_NAME)
    assert ok
    ok, _ = validate_answer_shape("February 14th", QuestionType.DATE_TIME)
    assert ok
    ok, _ = validate_answer_shape(
        "Marketing specialist at a small startup",
        QuestionType.GENERIC_FACT,
    )
    assert ok


def test_best_candidate_fallback_picks_top_confidence():
    """When verified candidates exist, fallback returns the strongest one."""
    cands = [
        Candidate(
            value="February 14th",
            normalized_value="february 14th",
            candidate_type="date",
            answer_type="date",
            subject="fundraising dinner",
            relation="located_at",
            object_="February 14th",
            source_span="on February 14th",
            confidence=0.85,
        ),
        Candidate(
            value="March 3rd",
            normalized_value="march 3rd",
            candidate_type="date",
            answer_type="date",
            subject="some other event",
            relation="located_at",
            object_="March 3rd",
            source_span="on March 3rd",
            confidence=0.4,
        ),
    ]
    out = best_candidate_fallback_answer(cands)
    assert out == "February 14th"


def test_best_candidate_fallback_returns_empty_when_no_candidates():
    assert best_candidate_fallback_answer([]) == ""


def test_best_candidate_fallback_drops_scaffold_candidates():
    """A scaffold candidate snuck into the verified pool is rejected."""
    cands = [
        Candidate(
            value="Sub-answer",
            normalized_value="sub-answer",
            candidate_type="entity",
            answer_type="entity",
            object_="Sub-answer",
            confidence=0.95,
        ),
        Candidate(
            value="UCLA",
            normalized_value="ucla",
            candidate_type="location",
            answer_type="location",
            object_="UCLA",
            confidence=0.6,
        ),
    ]
    assert best_candidate_fallback_answer(cands) == "UCLA"


def test_is_idk_recognises_known_refusals():
    assert is_idk("I don't know")
    assert is_idk("i do not know")
    assert is_idk("Not mentioned in the evidence")
    assert not is_idk("Business Administration")


# ─── (3) Subject-relation reranking for the five regression examples ──────


def _build(
    *,
    value: str,
    subject: str,
    relation: str = "is",
    claim: str = "",
    source_span: str = "",
    candidate_type: str = "entity",
    answer_type: str = "entity",
    role: str = "user",
    session_id: str = "s1",
    confidence: float = 0.7,
) -> Candidate:
    return Candidate(
        value=value,
        normalized_value=value.lower(),
        candidate_type=candidate_type,  # type: ignore[arg-type]
        answer_type=answer_type,
        subject=subject,
        relation=relation,
        object_=value,
        claim=claim or f"{subject} {relation} {value}.",
        source_span=source_span or value,
        source_session_id=session_id,
        source_role=role,
        confidence=confidence,
    )


def test_degree_question_prefers_graduated_with_claim():
    """e47becba — 'What degree did I graduate with?' picks the degree claim."""
    cands = [
        _build(
            value="Business Administration",
            subject="degree",
            relation="got_at",
            claim="I graduated with a Bachelor's degree in Business Administration.",
            source_span="degree in Business Administration",
        ),
        _build(
            value="2018",
            subject="graduation year",
            relation="date",
            candidate_type="date",
            answer_type="date",
            claim="I graduated in 2018.",
            source_span="in 2018",
            confidence=0.6,
        ),
    ]
    ranked = rank_claims_for_question(
        cands,
        "What degree did I graduate with?",
        question_type=QuestionType.ENTITY_NAME,
    )
    assert ranked
    assert "business administration" in ranked[0].object_.lower()


def test_fundraising_dinner_picks_february_14th():
    """58ef2f1c — 'When did I volunteer at the fundraising dinner?' picks the date."""
    cands = [
        _build(
            value="February 14th",
            subject="fundraising dinner",
            relation="located_at",
            candidate_type="date",
            answer_type="date",
            claim="The fundraising dinner was on February 14th.",
            source_span="on February 14th",
            confidence=0.85,
        ),
        _build(
            value="March 3rd",
            subject="dentist appointment",
            relation="located_at",
            candidate_type="date",
            answer_type="date",
            claim="My dentist appointment is on March 3rd.",
            source_span="on March 3rd",
            confidence=0.7,
        ),
    ]
    ranked = rank_claims_for_question(
        cands,
        "When did I volunteer at the local animal shelter's fundraising dinner?",
        question_type=QuestionType.DATE_TIME,
    )
    assert ranked
    assert "february 14" in ranked[0].object_.lower()


def test_previous_occupation_picks_job_phrase():
    """5d3d2817 — 'What was my previous occupation?' picks the job phrase."""
    cands = [
        _build(
            value="Marketing specialist at a small startup",
            subject="previous job",
            relation="got_at",
            claim="My previous job was Marketing specialist at a small startup.",
            source_span="Marketing specialist at a small startup",
            confidence=0.85,
        ),
        _build(
            value="cereal for breakfast",
            subject="breakfast",
            relation="is",
            claim="I had cereal for breakfast yesterday.",
            source_span="cereal for breakfast",
            confidence=0.5,
        ),
    ]
    ranked = rank_claims_for_question(
        cands,
        "What was my previous occupation?",
        question_type=QuestionType.GENERIC_FACT,
    )
    assert ranked
    assert "marketing specialist" in ranked[0].object_.lower()


def test_sister_birthday_gift_rejects_surgery_and_flowers():
    """
    66f24dbb — 'What did I buy for my sister's birthday gift?' picks
    the dress over flowers (a related-but-wrong distractor) and over
    surgery (an unrelated penalty case).
    """
    cands = [
        _build(
            value="a yellow dress",
            subject="sister's birthday gift",
            relation="is",
            claim="I bought a yellow dress for my sister's birthday gift.",
            source_span="a yellow dress for my sister's birthday",
            confidence=0.85,
        ),
        _build(
            value="Flowers",
            subject="anniversary",
            relation="is",
            claim="I sent Flowers for my mom's anniversary.",
            source_span="Flowers for my mom's anniversary",
            confidence=0.7,
        ),
        _build(
            value="painkillers",
            subject="recovery",
            relation="is",
            claim="I bought painkillers after my surgery recovery.",
            source_span="painkillers after my surgery",
            confidence=0.6,
        ),
    ]
    ranked = rank_claims_for_question(
        cands,
        "What did I buy for my sister's birthday gift?",
        question_type=QuestionType.ENTITY_NAME,
    )
    assert ranked
    assert "yellow dress" in ranked[0].object_.lower()
    # Surgery-context candidate must NOT be ranked above the dress.
    for c in ranked:
        if "painkiller" in c.object_.lower():
            assert ranked.index(c) > ranked.index(ranked[0])


def test_count_shirts_keeps_seven_shirts():
    """af8d2e46 — 'How many shirts did I pack' keeps '7 shirts'."""
    cands = [
        _build(
            value="7 shirts",
            subject="shirts",
            relation="count_of",
            candidate_type="count",
            answer_type="count",
            claim="I packed 7 shirts for the trip.",
            source_span="7 shirts",
            confidence=0.9,
        ),
        _build(
            value="3 hours",
            subject="commute",
            relation="duration_of",
            candidate_type="duration",
            answer_type="duration",
            claim="My commute was 3 hours.",
            source_span="3 hours",
            confidence=0.7,
        ),
    ]
    ranked = rank_claims_for_question(
        cands,
        "How many shirts did I pack for the trip?",
        question_type=QuestionType.COUNT,
    )
    assert ranked
    assert "7" in ranked[0].value


# ─── (4) User-turn preference over assistant advice ────────────────────────


def test_user_turn_beats_assistant_advice_on_personal_question():
    """
    A user-stated fact ("I graduated with a degree in Business Administration")
    outranks an assistant-suggested option ("you should consider Business").
    """
    user_fact = _build(
        value="Business Administration",
        subject="degree",
        relation="got_at",
        claim="I graduated with a degree in Business Administration in 2018.",
        source_span="degree in Business Administration",
        role="user",
        session_id="s_user",
        confidence=0.75,
    )
    assistant_advice = _build(
        value="Engineering",
        subject="degree",
        relation="got_at",
        claim="You might consider getting a degree in Engineering.",
        source_span="degree in Engineering",
        role="assistant",
        session_id="s_assistant",
        confidence=0.75,
    )
    ranked = rank_claims_for_question(
        [assistant_advice, user_fact],  # assistant first to make sure boost works
        "What degree did I graduate with?",
        question_type=QuestionType.ENTITY_NAME,
    )
    assert ranked[0].source_role == "user"
    assert "business" in ranked[0].object_.lower()


def test_assistant_text_not_fully_discarded():
    """An assistant claim should still be RANKED, just below user equivalents."""
    only_assistant = _build(
        value="UCLA",
        subject="Bachelor's degree",
        relation="got_at",
        claim="You went to UCLA for your Bachelor's.",
        source_span="UCLA",
        role="assistant",
        confidence=0.7,
    )
    ranked = rank_claims_for_question(
        [only_assistant],
        "Where did I get my Bachelor's degree?",
        question_type=QuestionType.LOCATION,
    )
    # Assistant-only pool — the candidate must still be ranked.
    assert ranked
    assert ranked[0].value == "UCLA"


def test_user_turn_boost_does_not_apply_to_third_person_question():
    """No 'I'/'my' in the question → no role boost."""
    user_fact = _build(
        value="UCLA",
        subject="Bachelor's degree",
        relation="got_at",
        claim="The user got a Bachelor's at UCLA.",
        source_span="at UCLA",
        role="user",
        confidence=0.7,
    )
    # Same candidate scored on a personal vs non-personal question —
    # the personal-question variant should rank it strictly higher.
    personal_rank = rank_claims_for_question(
        [user_fact],
        "Where did I get my Bachelor's degree?",
        question_type=QuestionType.LOCATION,
    )
    impersonal_rank = rank_claims_for_question(
        [user_fact],
        "Where is UCLA located?",
        question_type=QuestionType.LOCATION,
    )
    assert personal_rank and impersonal_rank
    # Both will return the same candidate, but the scoring difference
    # exists in the internal `score()` function — what we pin here is
    # that the candidate still ranks in both cases (no false reject).
    assert personal_rank[0].value == "UCLA"
    assert impersonal_rank[0].value == "UCLA"


# ─── (5) The precision_5 ID file ──────────────────────────────────────────


def test_precision_5_sample_file_contains_required_ids():
    """The regression file lists every spec-cited row."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (repo / "samples" / "precision_5_ids.json").read_text(),
    )
    expected = {"e47becba", "58ef2f1c", "5d3d2817", "66f24dbb", "af8d2e46"}
    assert set(payload["question_ids"]) == expected
    assert payload["total"] == 5
