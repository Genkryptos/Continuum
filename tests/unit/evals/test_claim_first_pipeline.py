"""
tests/unit/evals/test_claim_first_pipeline.py
=============================================
Focused tests for the claim-first evidence layer + broad task router
+ six reasoning heads.

Test groups (mirroring the spec's "Tests" section):

1. Scaffold / noise filtering (incl. transcript-timestamp junk).
2. Claim extraction with role / session metadata.
3. User-turn preference for autobiographical facts.
4. Assistant-turn preference for assistant-memory questions.
5. Task routing into the six broad modes.
6. Final-answer validation against source claims.
7. Fallback prompt when question type / candidate type is wrong.
8. Shortest-span extraction for the six regression examples.
"""
from __future__ import annotations

from continuum.core.types import ContextBundle, MemoryItem, TokenBudget
from evals.longmemeval.answer_post import validate_against_claims
from evals.longmemeval.claims import (
    Claim,
    extract_claims_from_context,
    is_transcript_timestamp_noise,
    rank_claims,
)
from evals.longmemeval.extractive_fallback import (
    build_extractive_prompt,
    validate_extracted_span,
)
from evals.longmemeval.reasoning_heads import (
    head_assistant_memory,
    head_fact_lookup,
    head_knowledge_update,
    head_multi_session_aggregate,
    head_preference_profile,
    head_temporal_reasoning,
)
from evals.longmemeval.task_router import TaskMode, route_task


def _bundle(items: list[MemoryItem]) -> ContextBundle:
    return ContextBundle(
        items=items, messages=[], tokens_used=0,
        budget=TokenBudget(
            total=4096,
            stm_reserved=1024,
            mtm_reserved=1024,
            ltm_reserved=1024,
            response_reserved=1024,
        ),
        tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
    )


def _item(
    *, session_id: str, content: str, role: str = "user",
    item_id: str = "",
) -> MemoryItem:
    return MemoryItem(
        id=item_id or f"item-{session_id}",
        content=content,
        session_id=session_id,
        metadata={"session_id": session_id, "role": role},
    )


# ─── (1) Scaffold / noise filtering ────────────────────────────────────────


def test_transcript_timestamp_noise_filter():
    assert is_transcript_timestamp_noise("02 share")
    assert is_transcript_timestamp_noise("04 principles")
    assert is_transcript_timestamp_noise("0:02")
    assert is_transcript_timestamp_noise("00:00:12")
    assert is_transcript_timestamp_noise("  3:14  ")
    assert not is_transcript_timestamp_noise(
        "I graduated with a degree in Business Administration."
    )
    assert not is_transcript_timestamp_noise("My favorite playlist")


def test_claim_extraction_skips_scaffold_and_timestamps():
    """Markdown headings + ``02 share`` style noise never become claims."""
    text = (
        "## Matched Turn\n"
        "- (user): I got my Bachelor's in Business Administration in 2018.\n"
        "## Nearby Context\n"
        "02 share\n"
        "0:14\n"
        "- (assistant): That sounds like a useful degree.\n"
    )
    items = [_item(session_id="s1", content=text)]
    claims = extract_claims_from_context(_bundle(items))
    texts = [c.text for c in claims]
    # Real content survives:
    assert any("Business Administration" in t for t in texts)
    # Scaffold gone:
    assert not any("Matched Turn" in t for t in texts)
    assert not any("Nearby Context" in t for t in texts)
    # Timestamp noise gone:
    assert not any(t.strip() in ("02 share", "0:14") for t in texts)


# ─── (2) Claim extraction with role / session metadata ────────────────────


def test_claims_carry_role_and_session_metadata():
    items = [
        _item(
            session_id="s_edu",
            content=(
                "- (user): I got my Bachelor's in Business Administration in 2018.\n"
                "- (assistant): That's a great field.\n"
            ),
        ),
    ]
    claims = extract_claims_from_context(_bundle(items))
    by_role = {c.source_role: c for c in claims}
    assert "user" in by_role
    assert "assistant" in by_role
    assert all(c.source_session_id == "s_edu" for c in claims)
    # Turn index is monotonically increasing for ordering downstream
    assert claims[0].turn_index < claims[-1].turn_index


def test_legacy_claim_extraction_strips_turn_n_role_prefix():
    """
    Tier 2A-1 regression pin: a Nearby Context line shaped as
    ``- turn 1 (user): Give me 100 prompt parameters`` must yield a
    claim whose ``text`` is the message body only — never the bundle
    scaffolding. Without this fix the legacy assistant-memory head
    returned the literal prefixed string as the answer (the 8752c811
    regression in the v7 run).
    """
    text = (
        "## Matched Turn\n"
        "- (user): What are the categories?\n"
        "## Nearby Context\n"
        "- turn 1 (user): Give me 100 prompt parameters for video "
        "production.\n"
        "- turn 2 (assistant): Sound effects, lighting, framing, and "
        "camera angles.\n"
    )
    items = [_item(session_id="s_dialog", content=text)]
    claims = extract_claims_from_context(_bundle(items))
    texts = [c.text for c in claims]
    # No claim should contain the raw bundle scaffolding.
    assert not any(t.lower().startswith("turn ") for t in texts), \
        f"role-prefix leaked into claim text: {texts}"
    assert not any("turn 1 (user)" in t for t in texts)
    assert not any("turn 2 (assistant)" in t for t in texts)
    # The actual content survives, with the right role tagging.
    by_role = {(c.source_role, c.text): c for c in claims}
    user_texts = [text for (role, text), _c in by_role.items() if role == "user"]
    assistant_texts = [text for (role, text), _c in by_role.items() if role == "assistant"]
    assert any("video production" in t.lower() for t in user_texts)
    assert any("sound effects" in t.lower() for t in assistant_texts)


def test_legacy_claim_extraction_strips_inline_role_prefix():
    """
    The bundle sometimes embeds the assistant reply inside a user
    turn as ``\\nassistant: text``. The legacy extractor must
    re-attribute the post-prefix sentence to ``assistant``.
    """
    text = (
        "- (user): I asked about my degree.\n"
        "assistant: Business Administration is a versatile field.\n"
    )
    items = [_item(session_id="s_inline", content=text)]
    claims = extract_claims_from_context(_bundle(items))
    by_role: dict[str, list[str]] = {}
    for c in claims:
        by_role.setdefault(c.source_role, []).append(c.text)
    assert any("degree" in t.lower() for t in by_role.get("user", []))
    assert any(
        "business administration" in t.lower()
        for t in by_role.get("assistant", [])
    )


def test_claim_extraction_preserves_content_for_business_admin():
    """The degree sentence survives extraction verbatim."""
    items = [_item(
        session_id="s1",
        content="## Matched Turn\n"
                "- (user): I graduated with a degree in Business "
                "Administration from a state university.\n",
    )]
    claims = extract_claims_from_context(_bundle(items))
    assert claims
    assert any(
        "Business Administration" in c.text for c in claims
    )


# ─── (3) User-turn preference for autobiographical facts ──────────────────


def test_user_turn_outranks_assistant_for_personal_question():
    claims = [
        Claim(
            text="You might consider getting a degree in Engineering.",
            source_session_id="s1", source_role="assistant", confidence=0.6,
        ),
        Claim(
            text="I got my degree in Business Administration in 2018.",
            source_session_id="s1", source_role="user", confidence=0.6,
        ),
    ]
    ranked = rank_claims(
        "What degree did I graduate with?", claims, prefer_role="user",
    )
    assert ranked[0].source_role == "user"
    assert "Business Administration" in ranked[0].text


# ─── (4) Assistant-turn preference for assistant-memory questions ─────────


def test_assistant_turn_boosted_when_question_asks_what_assistant_said():
    claims = [
        Claim(
            text="I really enjoyed that thriller you suggested.",
            source_session_id="s1", source_role="user", confidence=0.6,
        ),
        Claim(
            text="You should try \"The Silent Patient\" — it's a good thriller.",
            source_session_id="s1", source_role="assistant", confidence=0.6,
        ),
    ]
    ranked = rank_claims(
        "What book did you recommend to me?", claims,
    )
    # Auto-detected from "did you recommend" → assistant preferred.
    assert ranked[0].source_role == "assistant"
    assert "Silent Patient" in ranked[0].text


# ─── (5) Task routing into six broad modes ────────────────────────────────


def test_route_fact_lookup_for_simple_user_question():
    assert route_task(
        "Where did I get my Bachelor's degree?",
        question_type_hint="single-session-user",
    ) == TaskMode.FACT_LOOKUP


def test_route_assistant_memory_for_did_you_questions():
    assert route_task(
        "What book did you recommend last week?",
        question_type_hint="single-session-assistant",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_route_preference_for_favourite_questions():
    assert route_task(
        "What's my favorite restaurant?",
        question_type_hint="single-session-preference",
    ) == TaskMode.PREFERENCE_PROFILE


def test_route_knowledge_update_for_change_words():
    assert route_task(
        "What did I change my dog's name to?",
        question_type_hint="knowledge-update",
    ) == TaskMode.KNOWLEDGE_UPDATE


def test_route_multi_session_aggregate_for_how_many():
    assert route_task(
        "How many shirts did I pack across all my trips?",
        question_type_hint="multi-session",
    ) == TaskMode.MULTI_SESSION_AGGREGATE


def test_route_temporal_reasoning_for_how_long_ago():
    assert route_task(
        "How long ago did I start my job?",
        question_type_hint="temporal-reasoning",
    ) == TaskMode.TEMPORAL_REASONING


def test_router_wording_overrides_wrong_dataset_label():
    """Strong wording cues outrank a misleading dataset hint."""
    # Dataset says "single-session-user" but wording is temporal.
    assert route_task(
        "How long ago did I start the job?",
        question_type_hint="single-session-user",
    ) == TaskMode.TEMPORAL_REASONING


def test_router_falls_back_to_fact_lookup_when_neutral():
    """No wording cues + no dataset hint → FACT_LOOKUP."""
    assert route_task(
        "Where did I park?", question_type_hint=None,
    ) == TaskMode.FACT_LOOKUP


# ─── Tier 1A: dataset hint plumbing ───────────────────────────────────────


def test_dataset_hint_routes_ambiguous_single_session_assistant():
    """
    Surface-form-ambiguous question routes to ASSISTANT_MEMORY when the
    dataset hint says single-session-assistant. Without the hint it
    would default to FACT_LOOKUP.
    """
    q = "What was the title of the book?"
    assert route_task(q, question_type_hint=None) == TaskMode.FACT_LOOKUP
    assert route_task(
        q, question_type_hint="single-session-assistant",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_dataset_hint_routes_ambiguous_preference():
    """A bare "What X" with preference hint routes to PREFERENCE_PROFILE."""
    q = "What restaurant comes to mind for Italian food?"
    assert route_task(q, question_type_hint=None) == TaskMode.FACT_LOOKUP
    assert route_task(
        q, question_type_hint="single-session-preference",
    ) == TaskMode.PREFERENCE_PROFILE


def test_dataset_hint_routes_ambiguous_multi_session():
    """Multi-session aggregation when wording lacks explicit cues."""
    q = "What gifts have I bought?"
    assert route_task(q, question_type_hint=None) == TaskMode.FACT_LOOKUP
    assert route_task(
        q, question_type_hint="multi-session",
    ) == TaskMode.MULTI_SESSION_AGGREGATE


def test_dataset_hint_routes_ambiguous_knowledge_update():
    """Knowledge-update without 'changed/now/latest' wording cue."""
    q = "What is my pet's name?"
    assert route_task(q, question_type_hint=None) == TaskMode.FACT_LOOKUP
    assert route_task(
        q, question_type_hint="knowledge-update",
    ) == TaskMode.KNOWLEDGE_UPDATE


def test_is_multi_session_flag_forces_aggregate():
    """is_multi_session=True forces aggregate even without wording."""
    assert route_task(
        "What did I buy?",
        is_multi_session=True,
    ) == TaskMode.MULTI_SESSION_AGGREGATE


def test_dataset_hint_does_not_override_assistant_memory_cue():
    """Strong 'did you recommend' cue beats a wrong hint."""
    assert route_task(
        "What book did you recommend last week?",
        question_type_hint="single-session-user",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_assistant_memory_remind_me_routes_to_assistant():
    """``remind me what you recommended for X`` is assistant-memory."""
    assert route_task(
        "Remind me what you recommended for my next read.",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_assistant_memory_do_you_remember_routes_to_assistant():
    assert route_task(
        "Do you remember the book we discussed earlier?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_assistant_memory_we_talked_about_routes_to_assistant():
    assert route_task(
        "What was the playlist we talked about last week?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_assistant_memory_previous_conversation_routes_to_assistant():
    """
    The 4 wrong_route→TEMPORAL cases all carried 'previous|earlier|last'
    inside an assistant-memory frame. The new cue must beat the
    temporal regex.
    """
    assert route_task(
        "What was the advice in our previous conversation?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP
    assert route_task(
        "Can you check back on what you suggested in the earlier session?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_assistant_memory_what_was_the_x_you_gave_routes_to_assistant():
    """``what was the recipe you gave me`` — answer-recovery frame."""
    assert route_task(
        "What was the recipe you gave me for chocolate cake?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP
    assert route_task(
        "What were the three parameters you listed for me?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_assistant_memory_with_temporal_words_still_routes_correctly():
    """
    'last week' / 'earlier' / 'previous' co-occurring with an
    assistant-memory cue must not flip the route to TEMPORAL — this
    was the v7 regression source.
    """
    # 'last week' alone is temporal; with 'you recommended' it's
    # assistant-memory.
    assert route_task(
        "What book did you recommend last week?",
    ) == TaskMode.ASSISTANT_MEMORY_LOOKUP


def test_unknown_dataset_hint_falls_through_to_default():
    """Unrecognised hint string degrades gracefully to FACT_LOOKUP."""
    assert route_task(
        "Where did I park?",
        question_type_hint="totally-unknown-category",
    ) == TaskMode.FACT_LOOKUP


# ─── Tier 1C: refined failure buckets ─────────────────────────────────────


from evals.longmemeval.baseline import _classify_failure  # noqa: E402


def _stats(**kw):
    """Build a decomposition_stats dict for the tests."""
    return dict(kw)


def test_failure_llm_error_short_circuits():
    assert _classify_failure(
        answer="[error: timeout]", error=None,
        retrieved=["s"], expected=["s"],
    ) == "llm_error"


def test_failure_missing_fact_when_no_expected_session_retrieved():
    assert _classify_failure(
        answer="anything", error=None,
        retrieved=["s_other"], expected=["s_gold"],
    ) == "missing_fact"


def test_failure_wrong_route_when_actual_route_disagrees():
    """Assistant-memory question routed as FACT_LOOKUP → wrong_route."""
    assert _classify_failure(
        answer="something",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="single-session-assistant",
        decomposition_stats=_stats(
            mode="claim_first",
            claim_first_route="FACT_LOOKUP",
        ),
    ) == "wrong_route"


def test_failure_wrong_role_assistant_question_from_user_turn():
    """
    Question type single-session-assistant; matched_claim's
    source_role is "user" — the answer came from the wrong speaker.
    """
    assert _classify_failure(
        answer="The Three-Body Problem",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="single-session-assistant",
        decomposition_stats=_stats(
            mode="claim_first_structured",
            claim_first_route="ASSISTANT_MEMORY_LOOKUP",
            structured_matched_claim={"source_role": "user", "text": "..."},
        ),
    ) == "wrong_role"


def test_failure_wrong_role_user_question_from_assistant_turn():
    assert _classify_failure(
        answer="Marketing specialist",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="single-session-user",
        decomposition_stats=_stats(
            mode="claim_first_structured",
            claim_first_route="FACT_LOOKUP",
            structured_matched_claim={"source_role": "assistant", "text": "..."},
        ),
    ) == "wrong_role"


def test_failure_missing_aggregation_for_multi_session_single_span():
    """Multi-session that returned one short noun phrase → not aggregated."""
    assert _classify_failure(
        answer="yellow dress",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="multi-session",
        decomposition_stats=_stats(
            mode="claim_first_structured",
            claim_first_route="MULTI_SESSION_AGGREGATE",
        ),
    ) == "missing_aggregation"


def test_failure_missing_temporal_for_bare_date_answer():
    """Temporal question + a bare date answer → missing arithmetic."""
    assert _classify_failure(
        answer="February 14th",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="temporal-reasoning",
        decomposition_stats=_stats(
            mode="claim_first_structured",
            claim_first_route="TEMPORAL_REASONING",
        ),
    ) == "missing_temporal_computation"


def test_failure_stale_fact_for_knowledge_update_with_used_to():
    """
    Knowledge-update question; the matched claim's text contains
    "used to" — the stale half of the update pair was selected.
    """
    assert _classify_failure(
        answer="Bella",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="knowledge-update",
        decomposition_stats=_stats(
            mode="claim_first_structured",
            claim_first_route="KNOWLEDGE_UPDATE",
            structured_matched_claim={
                "source_role": "user",
                "text": "My dog used to be called Bella but we renamed her.",
            },
        ),
    ) == "stale_fact"


def test_failure_wrong_span_when_route_right_session_right():
    """
    Right route, right session, but the answer doesn't match gold and
    no other specific bucket fires → wrong_span.
    """
    assert _classify_failure(
        answer="Best Buy",
        error=None,
        retrieved=["s_gold"], expected=["s_gold"],
        question_type="single-session-user",
        decomposition_stats=_stats(
            mode="claim_first",
            claim_first_route="FACT_LOOKUP",
        ),
        expected_answer="a yellow dress",
    ) == "wrong_span"


def test_failure_backward_compat_without_new_kwargs():
    """
    Old call site that doesn't pass the new kwargs still classifies
    correctly. With Tier 1C the right-session-wrong-answer case lands
    in ``wrong_span`` — more precise than the legacy
    ``wrong_retrieval`` catch-all.
    """
    assert _classify_failure(
        answer="anything", error=None,
        retrieved=["s"], expected=["s"],
    ) == "wrong_span"
    # Pure retrieval miss still classifies as missing_fact.
    assert _classify_failure(
        answer="anything", error=None,
        retrieved=["s_other"], expected=["s_gold"],
    ) == "missing_fact"


# ─── (6) Final-answer validation against claims ───────────────────────────


def test_validation_rejects_empty_and_scaffold():
    ok, reason = validate_against_claims("", ["x"])
    assert not ok and reason == "empty_answer"
    ok, reason = validate_against_claims("Sub-answer", ["x"])
    assert not ok and reason == "answer_is_scaffold"
    ok, reason = validate_against_claims("I don't know", ["x"])
    assert not ok and "idk" in reason


def test_validation_rejects_unsupported_answer():
    ok, reason = validate_against_claims(
        "Berkeley", ["I got my Bachelor's at UCLA in 2018."],
    )
    assert not ok
    assert reason == "unsupported_by_claim"


def test_validation_accepts_substring_answer():
    ok, _ = validate_against_claims(
        "Business Administration",
        ["I got my degree in Business Administration in 2018."],
    )
    assert ok


def test_validation_accepts_token_overlap_paraphrase():
    """High-token-overlap paraphrase passes — semantic match."""
    ok, _ = validate_against_claims(
        "yellow dress",
        ["I bought a yellow dress for my sister's birthday."],
    )
    assert ok


def test_validation_accepts_when_require_support_false():
    ok, _ = validate_against_claims("UCLA", [], require_support=False)
    assert ok


# ─── (7) Fallback prompt + extractive-span validator ──────────────────────


def test_extractive_fallback_prompt_includes_question_and_claims():
    claims = [
        Claim(
            text="I graduated with a Bachelor's in Business Administration in 2018.",
            source_session_id="s_edu", source_role="user",
        ),
        Claim(
            text="I worked at a small startup before joining the agency.",
            source_session_id="s_work", source_role="user",
        ),
    ]
    prompt = build_extractive_prompt(
        "What degree did I graduate with?", claims, max_claims=2,
    )
    assert "What degree did I graduate with?" in prompt
    assert "Business Administration" in prompt
    assert "session=s_edu" in prompt
    assert "session=s_work" in prompt
    assert "SHORTEST" in prompt or "shortest" in prompt.lower()


def test_extractive_fallback_validator_accepts_substring():
    claims = [
        Claim(
            text="I bought a yellow dress for my sister's birthday.",
            source_session_id="s1", source_role="user",
        ),
    ]
    assert validate_extracted_span("a yellow dress", claims)
    assert validate_extracted_span("yellow dress", claims)
    assert not validate_extracted_span("a red coat", claims)


def test_extractive_fallback_prompt_empty_when_no_claims():
    assert build_extractive_prompt("anything", []) == ""


# ─── (8) Shortest-span extraction for the six regression examples ────────


def test_e47becba_business_administration():
    """FACT_LOOKUP — degree fact."""
    claims = [Claim(
        text="I graduated with a degree in Business Administration from a state university in 2018.",
        source_session_id="s1", source_role="user",
    )]
    out = head_fact_lookup(claims, "What degree did I graduate with?")
    assert "Business Administration" in out
    # Shortest faithful — must not return the entire sentence.
    assert len(out) < 60


def test_58ef2f1c_february_14th():
    """FACT_LOOKUP — date fact."""
    claims = [Claim(
        text="The animal shelter's fundraising dinner was on February 14th, 2024.",
        source_session_id="s1", source_role="user",
    )]
    out = head_fact_lookup(
        claims,
        "When did I volunteer at the local animal shelter's fundraising dinner?",
    )
    assert "February" in out and "14" in out


def test_5d3d2817_marketing_specialist():
    """FACT_LOOKUP — previous occupation."""
    claims = [Claim(
        text="Before this role I was a Marketing specialist at a small startup downtown.",
        source_session_id="s1", source_role="user",
    )]
    out = head_fact_lookup(claims, "What was my previous occupation?")
    assert "Marketing specialist" in out


def test_66f24dbb_yellow_dress():
    """FACT_LOOKUP — birthday gift (object after 'bought')."""
    claims = [Claim(
        text="I bought a yellow dress for my sister's birthday last weekend.",
        source_session_id="s1", source_role="user",
    )]
    out = head_fact_lookup(
        claims, "What did I buy for my sister's birthday gift?",
    )
    assert "yellow dress" in out.lower()


def test_af8d2e46_seven_shirts():
    """MULTI_SESSION_AGGREGATE — count with unit, single session here."""
    claims = [Claim(
        text="I packed 7 shirts for the trip last week.",
        source_session_id="s1", source_role="user",
    )]
    out = head_multi_session_aggregate(
        claims, "How many shirts did I pack for the trip?",
    )
    assert "7" in out


def test_b86304ba_triple_what_i_paid():
    """FACT_LOOKUP — comparative value fact."""
    claims = [Claim(
        text="Today the painting is worth triple what I paid for it back in 2015.",
        source_session_id="s1", source_role="user",
    )]
    out = head_fact_lookup(claims, "What is the painting worth today?")
    assert "triple" in out.lower()
    assert "paid" in out.lower()


# ─── (extra) Each head never returns "I don't know" when claims exist ────


def test_no_head_returns_idk_when_claims_exist():
    claims = [Claim(
        text="I went to the park yesterday afternoon.",
        source_session_id="s1", source_role="user",
    )]
    for head in (
        head_fact_lookup,
        head_assistant_memory,
        head_preference_profile,
        head_knowledge_update,
    ):
        out = head(claims, "Where did I go yesterday?")
        assert out != ""
        assert "i don't know" not in out.lower()


def test_head_returns_empty_when_no_claims():
    """An honest abstain only happens when truly no claims exist."""
    for head in (
        head_fact_lookup,
        head_assistant_memory,
        head_preference_profile,
        head_knowledge_update,
        head_multi_session_aggregate,
        head_temporal_reasoning,
    ):
        assert head([], "anything") == ""


# ─── Tier 2B-2: assistant-memory anchor + 2B-3: ordinal extraction ────────


def _turn_claims(
    *texts: str,
    session_id: str,
    role: str,
    base_turn_index: int,
) -> list[Claim]:
    """Build a contiguous run of Claims with shared role/session and
    monotonically-increasing turn_index — the same shape
    extract_claims_from_context produces from one item line."""
    return [
        Claim(
            text=t,
            source_session_id=session_id,
            source_role=role,
            turn_index=base_turn_index + i,
        )
        for i, t in enumerate(texts)
    ]


def test_assistant_memory_picks_response_after_matching_user_turn():
    """
    Two user turns ask different things; the assistant answers each
    in the following turn. The head must pick the assistant turn
    whose preceding user turn matches the question — not the other.
    """
    claims = [
        *_turn_claims(
            "Recommend a restaurant in Rome.",
            session_id="s", role="user", base_turn_index=100,
        ),
        *_turn_claims(
            "Roscioli is great for Roman pasta.",
            session_id="s", role="assistant", base_turn_index=200,
        ),
        *_turn_claims(
            "What about a hotel in Barcelona?",
            session_id="s", role="user", base_turn_index=300,
        ),
        *_turn_claims(
            "Hotel Casa Camper has a quiet courtyard.",
            session_id="s", role="assistant", base_turn_index=400,
        ),
    ]
    # Question matches the Rome user turn — answer must come from
    # the Roscioli assistant turn, not the Casa Camper one.
    out = head_assistant_memory(
        claims, "What Italian restaurant did you recommend in Rome?",
    )
    assert "Roscioli" in out
    assert "Casa Camper" not in out


def test_assistant_memory_skips_unrelated_user_turns():
    """
    Even if an unrelated assistant turn has high overlap with the
    question, the anchor logic prefers the assistant response
    immediately after the matching user turn.
    """
    claims = [
        *_turn_claims(
            "What book should I read next?",
            session_id="s", role="user", base_turn_index=100,
        ),
        *_turn_claims(
            "Try The Three-Body Problem by Liu Cixin.",
            session_id="s", role="assistant", base_turn_index=200,
        ),
        *_turn_claims(
            "Tell me about something unrelated entirely.",
            session_id="s", role="user", base_turn_index=300,
        ),
        # Decoy: matches "book" topic but isn't the answer.
        *_turn_claims(
            "Books I'm reading: Dune, Foundation, and book trivia.",
            session_id="s", role="assistant", base_turn_index=400,
        ),
    ]
    out = head_assistant_memory(claims, "What book did you recommend?")
    assert "Three-Body" in out or "Liu Cixin" in out


def test_ordinal_extraction_seventh_job():
    """
    Anchored assistant response contains a numbered list; the
    question asks for the 7th item — that's what comes back.
    Mirrors the 1903aded shape.
    """
    user = _turn_claims(
        "Brainstorm ideas for work from home jobs for seniors.",
        session_id="s", role="user", base_turn_index=100,
    )
    # Assistant turn: ten list items as separate claims.
    jobs = [
        "Customer service representative", "Bookkeeper", "Online tutor",
        "Virtual assistant", "Proofreader", "Translator",
        "Transcriptionist", "Data entry clerk", "Social media manager",
        "Pet sitter",
    ]
    assistant = _turn_claims(
        *jobs, session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant,
        "What was the 7th job you suggested?",
    )
    assert "Transcriptionist" in out


def test_ordinal_extraction_explicit_numbering_survives():
    """``1. Foo\\n2. Bar`` — the head returns item N even when the
    explicit number prefix survived sentence splitting."""
    user = _turn_claims(
        "Give me three ingredient suggestions.",
        session_id="s", role="user", base_turn_index=100,
    )
    assistant = _turn_claims(
        "1. Sun-dried tomatoes",
        "2. Fresh mozzarella",
        "3. Extra-virgin olive oil",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant,
        "What was the second ingredient you suggested?",
    )
    assert "Fresh mozzarella" in out
    assert "tomato" not in out.lower()


def test_ordinal_extraction_last_picks_final_item():
    """``last`` resolves to the highest-indexed item / final claim."""
    user = _turn_claims(
        "Give me five tips.",
        session_id="s", role="user", base_turn_index=100,
    )
    assistant = _turn_claims(
        "Stretch every morning",
        "Drink water",
        "Get sunlight",
        "Read daily",
        "Sleep eight hours",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant, "What was the last tip you mentioned?",
    )
    assert "Sleep eight hours" in out


def test_ordinal_falls_through_to_span_when_no_list_match():
    """
    Question asks for the 27th item but the assistant turn only has
    3 items. The ordinal extraction returns nothing; the span
    extractor takes over.
    """
    user = _turn_claims(
        "Suggest a movie to watch tonight.",
        session_id="s", role="user", base_turn_index=100,
    )
    assistant = _turn_claims(
        "Watch The Princess Bride for a fun classic.",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant, "What was the 27th movie you suggested?",
    )
    # Ordinal couldn't resolve, so the span extractor took over —
    # the only assistant claim survives as the answer.
    assert "Princess Bride" in out


def test_assistant_memory_returns_span_when_present_at_body_start():
    """
    Tier 2B-4: when the span IS the subject of the assistant claim,
    return it directly — that's the high-confidence case (the
    "Andy" / "Vocal Prayer and Meditation" / "2-3 eggs" v10
    regressions). The body verbatim was clobbering these.
    """
    user = _turn_claims(
        "What's a good name for my new puppy?",
        session_id="s", role="user", base_turn_index=100,
    )
    assistant = _turn_claims(
        # Span extractor pulls "Andy" via the noun-phrase head.
        "Andy is a great choice for a puppy name.",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant, "What name did you suggest for the puppy?",
    )
    assert "Andy" in out


def test_assistant_memory_rejects_nationality_adjective_alone():
    """
    Tier 2B-4: the v9 failure where span extractor returned ``Roman``
    from ``"Roscioli is great for Roman pasta"``. With the
    subject-position heuristic, the body wins — the substring scorer
    accepts ``"Roscioli"`` as substring.
    """
    user = _turn_claims(
        "Recommend a restaurant in Rome.",
        session_id="s", role="user", base_turn_index=100,
    )
    assistant = _turn_claims(
        "Roscioli is great for Roman pasta.",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant,
        "What restaurant did you recommend in Rome?",
    )
    assert "Roscioli" in out
    # The mid-body adjective must NOT be returned alone.
    assert out.strip() != "Roman"


def test_assistant_memory_returns_span_for_numeric_count():
    """
    The e8a79c70 regression shape — short numeric answer ("2-3 eggs")
    that span extraction can find. Don't let the body's verbose
    intro clobber it.
    """
    user = _turn_claims(
        "Give me a quick omelette recipe.",
        session_id="s", role="user", base_turn_index=100,
    )
    # Assistant's first sentence is intro; second sentence has the
    # actual numeric answer.
    assistant = _turn_claims(
        "Sure, here's a quick recipe for you to try.",
        "Use 2-3 eggs, beaten with a splash of milk.",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant, "How many eggs did you suggest using?",
    )
    # Either the span ("2-3" / "2-3 eggs") OR the full second-sentence
    # body containing "2-3 eggs" is acceptable to the substring
    # scorer. The intro sentence ("Sure, here's...") must NOT win.
    assert "2-3" in out or "2-3 eggs" in out.lower()
    assert "sure, here" not in out.lower()


def test_assistant_memory_soft_intro_demoted_in_favour_of_real_content():
    """
    Sentences that match the soft-intro pattern but escape the hard
    intro filter (because they have substantive tails) get walked
    PAST when a non-intro sibling claim is present in the same turn.
    """
    user = _turn_claims(
        "Suggest a book for me.",
        session_id="s", role="user", base_turn_index=100,
    )
    assistant = _turn_claims(
        # Soft intro — has tail words so is_generic_intro lets it
        # through, but it's still prefatory.
        "Sure, here are some great picks from my reading list.",
        # Real answer.
        "Vocal Prayer and Meditation by Karen Armstrong is a favorite.",
        session_id="s", role="assistant", base_turn_index=200,
    )
    out = head_assistant_memory(
        user + assistant, "What book did you recommend?",
    )
    assert "Vocal Prayer" in out or "Karen Armstrong" in out


def test_assistant_memory_falls_back_when_no_user_anchor_present():
    """
    Bundles where every claim is system-roled (no per-turn role
    parsing) should still produce an answer via the legacy ranker —
    no regression for content without clean turn structure.
    """
    # No user turn, no role tags — pure system-role content.
    claims = [
        Claim(
            text="The recommended book was The Three-Body Problem.",
            source_session_id="s", source_role="system",
            turn_index=100,
        ),
    ]
    out = head_assistant_memory(
        claims, "What book did you recommend?",
    )
    # Falls through to legacy ranker; non-empty answer.
    assert out != ""
    assert "Three-Body" in out


# ─── (extra) KNOWLEDGE_UPDATE picks the latest/current claim ──────────────


def test_knowledge_update_prefers_current_marker():
    claims = [
        Claim(
            text="Previously I worked at a small startup downtown.",
            source_session_id="s_old", source_role="user", turn_index=10,
        ),
        Claim(
            text="Currently I work at a large bank in the city.",
            source_session_id="s_new", source_role="user", turn_index=50,
        ),
    ]
    out = head_knowledge_update(claims, "Where do I work?")
    assert "bank" in out.lower()
    assert "startup" not in out.lower()


# ─── (extra) PREFERENCE_PROFILE returns a concise profile ─────────────────


def test_preference_profile_summarises_likes_and_avoids():
    claims = [
        Claim(
            text="I love iced coffee in the morning.",
            source_session_id="s1", source_role="user",
        ),
        Claim(
            text="I really enjoy long walks on weekends.",
            source_session_id="s2", source_role="user",
        ),
        Claim(
            text="I hate mushrooms.",
            source_session_id="s3", source_role="user",
        ),
    ]
    out = head_preference_profile(claims, "What do I prefer?")
    assert "iced coffee" in out.lower()
    assert "long walks" in out.lower() or "walks" in out.lower()
    assert "mushrooms" in out.lower()


# ─── (extra) MULTI_SESSION_AGGREGATE sums across distinct sessions ────────


def test_multi_session_aggregate_sums_per_session():
    claims = [
        Claim(
            text="On the Tokyo trip I packed 7 shirts.",
            source_session_id="s_tokyo", source_role="user",
        ),
        Claim(
            text="For the Paris trip I packed 5 shirts.",
            source_session_id="s_paris", source_role="user",
        ),
        # Same trip mentioned again — should NOT double-count.
        Claim(
            text="I still remember packing 7 shirts for Tokyo.",
            source_session_id="s_tokyo", source_role="user",
        ),
    ]
    out = head_multi_session_aggregate(
        claims, "How many shirts did I pack across all trips?",
    )
    # 7 (Tokyo) + 5 (Paris) = 12 shirts
    assert "12" in out
    assert "shirt" in out.lower()


# ─── precision_claim_first_ids.json file check ────────────────────────────


# ─── b86304ba regression — the live failure ───────────────────────────────


_B86_QUESTION = (
    "How much is the painting of a sunset worth in terms of the "
    "amount I paid for it?"
)
_B86_TOP_CLAIM = Claim(
    text=(
        "I was thinking about my flea market find, and I realized "
        "that it's actually worth triple what I paid for it, which "
        "is amazing!"
    ),
    source_session_id="s_painting",
    source_role="user",
    confidence=0.85,
)
# A transcript-scrape distractor that previously produced "110 for".
_B86_NOISE_CLAIM = Claim(
    text="0:02 110 for the introduction. Anyway, moving on.",
    source_session_id="s_chat_distractor",
    source_role="assistant",
    confidence=0.3,
)


def test_b86304ba_routes_to_fact_lookup_not_aggregate():
    """
    "How much is X worth..." is a single fact, not an aggregation.
    Routing it to MULTI_SESSION_AGGREGATE is what triggered the "110
    for" bug — the count head latched onto a transcript-junk
    ``<num> <stopword>`` pattern.
    """
    mode = route_task(_B86_QUESTION)
    assert mode == TaskMode.FACT_LOOKUP, (
        f"expected FACT_LOOKUP for a worth-question, got {mode}"
    )


def test_b86304ba_fact_lookup_extracts_triple_what_i_paid():
    """The FACT_LOOKUP head's value-phrase extractor returns the
    shortest faithful span from the top claim."""
    span = head_fact_lookup([_B86_TOP_CLAIM], _B86_QUESTION)
    assert "triple" in span.lower()
    assert "paid" in span.lower()
    # Shortest-faithful: the head must NOT echo the whole sentence.
    assert "flea market" not in span.lower()
    assert "amazing" not in span.lower()


def test_b86304ba_validator_rejects_110_for():
    """The validator must reject ``"110 for"`` as transcript junk
    even though both answer and claim share the token ``"for"``."""
    from evals.longmemeval.answer_post import validate_against_claims
    claims = [_B86_TOP_CLAIM.text, _B86_NOISE_CLAIM.text]
    ok, reason = validate_against_claims("110 for", claims)
    assert not ok, "validator must reject '110 for'"
    # The reason should pin the failure mode for the trace log.
    assert reason in ("answer_is_timestamp_junk", "unsupported_by_claim")


def test_b86304ba_validator_accepts_correct_span():
    """Sanity: the validator passes the actual answer span."""
    from evals.longmemeval.answer_post import validate_against_claims
    ok, _ = validate_against_claims(
        "triple what I paid for it",
        [_B86_TOP_CLAIM.text],
    )
    assert ok


def test_b86304ba_end_to_end_through_router_and_head():
    """Full claim-first path on the b86304ba bundle — route + head
    + validation must end at the correct span, not at '110 for'."""
    from evals.longmemeval.answer_post import validate_against_claims
    from evals.longmemeval.claims import rank_claims

    bundle = [_B86_NOISE_CLAIM, _B86_TOP_CLAIM]  # distractor first
    mode = route_task(_B86_QUESTION)
    assert mode == TaskMode.FACT_LOOKUP

    ranked = rank_claims(_B86_QUESTION, bundle)
    span = head_fact_lookup(ranked, _B86_QUESTION)
    assert span
    assert "triple" in span.lower() and "paid" in span.lower()
    assert "110" not in span

    ok, reason = validate_against_claims(
        span, [c.text for c in ranked[:5]],
    )
    assert ok, f"validator rejected the correct span: {reason}"


def test_validator_rejects_timestamp_junk_patterns():
    """Hard reject on '<num> <stopword>' patterns regardless of claim."""
    from evals.longmemeval.answer_post import validate_against_claims
    claim_text = "Anything goes here — for example a long sentence."
    for junk in ("110 for", "02 share", "04 principles", "7 every"):
        ok, reason = validate_against_claims(junk, [claim_text])
        assert not ok, f"junk passed: {junk!r}"
        assert reason in (
            "answer_is_timestamp_junk", "unsupported_by_claim",
        )


def test_b86304ba_terminal_no_postprocess_overwrite():
    """
    Regression for the smoke-vs-full determinism bug: once claim-first
    returns ``"triple what I paid for it"``, ``answer_question`` must
    skip ``_postprocess`` so the COUNT override / LLM repair can't
    overwrite the span with a numeric distractor like ``"6"``.

    This test simulates the adapter without an LLM by stubbing
    `_answer_claim_first` to set the success flag and return the span,
    then asserting `answer_question` returns it unmodified.
    """
    import asyncio

    from evals.longmemeval.bootstrap_ollama import _DecomposedAnsweringAdapter

    # Build a bare instance without going through __init__ (which
    # requires a real session / llm). We only need the answer_question
    # path which doesn't touch storage when claim-first short-circuits.
    adapter = object.__new__(_DecomposedAnsweringAdapter)
    adapter.last_decomposition_stats = {}
    adapter.last_telemetry = {}
    adapter.claim_first = True
    adapter.evidence_packet = False
    adapter._claim_first_used = False

    expected_span = "triple what I paid for it"

    async def fake_inner(question):
        adapter._claim_first_used = True
        return expected_span

    async def fake_postprocess(*args, **kwargs):
        # If this fires, the determinism bug is back — fail loudly.
        raise AssertionError(
            "postprocess must NOT run when claim-first succeeded",
        )

    adapter._answer_question_inner = fake_inner
    adapter._postprocess = fake_postprocess
    # answer_question references current_counter/end_row_telemetry —
    # the real implementations work without any setup (ContextVar default
    # is None and end_row_telemetry returns None when no counter is set).

    result = asyncio.run(adapter.answer_question(
        "How much is the painting of a sunset worth in terms of "
        "the amount I paid for it?",
    ))
    assert result == expected_span
    # And the trace state reflects the terminal claim-first decision.
    assert adapter.last_decomposition_stats.get("validator_reason") == \
        "claim_first_terminal"
    assert adapter.last_decomposition_stats.get("regeneration_attempted") is False


def test_b86304ba_numeric_only_answer_rejected_by_validator():
    """
    A bare "6" must fail validation against the painting claim — the
    claim is qualitative ("triple what I paid"), so a numeric answer
    is unsupported.
    """
    from evals.longmemeval.answer_post import validate_against_claims

    claim_text = (
        "I was thinking about my flea market find, and I realized "
        "that it's actually worth triple what I paid for it, which "
        "is amazing!"
    )
    ok, reason = validate_against_claims("6", [claim_text])
    assert not ok
    assert reason == "unsupported_by_claim"


def test_b86304ba_correct_span_passes_validation_against_supporting_claim():
    """The qualitative span passes validation — sanity that we didn't
    over-tighten the validator while fixing the determinism bug."""
    from evals.longmemeval.answer_post import validate_against_claims

    claim_text = (
        "I realized that it's actually worth triple what I paid for it"
    )
    for span in (
        "triple what I paid for it",
        "worth triple what I paid for it",
        "The painting is worth triple what I paid for it.",
    ):
        ok, reason = validate_against_claims(span, [claim_text])
        assert ok, f"valid span rejected: {span!r} ({reason})"


# ─── 58ef2f1c — date specificity (February 14th, not May) ─────────────────


def test_58ef2f1c_prefers_specific_date_over_bare_month():
    """
    The same claim contains both "May" (incidentally) and "February
    14th" (the actual answer). The date extractor must return the
    more-specific date regardless of which appears first.
    """
    claim = Claim(
        text=(
            "I might be free in May too, but the animal shelter's "
            "fundraising dinner was on February 14th, so I went then."
        ),
        source_session_id="s1", source_role="user", confidence=0.8,
    )
    out = head_fact_lookup(
        [claim],
        "When did I volunteer at the local animal shelter's "
        "fundraising dinner?",
    )
    assert "February" in out
    assert "14" in out
    assert out.strip().lower() != "may"


def test_58ef2f1c_specific_date_outranks_other_claim_with_bare_month():
    """
    Two claims, one with a bare month (distractor) and one with the
    specific date (answer). The head must pick the specific-date claim.
    """
    distractor = Claim(
        text="In May I usually attend the spring potluck at work.",
        source_session_id="s_potluck", source_role="user", confidence=0.6,
    )
    correct = Claim(
        text="The animal shelter's fundraising dinner was on February 14th.",
        source_session_id="s_shelter", source_role="user", confidence=0.85,
    )
    out = head_fact_lookup(
        [distractor, correct],  # distractor ranked first by accident
        "When did I volunteer at the fundraising dinner?",
    )
    assert "February" in out
    assert "May" not in out


# ─── 5d3d2817 — user-request claims must not outrank real occupation ───────


def test_5d3d2817_user_request_claim_does_not_win():
    """
    A user-question claim that shares "previous" with the benchmark
    question must NOT outrank the actual occupation claim.
    """
    request = Claim(
        text=(
            "Can you help me organize my sales data from previous "
            "markets?"
        ),
        source_session_id="s_request", source_role="user", confidence=0.7,
    )
    occupation = Claim(
        text=(
            "I used to be a Marketing specialist at a small startup "
            "before I joined the agency."
        ),
        source_session_id="s_occupation", source_role="user", confidence=0.85,
    )
    out = head_fact_lookup(
        [request, occupation],
        "What was my previous occupation?",
    )
    assert "Marketing specialist" in out
    assert "Can you help" not in out


def test_5d3d2817_extracts_full_phrase_with_company_qualifier():
    """The 'at a small startup' qualifier survives the extractor."""
    claim = Claim(
        text=(
            "I used to be a Marketing specialist at a small startup "
            "before I joined the agency."
        ),
        source_session_id="s1", source_role="user", confidence=0.85,
    )
    out = head_fact_lookup(
        [claim], "What was my previous occupation?",
    )
    assert "Marketing specialist" in out
    assert "startup" in out.lower()


def test_5d3d2817_long_responsibility_claim_does_not_leak_as_prose():
    """
    When the top-ranked claim is a long responsibilities-prose
    sentence (no extractable title) AND no sibling claim has a clean
    'I was a X' frame either, ``head_fact_lookup`` must NOT return
    the prose body as a fallback — that produces the
    ``answer_is_analysis_prose`` validator rejection observed in
    the live log.

    Returning empty here lets the orchestrator's extractive LLM
    fallback try the top claims for a span.
    """
    responsibilities = Claim(
        text=(
            "In my previous role at the startup, I was responsible "
            "for managing marketing campaigns and product launches "
            "across the year and coordinating with the design team."
        ),
        source_session_id="s_resp", source_role="user", confidence=0.8,
    )
    out = head_fact_lookup(
        [responsibilities], "What was my previous occupation?",
    )
    # The prose body is too long to be a valid fact-lookup answer.
    # The head must return empty so the orchestrator's extractive
    # LLM fallback can run, not the whole responsibility sentence.
    assert out == "" or len(out.split()) <= 20, (
        f"head returned long prose body ({len(out.split())} words): {out!r}"
    )


def test_5d3d2817_short_claim_body_still_falls_back_through():
    """
    Sanity: a SHORT claim body (≤ 20 words) is still a valid fallback
    when no regex matches — we didn't tighten too far.
    """
    short_claim = Claim(
        text="I went to the park yesterday afternoon.",
        source_session_id="s1", source_role="user", confidence=0.6,
    )
    out = head_fact_lookup([short_claim], "Where did I go yesterday?")
    # The original whole-body fallback still kicks in for short claims.
    assert out != ""
    assert "park" in out.lower()


def test_5d3d2817_validator_rejects_user_request_answer():
    """A user-request answer must fail validation, even if claims exist."""
    ok, reason = validate_against_claims(
        "Can you help me organize my sales data from previous markets?",
        [
            "I used to be a Marketing specialist at a small startup.",
            "Can you help me organize my sales data from previous markets?",
        ],
    )
    assert not ok
    assert reason == "answer_is_user_request"


# ─── 66f24dbb — heading-only "**Sister's Birthday**" must not win ──────────


def test_66f24dbb_heading_claim_is_skipped():
    """
    A ``**Sister's Birthday**`` heading line shares "sister" + "birthday"
    with the question but isn't a memory — it's a label. The head must
    skip it and use the real purchase claim.
    """
    heading = Claim(
        text="**Sister's Birthday**",
        source_session_id="s_heading", source_role="user", confidence=0.5,
    )
    purchase = Claim(
        text="I bought my sister a yellow dress for her birthday last weekend.",
        source_session_id="s_purchase", source_role="user", confidence=0.85,
    )
    out = head_fact_lookup(
        [heading, purchase],
        "What did I buy for my sister's birthday gift?",
    )
    assert "yellow dress" in out.lower()
    assert "**" not in out
    assert "sister" not in out.lower() or "yellow" in out.lower()


def test_66f24dbb_validator_rejects_heading_answer():
    """A markdown heading answer must fail validation."""
    ok, reason = validate_against_claims(
        "**Sister's Birthday**",
        ["I bought my sister a yellow dress for her birthday."],
    )
    assert not ok
    assert reason == "answer_is_heading"


def test_66f24dbb_validator_rejects_label_metadata_answer():
    """A label-only answer like ``Occasion: ...`` fails too."""
    ok, reason = validate_against_claims(
        "Occasion: Sister's birthday",
        ["I bought my sister a yellow dress."],
    )
    assert not ok
    assert reason == "answer_is_heading"


def test_66f24dbb_extracts_dress_from_indirect_object_pattern():
    """``I bought my sister a yellow dress`` → ``a yellow dress``."""
    claim = Claim(
        text="I bought my sister a yellow dress for her birthday last weekend.",
        source_session_id="s1", source_role="user", confidence=0.85,
    )
    out = head_fact_lookup(
        [claim], "What did I buy for my sister's birthday gift?",
    )
    assert "yellow dress" in out.lower()


# ─── Helper functions standalone ──────────────────────────────────────────


def test_is_user_request_sentence_detects_common_shapes():
    from evals.longmemeval.scaffold_filter import is_user_request_sentence
    assert is_user_request_sentence("Can you help me with this?")
    assert is_user_request_sentence("Could you tell me more?")
    assert is_user_request_sentence("Please summarise the meeting.")
    assert is_user_request_sentence("Help me organize my notes.")
    assert is_user_request_sentence("Any advice on the new role?")
    # Not requests:
    assert not is_user_request_sentence(
        "I used to be a Marketing specialist at a small startup."
    )
    assert not is_user_request_sentence(
        "The fundraising dinner was on February 14th."
    )


def test_is_label_metadata_line_detects_common_shapes():
    from evals.longmemeval.scaffold_filter import is_label_metadata_line
    assert is_label_metadata_line("Occasion: Sister's birthday")
    assert is_label_metadata_line("Topic: Marketing")
    assert is_label_metadata_line("Date: February 14th")
    assert not is_label_metadata_line(
        "I bought my sister a yellow dress for her birthday."
    )


def test_aggregate_router_still_fires_on_true_aggregation():
    """We tightened the router; verify true-aggregate questions still route correctly."""
    assert route_task(
        "How many shirts did I pack in total across all my trips?",
    ) == TaskMode.MULTI_SESSION_AGGREGATE
    assert route_task(
        "List all the cities I visited.",
    ) == TaskMode.MULTI_SESSION_AGGREGATE
    # Bare "how many" alone is FACT_LOOKUP — single session.
    assert route_task(
        "How many shirts did I pack for the trip?",
    ) == TaskMode.FACT_LOOKUP


def test_precision_claim_first_sample_file_contents():
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (repo / "samples" / "precision_claim_first_ids.json").read_text(),
    )
    expected = {
        "e47becba", "58ef2f1c", "5d3d2817",
        "66f24dbb", "af8d2e46", "b86304ba",
    }
    assert set(payload["question_ids"]) == expected
    assert payload["total"] == 6
