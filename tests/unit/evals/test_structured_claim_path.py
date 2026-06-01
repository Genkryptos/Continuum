"""
tests/unit/evals/test_structured_claim_path.py
==============================================
Tests for the structured-claim layer:

* :mod:`evals.longmemeval.structured_claims` — Claim dataclass +
  per-relation extractors.
* :mod:`evals.longmemeval.query_intent` — question → intent table.
* :mod:`evals.longmemeval.structured_finalizer` — intent + claims
  → final answer string.

Acceptance scenarios (from the spec):

1. **Yellow dress / sister's birthday** — bought_gift_for relation
   with recipient+occasion constraints.
2. **Previous occupation** — previous_occupation relation, must beat
   any user-question / topic-overlap distractor.
3. **Business Administration degree** — degree_in relation, full
   field name preserved.
4. **Painting worth triple what I paid** — value_compared_to_paid
   relation, comparative phrase preserved verbatim.
5. **Costa Rica 7 shirts** — count_packed relation with item+
   destination constraints.
6. **Rejection** — headings / user-questions / transcript-timestamps
   never become claims and never become final answers.
"""

from __future__ import annotations

from evals.longmemeval.query_intent import parse_intent
from evals.longmemeval.structured_claims import (
    StructuredClaim,
    extract_structured_claims,
)
from evals.longmemeval.structured_finalizer import (
    finalize_fact_answer,
    rank_structured_claims,
)


def _metadata(session_id: str = "s1", role: str = "user", idx: int = 0):
    return {
        "source_session_id": session_id,
        "source_role": role,
        "source_turn_index": idx,
    }


# ─── (1) Yellow dress / sister's birthday ─────────────────────────────────


def test_bought_gift_for_extracts_recipient_and_occasion():
    """Indirect object recipient + trailing occasion."""
    claims = extract_structured_claims(
        "I bought my sister a yellow dress for her birthday last weekend.",
        source_session_id="s_gift",
        source_role="user",
        source_turn_index=0,
    )
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift, "no bought_gift_for claim extracted"
    c = gift[0]
    assert "yellow dress" in c.object_.lower()
    assert c.qualifiers.get("recipient") == "sister"


def test_bought_gift_for_alt_phrasing():
    """``for my sister's birthday`` style."""
    claims = extract_structured_claims(
        "I bought a yellow dress for my sister's birthday last weekend.",
        **_metadata(),
    )
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift
    assert "yellow dress" in gift[0].object_.lower()
    assert gift[0].qualifiers.get("recipient") == "sister"
    assert gift[0].qualifiers.get("occasion") == "birthday"


def test_intent_parses_birthday_gift_question():
    intent = parse_intent("What did I buy for my sister's birthday gift?")
    assert intent.relation == "bought_gift_for"
    assert intent.constraints.get("recipient") == "sister"
    assert intent.constraints.get("occasion") == "birthday"
    assert intent.answer_shape == "purchased_item"


def test_finalize_yellow_dress_birthday_gift():
    """End-to-end: gift question + gift claim → ``"a yellow dress"``."""
    intent = parse_intent("What did I buy for my sister's birthday gift?")
    claims = extract_structured_claims(
        "I bought my sister a yellow dress for her birthday.",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "yellow dress" in answer.lower()


# ─── (2) Previous occupation ──────────────────────────────────────────────


def test_previous_occupation_extracts_title_with_company():
    claims = extract_structured_claims(
        "I used to be a Marketing specialist at a small startup before I joined the agency.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "marketing specialist" in occ[0].object_.lower()
    # Company qualifier is captured separately when present.
    assert (
        "startup" in occ[0].object_.lower()
        or "startup" in occ[0].qualifiers.get("company", "").lower()
    )


def test_intent_parses_previous_occupation():
    intent = parse_intent("What was my previous occupation?")
    assert intent.relation == "previous_occupation"
    assert intent.answer_shape == "occupation_title"


def test_finalize_previous_occupation():
    intent = parse_intent("What was my previous occupation?")
    claims = extract_structured_claims(
        "I used to be a Marketing specialist at a small startup.",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "Marketing specialist" in answer
    assert "startup" in answer.lower()


def test_finalize_previous_occupation_ignores_distractor_claim():
    """
    A claim with the topic word but no relation match (e.g. a generic
    sentence about marketing) must NOT outrank the actual occupation
    claim. Relation-first ranking is the contract.
    """
    intent = parse_intent("What was my previous occupation?")
    claims = extract_structured_claims(
        "I was working on marketing campaigns at the agency. "
        "I used to be a Marketing specialist at a small startup.",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "Marketing specialist" in answer


# ─── (3) Business Administration degree ───────────────────────────────────


def test_degree_in_business_administration():
    claims = extract_structured_claims(
        "I graduated with a degree in Business Administration in 2018.",
        **_metadata(),
    )
    deg = [c for c in claims if c.relation == "degree_in"]
    assert deg
    assert "Business Administration" in deg[0].object_


def test_intent_parses_degree_question():
    intent = parse_intent("What degree did I graduate with?")
    assert intent.relation == "degree_in"
    assert intent.answer_shape == "field_of_study"


def test_finalize_business_administration():
    intent = parse_intent("What degree did I graduate with?")
    claims = extract_structured_claims(
        "I graduated with a Bachelor's degree in Business Administration "
        "from a state university in 2018.",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "Business Administration" in answer


# ─── (4) Painting worth triple what I paid ────────────────────────────────


def test_value_vs_paid_extracts_comparative_phrase():
    claims = extract_structured_claims(
        "Today the painting is worth triple what I paid for it.",
        **_metadata(),
    )
    val = [c for c in claims if c.relation == "value_compared_to_paid"]
    assert val
    assert "triple" in val[0].object_.lower()
    assert "paid" in val[0].object_.lower()


def test_intent_parses_value_question():
    intent = parse_intent(
        "How much is the painting of a sunset worth in terms of the amount I paid for it?"
    )
    assert intent.relation == "value_compared_to_paid"
    assert intent.answer_shape == "comparative_phrase"


def test_finalize_value_vs_paid():
    intent = parse_intent(
        "How much is the painting of a sunset worth in terms of the amount I paid for it?"
    )
    claims = extract_structured_claims(
        "I was thinking about my flea market find, and I realized "
        "that the painting is worth triple what I paid for it, "
        "which is amazing!",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "triple" in answer.lower()
    assert "paid" in answer.lower()


# ─── (5) Costa Rica 7 shirts ──────────────────────────────────────────────


def test_count_packed_extracts_count_and_destination():
    claims = extract_structured_claims(
        "I packed 7 shirts for the trip to Costa Rica.",
        **_metadata(),
    )
    counts = [c for c in claims if c.relation == "count_packed"]
    assert counts
    assert "7" in counts[0].object_
    assert counts[0].qualifiers.get("item") == "shirt"


def test_intent_parses_count_packed_question():
    intent = parse_intent("How many shirts did I pack for the trip to Costa Rica?")
    assert intent.relation == "count_packed"
    assert intent.answer_shape == "count_with_unit"
    assert intent.constraints.get("item") == "shirt"


def test_finalize_count_packed():
    intent = parse_intent("How many shirts did I pack for the trip to Costa Rica?")
    claims = extract_structured_claims(
        "I packed 7 shirts for the trip to Costa Rica last week.",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "7" in answer
    assert "shirt" in answer.lower()


# ─── (6) Rejection of headings / questions / transcript timestamps ────────


def test_headings_do_not_become_claims():
    """``**Sister's Birthday**`` is a scaffold line — no claim emitted."""
    claims = extract_structured_claims(
        "**Sister's Birthday**",
        **_metadata(),
    )
    assert claims == []


def test_label_lines_do_not_become_claims():
    """``Occasion: Sister's birthday`` is label metadata."""
    claims = extract_structured_claims(
        "Occasion: Sister's birthday",
        **_metadata(),
    )
    assert claims == []


def test_user_request_sentences_do_not_become_claims():
    """``Can you help me…`` is a request, not a memory claim."""
    claims = extract_structured_claims(
        "Can you help me organize my sales data from previous markets?",
        **_metadata(),
    )
    assert claims == []


def test_transcript_timestamps_do_not_become_claims():
    """``02 share`` style noise is dropped."""
    claims = extract_structured_claims(
        "02 share\n0:14\n## Matched Turn\n",
        **_metadata(),
    )
    assert claims == []


def test_finalize_falls_back_to_none_when_no_matching_relation():
    """
    No structured claim has the asked relation → finalizer returns
    None. The adapter treats None as "fall through to the legacy
    head" rather than guessing.
    """
    intent = parse_intent("What degree did I graduate with?")
    claims = extract_structured_claims(
        "I went to the park yesterday afternoon.",
        **_metadata(),
    )
    result = finalize_fact_answer(intent, claims)
    assert result is None


def test_finalize_returns_none_when_intent_unmatched():
    """Unrecognised question shape → unmatched intent → no answer."""
    intent = parse_intent("Something completely unrelated.")
    assert not intent.matched
    result = finalize_fact_answer(
        intent,
        [
            StructuredClaim(
                text="anything",
                subject="user",
                relation="previous_occupation",
                object_="something",
                qualifiers={},
                source_session_id="s",
                source_role="user",
                source_turn_index=0,
                source_span="anything",
                confidence=0.9,
            )
        ],
    )
    assert result is None


# ─── Ranking semantics ────────────────────────────────────────────────────


def test_relation_match_outweighs_topic_overlap():
    """
    Two claims, only one has the right relation. The relation-match
    claim wins regardless of any other signal.
    """
    intent = parse_intent("What was my previous occupation?")
    relation_match = StructuredClaim(
        text="I used to be a Marketing specialist at a small startup.",
        subject="user",
        relation="previous_occupation",
        object_="Marketing specialist at a small startup",
        qualifiers={},
        source_session_id="s_match",
        source_role="user",
        source_turn_index=0,
        source_span="Marketing specialist at a small startup",
        confidence=0.5,
    )
    relation_mismatch = StructuredClaim(
        text="My previous house was on Maple Street.",
        subject="user",
        relation="home_location",
        object_="Maple Street",
        qualifiers={},
        source_session_id="s_mismatch",
        source_role="user",
        source_turn_index=0,
        source_span="Maple Street",
        confidence=0.95,  # higher confidence!
    )
    ranked = rank_structured_claims(
        intent,
        [relation_mismatch, relation_match],
    )
    # The relation_mismatch claim must be dropped (score ≤ 0); only
    # relation_match survives.
    assert len(ranked) == 1
    assert ranked[0].relation == "previous_occupation"


def test_constraint_match_breaks_relation_ties():
    """
    When two claims share the relation, the one whose qualifiers
    match the intent's constraints wins.
    """
    intent = parse_intent("What did I buy for my sister's birthday gift?")
    sister_gift = StructuredClaim(
        text="I bought my sister a yellow dress for her birthday.",
        subject="user",
        relation="bought_gift_for",
        object_="a yellow dress",
        qualifiers={"recipient": "sister", "occasion": "birthday"},
        source_session_id="s1",
        source_role="user",
        source_turn_index=0,
        source_span="a yellow dress",
        confidence=0.7,
    )
    brother_gift = StructuredClaim(
        text="I bought my brother a watch for his graduation.",
        subject="user",
        relation="bought_gift_for",
        object_="a watch",
        qualifiers={"recipient": "brother", "occasion": "graduation"},
        source_session_id="s2",
        source_role="user",
        source_turn_index=0,
        source_span="a watch",
        confidence=0.9,
    )
    ranked = rank_structured_claims(intent, [brother_gift, sister_gift])
    assert ranked[0].qualifiers.get("recipient") == "sister"


def test_user_role_boost_breaks_score_ties():
    """A user-stated claim outranks the same claim from an assistant turn."""
    intent = parse_intent("What was my previous occupation?")
    user_claim = StructuredClaim(
        text="I used to be a designer.",
        subject="user",
        relation="previous_occupation",
        object_="designer",
        qualifiers={},
        source_session_id="s_user",
        source_role="user",
        source_turn_index=0,
        source_span="designer",
        confidence=0.7,
    )
    assistant_claim = StructuredClaim(
        text="You were a designer, right?",
        subject="user",
        relation="previous_occupation",
        object_="designer",
        qualifiers={},
        source_session_id="s_asst",
        source_role="assistant",
        source_turn_index=0,
        source_span="designer",
        confidence=0.7,
    )
    ranked = rank_structured_claims(intent, [assistant_claim, user_claim])
    assert ranked[0].source_role == "user"


# ─── Claim dataclass serialisation ────────────────────────────────────────


def test_structured_claim_to_dict_uses_object_not_underscored():
    """JSON shape uses ``object``, not ``object_``."""
    c = StructuredClaim(
        text="x",
        subject="user",
        relation="r",
        object_="o",
        qualifiers={"k": "v"},
        source_session_id="s",
        source_role="user",
        source_turn_index=1,
        source_span="o",
        confidence=0.5,
    )
    d = c.to_dict()
    assert d["object"] == "o"
    assert d["qualifiers"] == {"k": "v"}
    assert "object_" not in d


# ─── Extended occupation intent coverage ──────────────────────────────────


def test_intent_covers_what_role_did_i_have_before():
    intent = parse_intent("What role did I have before?")
    assert intent.relation == "previous_occupation"
    assert intent.answer_shape == "occupation_title"


def test_intent_covers_what_did_i_used_to_work_as():
    intent = parse_intent("What did I used to work as?")
    assert intent.relation == "previous_occupation"


def test_intent_covers_what_did_i_do_for_work_before():
    intent = parse_intent("What did I do for work before?")
    assert intent.relation == "previous_occupation"


def test_intent_covers_past_position():
    intent = parse_intent("What was my past position?")
    assert intent.relation == "previous_occupation"


# ─── Responsibility-only sentences must NOT become occupation claims ──────


def test_responsibility_sentence_is_rejected():
    """``In my previous role, I was responsible for…`` produces no claim."""
    claims = extract_structured_claims(
        "In my previous role at the startup, I was responsible for "
        "managing marketing campaigns and product launches.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ == [], (
        f"responsibility sentence wrongly produced occupation claim: {[c.object_ for c in occ]}"
    )


def test_title_claim_outranks_responsibility_sentence():
    """
    Bundle has both a responsibility-only sentence (no title) and a
    real title claim. The title claim must win — and the responsibility
    sentence must not even enter the candidate pool.
    """
    intent = parse_intent("What was my previous occupation?")
    text = (
        "In my previous role at the startup, I was responsible for "
        "managing marketing campaigns and product launches. "
        "I used to be a Marketing specialist at a small startup."
    )
    claims = extract_structured_claims(text, **_metadata())
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "Marketing specialist" in answer
    assert "responsible" not in answer.lower()
    assert "managing" not in answer.lower()


# ─── Extended occupation lead-in coverage ─────────────────────────────────


def test_occupation_extract_served_as_frame():
    """``I served as a X`` lead-in."""
    claims = extract_structured_claims(
        "Previously I served as a Marketing specialist at a small startup.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "Marketing specialist" in occ[0].object_


def test_occupation_extract_held_the_position_frame():
    """``I held the position/role of X`` lead-in."""
    claims = extract_structured_claims(
        "I held the position of Marketing specialist at a small startup.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "Marketing specialist" in occ[0].object_


def test_occupation_extract_most_recently_frame():
    """``Most recently, I (was|held|served as) X``."""
    claims = extract_structured_claims(
        "Most recently, I was a Marketing specialist at a small startup.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "Marketing specialist" in occ[0].object_


def test_occupation_extract_before_x_i_was_frame():
    """``Before X, I was a Y`` clause-prefix frame."""
    claims = extract_structured_claims(
        "Before joining the agency, I was a Marketing specialist at a small startup.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "Marketing specialist" in occ[0].object_


def test_occupation_extract_former_self_description():
    """``I'm a former X`` ↔ past occupation."""
    claims = extract_structured_claims(
        "I'm a former Marketing specialist at a small startup.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "Marketing specialist" in occ[0].object_


# ─── Standalone ``<title> at <company>`` fallback (strategy 2) ─────────────


def test_occupation_extract_title_at_company_with_past_tense_marker():
    """
    Sentence has no explicit lead-in frame but does have 1st-person
    + past-tense markers, so the standalone title-at-company pattern
    fires.
    """
    claims = extract_structured_claims(
        "Before this gig, my title was Marketing specialist at a small "
        "startup, and I loved the team.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert "Marketing specialist" in occ[0].object_


def test_occupation_standalone_blocked_without_past_tense():
    """
    Standalone pattern requires past-tense markers — without them
    we'd match current-job statements as previous occupations.
    """
    claims = extract_structured_claims(
        "My current role is Marketing specialist at a small startup.",
        **_metadata(),
    )
    # No past-tense / "I was" / "previously" markers, so the standalone
    # title pattern should NOT emit a previous_occupation claim.
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ == []


def test_occupation_standalone_blocked_without_first_person():
    """
    Standalone pattern requires 1st-person — third-person facts
    about other people should not become "my previous occupation".
    """
    claims = extract_structured_claims(
        "She used to be a Marketing specialist at a small startup.",
        **_metadata(),
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ == []


# ─── Section-scoped gift extraction ───────────────────────────────────────


def test_section_scoped_gift_under_bold_heading():
    """
    Markdown bold heading ``**Sister's Birthday**`` followed by a
    purchase line emits a ``bought_gift_for`` claim with
    ``recipient=sister, occasion=birthday`` inherited from the heading.
    """
    text = "**Sister's Birthday**\n- I bought a yellow dress at the boutique downtown.\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift
    assert "yellow dress" in gift[0].object_.lower()
    assert gift[0].qualifiers.get("recipient") == "sister"
    assert gift[0].qualifiers.get("occasion") == "birthday"


def test_section_scoped_gift_under_label_metadata():
    text = "Occasion: Sister's birthday\nI got her a yellow dress from a small shop.\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift
    assert "yellow dress" in gift[0].object_.lower()


def test_per_sentence_gift_rejects_groceries_recovery_context():
    """
    ``I got her some groceries when she was recovering from surgery``
    must NOT become a gift claim — even though the verb + recipient
    pattern matches, the topic is clearly not a gift.
    """
    claims = extract_structured_claims(
        "I got her some groceries and flowers when she was recovering from surgery.",
        **_metadata(),
    )
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift == []


def test_per_sentence_gift_requires_occasion():
    """
    ``I got her a yellow dress`` (no occasion) must NOT become a gift
    claim — the per-sentence path requires an occasion. The
    section-scoped extractor handles the no-occasion-in-sentence case
    via the inherited section context.
    """
    claims = extract_structured_claims(
        "I got her a yellow dress because she likes pastels.",
        **_metadata(),
    )
    # No occasion → no per-sentence gift claim.
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift == []


def test_per_sentence_gift_with_occasion_still_emits():
    """Sanity: occasion-bearing sentences continue to work as before."""
    claims = extract_structured_claims(
        "I bought my sister a yellow dress for her birthday last weekend.",
        **_metadata(),
    )
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift
    assert gift[0].qualifiers.get("occasion") == "birthday"


def test_section_scoped_bare_item_under_heading():
    """
    ``**Sister's Birthday**`` followed by a bare item bullet (no verb)
    emits a ``bought_gift_for`` claim. This handles bundles where the
    user just listed gifts under a heading.
    """
    text = "**Sister's Birthday**\n- a yellow dress\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift
    assert "yellow dress" in gift[0].object_.lower()
    assert gift[0].qualifiers.get("recipient") == "sister"
    assert gift[0].qualifiers.get("occasion") == "birthday"


def test_section_scoped_real_66f24dbb_bundle_shape():
    """
    Regression for the exact bundle shape observed in the live
    66f24dbb row: a bold heading PLUS a reinforcing ``* Occasion:``
    line PLUS one or more ``* <Label>: <value>`` rows where one of
    the labels is ``Gift:``. The section extractor must:

      * Treat both heading shapes as context-setters (no item emitted
        from a heading itself).
      * Skip non-item label rows (``Recipient: Sister``).
      * Pick up the ``Gift: <noun phrase>`` row as the claim.

    AND a distractor sentence with the same indirect-object pattern
    but no occasion must NOT outrank it (groceries-during-surgery).
    """
    text = (
        "**Sister's Birthday**\n"
        "* Occasion: Sister's birthday\n"
        "* Recipient: Sister\n"
        "* Gift: a yellow dress\n"
        "* Where: a boutique downtown\n"
        "\n"
        # Distractor sentence elsewhere in the bundle:
        "I got her some groceries and flowers when she was recovering from surgery.\n"
    )
    claims = extract_structured_claims(text, **_metadata())
    gifts = [c for c in claims if c.relation == "bought_gift_for"]
    # The distractor must not emit a claim AT ALL (no occasion +
    # surgery/recovery context).
    assert not any("groceries" in c.object_.lower() for c in gifts), (
        "groceries claim must be rejected by per-sentence gift extractor"
    )
    # The section-bound dress claim must be present and carry both
    # inherited qualifiers.
    dress = [c for c in gifts if "yellow dress" in c.object_.lower()]
    assert dress, (
        f"yellow dress not extracted from sectioned bundle; got: {[c.object_ for c in gifts]}"
    )
    c = dress[0]
    assert c.qualifiers.get("recipient") == "sister"
    assert c.qualifiers.get("occasion") == "birthday"


def test_finalize_66f24dbb_picks_yellow_dress_not_groceries():
    """End-to-end: intent + bundle → ``a yellow dress``."""
    intent = parse_intent("What did I buy for my sister's birthday gift?")
    text = (
        "**Sister's Birthday**\n"
        "* Occasion: Sister's birthday\n"
        "* Gift: a yellow dress\n"
        "\n"
        "I got her some groceries and flowers when she was recovering from surgery.\n"
    )
    claims = extract_structured_claims(text, **_metadata())
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    answer, _diag = result
    assert "yellow dress" in answer.lower()
    assert "groceries" not in answer.lower()


def test_gift_sentence_initial_prefix_66f24dbb_live_shape():
    """
    Live 66f24dbb wording: the gold answer lives in a user turn shaped
    as ``For my <recipient>'s <occasion>, I got her <item> and ...``.
    The verb-first regex misses this because the occasion clause
    leads, not the verb.
    """
    text = "For my sister's birthday, I got her a yellow dress and a pair of earrings to match."
    claims = extract_structured_claims(text, **_metadata())
    gifts = [c for c in claims if c.relation == "bought_gift_for"]
    assert gifts, "no bought_gift_for claim emitted for prefix form"
    c = gifts[0]
    assert "yellow dress" in c.object_.lower()
    assert "earrings" not in c.object_.lower()  # multi-item: keep only the first
    assert c.qualifiers.get("recipient") == "sister"
    assert c.qualifiers.get("occasion") == "birthday"
    # End-to-end finalizer.
    intent = parse_intent("What did I buy for my sister's birthday gift?")
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    assert "yellow dress" in result[0].lower()


def test_previous_occupation_in_my_previous_role_as_5d3d2817_live_shape():
    """
    Live 5d3d2817 wording: ``I've used Trello in my previous role as
    a marketing specialist at a small startup and I'm familiar with
    its features.``
    Two structural gaps it exposes:
      1. ``in my previous role as a <title>`` lead-in not covered by
         older "I was a" / "my previous job was a" frames.
      2. Trailing ``and I'm ...`` clause would otherwise swallow into
         the title capture — END must stop at "and I"/"and I'm".
    """
    text = (
        "I've used Trello in my previous role as a marketing specialist "
        "at a small startup and I'm familiar with its features."
    )
    claims = extract_structured_claims(text, **_metadata())
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ, "no previous_occupation claim emitted for prepositional form"
    c = occ[0]
    obj = c.object_.lower()
    assert "marketing specialist" in obj
    assert "small startup" in obj
    # The "and I'm familiar with its features" tail must not bleed in.
    assert "familiar" not in obj
    assert "features" not in obj
    intent = parse_intent("What was my previous occupation?")
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    assert "marketing specialist" in result[0].lower()


def test_section_scoped_plus_bullet_yellow_dress_live_wiki_shape():
    """
    Live 66f24dbb wiki-bundle shape: the assistant rendered the gift
    list with ``\\t+ Yellow dress`` (tab + plus-sign sub-bullet),
    embedded inside a user turn. The section walker must accept
    ``+`` as a bullet character, not just ``-`` / ``*``.
    """
    text = (
        "I got her some groceries and flowers when she was recovering "
        "from surgery.\n"
        "assistant:That's very thoughtful of you!\n"
        "\n"
        "Here's the updated list:\n"
        "\n"
        "**Sister's Birthday**\n"
        "\n"
        "* Occasion: Sister's birthday\n"
        "* Gift(s):\n"
        "\t+ Yellow dress\n"
        "\t+ Pair of earrings (matching the dress)\n"
    )
    claims = extract_structured_claims(text, **_metadata())
    gifts = [c for c in claims if c.relation == "bought_gift_for"]
    assert any("yellow dress" in c.object_.lower() for c in gifts), (
        f"section walker missed '+ Yellow dress'; got {[c.object_ for c in gifts]}"
    )
    # The surgery / groceries sentence must NOT bind a gift claim.
    assert not any("groceries" in c.object_.lower() for c in gifts)
    intent = parse_intent("What did I buy for my sister's birthday gift?")
    result = finalize_fact_answer(intent, claims)
    assert result is not None
    assert "yellow dress" in result[0].lower()


def test_wiki_bundle_per_turn_role_tags_user_claim_as_user():
    """
    Tier 1B regression: a wiki ``Matched Turn`` line shaped as
    ``- (user): I used to be a Marketing specialist at a startup.``
    must produce a claim whose ``source_role == "user"`` — not the
    outer page's ``"system"`` role. Without this fix the ranker's
    ``+0.5`` user-role boost never fires on wiki-bundled claims.
    """
    text = (
        "# Evidence Window for sess_abc\n"
        "\n"
        "## Matched Turn\n"
        "- (user): I used to be a Marketing specialist at a startup.\n"
    )
    # Outer metadata mimics how the wiki page is constructed —
    # source_role="system" because the page itself is a system message.
    claims = extract_structured_claims(
        text,
        source_session_id="sess_abc",
        source_role="system",
        source_turn_index=0,
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ, "no previous_occupation claim extracted"
    assert occ[0].source_role == "user", (
        f"expected user role from per-line prefix; got {occ[0].source_role!r}"
    )


def test_wiki_bundle_per_turn_role_tags_assistant_claim_as_assistant():
    """
    Per-line ``- (assistant): ...`` tags the claim as assistant.
    Critical for the assistant-memory category, where the answer
    must come from an assistant turn, not a user turn.
    """
    text = (
        "## Nearby Context\n"
        "- turn 5 (assistant): I recommended The Three-Body Problem "
        "for your next read.\n"
    )
    # The claim itself may not match any of the existing relations
    # (this is a recommendation, not a previous_occupation / degree /
    # gift), but if any claim were extracted from this line it must
    # carry the assistant role. Use _sentences_of directly to assert
    # the role parsing is wired through.
    from evals.longmemeval.structured_claims import _sentences_of

    pairs = _sentences_of(text)
    # The header / blank lines drop out; the bullet line yields the
    # sentence with its parsed role.
    roles = {role for _sent, role in pairs if role}
    assert "assistant" in roles


def test_wiki_bundle_mixed_roles_preserve_per_line_attribution():
    """
    A bundle that interleaves user and assistant turns must produce
    claims with the right per-line role — not a single global role.
    """
    text = (
        "## Matched Turn\n"
        "- (user): I bought my sister a yellow dress for her birthday.\n"
        "\n"
        "## Nearby Context\n"
        "- turn 4 (user): I used to be a Marketing specialist at a startup.\n"
        "- turn 5 (assistant): That's interesting! What field is your "
        "current role in?\n"
    )
    claims = extract_structured_claims(
        text,
        source_session_id="s",
        source_role="system",
    )
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert gift and gift[0].source_role == "user"
    assert occ and occ[0].source_role == "user"


def test_inline_role_prefix_after_user_turn_tags_as_assistant():
    """
    The live bundle sometimes carries an embedded ``\\nassistant:``
    inline-role prefix inside a user turn's body (where the wrapper
    stuffed the assistant response into the user content). Anything
    AFTER that prefix should be tagged ``assistant``.
    """
    text = (
        "I'm asking the user about their week.\n"
        "assistant: I used to be a Marketing specialist at a startup.\n"
    )
    claims = extract_structured_claims(
        text,
        source_session_id="s",
        source_role="user",
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ, "no previous_occupation claim extracted"
    assert occ[0].source_role == "assistant", (
        f"inline 'assistant:' prefix should re-tag the sentence; got {occ[0].source_role!r}"
    )


def test_default_role_preserved_when_no_role_prefix_present():
    """
    A line with no role prefix inherits the caller-default
    ``source_role``. This preserves backward-compat with old test
    fixtures and with non-wiki content paths.
    """
    text = "I used to be a Marketing specialist at a startup."
    claims = extract_structured_claims(
        text,
        source_session_id="s",
        source_role="user",
    )
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert occ
    assert occ[0].source_role == "user"


def test_previous_occupation_in_new_role_does_not_emit():
    """
    Negative pin: ``in my new role as a senior marketing analyst``
    must NOT emit a ``previous_occupation`` claim — it describes the
    current role. The lead-in is gated on previous|former|past|old|
    last, never new|current.
    """
    text = (
        "I'm thinking of automating some of the workflows in my new "
        "role as a senior marketing analyst."
    )
    claims = extract_structured_claims(text, **_metadata())
    occ = [c for c in claims if c.relation == "previous_occupation"]
    assert not occ, f"unexpected previous_occupation claim: {[c.object_ for c in occ]}"


def test_section_scoped_went_with_verb():
    """
    Less-common purchase verbs (``went with`` / ``settled on`` /
    ``ended up with``) now fire under a section context.
    """
    text = "**Sister's Birthday**\n- I went with a yellow dress in the end.\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift
    assert "yellow dress" in gift[0].object_.lower()


def test_section_scoped_extraction_rejects_groceries_line():
    """A line about groceries under the birthday section is not a gift."""
    text = "**Sister's Birthday**\n- I picked up some groceries for the party prep.\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift == []


def test_section_scoped_extraction_rejects_surgery_line():
    text = "**Sister's Birthday**\n- I bought some painkillers after my surgery.\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    assert gift == []


def test_section_scoped_extraction_rejects_store_name_as_item():
    """``I went to Best Buy`` under the birthday section — the *store*
    is not the gift."""
    text = "**Sister's Birthday**\n- I bought the gift at Best Buy.\n"
    claims = extract_structured_claims(text, **_metadata())
    gift = [c for c in claims if c.relation == "bought_gift_for"]
    # Either no claim, or the item isn't Best Buy.
    for c in gift:
        assert "best buy" not in c.object_.lower()


def test_section_context_clears_after_distance():
    """
    A heading binds at most ~6 lines of context. A purchase 10 lines
    later under no section must NOT inherit the stale recipient.
    """
    text = (
        "**Sister's Birthday**\n"
        + "\n".join(f"- random line {i}" for i in range(10))
        + "\n- I bought a watch for myself.\n"
    )
    claims = extract_structured_claims(text, **_metadata())
    # No section-scoped claim should carry through to the watch line.
    gift = [c for c in claims if c.relation == "bought_gift_for" and "watch" in c.object_.lower()]
    assert gift == []


# ─── Answerability gate ───────────────────────────────────────────────────


def test_answerability_gate_rejects_store_for_purchased_item():
    """A ``purchased_item`` answer can't be a store name."""
    from evals.longmemeval.structured_finalizer import _answer_shape_ok

    assert not _answer_shape_ok("Best Buy", "purchased_item")
    assert not _answer_shape_ok("Amazon", "purchased_item")
    assert _answer_shape_ok("a yellow dress", "purchased_item")


def test_answerability_gate_rejects_heading_for_purchased_item():
    from evals.longmemeval.structured_finalizer import _answer_shape_ok

    assert not _answer_shape_ok("**Sister's Birthday**", "purchased_item")
    assert not _answer_shape_ok("Occasion: Sister's birthday", "purchased_item")


def test_answerability_gate_requires_title_head_noun_for_occupation():
    """``responsible for X`` is not an occupation_title."""
    from evals.longmemeval.structured_finalizer import _answer_shape_ok

    assert _answer_shape_ok("Marketing specialist at a small startup", "occupation_title")
    assert _answer_shape_ok("software engineer", "occupation_title")
    assert not _answer_shape_ok("responsible for marketing", "occupation_title")
    assert not _answer_shape_ok("managing campaigns", "occupation_title")


def test_finalizer_walks_past_shape_mismatched_claim():
    """
    Two claims tied on relation — first has scaffold object,
    second has a clean title. The gate skips the first and returns
    the second.
    """
    intent = parse_intent("What was my previous occupation?")
    bad = StructuredClaim(
        text="**My Old Job**",
        subject="user",
        relation="previous_occupation",
        object_="**My Old Job**",
        qualifiers={},
        source_session_id="s1",
        source_role="user",
        source_turn_index=0,
        source_span="**My Old Job**",
        confidence=0.95,
    )
    good = StructuredClaim(
        text="I was a Marketing specialist at a small startup.",
        subject="user",
        relation="previous_occupation",
        object_="Marketing specialist at a small startup",
        qualifiers={},
        source_session_id="s1",
        source_role="user",
        source_turn_index=1,
        source_span="Marketing specialist at a small startup",
        confidence=0.85,
    )
    result = finalize_fact_answer(intent, [bad, good])
    assert result is not None
    answer, diag = result
    assert "Marketing specialist" in answer
    # And the diagnostic surfaces what was rejected.
    assert any("shape_mismatch" in r["reason"] for r in diag["rejected_by_answerability_gate"])


def test_query_intent_to_dict_includes_matched_flag():
    intent = parse_intent("What was my previous occupation?")
    d = intent.to_dict()
    assert d["matched"] is True
    assert d["relation"] == "previous_occupation"
