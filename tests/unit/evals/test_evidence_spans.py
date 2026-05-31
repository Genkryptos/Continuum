"""
tests/unit/evals/test_evidence_spans.py
=======================================
Unit tests for :mod:`evals.longmemeval.evidence_spans`.

Covers the failure modes the 500-row analysis identified as the
dominant precision loss on multi-session / temporal-reasoning rows:
the correct session was retrieved, but the wide evidence window let
the answerer latch onto a *nearby* number / location / name that
wasn't the one the question asked about.

The acceptance scenarios come straight from the spec:

* **IKEA bookshelf**: window contains both "the bookshelf took 4 hours"
  and "the IKEA table took 2 hours" — selector must surface the
  4-hour span, not the 2-hour one.
* **Bachelor's at UCLA**: window contains a stated fact ("I got my
  Bachelor's at UCLA") and a *list of options* for Master's programs
  ("USC, Berkeley, …") — selector must surface UCLA, not the options.
"""

from __future__ import annotations

from continuum.core.types import MemoryItem
from evals.longmemeval.evidence_spans import (
    EvidenceSpan,
    render_spans_for_prompt,
    select_spans,
)
from evals.longmemeval.question_type import QuestionType


def _wiki_window(session_id: str, body: str, *, item_id: str = "") -> MemoryItem:
    """Build a MemoryItem that mirrors WikiMemoryRetriever._evidence_window."""
    return MemoryItem(
        id=item_id or f"item-{session_id}",
        content=body,
        session_id=session_id,
        metadata={"session_id": session_id, "role": "user"},
    )


# ─── Acceptance: IKEA bookshelf duration ───────────────────────────────────


def test_ikea_bookshelf_selects_correct_duration_span():
    """
    The wide window mentions two durations — 4 hours for the bookshelf
    (correct) and 2 hours for an unrelated table (distractor). The
    selector must surface the 4-hour span.
    """
    window = _wiki_window(
        "s_ikea_bookshelf",
        "# Evidence Window for s_ikea_bookshelf\n\n"
        "## Matched Turn\n"
        "- (user): Assembling the IKEA bookshelf took 4 hours from start to finish.\n\n"
        "## Nearby Context\n"
        "- (user): Last week I put together a small IKEA table — that one only took 2 hours.\n"
        "- (assistant): Great, glad you got both done.\n"
        "- (user): The bookshelf was much more complicated because of the shelves.\n",
    )

    spans = select_spans(
        "How long did the IKEA bookshelf take to assemble?",
        [window],
        question_type=QuestionType.DURATION,
    )

    assert spans, "selector returned nothing for a question its window clearly answers"
    top = spans[0]
    assert "4 hours" in top.text
    assert "2 hours" not in top.text
    assert top.source_session_id == "s_ikea_bookshelf"
    # And the distractor "2 hours" must not appear in any returned span.
    assert not any("2 hours" in s.text for s in spans)


def test_ikea_bookshelf_preserves_exact_text():
    """The span text must be the verbatim substring, not a paraphrase."""
    src = "Assembling the IKEA bookshelf took 4 hours from start to finish."
    window = _wiki_window(
        "s_ikea_bookshelf",
        f"## Matched Turn\n- (user): {src}\n",
    )
    spans = select_spans(
        "How long did the IKEA bookshelf take?",
        [window],
        question_type=QuestionType.DURATION,
    )
    assert spans
    assert spans[0].text == src


# ─── Acceptance: Bachelor's at UCLA distractor ─────────────────────────────


def test_bachelors_at_ucla_beats_masters_options():
    """
    Window contains the stated fact (Bachelor's at UCLA) and several
    hypothetical/option lines about Master's programs at other schools.
    The selector must surface UCLA, not the Master's options.
    """
    window = _wiki_window(
        "s_education",
        "## Matched Turn\n"
        "- (user): I got my Bachelor's degree at UCLA in 2018.\n\n"
        "## Nearby Context\n"
        "- (user): I'm looking at Master's programs — options include USC, Berkeley, and Stanford.\n"
        "- (assistant): Did you decide between them yet?\n"
        "- (user): Not yet, I'd consider either USC or Berkeley.\n",
    )

    spans = select_spans(
        "Where did I get my Bachelor's degree?",
        [window],
        question_type=QuestionType.LOCATION,
    )

    assert spans
    top = spans[0]
    assert "UCLA" in top.text
    assert "USC" not in top.text and "Berkeley" not in top.text
    # The distractor lines (options / questions) must not crowd out
    # the stated fact even if they share content words like "Master's".
    assert not any("USC" in s.text or "Berkeley" in s.text or "Stanford" in s.text for s in spans)


# ─── Empty-selection forces abstain ────────────────────────────────────────


def test_no_overlap_returns_empty():
    """
    When the window shares no content words with the question, the
    selector must return [] so the caller can force abstain.
    """
    window = _wiki_window(
        "s_irrelevant",
        "## Matched Turn\n"
        "- (user): I had cereal for breakfast.\n\n"
        "## Nearby Context\n"
        "- (user): The weather was nice yesterday.\n",
    )
    spans = select_spans(
        "What was my flight number from Tokyo to San Francisco?",
        [window],
        question_type=QuestionType.ENTITY_NAME,
    )
    assert spans == []


def test_only_distractors_returns_empty():
    """
    When every overlap candidate is a hypothetical / question, the
    score after penalty must drop below zero and the selector returns
    [] rather than surfacing a misleading span.
    """
    window = _wiki_window(
        "s_options",
        "## Matched Turn\n"
        "- (user): I might switch to a different gym — maybe Equinox or "
        "Crunch, I'm considering options.\n"
        "- (assistant): Have you decided which gym you'd prefer?\n",
    )
    spans = select_spans(
        "Which gym do I go to?",
        [window],
        question_type=QuestionType.LOCATION,
    )
    # All lines are option-listing / questions → penalty drops score
    # to ≤ 0, selector returns nothing.
    assert spans == []


def test_empty_items_returns_empty():
    assert select_spans("anything", [], question_type=QuestionType.COUNT) == []


# ─── Source attribution ────────────────────────────────────────────────────


def test_session_id_and_turn_id_propagate():
    window = MemoryItem(
        id="turn-42",
        content="## Matched Turn\n- (user): The conference was in Berlin in 2023.\n",
        session_id="s_berlin",
        metadata={"session_id": "s_berlin", "role": "user"},
    )
    spans = select_spans(
        "Where was the conference held?",
        [window],
        question_type=QuestionType.LOCATION,
    )
    assert spans
    s = spans[0]
    assert s.source_session_id == "s_berlin"
    assert s.source_turn_id == "turn-42"
    assert s.role == "user"


def test_role_falls_back_when_bullet_has_no_role_prefix():
    """A bullet without `(role):` falls back to the item's metadata role."""
    window = MemoryItem(
        id="turn-x",
        content="- The marathon took 4 hours 12 minutes.",
        session_id="s_marathon",
        metadata={"session_id": "s_marathon", "role": "user"},
    )
    spans = select_spans(
        "How long did the marathon take?",
        [window],
        question_type=QuestionType.DURATION,
    )
    assert spans
    assert spans[0].role == "user"


# ─── Cap + dedup ───────────────────────────────────────────────────────────


def test_max_spans_cap_respected():
    """At most ``max_spans`` results, regardless of how many candidates."""
    lines = "\n".join(f"- (user): The bookshelf took {n} hours session {n}." for n in range(1, 10))
    window = _wiki_window("s_many", f"## Matched Turn\n{lines}\n")
    spans = select_spans(
        "How long did the bookshelf take?",
        [window],
        max_spans=2,
        question_type=QuestionType.DURATION,
    )
    assert len(spans) <= 2


def test_dedupes_substring_repeats():
    """Wiki windows repeat the matched turn in 'Nearby Context'; dedup it."""
    repeated = "The IKEA bookshelf took 4 hours to assemble."
    window = _wiki_window(
        "s_dup",
        f"## Matched Turn\n- (user): {repeated}\n\n## Nearby Context\n- (user): {repeated}\n",
    )
    spans = select_spans(
        "How long did the IKEA bookshelf take?",
        [window],
        question_type=QuestionType.DURATION,
    )
    # The selector should keep one copy, not two near-identical lines.
    assert len(spans) == 1


# ─── Shape-aware bonuses ───────────────────────────────────────────────────


def test_count_question_prefers_span_with_number():
    """COUNT shape bonus surfaces the numbered span over a tie-overlap one."""
    window = _wiki_window(
        "s_count",
        "## Matched Turn\n"
        "- (user): I own shirts in many colours.\n"
        "- (user): I have 7 shirts in my closet right now.\n",
    )
    spans = select_spans(
        "How many shirts do I own?",
        [window],
        question_type=QuestionType.COUNT,
    )
    assert spans
    assert "7" in spans[0].text


def test_date_time_question_prefers_dated_span():
    window = _wiki_window(
        "s_date",
        "## Matched Turn\n"
        "- (user): I started the new job recently.\n"
        "- (user): I started the new job on March 14th, 2024.\n",
    )
    spans = select_spans(
        "When did I start the new job?",
        [window],
        question_type=QuestionType.DATE_TIME,
    )
    assert spans
    assert "March" in spans[0].text


# ─── Rendering for the prompt ──────────────────────────────────────────────


def test_render_spans_for_prompt_includes_session_role_and_text():
    s = EvidenceSpan(
        text="4 hours.",
        source_session_id="s1",
        source_turn_id="t1",
        role="user",
        score=1.2,
        reason="overlap=2/4 has_duration",
    )
    out = render_spans_for_prompt([s])
    assert "[session=s1]" in out
    assert "(user)" in out
    assert "4 hours." in out


def test_render_spans_for_prompt_empty_returns_empty_string():
    assert render_spans_for_prompt([]) == ""
