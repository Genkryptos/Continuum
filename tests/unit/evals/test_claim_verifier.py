"""
tests/unit/evals/test_claim_verifier.py
=======================================
Acceptance tests for the verifier stage that runs between the
grounded-claim extractor and the final answerer.

Three spec scenarios + supporting unit tests:

1. **Painting "1 made"** — a ``count`` candidate (``value="1"``,
   subject="painting") leaks into a ``"what is the painting worth?"``
   question because the subject mentions "painting". The verifier
   must FAIL it on answer-type mismatch.
2. **Stanford for Bachelor's** — a hypothetical ``location`` candidate
   (``"Stanford"`` from "looking at Stanford for Master's") survives
   the type gate for a ``"Where did I get my Bachelor's?"`` question.
   The verifier must FAIL it on subject + relation grounds.
3. **IKEA bookshelf "4 hours"** — the factual duration claim whose
   subject contains "bookshelf" must PASS for the bookshelf question.

The remaining tests pin each individual check so a regression in one
of the 5 stages surfaces immediately.
"""
from __future__ import annotations

import pytest

from evals.longmemeval.candidates import (
    Candidate,
    extract_durations,
    extract_locations,
)
from evals.longmemeval.claim_verifier import (
    VerifierResult,
    filter_passing,
    verify_claim,
    verify_claims,
    verify_with_llm,
)
from evals.longmemeval.question_type import QuestionType

# ─── ACCEPTANCE 1 — Painting "1 made" rejected ─────────────────────────────


def test_painting_count_candidate_rejected_for_worth_question():
    """
    A ``count`` candidate carrying ``value="1"`` cannot answer a
    ``"what is the painting worth?"`` question even when its subject
    happens to be "the painting". The verifier must FAIL it on
    answer-type incompatibility.
    """
    bogus_count = Candidate(
        value="1 made",
        normalized_value="1",
        candidate_type="count",
        answer_type="count",
        unit="painting",
        subject="the painting",
        relation="count_of",
        object_="1",
        claim="I made 1 painting last summer.",
        source_span="1 painting",
        confidence=0.7,
    )
    # The real "worth" claim that should survive.
    worth_claim = Candidate(
        value="triple what I paid",
        normalized_value="triple what i paid",
        candidate_type="value",
        answer_type="value",
        subject="painting",
        relation="is_worth",
        object_="triple what I paid",
        claim="The painting is worth triple what I paid.",
        source_span="is worth triple what I paid",
        confidence=0.8,
    )

    results = verify_claims(
        "What is the painting worth?",
        [bogus_count, worth_claim],
        question_type=QuestionType.GENERIC_FACT,
    )
    by_value = {r.candidate.value: r for r in results}
    assert by_value["1 made"].verdict == "FAIL"
    assert "answer_type" in by_value["1 made"].reason or \
           "wrong_relation" in by_value["1 made"].reason
    assert by_value["triple what I paid"].verdict == "PASS"


# ─── ACCEPTANCE 2 — Stanford for Bachelor's rejected ───────────────────────


def test_stanford_rejected_for_bachelors_question():
    """
    Hypothetical ``location`` claim about Master's cannot answer the
    Bachelor's question. The verifier should reject Stanford on
    subject mismatch + hypothetical relation grounds.
    """
    text = (
        "I got my Bachelor's degree at UCLA in 2018. "
        "I'm currently looking at Stanford for Master's programs."
    )
    cs = extract_locations(text, source_session_id="s_edu")
    ucla = next(c for c in cs if c.object_.lower() == "ucla")
    stanford = next(c for c in cs if c.object_.lower() == "stanford")

    results = verify_claims(
        "Where did I get my Bachelor's degree?",
        cs,
        question_type=QuestionType.LOCATION,
    )
    by_obj = {r.candidate.object_.lower(): r for r in results}
    assert by_obj["ucla"].verdict == "PASS"
    assert by_obj["stanford"].verdict == "FAIL"
    # The reason should mention the subject mismatch or the
    # hypothetical relation — both are valid signals here.
    fail_reason = by_obj["stanford"].reason
    assert (
        "subject_mismatch" in fail_reason
        or "hypothetical" in fail_reason
        or "weaker" in fail_reason
    ), f"unexpected reason: {fail_reason}"
    # Confirm filter_passing returns only UCLA.
    passing = filter_passing(results)
    assert ucla in passing
    assert stanford not in passing


# ─── ACCEPTANCE 3 — IKEA bookshelf "4 hours" accepted ──────────────────────


def test_ikea_bookshelf_duration_accepted():
    """The factual bookshelf duration claim must PASS."""
    text = (
        "Assembling the IKEA bookshelf took 4 hours from start to finish. "
        "Last week I put together a small IKEA table — that one only took 2 hours."
    )
    cs = extract_durations(text, source_session_id="s_ikea")
    results = verify_claims(
        "How long did the IKEA bookshelf take to assemble?",
        cs,
        question_type=QuestionType.DURATION,
    )
    by_object = {r.candidate.object_: r for r in results}
    assert by_object["4 hours"].verdict == "PASS"
    # The table claim has subject "IKEA table"; the question is about
    # the bookshelf. Either check 2 (subject_mismatch — they share
    # "IKEA" but not "bookshelf") or check 5 (weaker_question_overlap
    # vs the bookshelf claim) is a valid rejection path. We accept
    # both because the resulting verdict is correct either way.
    table_result = by_object["2 hours"]
    assert table_result.verdict == "FAIL"
    assert any(token in table_result.reason for token in (
        "subject_mismatch",
        "weaker_question_overlap",
    )), f"unexpected reason: {table_result.reason}"


# ─── Individual check coverage ─────────────────────────────────────────────


def test_check1_span_no_overlap_rejected():
    """Source span sharing no content words with the question fails check 1."""
    c = Candidate(
        value="42", normalized_value="42", candidate_type="count",
        answer_type="count", unit="apples", subject="apples",
        relation="count_of", object_="42",
        claim="I have 42 apples in the basket.",
        source_span="42 apples in the basket",
        confidence=0.9,
    )
    result = verify_claim(
        "How many oranges did I buy at the supermarket?",
        c,
        question_type=QuestionType.COUNT,
    )
    assert result.verdict == "FAIL"
    assert "span_no_overlap" in result.reason or \
           "subject_mismatch" in result.reason


def test_check2_subject_mismatch_rejected():
    """Right answer-type but wrong subject fails check 2.

    The candidate's claim mentions "Bachelor's" by accident — for
    instance a sentence that compares Bachelor's and Master's. That
    means check 1 passes (claim overlaps with the question's content
    words) and check 2 has to do the rejection work on
    ``subject_mismatch`` rather than letting check 1 catch it.
    """
    c = Candidate(
        value="Berkeley", normalized_value="Berkeley",
        candidate_type="location", answer_type="location",
        subject="Master's program",
        relation="located_at", object_="Berkeley",
        # Note: the *claim* mentions Bachelor's so check 1 passes;
        # the *subject* is "Master's program" so check 2 must fire.
        claim="My Bachelor's was elsewhere; my Master's program is at Berkeley.",
        source_span="at Berkeley",
        confidence=0.7,
    )
    result = verify_claim(
        "Where did I get my Bachelor's degree?",
        c,
        question_type=QuestionType.LOCATION,
    )
    assert result.verdict == "FAIL"
    assert "subject_mismatch" in result.reason


def test_check3_answer_type_mismatch_rejected():
    """Wrong answer_type for the question type fails check 3."""
    c = Candidate(
        value="March 14, 2024", normalized_value="march 14 2024",
        candidate_type="date", answer_type="date",
        subject="job",
        relation="located_at", object_="March 14, 2024",
        claim="I started the job on March 14, 2024.",
        source_span="March 14, 2024",
        confidence=0.85,
    )
    # Asking a LOCATION question — a date candidate must fail the type check
    result = verify_claim(
        "Where did I start the job?",
        c,
        question_type=QuestionType.LOCATION,
    )
    assert result.verdict == "FAIL"
    assert "answer_type_mismatch" in result.reason


def test_check3_wrong_relation_for_keyword_rejected():
    """``paid`` candidate fails for a ``worth`` question."""
    c = Candidate(
        value="200 dollars", normalized_value="200",
        candidate_type="value", answer_type="amount",
        subject="painting",
        relation="paid", object_="200 dollars",
        claim="I paid 200 dollars for the painting.",
        source_span="paid 200 dollars for the painting",
        confidence=0.7,
    )
    result = verify_claim(
        "What is the painting worth today?",
        c,
        question_type=QuestionType.GENERIC_FACT,
    )
    assert result.verdict == "FAIL"
    assert "wrong_relation_for_keyword" in result.reason


def test_check4_hypothetical_rejected_for_completed_question():
    """``looking_at`` relation fails for a "did I" / "got" question."""
    c = Candidate(
        value="Stanford", normalized_value="Stanford",
        candidate_type="location", answer_type="location",
        subject="Master's",
        relation="looking_at", object_="Stanford",
        claim="I'm looking at Stanford for Master's programs.",
        source_span="looking at Stanford",
        confidence=0.4,
    )
    result = verify_claim(
        "Where did I get my degree?",
        c,
        question_type=QuestionType.LOCATION,
    )
    assert result.verdict == "FAIL"


def test_check5_no_stronger_conflict_drops_hypothetical():
    """
    When two candidates share the subject (overlap), the one with a
    hypothetical relation is dropped in favour of the factual one.
    """
    ucla = Candidate(
        value="UCLA", normalized_value="UCLA",
        candidate_type="location", answer_type="location",
        subject="Bachelor's degree",
        relation="got_at", object_="UCLA",
        claim="I got my Bachelor's degree at UCLA.",
        source_span="at UCLA",
        confidence=0.9,
    )
    stanford = Candidate(
        value="Stanford", normalized_value="Stanford",
        candidate_type="location", answer_type="location",
        # SAME subject as ucla to trigger check 5
        subject="Bachelor's degree",
        relation="looking_at", object_="Stanford",
        claim="I'm looking at Stanford for the Bachelor's program too.",
        source_span="looking at Stanford",
        confidence=0.4,
    )
    results = verify_claims(
        "Where did I get my Bachelor's degree?",
        [ucla, stanford],
        question_type=QuestionType.LOCATION,
    )
    by_obj = {r.candidate.object_: r for r in results}
    assert by_obj["UCLA"].verdict == "PASS"
    assert by_obj["Stanford"].verdict == "FAIL"


def test_no_question_tokens_passes_trivially():
    """When the question has no content words, every check is a pass-through."""
    c = Candidate(
        value="x", normalized_value="x", candidate_type="entity",
        answer_type="entity", subject="thing", relation="is",
        object_="x", claim="It is x.", source_span="x", confidence=0.5,
    )
    # Question is all stopwords / unparseable
    result = verify_claim("the", c)
    assert result.verdict == "PASS"


# ─── filter_passing helper ─────────────────────────────────────────────────


def test_filter_passing_preserves_order_and_drops_fails():
    candidates = [
        Candidate(
            value="A", normalized_value="a", candidate_type="count",
            answer_type="count", subject="apples", relation="count_of",
            object_="42", source_span="42 apples", confidence=0.9,
        ),
        Candidate(
            value="B", normalized_value="b", candidate_type="location",
            answer_type="location", subject="oranges",
            relation="located_at", object_="Berkeley",
            source_span="at Berkeley", confidence=0.9,
        ),
    ]
    results = verify_claims(
        "How many oranges do I have?",
        candidates,
        question_type=QuestionType.COUNT,
    )
    passing = filter_passing(results)
    # B has answer_type=location → fails the COUNT type gate.
    # A has answer_type=count but subject="apples" ≠ "oranges" — fails subject.
    # Both fail; passing is empty. (The test exists to confirm the
    # filter discards results cleanly.)
    assert passing == []
    # And the result set still records both verdicts.
    assert len(results) == 2
    assert all(r.verdict == "FAIL" for r in results)


# ─── Optional LLM verifier ─────────────────────────────────────────────────


class _StubLLMVerifier:
    """Fake LLM verifier — flips PASS to FAIL for a configured value."""

    def __init__(self, reject_values: set[str]) -> None:
        self.reject_values = reject_values
        self.calls = 0

    async def verify(
        self, question: str, candidate: Candidate,
    ) -> tuple[str, str]:
        self.calls += 1
        if candidate.value in self.reject_values:
            return "FAIL", "stub_rejected"
        return "PASS", "stub_accepted"


@pytest.mark.asyncio
async def test_llm_verifier_only_called_on_deterministic_passes():
    """A candidate that already failed deterministic checks isn't re-checked."""
    ucla = Candidate(
        value="UCLA", normalized_value="UCLA",
        candidate_type="location", answer_type="location",
        subject="Bachelor's degree", relation="got_at", object_="UCLA",
        claim="I got my Bachelor's degree at UCLA in 2018.",
        source_span="at UCLA", confidence=0.9,
    )
    stanford = Candidate(
        value="Stanford", normalized_value="Stanford",
        candidate_type="location", answer_type="location",
        subject="Master's", relation="looking_at", object_="Stanford",
        claim="I'm looking at Stanford for my Master's.",
        source_span="looking at Stanford", confidence=0.4,
    )
    stub = _StubLLMVerifier(reject_values={"UCLA"})
    results = await verify_with_llm(
        "Where did I get my Bachelor's degree?",
        [ucla, stanford],
        stub,
        question_type=QuestionType.LOCATION,
    )
    by_value = {r.candidate.value: r for r in results}
    # Stanford was rejected deterministically — LLM not called on it.
    assert by_value["Stanford"].verdict == "FAIL"
    assert "llm" not in by_value["Stanford"].reason  # came from deterministic stage
    # UCLA passed deterministic; LLM rejected it via the stub.
    assert by_value["UCLA"].verdict == "FAIL"
    assert "stub_rejected" in by_value["UCLA"].reason
    # Only one LLM call was made — the deterministic stage spared one.
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_llm_verifier_error_falls_back_to_deterministic_pass():
    """If the LLM raises, we keep the deterministic verdict."""

    class _BrokenLLM:
        async def verify(self, q, c):
            raise RuntimeError("provider down")

    ucla = Candidate(
        value="UCLA", normalized_value="UCLA",
        candidate_type="location", answer_type="location",
        subject="Bachelor's degree", relation="got_at", object_="UCLA",
        claim="I got my Bachelor's degree at UCLA in 2018.",
        source_span="at UCLA", confidence=0.9,
    )
    results = await verify_with_llm(
        "Where did I get my Bachelor's degree?",
        [ucla],
        _BrokenLLM(),
        question_type=QuestionType.LOCATION,
    )
    assert len(results) == 1
    assert results[0].verdict == "PASS"
    assert "llm_verifier_error" in results[0].reason


# ─── to_dict serialization ─────────────────────────────────────────────────


def test_verifier_result_serializes_cleanly():
    c = Candidate(
        value="UCLA", normalized_value="UCLA", candidate_type="location",
        answer_type="location", subject="Bachelor's", relation="got_at",
        object_="UCLA", source_span="at UCLA", confidence=0.9,
    )
    r = VerifierResult(c, "PASS", "ok")
    d = r.to_dict()
    assert d["verdict"] == "PASS"
    assert d["reason"] == "ok"
    assert d["candidate"]["object"] == "UCLA"
    assert d["candidate"]["subject"] == "Bachelor's"
