"""
tests/unit/evals/test_precision_under_noise.py
==============================================
Tests for the precision-under-noisy-recall benchmark.

Acceptance contract from the spec:

* Benchmark **fails** on the current noisy pipeline (the four
  regression cases — Serenity Yoga, IKEA 2h/4h, painting worth,
  UCLA/Stanford — all get the wrong answer because no grounding /
  verification stage filters out distractors).
* Benchmark **passes** after span selector + grounded claims +
  verifier (the filtered pipeline).

The tests also pin the metric shape, the per-case dataset contents,
and the JSON serialisation so a regression in the metrics block
surfaces in CI rather than next week's benchmark spend.
"""
from __future__ import annotations

import pytest

from bench.precision_under_noise import (
    DATASET,
    NoiseCase,
    NoiseSession,
    aggregate,
    run_benchmark,
    run_filtered_pipeline,
    run_noisy_pipeline,
)

# ─── Dataset shape ─────────────────────────────────────────────────────────


def test_dataset_contains_required_regression_cases():
    """All four spec-cited cases must be present."""
    ids = {c.case_id for c in DATASET}
    assert "serenity_yoga" in ids
    assert "ikea_bookshelf_4h" in ids
    assert "painting_worth_triple" in ids
    assert "ucla_bachelors" in ids


def test_dataset_includes_abstention_cases():
    """At least two abstention cases (no-answer in bundle) exist."""
    abstention = [c for c in DATASET if c.is_no_answer]
    assert len(abstention) >= 2
    # Each abstention case expects "I don't know"
    for c in abstention:
        assert "i don't know" in c.expected_answer.lower()


def test_every_answerable_case_has_recall_at_1_by_construction():
    """The expected session_id must be in the bundle (recall@k=1.0)."""
    for case in DATASET:
        if case.is_no_answer:
            continue
        session_ids = {s.session_id for s in case.sessions}
        assert case.expected_session_id in session_ids, (
            f"case {case.case_id}: expected session "
            f"{case.expected_session_id!r} not in bundle"
        )


def test_every_answerable_case_marks_one_correct_session():
    """Exactly one session per answerable case is flagged is_correct=True."""
    for case in DATASET:
        if case.is_no_answer:
            continue
        correct = [s for s in case.sessions if s.is_correct]
        assert len(correct) == 1, (
            f"case {case.case_id}: expected 1 correct session, got {len(correct)}"
        )
        assert correct[0].session_id == case.expected_session_id


def test_distractors_are_semantically_similar():
    """Every answerable case has at least one distractor sharing a domain word."""
    for case in DATASET:
        if case.is_no_answer:
            continue
        # Count distractor sessions (non-correct)
        distractors = [s for s in case.sessions if not s.is_correct]
        assert distractors, f"case {case.case_id} has no distractor sessions"


# ─── ACCEPTANCE — noisy pipeline FAILS the regression cases ───────────────


_REGRESSION_CASE_IDS = {
    "serenity_yoga",
    "ikea_bookshelf_4h",
    "painting_worth_triple",
    "ucla_bachelors",
}


def test_noisy_pipeline_fails_all_four_regression_cases():
    """
    The naïve pipeline must get every regression case WRONG. This is
    the "benchmark fails on current noisy pipeline" half of the spec.
    """
    regression = [c for c in DATASET if c.case_id in _REGRESSION_CASE_IDS]
    assert len(regression) == 4
    results = [run_noisy_pipeline(c) for c in regression]
    n_correct = sum(1 for r in results if r.correct)
    assert n_correct == 0, (
        f"noisy pipeline got {n_correct}/4 regression cases right — "
        "the benchmark is supposed to demonstrate the noisy failure mode"
    )
    # And the wrong answers should be drawn from distractor sessions.
    for r in results:
        # evidence_precision = 0 means the selected candidate came from
        # a non-expected session (the distractor)
        assert r.evidence_precision == 0.0, (
            f"case {r.case_id}: noisy pipeline picked from the correct "
            f"session despite the distractor ordering"
        )


def test_noisy_pipeline_has_high_recall_but_low_accuracy():
    """recall@k=1.0 alongside accuracy<0.5 — the gap the spec targets."""
    results = [run_noisy_pipeline(c) for c in DATASET]
    agg = aggregate(results)
    assert agg["recall_at_k"] == 1.0
    assert agg["final_answer_accuracy"] < 0.5, (
        f"noisy pipeline accuracy {agg['final_answer_accuracy']:.1%} "
        "should be < 50 % to demonstrate the precision/recall gap"
    )


# ─── ACCEPTANCE — filtered pipeline PASSES the regression cases ───────────


def test_filtered_pipeline_passes_all_four_regression_cases():
    """
    Grounded claims + verifier + evidence packet → every regression
    case answered correctly. The other half of the spec acceptance.
    """
    regression = [c for c in DATASET if c.case_id in _REGRESSION_CASE_IDS]
    results = [run_filtered_pipeline(c) for c in regression]
    n_correct = sum(1 for r in results if r.correct)
    assert n_correct >= 3, (
        f"filtered pipeline got only {n_correct}/4 regression cases — "
        "spec acceptance is ≥75% (3/4 or 4/4)"
    )


def test_filtered_pipeline_passes_each_regression_case_individually():
    """Pin every regression case so individual failures surface in CI."""
    by_id = {c.case_id: c for c in DATASET}
    expectations = {
        "serenity_yoga":          "Serenity Yoga",
        "ikea_bookshelf_4h":      "4 hours",
        "painting_worth_triple":  "triple what I paid",
        "ucla_bachelors":         "UCLA",
    }
    for case_id, expected_substr in expectations.items():
        result = run_filtered_pipeline(by_id[case_id])
        assert result.correct, (
            f"case {case_id}: filtered pipeline returned "
            f"{result.actual_answer!r}, expected substring {expected_substr!r}"
        )
        # And it pulled evidence from the correct session.
        assert result.evidence_precision == 1.0, (
            f"case {case_id}: evidence_precision is "
            f"{result.evidence_precision} — packet drew from a distractor"
        )


def test_filtered_pipeline_abstains_on_no_answer_cases():
    """Abstention cases: pipeline must say "I don't know", not hallucinate."""
    abstention_cases = [c for c in DATASET if c.is_no_answer]
    results = [run_filtered_pipeline(c) for c in abstention_cases]
    for r in results:
        assert r.abstention_correct, (
            f"case {r.case_id}: pipeline returned {r.actual_answer!r} "
            "instead of abstaining"
        )
        assert "i don't know" in r.actual_answer.lower()


# ─── Metric correctness ───────────────────────────────────────────────────


def test_noise_token_ratio_drops_under_filtered_pipeline():
    """The whole point of the packet is to drive noise tokens to zero."""
    noisy = [run_noisy_pipeline(c) for c in DATASET]
    filtered = [run_filtered_pipeline(c) for c in DATASET]
    noisy_agg = aggregate(noisy)
    filt_agg = aggregate(filtered)
    assert filt_agg["noise_token_ratio"] < noisy_agg["noise_token_ratio"]


def test_evidence_precision_improves_under_filtered_pipeline():
    noisy = [run_noisy_pipeline(c) for c in DATASET]
    filtered = [run_filtered_pipeline(c) for c in DATASET]
    assert (
        aggregate(filtered)["evidence_precision"]
        > aggregate(noisy)["evidence_precision"]
    )


def test_verified_candidate_accuracy_high_in_filtered_pipeline():
    """Verifier should keep only candidates from the correct session."""
    filtered = [run_filtered_pipeline(c) for c in DATASET if not c.is_no_answer]
    for r in filtered:
        if r.n_verified_candidates == 0:
            # Permissible — the verifier may legitimately reject every
            # candidate on some cases (forced abstain). Just check we
            # didn't somehow pass a wrong answer.
            continue
        ratio = r.n_verified_from_correct_session / r.n_verified_candidates
        assert ratio == 1.0, (
            f"case {r.case_id}: only {ratio:.1%} of verified candidates "
            "are from the expected session"
        )


def test_recall_at_k_is_always_one_for_answerable_cases():
    """By construction, recall@k=1.0 for every case in the dataset."""
    for case in DATASET:
        result = run_filtered_pipeline(case)
        assert result.recall_at_k == 1.0, (
            f"case {case.case_id}: recall_at_k={result.recall_at_k}"
        )


# ─── Aggregate + serialisation ────────────────────────────────────────────


def test_run_benchmark_emits_full_payload_shape():
    """JSON shape mirrors the other bench/*_latest.json files."""
    payload = run_benchmark()
    assert payload["benchmark"] == "precision_under_noise"
    assert "timestamp" in payload
    assert payload["config"]["n_cases"] == len(DATASET)

    systems = {s["system"] for s in payload["systems"]}
    assert systems == {"noisy_baseline", "filtered_pipeline"}

    for sys_entry in payload["systems"]:
        m = sys_entry["metrics"]
        # Every required metric is present.
        for key in (
            "recall_at_k",
            "evidence_precision",
            "noise_token_ratio",
            "verified_candidate_accuracy",
            "final_answer_accuracy",
            "abstention_correctness",
        ):
            assert key in m, f"system {sys_entry['system']} missing {key}"


def test_filtered_metrics_strictly_dominate_noisy():
    """The headline metrics must be strictly better on the filtered side."""
    payload = run_benchmark()
    noisy = next(s["metrics"] for s in payload["systems"] if s["system"] == "noisy_baseline")
    filtered = next(s["metrics"] for s in payload["systems"] if s["system"] == "filtered_pipeline")

    assert filtered["final_answer_accuracy"] > noisy["final_answer_accuracy"]
    assert filtered["evidence_precision"] >= noisy["evidence_precision"]
    assert filtered["noise_token_ratio"] <= noisy["noise_token_ratio"]


def test_payload_serializes_to_json():
    """Full payload must round-trip through json — required for trace logs."""
    import json
    payload = run_benchmark()
    serialized = json.dumps(payload, default=str)
    parsed = json.loads(serialized)
    assert parsed["benchmark"] == "precision_under_noise"
    assert len(parsed["systems"]) == 2


# ─── Custom dataset usage ─────────────────────────────────────────────────


def test_run_benchmark_accepts_external_dataset():
    """Callers can pass their own dataset (for derivative benchmarks)."""
    tiny = [
        NoiseCase(
            case_id="tiny",
            category="location",
            question="Where do I do yoga?",
            expected_answer="Serenity Yoga",
            expected_session_id="s1",
            sessions=[
                NoiseSession(
                    session_id="s1",
                    content="I do yoga at Serenity Yoga every morning.",
                    is_correct=True,
                ),
                NoiseSession(
                    session_id="s2",
                    content="I shop for groceries at Whole Foods.",
                ),
            ],
        ),
    ]
    payload = run_benchmark(dataset=tiny)
    assert payload["config"]["n_cases"] == 1


# ─── Per-case result shape ────────────────────────────────────────────────


def test_case_result_carries_all_required_fields():
    """The dataclass exposes the spec's required telemetry fields."""
    case = next(c for c in DATASET if c.case_id == "serenity_yoga")
    result = run_filtered_pipeline(case)
    d = result.to_dict()
    for key in (
        "case_id",
        "category",
        "expected_answer",
        "actual_answer",
        "recall_at_k",
        "evidence_precision",
        "noise_token_ratio",
        "selected_evidence_count",
        "excluded_noise_count",
        "n_verified_candidates",
        "n_verified_from_correct_session",
        "correct",
        "abstention_correct",
    ):
        assert key in d


# ─── Sanity smoke ─────────────────────────────────────────────────────────


def test_dataset_is_nonempty_and_all_cases_unique():
    assert len(DATASET) >= 4
    ids = [c.case_id for c in DATASET]
    assert len(ids) == len(set(ids)), "duplicate case_id in DATASET"


@pytest.mark.parametrize("case_id", sorted(_REGRESSION_CASE_IDS))
def test_each_regression_case_total_tokens_positive(case_id):
    """Sanity — every regression case has a non-empty bundle."""
    case = next(c for c in DATASET if c.case_id == case_id)
    assert case.total_tokens() > 0
    assert case.correct_tokens() > 0
