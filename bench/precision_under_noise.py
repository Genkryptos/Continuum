"""
bench/precision_under_noise.py
==============================
Precision-under-noisy-recall benchmark.

A failure mode the existing benchmarks don't cover
--------------------------------------------------
``bench/retrieval_quality.py`` measures *recall@k* — how often the
expected session is in the top-k retrieved. Continuum already does
well there (recall is the easy half of LongMemEval). What it does
NOT measure is what happens **between** "recall=1.0" and "the model
gave the right answer": a noisy bundle full of distractor sessions
that lexically resemble the correct one, picked up by retrieval
because they share vocabulary with the question.

This benchmark targets that gap. Every case is constructed so that:

1. The expected answer-bearing session is present in the bundle
   (recall@k = 1.0 by construction).
2. Several semantically similar distractor sessions are also present
   (same domain words, same answer shape, often higher rank).
3. The naïve pipeline picks an answer from the distractor; the
   filtered pipeline (grounded claims + verifier + evidence packet)
   picks the right one.

What it measures
----------------
* **recall@k** — fraction of cases where the expected session_id is
  in the input bundle. By construction this is always 1.0 here; the
  metric exists to document that "recall=1.0 doesn't imply
  accuracy=1.0".
* **evidence_precision** — fraction of selected evidence whose
  source_session_id equals the expected one. The noisy pipeline
  scores low; the filtered pipeline scores high.
* **noise_token_ratio** — tokens contributed by distractor sessions
  divided by total tokens passed to the final-answer prompt. The
  whole point of the packet is to drive this near zero.
* **verified_candidate_accuracy** — of the verifier's PASS set,
  what fraction came from the expected session?
* **final_answer_accuracy** — substring match against the gold
  answer. The headline metric.
* **abstention_correctness** — on no-answer cases (the expected
  answer is "I don't know"), did the pipeline abstain rather than
  hallucinate?

Regression cases the spec asked for
-----------------------------------
* **Serenity Yoga / Whole Foods** — both are real "Where" answers
  in the bundle; only one is about yoga.
* **IKEA 2 hours vs 4 hours** — same brand, different objects.
* **Painting worth triple** — qualitative value beats numeric paid.
* **UCLA vs Stanford** — factual completed vs hypothetical option.

Plus two abstention cases (the question's answer isn't in the
bundle, so abstaining is the correct behaviour).

Acceptance
----------
* The **noisy** pipeline scores < 50 % final-answer accuracy on the
  4 regression cases.
* The **filtered** pipeline scores ≥ 75 % on the same cases.

Both pipelines run hermetically — no LLM calls, no network. The
noisy pipeline is the deterministic worst-case stand-in for a
broad-context LLM with no grounding: it picks the answer-shaped
span that appears earliest in the concatenated bundle.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from evals.longmemeval.candidates import (
    extract_candidates_from_text,
    filter_candidates_for_question,
)
from evals.longmemeval.claim_verifier import filter_passing, verify_claims
from evals.longmemeval.evidence_packet import build_evidence_packet
from evals.longmemeval.question_type import classify

log = logging.getLogger(__name__)


# ─── Dataset shapes ────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class NoiseSession:
    """One session window in a benchmark case. ``content`` is the user/assistant text."""

    session_id: str
    content: str
    is_correct: bool = False


@dataclasses.dataclass(frozen=True)
class NoiseCase:
    """One precision-under-noise benchmark case."""

    case_id: str
    category: str
    question: str
    expected_answer: str
    expected_session_id: str
    sessions: list[NoiseSession]
    is_no_answer: bool = False

    def total_tokens(self) -> int:
        """Approximate token count via whitespace split — fine for ratios."""
        return sum(len(s.content.split()) for s in self.sessions)

    def correct_tokens(self) -> int:
        return sum(
            len(s.content.split()) for s in self.sessions if s.is_correct
        )


# ─── The dataset — 4 regression cases + 2 abstention cases ────────────────


DATASET: list[NoiseCase] = [
    # ─── 1. Serenity Yoga / Whole Foods ──────────────────────────────────
    NoiseCase(
        case_id="serenity_yoga",
        category="location",
        question="Where do I go for yoga?",
        expected_answer="Serenity Yoga",
        expected_session_id="s_yoga",
        sessions=[
            # Distractor RANKED FIRST so the noisy pipeline picks it.
            NoiseSession(
                session_id="s_groceries",
                content=(
                    "I shop for groceries at Whole Foods every Saturday "
                    "morning before the crowds."
                ),
            ),
            NoiseSession(
                session_id="s_yoga",
                content=(
                    "I do yoga at Serenity Yoga every Tuesday morning "
                    "for my class."
                ),
                is_correct=True,
            ),
            NoiseSession(
                session_id="s_dining",
                content=(
                    "My favorite restaurant is Mediterranean Mood, "
                    "downtown near the river."
                ),
            ),
        ],
    ),
    # ─── 2. IKEA 2 hours vs 4 hours ──────────────────────────────────────
    NoiseCase(
        case_id="ikea_bookshelf_4h",
        category="duration",
        question="How long did the IKEA bookshelf take to assemble?",
        expected_answer="4 hours",
        expected_session_id="s_ikea_bookshelf",
        sessions=[
            # Distractor first — the noisy pipeline sees "2 hours" first.
            NoiseSession(
                session_id="s_ikea_table",
                content=(
                    "Last week I put together a small IKEA table — "
                    "that one only took 2 hours."
                ),
            ),
            NoiseSession(
                session_id="s_ikea_bookshelf",
                content=(
                    "Assembling the IKEA bookshelf took 4 hours from "
                    "start to finish."
                ),
                is_correct=True,
            ),
            NoiseSession(
                session_id="s_ikea_dresser",
                content=(
                    "I helped my friend with an IKEA dresser, which was "
                    "3 hours of work."
                ),
            ),
        ],
    ),
    # ─── 3. Painting worth triple (not the amount paid) ──────────────────
    NoiseCase(
        case_id="painting_worth_triple",
        category="value",
        question="What is the painting worth today?",
        expected_answer="triple what I paid",
        expected_session_id="s_painting_worth",
        sessions=[
            NoiseSession(
                session_id="s_painting_paid",
                content=(
                    "I paid 200 dollars for the painting back in 2015 "
                    "at a small gallery."
                ),
            ),
            NoiseSession(
                session_id="s_painting_worth",
                content=(
                    "Today the painting is worth triple what I paid "
                    "after the artist became famous."
                ),
                is_correct=True,
            ),
        ],
    ),
    # ─── 4. UCLA Bachelor's vs Stanford Master's options ─────────────────
    NoiseCase(
        case_id="ucla_bachelors",
        category="location",
        question="Where did I get my Bachelor's degree?",
        expected_answer="UCLA",
        expected_session_id="s_ucla",
        sessions=[
            # Stanford (hypothetical, Master's) ranked first
            NoiseSession(
                session_id="s_stanford_masters",
                content=(
                    "I'm currently looking at Stanford for my Master's "
                    "program — they have a great applied math track."
                ),
            ),
            NoiseSession(
                session_id="s_ucla",
                content=(
                    "I got my Bachelor's degree at UCLA in 2018, "
                    "majored in computer science."
                ),
                is_correct=True,
            ),
            NoiseSession(
                session_id="s_usc_summer",
                content=(
                    "I attended USC for a summer program in 2019 "
                    "on machine learning."
                ),
            ),
        ],
    ),
    # ─── 5. Abstention — question's answer isn't in the bundle ───────────
    NoiseCase(
        case_id="podcast_no_answer",
        category="abstention",
        question="What is the name of my podcast?",
        expected_answer="I don't know",
        expected_session_id="",
        is_no_answer=True,
        sessions=[
            NoiseSession(
                session_id="s_spotify_playlist",
                content=(
                    "My Spotify playlist is called Summer Vibes and "
                    "I listen to it on commutes."
                ),
            ),
            NoiseSession(
                session_id="s_apple_news",
                content=(
                    "I read Apple's quarterly report; their market "
                    "cap is around 2 trillion dollars."
                ),
            ),
        ],
    ),
    # ─── 6. Abstention — duration question with no duration evidence ──────
    NoiseCase(
        case_id="commute_no_answer",
        category="abstention",
        question="How long was my commute yesterday?",
        expected_answer="I don't know",
        expected_session_id="",
        is_no_answer=True,
        sessions=[
            NoiseSession(
                session_id="s_breakfast",
                content=(
                    "I had cereal for breakfast yesterday — went with "
                    "granola and milk."
                ),
            ),
            NoiseSession(
                session_id="s_movie",
                content=(
                    "I watched a great documentary last night about "
                    "the history of the printing press."
                ),
            ),
        ],
    ),
]


# ─── Per-case result + pipeline outputs ───────────────────────────────────


@dataclasses.dataclass
class CaseResult:
    """Per-case evaluation outcome with all metrics."""

    case_id: str
    category: str
    expected_answer: str
    expected_session_id: str
    is_no_answer: bool

    actual_answer: str
    recall_at_k: float
    evidence_precision: float
    noise_token_ratio: float
    n_verified_candidates: int
    n_verified_from_correct_session: int
    selected_evidence_count: int
    excluded_noise_count: int
    correct: bool
    abstention_correct: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ─── Pipeline 1 — noisy (no verification, position-based pick) ────────────


def _normalised(text: str) -> str:
    """Lowercase + collapse whitespace for substring comparison."""
    return " ".join(text.lower().split())


def _answer_matches(actual: str, expected: str) -> bool:
    """Liberal substring match against the gold answer."""
    if not actual or not expected:
        return False
    a, e = _normalised(actual), _normalised(expected)
    return e in a or a in e


def run_noisy_pipeline(case: NoiseCase) -> CaseResult:
    """
    Naïve baseline: concatenate every session in the bundle, extract
    candidates with no grounding, pick the one whose source text
    appears EARLIEST in the concatenated bundle.

    This simulates a broad-context LLM with no subject/relation
    awareness — exactly the failure mode the pipeline lacked before
    the grounded-claim / verifier work.
    """
    q_type = classify(case.question)

    # Extract candidates per-session so source_session_id is preserved.
    all_candidates = []
    for s in case.sessions:
        all_candidates.extend(extract_candidates_from_text(
            s.content, source_session_id=s.session_id,
        ))
    relevant = filter_candidates_for_question(all_candidates, q_type)

    # Concatenate in bundle order so position reflects rank.
    bundle = " ".join(s.content for s in case.sessions)
    # Pick the candidate whose source text appears EARLIEST.
    if relevant:
        def position(c) -> int:  # type: ignore[no-untyped-def]
            anchor = c.source_span or c.value
            idx = bundle.find(anchor)
            return idx if idx >= 0 else len(bundle)
        relevant.sort(key=position)
        chosen = relevant[0]
        actual_answer = chosen.object_ or chosen.value
        chosen_sid = chosen.source_session_id
    else:
        actual_answer = ""
        chosen_sid = ""

    # Metrics
    expected_present = any(s.session_id == case.expected_session_id for s in case.sessions)
    recall_at_k = 1.0 if (expected_present or case.is_no_answer) else 0.0
    correct = (
        case.is_no_answer if not actual_answer
        else _answer_matches(actual_answer, case.expected_answer)
    )
    # Evidence precision: did we draw from the correct session?
    if case.is_no_answer:
        # On a no-answer case, every selected evidence is noise — so
        # evidence_precision is 0 unless we picked nothing.
        evidence_precision = 1.0 if not actual_answer else 0.0
    else:
        evidence_precision = (
            1.0 if chosen_sid == case.expected_session_id else 0.0
        )
    # Noise token ratio over the *passed-to-answerer* context — for the
    # noisy pipeline the whole bundle is passed through.
    total = max(1, case.total_tokens())
    noise_tokens = total - case.correct_tokens()
    noise_token_ratio = noise_tokens / total

    abstention_correct = (
        case.is_no_answer and (
            not actual_answer or _normalised(actual_answer).startswith("i don't know")
        )
    )

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_answer=case.expected_answer,
        expected_session_id=case.expected_session_id,
        is_no_answer=case.is_no_answer,
        actual_answer=actual_answer or "(empty)",
        recall_at_k=recall_at_k,
        evidence_precision=evidence_precision,
        noise_token_ratio=noise_token_ratio,
        n_verified_candidates=0,  # noisy pipeline does not verify
        n_verified_from_correct_session=0,
        selected_evidence_count=len(relevant),
        excluded_noise_count=0,
        correct=correct,
        abstention_correct=abstention_correct,
    )


# ─── Pipeline 2 — filtered (verifier + evidence packet) ───────────────────


def run_filtered_pipeline(case: NoiseCase) -> CaseResult:
    """
    The grounded pipeline: extract candidates with the grounded-claim
    fields, run the verifier, build the evidence packet, pick the
    top-confidence claim's object as the answer.

    Abstains (returns ``"I don't know"``) when the packet is empty
    — i.e. the verifier rejected every candidate.
    """
    q_type = classify(case.question)

    # Per-session extraction so source_session_id stays attached.
    all_candidates = []
    for s in case.sessions:
        all_candidates.extend(extract_candidates_from_text(
            s.content, source_session_id=s.session_id,
        ))

    results = verify_claims(case.question, all_candidates, question_type=q_type)
    verified = filter_passing(results)
    packet = build_evidence_packet(
        case.question, verified, all_candidates=all_candidates,
    )

    if packet.is_empty:
        actual_answer = "I don't know"
        chosen_sid = ""
    else:
        top_claim = packet.claims[0]
        actual_answer = top_claim.object_ or top_claim.claim
        chosen_sid = top_claim.source_session_id

    # Metrics
    expected_present = any(
        s.session_id == case.expected_session_id for s in case.sessions
    )
    recall_at_k = 1.0 if (expected_present or case.is_no_answer) else 0.0
    correct = (
        _answer_matches(actual_answer, case.expected_answer)
        if not case.is_no_answer
        else _normalised(actual_answer).startswith("i don't know")
    )
    if case.is_no_answer:
        evidence_precision = 1.0 if packet.is_empty else 0.0
    elif packet.is_empty:
        evidence_precision = 0.0
    else:
        evidence_precision = (
            1.0 if chosen_sid == case.expected_session_id else 0.0
        )
    # Noise token ratio over the packet-rendered prompt: count the
    # tokens contributed by claims sourced from non-expected sessions
    # against the total tokens of the packet's claim text.
    if packet.is_empty:
        noise_token_ratio = 0.0
    else:
        total_pkt_tokens = sum(
            len(c.claim.split()) + len(c.source_span.split())
            for c in packet.claims
        )
        noise_pkt_tokens = sum(
            len(c.claim.split()) + len(c.source_span.split())
            for c in packet.claims
            if c.source_session_id != case.expected_session_id
        )
        noise_token_ratio = noise_pkt_tokens / max(1, total_pkt_tokens)

    n_verified_from_correct = sum(
        1 for c in verified if c.source_session_id == case.expected_session_id
    )
    abstention_correct = (
        case.is_no_answer and _normalised(actual_answer).startswith("i don't know")
    )

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_answer=case.expected_answer,
        expected_session_id=case.expected_session_id,
        is_no_answer=case.is_no_answer,
        actual_answer=actual_answer,
        recall_at_k=recall_at_k,
        evidence_precision=evidence_precision,
        noise_token_ratio=noise_token_ratio,
        n_verified_candidates=len(verified),
        n_verified_from_correct_session=n_verified_from_correct,
        selected_evidence_count=packet.selected_evidence_count,
        excluded_noise_count=packet.excluded_noise_count,
        correct=correct,
        abstention_correct=abstention_correct,
    )


# ─── Aggregate metrics ─────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate per-case results into the headline metric dict."""
    n_total = len(results)
    answerable = [r for r in results if not r.is_no_answer]
    no_answer = [r for r in results if r.is_no_answer]

    verified_acc_values = [
        r.n_verified_from_correct_session / r.n_verified_candidates
        for r in answerable
        if r.n_verified_candidates > 0
    ]

    return {
        "n_cases": n_total,
        "n_answerable": len(answerable),
        "n_abstention": len(no_answer),
        "recall_at_k": _mean([r.recall_at_k for r in results]),
        "evidence_precision": _mean([r.evidence_precision for r in answerable]),
        "noise_token_ratio": _mean([r.noise_token_ratio for r in answerable]),
        "verified_candidate_accuracy": _mean(verified_acc_values),
        "final_answer_accuracy": (
            sum(1 for r in results if r.correct) / n_total if n_total else 0.0
        ),
        "abstention_correctness": (
            sum(1 for r in no_answer if r.abstention_correct) / len(no_answer)
            if no_answer else None
        ),
        "answerable_accuracy": (
            sum(1 for r in answerable if r.correct) / len(answerable)
            if answerable else 0.0
        ),
        "selected_evidence_avg": _mean([
            float(r.selected_evidence_count) for r in answerable
        ]),
        "excluded_noise_avg": _mean([
            float(r.excluded_noise_count) for r in answerable
        ]),
    }


# ─── Runner ───────────────────────────────────────────────────────────────


def run_benchmark(
    dataset: list[NoiseCase] | None = None,
) -> dict[str, Any]:
    """Run both pipelines over the dataset and return the metric block."""
    cases = dataset if dataset is not None else DATASET
    noisy_results = [run_noisy_pipeline(c) for c in cases]
    filtered_results = [run_filtered_pipeline(c) for c in cases]
    return {
        "benchmark": "precision_under_noise",
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "config": {"n_cases": len(cases)},
        "systems": [
            {
                "system": "noisy_baseline",
                "available": True,
                "metrics": aggregate(noisy_results),
                "per_case": [r.to_dict() for r in noisy_results],
                "note": (
                    "No grounded claims, no verifier — picks the earliest "
                    "answer-shaped candidate in the concatenated bundle. "
                    "Stand-in for a broad-context LLM with no subject/relation awareness."
                ),
            },
            {
                "system": "filtered_pipeline",
                "available": True,
                "metrics": aggregate(filtered_results),
                "per_case": [r.to_dict() for r in filtered_results],
                "note": (
                    "Grounded claims + claim verifier + minimal evidence packet. "
                    "Picks the top-confidence verified claim's object."
                ),
            },
        ],
    }


def _write_results(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["timestamp"]
    path = out_dir / f"precision_under_noise_{stamp}.json"
    latest = out_dir / "precision_under_noise_latest.json"
    blob = json.dumps(payload, indent=2)
    path.write_text(blob)
    latest.write_text(blob)
    return path


def _print_summary(payload: dict[str, Any]) -> None:
    print("=" * 64)
    print("Precision-under-noisy-recall benchmark")
    print(f"  cases: {payload['config']['n_cases']}")
    for sys in payload["systems"]:
        m = sys["metrics"]
        print(f"\n  ── {sys['system']} " + "─" * (50 - len(sys["system"])))
        print(f"    recall@k:                    {m['recall_at_k']:.1%}")
        print(f"    evidence_precision:          {m['evidence_precision']:.1%}")
        print(f"    noise_token_ratio:           {m['noise_token_ratio']:.1%}")
        print(f"    verified_candidate_accuracy: {m['verified_candidate_accuracy']:.1%}")
        print(f"    final_answer_accuracy:       {m['final_answer_accuracy']:.1%}")
        if m["abstention_correctness"] is not None:
            print(f"    abstention_correctness:      {m['abstention_correctness']:.1%}")
        print(f"    answerable_accuracy:         {m['answerable_accuracy']:.1%}")
    print("=" * 64)


def main() -> int:  # pragma: no cover — CLI wrapper
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    payload = run_benchmark()
    out_path = _write_results(payload, Path(__file__).parent / "results")
    _print_summary(payload)
    print(f"\nresults: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DATASET",
    "CaseResult",
    "NoiseCase",
    "NoiseSession",
    "aggregate",
    "run_benchmark",
    "run_filtered_pipeline",
    "run_noisy_pipeline",
]
