"""
tests/unit/evals/test_longmemeval_baseline.py
=============================================
Unit tests for the LongMemEval baseline runner. All network /
LongMemEval / LLM calls are stubbed via injected fakes; tests run
offline in well under a second.

Coverage
--------
* ``default_scorer`` substring + token-overlap semantics
* ``run_baseline`` happy path: per-row metrics + aggregate accuracy /
  recall / latency p95 / cost
* Failure categorisation: llm_error, missing_fact, wrong_retrieval, other
* JSON + failures-CSV are written to the requested directory
* Adapter exception is captured (not raised) and classified as llm_error
* ``--limit`` short-circuits the loop
* `last_ctx` is used to extract retrieved session ids
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from evals.longmemeval.baseline import (
    BaselineResults,
    EvalRow,
    default_scorer,
    run_baseline,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """
    Pre-programmed adapter: returns scripted answers + simulates a
    retrieved context bundle so the baseline runner has session ids to
    score recall against.
    """

    def __init__(
        self,
        *,
        answer: str = "the answer",
        retrieved_session_ids: list[str] | None = None,
        raise_on_answer: bool = False,
    ) -> None:
        self.answer = answer
        self.raise_on_answer = raise_on_answer
        self.last_ctx: ContextBundle | None = _make_ctx(
            retrieved_session_ids or []
        )
        self.process_calls = 0
        self.answer_calls = 0

    async def process_conversation(self, messages: list[dict[str, Any]]) -> None:
        self.process_calls += 1

    async def answer_question(self, question: str) -> str:
        self.answer_calls += 1
        if self.raise_on_answer:
            raise RuntimeError("adapter exploded")
        return self.answer


def _make_ctx(session_ids: list[str]) -> ContextBundle:
    items = [
        MemoryItem(
            id=str(uuid.uuid4()),
            content=f"row from {sid}",
            tier=MemoryTier.LTM,
            metadata={"session_id": sid},
        )
        for sid in session_ids
    ]
    return ContextBundle(
        items=items,
        messages=[],
        tokens_used=0,
        budget=TokenBudget(
            total=4000, stm_reserved=500, mtm_reserved=500,
            ltm_reserved=2000, response_reserved=100,
        ),
    )


def _row(
    qid: str,
    expected: str,
    expected_sids: list[str] | None = None,
) -> EvalRow:
    return EvalRow(
        question_id=qid,
        question=f"What about {qid}?",
        expected_answer=expected,
        messages=[{"role": "user", "content": "context goes here"}],
        answer_session_ids=expected_sids or [],
    )


# ---------------------------------------------------------------------------
# default_scorer
# ---------------------------------------------------------------------------


def test_default_scorer_substring_match() -> None:
    assert default_scorer("The answer is Paris.", "Paris")
    assert default_scorer("paris", "Paris")  # case-insensitive
    assert default_scorer("Paris", "the answer is Paris")  # either direction


def test_default_scorer_token_overlap() -> None:
    # No substring containment but ≥ 70 % overlap on the expected tokens.
    assert default_scorer(
        "Bob Smith works at Acme Corp.",
        "Bob Smith Acme",
    )


def test_default_scorer_rejects_unrelated_answers() -> None:
    assert not default_scorer("I don't know", "Paris")
    assert not default_scorer("", "Paris")


def test_default_scorer_handles_empty_expected() -> None:
    assert not default_scorer("anything", "")


# ---------------------------------------------------------------------------
# run_baseline — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_baseline_happy_path(tmp_path: Path) -> None:
    rows = [
        _row("q1", "Paris", expected_sids=["s1"]),
        _row("q2", "Tokyo", expected_sids=["s2"]),
    ]

    # First row answer correctly + retrieves s1; second is wrong + s2 was
    # actually retrieved → wrong_retrieval.
    answers = {"q1": "Paris is the capital.", "q2": "Berlin"}
    retrieved = {"q1": ["s1"], "q2": ["s2", "s3"]}

    adapters: list[_FakeAdapter] = []

    def factory() -> _FakeAdapter:
        # rows iterate in order; advance via len.
        row = rows[len(adapters)]
        a = _FakeAdapter(
            answer=answers[row.question_id],
            retrieved_session_ids=retrieved[row.question_id],
        )
        adapters.append(a)
        return a

    def loader(_name: str) -> list[EvalRow]:
        return rows

    results = await run_baseline(
        dataset="longmemeval-s",
        adapter_factory=factory,
        dataset_loader=loader,
        output_dir=tmp_path,
        answerer="gpt-4o-mini",
        now=lambda: dt.datetime(2026, 5, 20, tzinfo=dt.UTC),
        token_counter=lambda s: len(s.split()),
    )

    # Each row got a fresh adapter + one process/answer pair.
    assert len(adapters) == 2
    assert all(a.process_calls == 1 and a.answer_calls == 1 for a in adapters)

    # Metrics
    assert results.accuracy == 0.5
    # Recall: q1 has 1/1, q2 has 1/1 → mean 1.0
    assert results.recall == 1.0
    assert results.latency_p50 >= 0.0
    assert results.latency_p95 >= results.latency_p50

    # Files
    json_path = tmp_path / "baseline_2026-05-20.json"
    csv_path = tmp_path / "baseline_failures.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["metrics"]["n_questions"] == 2
    assert payload["metrics"]["accuracy"] == 0.5
    assert payload["dataset"] == "longmemeval-s"

    # Failure CSV: one wrong row (q2), classified as wrong_retrieval.
    with csv_path.open() as fh:
        reader = list(csv.DictReader(fh))
    assert len(reader) == 1
    assert reader[0]["question_id"] == "q2"
    assert reader[0]["failure_category"] == "wrong_retrieval"


# ---------------------------------------------------------------------------
# Failure categorisation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_fact_when_no_expected_ids_retrieved(
    tmp_path: Path,
) -> None:
    rows = [_row("q1", "Paris", expected_sids=["s1", "s2"])]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(
            answer="Berlin",
            retrieved_session_ids=["s9", "s8"],  # none of the expected
        )

    results = await run_baseline(
        dataset="d",
        adapter_factory=factory,
        dataset_loader=lambda _n: rows,
        output_dir=tmp_path,
    )
    [r] = results.rows
    assert not r.correct
    assert r.failure_category == "missing_fact"


@pytest.mark.asyncio
async def test_llm_error_when_answer_is_error_string(tmp_path: Path) -> None:
    rows = [_row("q1", "Paris", expected_sids=["s1"])]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(
            answer="[error: model down]",
            retrieved_session_ids=["s1"],
        )

    results = await run_baseline(
        dataset="d",
        adapter_factory=factory,
        dataset_loader=lambda _n: rows,
        output_dir=tmp_path,
    )
    [r] = results.rows
    assert not r.correct
    assert r.failure_category == "llm_error"


@pytest.mark.asyncio
async def test_adapter_exception_captured_as_llm_error(tmp_path: Path) -> None:
    rows = [_row("q1", "Paris", expected_sids=["s1"])]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(
            answer="anything",
            retrieved_session_ids=["s1"],
            raise_on_answer=True,
        )

    results = await run_baseline(
        dataset="d",
        adapter_factory=factory,
        dataset_loader=lambda _n: rows,
        output_dir=tmp_path,
    )
    [r] = results.rows
    assert not r.correct
    assert r.failure_category == "llm_error"
    assert r.error is not None and "adapter exploded" in r.error


@pytest.mark.asyncio
async def test_classifies_other_when_no_expected_ids(tmp_path: Path) -> None:
    rows = [_row("q1", "Paris", expected_sids=[])]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(answer="Berlin", retrieved_session_ids=["s1"])

    results = await run_baseline(
        dataset="d",
        adapter_factory=factory,
        dataset_loader=lambda _n: rows,
        output_dir=tmp_path,
    )
    [r] = results.rows
    assert r.failure_category == "other"


# ---------------------------------------------------------------------------
# Cost & limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_uses_price_table(tmp_path: Path) -> None:
    rows = [_row(f"q{i}", "yes") for i in range(3)]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(answer="yes")  # 1 token in the answer

    results = await run_baseline(
        dataset="d",
        adapter_factory=factory,
        dataset_loader=lambda _n: rows,
        output_dir=tmp_path,
        answerer="gpt-4o-mini",
        token_counter=lambda s: 1000 if s else 0,
        price_per_1k_out={"gpt-4o-mini": 0.5},
    )
    # 3 rows × 1000 output tokens × $0.5 / 1K = $1.50
    assert results.total_cost_usd == pytest.approx(1.50)


@pytest.mark.asyncio
async def test_limit_caps_iteration(tmp_path: Path) -> None:
    rows = [_row(f"q{i}", "yes") for i in range(10)]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(answer="yes")

    results = await run_baseline(
        dataset="d",
        adapter_factory=factory,
        dataset_loader=lambda _n: rows,
        output_dir=tmp_path,
        limit=3,
    )
    assert len(results.rows) == 3


# ---------------------------------------------------------------------------
# BaselineResults aggregates
# ---------------------------------------------------------------------------


def test_baseline_results_aggregates_empty() -> None:
    results = BaselineResults(
        dataset="d", answerer="a", rows=[],
        started_at="", finished_at="",
    )
    # Aggregations don't divide by zero on empty input.
    assert results.accuracy == 0.0
    assert results.avg_tokens == 0.0
    assert results.recall == 0.0
    assert results.latency_p50 == 0.0
    assert results.latency_p95 == 0.0
    assert results.total_cost_usd == 0.0


def test_baseline_results_breakdown_in_payload() -> None:
    from evals.longmemeval.baseline import RowResult

    rows = [
        RowResult(
            question_id="a", correct=False, latency_ms=10.0,
            answer_tokens=5, cost_usd=0.01,
            retrieved_session_ids=[], expected_session_ids=["s1"],
            answer="x", expected_answer="y",
            failure_category="missing_fact",
        ),
        RowResult(
            question_id="b", correct=False, latency_ms=20.0,
            answer_tokens=5, cost_usd=0.01,
            retrieved_session_ids=["s1"], expected_session_ids=["s1"],
            answer="x", expected_answer="y",
            failure_category="wrong_retrieval",
        ),
        RowResult(
            question_id="c", correct=True, latency_ms=30.0,
            answer_tokens=5, cost_usd=0.01,
            retrieved_session_ids=["s1"], expected_session_ids=["s1"],
            answer="y", expected_answer="y",
            failure_category=None,
        ),
    ]
    res = BaselineResults(
        dataset="d", answerer="a", rows=rows,
        started_at="", finished_at="",
    )
    payload = res.to_dict()
    assert payload["metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert payload["metrics"]["failure_breakdown"]["missing_fact"] == 1
    assert payload["metrics"]["failure_breakdown"]["wrong_retrieval"] == 1


def test_eval_row_and_row_result_are_dataclasses() -> None:
    # Asdict makes JSON serialisation straightforward — guard the contract.
    row = _row("q", "e")
    assert dataclasses.is_dataclass(row)
    assert dataclasses.asdict(row)["question_id"] == "q"
