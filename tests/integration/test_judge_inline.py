"""
tests/integration/test_judge_inline.py
======================================
Smoke that the inline LLM judge wiring in ``baseline.run_baseline`` runs
both scorers and persists both numbers to the result JSON.

This is hermetic: a fake adapter returns scripted answers and a fake
judge returns scripted verdicts. The point is the **wiring**, not any
particular model output. The actual LLM judge implementation
(:class:`evals.longmemeval.judge.LLMJudgeScorer`) is covered in its own
unit tests.

A 25-row run exercises every meaningful combination:

* substring agrees with judge
* substring is wrong, judge is right (paraphrase recovery — the whole
  reason the judge was promoted)
* substring is right, judge says no (model spat out a related but wrong
  fact)
* adapter explodes and the judge is skipped (verdict stays ``None``)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, TokenBudget
from evals.longmemeval.baseline import EvalRow, run_baseline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self, *, answer: str, retrieved: list[str], raise_on_answer: bool = False):
        self.answer = answer
        self.raise_on_answer = raise_on_answer
        self.last_ctx = _ctx(retrieved)

    async def process_conversation(self, messages: list[dict[str, Any]]) -> None:
        return None

    async def answer_question(self, question: str) -> str:
        if self.raise_on_answer:
            raise RuntimeError("boom")
        return self.answer


class _FakeJudge:
    """Deterministic judge keyed on ``(question_id, expected, actual)``.

    Tracks call count so the test can assert the judge fired inline for
    every answerable row (and was skipped for the error row).
    """

    def __init__(self, verdicts: dict[str, bool]):
        self._verdicts = verdicts
        self.calls = 0

    async def is_correct(self, question: str, expected: str, actual: str) -> bool:
        self.calls += 1
        return self._verdicts.get(actual, False)


def _ctx(session_ids: list[str]) -> ContextBundle:
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


def _eval_row(qid: str, expected: str) -> EvalRow:
    return EvalRow(
        question_id=qid,
        question=f"Q for {qid}?",
        expected_answer=expected,
        messages=[{"role": "user", "content": "context"}],
        answer_session_ids=[f"s_{qid}"],
    )


def _build_25() -> tuple[
    list[EvalRow],
    dict[str, tuple[str, list[str], bool]],
    dict[str, bool],
]:
    """
    Build 25 rows + a script of adapter answers + a script of judge verdicts.

    Layout:
    * rows 1-10  : agree (substring True, judge True)
    * rows 11-17 : substring wrong, judge right (paraphrase)
    * rows 18-22 : substring right, judge wrong (lucky overlap)
    * rows 23-24 : both wrong
    * row 25     : adapter error → judge skipped
    """
    rows: list[EvalRow] = []
    answers: dict[str, tuple[str, list[str], bool]] = {}
    verdicts: dict[str, bool] = {}

    def add(qid: str, expected: str, answer: str, *, judge: bool, raise_: bool = False):
        rows.append(_eval_row(qid, expected))
        answers[qid] = (answer, [f"s_{qid}"], raise_)
        verdicts[answer] = judge

    # 1-10 : both right
    for i in range(1, 11):
        add(f"q{i:02d}", f"Paris-{i}", f"Paris-{i} is the capital.", judge=True)

    # 11-17 : substring miss (no token overlap with expected), judge catches it
    for i in range(11, 18):
        add(
            f"q{i:02d}",
            f"target-{i}",
            f"completely-different-wording-{i}",
            judge=True,
        )

    # 18-22 : substring hits, judge rejects (e.g. partial-but-wrong)
    for i in range(18, 23):
        add(
            f"q{i:02d}",
            f"answer-{i}",
            f"answer-{i} but actually wrong context",
            judge=False,
        )

    # 23-24 : both wrong
    for i in range(23, 25):
        add(f"q{i:02d}", f"truth-{i}", f"nope-{i}", judge=False)

    # 25 : adapter raises, judge should not be called for this row
    add("q25", "anything", "", judge=False, raise_=True)

    assert len(rows) == 25
    return rows, answers, verdicts


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_runs_inline_and_persists_both_metrics(tmp_path: Path) -> None:
    rows, answers, verdicts = _build_25()
    judge = _FakeJudge(verdicts)

    def factory() -> _FakeAdapter:
        # Walk `rows` in order; this matches the order run_baseline iterates.
        # We track which row we're on via a counter on the function object.
        idx = factory.calls  # type: ignore[attr-defined]
        factory.calls += 1  # type: ignore[attr-defined]
        ans, retrieved, raise_ = answers[rows[idx].question_id]
        return _FakeAdapter(answer=ans, retrieved=retrieved, raise_on_answer=raise_)

    factory.calls = 0  # type: ignore[attr-defined]

    def loader(_dataset: str):
        return iter(rows)

    out_path = tmp_path / "smoke_judge.json"
    results = await run_baseline(
        dataset="longmemeval-s",
        adapter_factory=factory,
        dataset_loader=loader,
        judge=judge,
        output_file=out_path,
    )

    # ── wiring assertions ──────────────────────────────────────────────────
    # 24 non-error rows had a non-empty answer → 24 judge calls. The error
    # row produces "" so the judge is skipped (judge_correct stays None).
    assert judge.calls == 24
    assert len(results.rows) == 25

    error_row = next(r for r in results.rows if r.question_id == "q25")
    assert error_row.judge_correct is None
    assert error_row.substring_correct is False

    # 24 rows that should carry a judge verdict.
    judged = [r for r in results.rows if r.judge_correct is not None]
    assert len(judged) == 24

    # Hand-counted from _build_25:
    # judge True: rows 1-17 (17 rows)   substring True: 1-10, 18-22 (15 rows)
    expected_judged_correct = 17
    expected_substring_correct = 15
    assert sum(1 for r in judged if r.judge_correct) == expected_judged_correct
    assert sum(1 for r in results.rows if r.substring_correct) == expected_substring_correct

    # ── persisted JSON carries both keys ───────────────────────────────────
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    metrics = payload["metrics"]
    assert "substring_accuracy" in metrics
    assert "judged_accuracy" in metrics
    assert "judged_row_count" in metrics
    assert metrics["judged_row_count"] == 24
    # 17/24 judge-correct vs 15/25 substring-correct (substring counts the
    # error row as wrong, so denominator is the full 25).
    assert metrics["judged_accuracy"] == pytest.approx(17 / 24)
    assert metrics["substring_accuracy"] == pytest.approx(15 / 25)
    # Primary accuracy mirrors the judge whenever it ran.
    assert metrics["accuracy"] == pytest.approx(17 / 25)

    # Per-row records carry the dual-track fields too.
    sample = next(r for r in payload["rows"] if r["question_id"] == "q11")
    assert sample["substring_correct"] is False
    assert sample["judge_correct"] is True
    assert sample["correct"] is True  # primary mirrors judge


@pytest.mark.asyncio
async def test_no_judge_falls_back_to_substring(tmp_path: Path) -> None:
    """Sanity: when ``judge`` is None, the JSON still has both keys —
    ``judged_accuracy`` just comes out as ``null``."""
    rows = [_eval_row("q1", "Paris")]

    def factory() -> _FakeAdapter:
        return _FakeAdapter(answer="The answer is Paris.", retrieved=["s_q1"])

    out_path = tmp_path / "no_judge.json"
    await run_baseline(
        dataset="longmemeval-s",
        adapter_factory=factory,
        dataset_loader=lambda _d: iter(rows),
        output_file=out_path,
    )
    payload = json.loads(out_path.read_text())
    assert payload["metrics"]["substring_accuracy"] == 1.0
    assert payload["metrics"]["judged_accuracy"] is None
    assert payload["metrics"]["judged_row_count"] == 0
