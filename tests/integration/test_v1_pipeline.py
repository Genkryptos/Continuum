"""
tests/integration/test_v1_pipeline.py
======================================
v1 regression gate — runs the **actual v1 pipeline** (direct retrieval
+ hybrid + session-aware + LTM supersession, judged) on a fixed,
committed 10-question LongMemEval-S subset and asserts judged accuracy
stays above a conservative floor.

Why direct mode, not the IterativeReasoner
------------------------------------------
v1 ships ``--reasoner direct``. The IterativeReasoner was built,
A/B-tested, and **cut** as net-negative (see
``findings/reasoning_loop_2026-06.md``). The regression gate locks the
pipeline that actually shipped, not the one that lost.

Why this skips in CI
--------------------
A judged-accuracy gate needs a live answerer LLM. Like the Postgres
integration tests (which skip when the DB is unreachable), this skips
unless ``OPENROUTER_API_KEY`` is set and the dataset is present — so
default CI is green without keys, and a keyed nightly/manual run
enforces the gate. Run it explicitly with:

    OPENROUTER_API_KEY=… python3.12 -m pytest \\
        tests/integration/test_v1_pipeline.py -v

Threshold
---------
The 10 fixture questions were all answered correctly by v1, across
categories. We assert ``>= 0.6`` (6/10): high enough that a real
regression (a retrieval break, a truncation bug, a judge misfire)
trips it, low enough that 1-2 nondeterministic model flips don't.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / "tests/integration/fixtures/v1_regression_ids.json"
_DATASET = _REPO / "evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json"

#: Conservative floor — fixture questions are all known-correct; this
#: catches real regressions, not single-question model noise.
_MIN_JUDGED_ACCURACY = 0.6


def _skip_reasons() -> list[str]:
    reasons = []
    if not os.environ.get("OPENROUTER_API_KEY"):
        reasons.append("OPENROUTER_API_KEY not set")
    if not _DATASET.exists():
        reasons.append(f"dataset missing at {_DATASET}")
    if not _FIXTURE.exists():
        reasons.append(f"fixture missing at {_FIXTURE}")
    return reasons


@pytest.mark.asyncio
async def test_v1_pipeline_holds_accuracy_floor(tmp_path: Path) -> None:
    reasons = _skip_reasons()
    if reasons:
        pytest.skip("; ".join(reasons))

    from evals.longmemeval.baseline import run_baseline
    from evals.longmemeval.bootstrap_ollama import (
        OpenRouterLLM,
        _Embedder,
        load_longmemeval_rows,
        make_adapter_factory,
    )
    from evals.longmemeval.judge import LLMJudgeScorer

    wanted = set(json.loads(_FIXTURE.read_text())["question_ids"])
    rows = [r for r in load_longmemeval_rows(_DATASET) if r.question_id in wanted]
    assert rows, "no fixture questions matched the dataset"

    answerer = OpenRouterLLM(model="openai/gpt-oss-120b", rpm=30)
    # Non-reasoning judge (reasoning models break the yes/no judge).
    judge = LLMJudgeScorer(
        llm=OpenRouterLLM(model="meta-llama/llama-3.3-70b-instruct", rpm=15),
        max_tokens=8,
    )

    async def row_scorer(row, answer):  # type: ignore[no-untyped-def]
        return bool(answer) and await judge.is_correct(row.question, row.expected_answer, answer)

    embedder = _Embedder()
    factory = make_adapter_factory(
        llm=answerer,
        embedder=embedder,
        top_k=80,
        answer_max_tokens=2048,
        retriever_kind="hybrid",
        session_aware_retrieval=True,
        session_top_k=12,
        turns_per_session=6,
        direct_answer=True,
        store_factory=_in_memory_ltm_store_factory(embedder),
        direct_max_context_chars=64000,
    )

    results = await run_baseline(
        dataset="longmemeval-s",
        adapter_factory=factory,
        dataset_loader=lambda _d: rows,
        output_dir=tmp_path,
        answerer="openai/gpt-oss-120b",
        row_scorer=row_scorer,
    )

    acc = results.accuracy
    n = len(results.rows)
    assert n == len(rows), f"expected {len(rows)} rows, got {n}"
    assert acc >= _MIN_JUDGED_ACCURACY, (
        f"v1 pipeline regressed: judged accuracy {acc:.1%} on the "
        f"{n}-question fixture is below the {_MIN_JUDGED_ACCURACY:.0%} floor. "
        "These questions were all correct at v1 — investigate retrieval, "
        "truncation caps, or the judge before lowering this threshold."
    )


def _in_memory_ltm_store_factory(embedder):  # type: ignore[no-untyped-def]
    """Build the --use-ltm store_factory (in-memory LTM, heuristic promoter)."""
    from continuum.stores.in_memory.ltm import InMemoryLTM
    from evals.longmemeval.continuum_ltm_store import ContinuumLTMHaystackStore

    def factory():  # type: ignore[no-untyped-def]
        return ContinuumLTMHaystackStore(
            ltm=InMemoryLTM(),
            embedder=embedder,
            promoter=None,
            fact_extractor=None,
            llm_available=False,
            ltm_backend_label="in_memory",
        )

    return factory
