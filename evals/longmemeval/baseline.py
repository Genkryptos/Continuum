"""
evals/longmemeval/baseline.py
=============================
Run LongMemEval against Continuum **without** the Optimizer chain to
establish the baseline number every subsequent optimisation gets
compared to.

Pipeline (per question row)
---------------------------
1. Reset session / load the row's session history into Continuum via
   ``adapter.process_conversation``.
2. Time ``adapter.answer_question(question)``.
3. Score the answer (default: case-insensitive substring overlap; an
   ``LlmJudge``-style callable can be injected for nuanced scoring).
4. Estimate cost from the answer's token counts × per-1K price.
5. Categorise failures:
   * ``llm_error`` — adapter returned ``"[error: …]"`` (LLM call blew up)
   * ``missing_fact`` — none of the ground-truth session ids appear in
     the retrieved context
   * ``wrong_retrieval`` — ground-truth was retrieved but answer is wrong
   * ``other`` — unknown classification (rare; flagged for manual review)

Outputs
-------
* ``results/baseline_YYYY-MM-DD.json`` — aggregate metrics + per-row
  records (latency, tokens, cost, retrieved_session_ids, correct)
* ``results/baseline_failures.csv`` — one row per failed question with
  category, retrieved_ids, expected_session_ids, and the model output

Usage
-----
.. code-block:: bash

    python -m evals.longmemeval.baseline \\
        --dataset longmemeval-s --output results/

The script is *library-first*: the CLI just bridges argparse onto
:func:`run_baseline`, which the tests drive directly with fakes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import datetime as dt
import json
import logging
import re
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalRow:
    """
    One LongMemEval row, normalised to a shape the baseline runner can
    chew on without depending on the upstream package.

    Attributes
    ----------
    question_id:
        Stable identifier (``EvalRow.id`` upstream).
    question:
        Natural-language query asked of the system.
    expected_answer:
        Ground-truth string. The scorer compares against this.
    messages:
        Flattened OpenAI-format message list spanning every "haystack"
        session in chronological order. ``adapter.process_conversation``
        ingests this directly.
    answer_session_ids:
        Identifiers of the sessions that actually contain the answer —
        used to compute retrieval recall and classify failures.
    """

    question_id: str
    question: str
    expected_answer: str
    messages: list[dict[str, Any]]
    answer_session_ids: list[str] = dataclasses.field(default_factory=list)
    question_type: str = "unknown"
    user_id: str | None = None


@dataclasses.dataclass
class RowResult:
    """Per-row evaluation outcome."""

    question_id: str
    correct: bool
    latency_ms: float
    answer_tokens: int
    cost_usd: float
    retrieved_session_ids: list[str]
    expected_session_ids: list[str]
    answer: str
    expected_answer: str
    failure_category: str | None  # None when correct
    expected_session_count: int = 0
    retrieved_expected_session_count: int = 0
    partial_recall: float = 0.0
    missing_expected_session_ids: list[str] = dataclasses.field(default_factory=list)
    error: str | None = None  # populated on adapter exceptions
    question_type: str = "unknown"

    # Optional optimizer telemetry (populated only when the adapter
    # exposes ``last_optimizer_stats``). All counts are tokens.
    pre_opt_total_tokens: int = 0
    post_opt_total_tokens: int = 0
    pre_opt_stm_tokens: int = 0
    pre_opt_mtm_tokens: int = 0
    pre_opt_ltm_tokens: int = 0
    post_opt_stm_tokens: int = 0
    post_opt_mtm_tokens: int = 0
    post_opt_ltm_tokens: int = 0
    retrieved_count: int = 0
    retrieval_ms: float = 0.0
    optimizer_ms: float = 0.0
    strategy_savings: dict[str, int] = dataclasses.field(default_factory=dict)
    decomposition_stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Per-row LLM-call telemetry populated by `_DecomposedAnsweringAdapter`
    #: (real prompt/completion token counts, cost, validator state,
    #: regeneration attempts, abstain reason, wiki retrieval counts).
    #: The fix for `retrieved_count = 0` and `retrieval_ms = 0` despite
    #: non-empty `retrieved_session_ids` lives in this field.
    telemetry: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Dual-track verdicts so analyses can report both metrics from one run.
    #: ``substring_correct`` is the legacy cheap matcher; ``judge_correct``
    #: is the LLM judge's verdict (``None`` when the judge wasn't run or its
    #: call failed). The "primary" ``correct`` above mirrors the judge when
    #: it ran, falling back to substring otherwise.
    substring_correct: bool = False
    judge_correct: bool | None = None


@dataclasses.dataclass
class BaselineResults:
    """Aggregate metrics + per-row records — the JSON payload."""

    dataset: str
    answerer: str
    rows: list[RowResult]
    started_at: str
    finished_at: str

    @property
    def accuracy(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.correct) / len(self.rows)

    @property
    def substring_accuracy(self) -> float:
        """Accuracy under the cheap substring scorer (kept for reference)."""
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.substring_correct) / len(self.rows)

    @property
    def judged_accuracy(self) -> float | None:
        """Accuracy under the LLM judge. ``None`` if no row was judged."""
        judged = [r for r in self.rows if r.judge_correct is not None]
        if not judged:
            return None
        return sum(1 for r in judged if r.judge_correct) / len(judged)

    @property
    def judged_row_count(self) -> int:
        return sum(1 for r in self.rows if r.judge_correct is not None)

    @property
    def avg_tokens(self) -> float:
        if not self.rows:
            return 0.0
        return statistics.fmean(r.answer_tokens for r in self.rows)

    @property
    def recall(self) -> float:
        """Average per-row recall of ground-truth session ids."""
        if not self.rows:
            return 0.0
        per_row = []
        for r in self.rows:
            if not r.expected_session_ids:
                continue
            hit = sum(
                1 for sid in r.expected_session_ids
                if sid in r.retrieved_session_ids
            )
            per_row.append(hit / len(r.expected_session_ids))
        return statistics.fmean(per_row) if per_row else 0.0

    @property
    def latency_p50(self) -> float:
        if not self.rows:
            return 0.0
        return float(statistics.median(r.latency_ms for r in self.rows))

    @property
    def latency_p95(self) -> float:
        if not self.rows:
            return 0.0
        return _percentile([r.latency_ms for r in self.rows], 0.95)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.rows)

    @property
    def avg_context_tokens_pre(self) -> float:
        vals = [r.pre_opt_total_tokens for r in self.rows if r.pre_opt_total_tokens]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def avg_context_tokens_post(self) -> float:
        vals = [r.post_opt_total_tokens for r in self.rows if r.post_opt_total_tokens]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def avg_optimizer_ms(self) -> float:
        vals = [r.optimizer_ms for r in self.rows if r.optimizer_ms]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def strategy_savings_total(self) -> dict[str, int]:
        """Sum of token deltas attributed to each strategy across all rows."""
        out: dict[str, int] = {}
        for r in self.rows:
            for k, v in r.strategy_savings.items():
                out[k] = out.get(k, 0) + int(v)
        return out

    @property
    def by_question_type(self) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[RowResult]] = {}
        for row in self.rows:
            grouped.setdefault(row.question_type or "unknown", []).append(row)
        out: dict[str, dict[str, float]] = {}
        for question_type, rows in sorted(grouped.items()):
            correct = sum(1 for r in rows if r.correct)
            recall_vals: list[float] = []
            for r in rows:
                if not r.expected_session_ids:
                    continue
                hit = sum(
                    1 for sid in r.expected_session_ids
                    if sid in r.retrieved_session_ids
                )
                recall_vals.append(hit / len(r.expected_session_ids))
            out[question_type] = {
                "n_questions": len(rows),
                "accuracy": correct / len(rows) if rows else 0.0,
                "recall": statistics.fmean(recall_vals) if recall_vals else 0.0,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "answerer": self.answerer,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": {
                "n_questions": len(self.rows),
                "accuracy": self.accuracy,
                # Dual scorers — substring is kept around as a reference
                # column; judged_accuracy is the primary metric whenever a
                # judge was wired in (``None`` if no row was judged).
                "substring_accuracy": self.substring_accuracy,
                "judged_accuracy": self.judged_accuracy,
                "judged_row_count": self.judged_row_count,
                "avg_tokens": self.avg_tokens,
                "recall": self.recall,
                "latency_p50_ms": self.latency_p50,
                "latency_p95_ms": self.latency_p95,
                "total_cost_usd": self.total_cost_usd,
                "failure_breakdown": _failure_breakdown(self.rows),
                # Optimizer telemetry — zeroes when the optimizer is off.
                "avg_context_tokens_pre_opt": self.avg_context_tokens_pre,
                "avg_context_tokens_post_opt": self.avg_context_tokens_post,
                "context_token_reduction_pct": (
                    100.0 * (1 - self.avg_context_tokens_post
                             / self.avg_context_tokens_pre)
                    if self.avg_context_tokens_pre else 0.0
                ),
                "avg_optimizer_ms": self.avg_optimizer_ms,
                "strategy_savings_total": self.strategy_savings_total,
                "by_question_type": self.by_question_type,
            },
            "rows": [dataclasses.asdict(r) for r in self.rows],
        }


# Adapter contract — the baseline only depends on this shape, not on
# ``ContinuumAdapter`` directly, so tests can swap in fakes trivially.
AdapterFactory = Callable[[], Any]
DatasetLoader = Callable[[str], Iterable[EvalRow]]
Scorer = Callable[[str, str], bool]
RowScorer = Callable[[EvalRow, str], bool | Awaitable[bool]]






# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_scorer(answer: str, expected: str) -> bool:
    """
    Cheap substring / token-overlap match.

    LongMemEval's official scorer uses an LLM judge; we ship a free
    fallback that is "good enough" for baseline trends and lets tests
    run hermetically. Callers wire a real judge via the ``scorer``
    parameter.
    """
    if expected is None or expected == "" or not answer:
        return False
    # LongMemEval rows include ints / numeric-string answers for counting
    # and date questions. Coerce defensively so the scorer never crashes
    # mid-run on a non-string ground-truth.
    a = _normalise_answer_text(str(answer).strip().lower())
    e = _normalise_answer_text(str(expected).strip().lower())
    if not a or not e:
        return False
    if e in a or a in e:
        return True
    # Token-set overlap ≥ 70 % is a permissive sanity check.
    a_tokens = set(_tokenise(a))
    e_tokens = set(_tokenise(e))
    if not e_tokens:
        return False
    overlap = len(a_tokens & e_tokens) / len(e_tokens)
    return overlap >= 0.7


def _tokenise(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


_NUMBER_PHRASES = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "one and a half": "1.5",
    "two and a half": "2.5",
    "three and a half": "3.5",
    "four and a half": "4.5",
    "five and a half": "5.5",
    "six and a half": "6.5",
    "seven and a half": "7.5",
    "eight and a half": "8.5",
    "nine and a half": "9.5",
    "ten and a half": "10.5",
}


def _normalise_answer_text(text: str) -> str:
    out = text.lower()
    for phrase, value in sorted(
        _NUMBER_PHRASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        out = re.sub(rf"\b{re.escape(phrase)}\b", value, out)
    out = re.sub(r"(?<!\d)[^\w\s.]+|[.](?!\d)", " ", out)
    return re.sub(r"\s+", " ", out).strip()


#: Per-model USD price per 1 K **output** tokens (input cost ignored —
#: baseline answers are short and prompts dwarf the output cost
#: in the eventual cost optimisation).
_PRICE_PER_1K_OUT: dict[str, float] = {
    "gpt-4o-mini": 0.00060,
    "gpt-4o": 0.01000,
    "claude-3-haiku": 0.00125,
    "claude-3-5-haiku": 0.00400,
    "claude-3-5-sonnet": 0.01500,
    # Groq (per 1K output tokens; ÷1000 from their per-1M sheet)
    "llama-3.3-70b-versatile": 0.00079,
    "llama-3.1-8b-instant": 0.00008,
    "gemma2-9b-it": 0.00020,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_baseline(
    *,
    dataset: str,
    adapter_factory: AdapterFactory,
    dataset_loader: DatasetLoader,
    answerer: str = "gpt-4o-mini",
    output_dir: Path | str | None = None,
    output_file: Path | str | None = None,
    scorer: Scorer | None = None,
    row_scorer: RowScorer | None = None,
    judge: Any | None = None,
    limit: int | None = None,
    price_per_1k_out: dict[str, float] | None = None,
    now: Callable[[], dt.datetime] | None = None,
    token_counter: Callable[[str], int] | None = None,
    trace_writer: Any | None = None,
    run_id: str | None = None,
) -> BaselineResults:
    """
    Drive the baseline evaluation. Pure-async; never raises into the
    caller — per-row errors are captured into the ``RowResult``.

    Parameters
    ----------
    dataset:
        Dataset name (``"longmemeval-s"`` / ``"longmemeval-l"``).
    adapter_factory:
        Callable returning a *fresh* adapter for each row — Continuum
        baselines must not leak memory state between questions.
    dataset_loader:
        Callable taking a dataset name and yielding :class:`EvalRow`.
    answerer:
        Model name forwarded to logs + cost lookup. Defaults to
        ``"gpt-4o-mini"`` per the spec.
    output_dir:
        Directory to write the JSON + failures CSV. Defaults to
        ``./results``.
    scorer:
        Optional answer-vs-expected scorer. Defaults to
        :func:`default_scorer`. This is the *substring reference*
        scorer; its verdict is always recorded as
        ``RowResult.substring_correct``.
    judge:
        Optional LLM judge (anything exposing ``async is_correct(question,
        expected, actual) -> bool`` — typically
        :class:`evals.longmemeval.judge.LLMJudgeScorer`). When provided,
        the judge runs **inline** after every answer, its verdict is
        recorded as ``RowResult.judge_correct``, and the row's primary
        ``correct`` flag mirrors the judge.
    output_file:
        Explicit JSON path to write to. Overrides the default
        ``<output_dir>/baseline_<YYYY-MM-DD>.json`` naming.
    limit:
        Cap the number of rows processed (handy for smoke tests).
    price_per_1k_out:
        Override the cost table.
    now / token_counter:
        Injectable for deterministic tests.
    """
    scorer = scorer or default_scorer
    prices = price_per_1k_out or _PRICE_PER_1K_OUT
    out_root = Path(output_dir) if output_dir else Path("results")
    clock = now or _utc_now
    counter = token_counter or _default_token_counter
    effective_run_id = run_id or _utc_now().strftime("%Y%m%dT%H%M%S")

    started = clock()
    rows: list[RowResult] = []

    raw_rows = dataset_loader(dataset)
    for idx, row in enumerate(raw_rows):
        if limit is not None and idx >= limit:
            break
        result = await _run_one(
            row=row,
            adapter=adapter_factory(),
            scorer=scorer,
            row_scorer=row_scorer,
            judge=judge,
            answerer=answerer,
            prices=prices,
            count_tokens=counter,
        )
        rows.append(result)
        log.info(
            "row %d/%s id=%s correct=%s latency=%.0fms",
            idx + 1, "?" if limit is None else limit,
            row.question_id, result.correct, result.latency_ms,
        )
        # Per-row JSONL trace — `trace_writer` may be a TraceWriter or any
        # object that exposes `.write(row_dict)`. We avoid importing the
        # trace module here to keep baseline.py free of new top-level deps.
        if trace_writer is not None:
            try:
                from evals.longmemeval.trace import trace_row_from_result
                trace_writer.write(trace_row_from_result(
                    result,
                    run_id=effective_run_id,
                    dataset=dataset,
                    question_text=row.question,
                ))
            except Exception:  # pragma: no cover — never fail the run on trace I/O
                log.exception("trace writer failed for row %s", row.question_id)

    finished = clock()
    results = BaselineResults(
        dataset=dataset,
        answerer=answerer,
        rows=rows,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
    )

    _persist(results, out_root, clock, output_file=output_file)
    _print_summary(results)
    return results


async def _run_one(
    *,
    row: EvalRow,
    adapter: Any,
    scorer: Scorer,
    answerer: str,
    prices: dict[str, float],
    count_tokens: Callable[[str], int],
    row_scorer: RowScorer | None = None,
    judge: Any | None = None,
) -> RowResult:
    """Run a single question. All exceptions captured into the result."""
    error: str | None = None
    answer = ""
    retrieved: list[str] = []
    t0 = time.perf_counter()
    try:
        await _maybe_await(adapter.process_conversation(row.messages))
        # Pass dataset hints to the adapter so its router can
        # disambiguate categories (assistant-memory, preference,
        # temporal, multi-session, knowledge-update) that the
        # question's surface form alone can't reliably distinguish.
        # Set via attributes rather than a kwarg so the adapter
        # protocol stays single-arg.
        if hasattr(adapter, "dataset_question_type"):
            adapter.dataset_question_type = row.question_type
        if hasattr(adapter, "dataset_is_multi_session"):
            adapter.dataset_is_multi_session = (
                len(row.answer_session_ids or []) > 1
            )
        answer = await _maybe_await(adapter.answer_question(row.question))
    except Exception as exc:
        log.exception("row %s failed", row.question_id)
        error = repr(exc)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Retrieved session ids come from the adapter via an optional hook;
    # any debug surface set by our retriever path makes them available.
    retrieved = _extract_retrieved_session_ids(adapter)

    answer_tokens = count_tokens(answer)
    # ── dual scorers ───────────────────────────────────────────────────────
    # `substring_correct` is the reference (default_scorer or whatever the
    # caller injected as ``scorer``/``row_scorer``). The optional ``judge``
    # is the primary metric when wired up: its verdict becomes ``correct``.
    if row_scorer is not None:
        substring_correct = bool(answer) and bool(
            await _maybe_await(row_scorer(row, answer))
        )
    else:
        substring_correct = bool(answer) and scorer(answer, row.expected_answer)

    judge_correct: bool | None = None
    if judge is not None and answer:
        try:
            judge_correct = bool(
                await _maybe_await(
                    judge.is_correct(row.question, row.expected_answer, answer)
                )
            )
        except Exception:
            log.exception("judge call failed for %s", row.question_id)
            judge_correct = None
    correct = judge_correct if judge_correct is not None else substring_correct
    recall_diag = _recall_diagnostics(
        expected=row.answer_session_ids,
        retrieved=retrieved,
    )
    failure_category = (
        None if correct
        else _classify_failure(
            answer=answer, error=error,
            retrieved=retrieved, expected=row.answer_session_ids,
            question_type=row.question_type,
            decomposition_stats=(
                getattr(adapter, "last_decomposition_stats", None) or {}
            ),
            expected_answer=row.expected_answer,
        )
    )
    # Optional optimizer telemetry — the adapter publishes this dict on
    # itself after answer_question() if a chain ran. Absent in baseline.
    opt = getattr(adapter, "last_optimizer_stats", None) or {}
    # Per-row LLM telemetry (counter wraps every LLM client call). When
    # present, it's the source of truth for retrieved_count / latency /
    # tokens — the previous logs had zeros here while retrieved_session_ids
    # was non-empty. Falls back to optimizer stats when telemetry absent.
    telem = getattr(adapter, "last_telemetry", None) or {}
    # Cost: prefer the real telemetry figure (it adds across every LLM
    # call, including decompose + sub-answers + synthesis + repair).
    # Fallback to the legacy per-answer-token estimate for back-compat.
    measured_cost = float(telem.get("cost_usd", 0.0))
    legacy_cost = (answer_tokens / 1000.0) * prices.get(answerer, 0.0)
    cost_final = measured_cost if measured_cost > 0 else legacy_cost
    # retrieved_count: telemetry's selected_evidence_count is the
    # honest answer; the optimizer field is zero on the wiki+decompose
    # path. Use whichever is non-zero.
    retrieved_count_final = (
        int(telem.get("selected_evidence_count", 0))
        or int(opt.get("retrieved_count", 0))
        or len(retrieved)
    )
    return RowResult(
        question_id=row.question_id,
        question_type=row.question_type,
        correct=correct,
        latency_ms=latency_ms,
        answer_tokens=answer_tokens,
        cost_usd=cost_final,
        retrieved_session_ids=retrieved,
        expected_session_ids=list(row.answer_session_ids),
        answer=str(answer),
        expected_answer=row.expected_answer,
        failure_category=failure_category,
        expected_session_count=recall_diag["expected_session_count"],
        retrieved_expected_session_count=recall_diag[
            "retrieved_expected_session_count"
        ],
        partial_recall=recall_diag["partial_recall"],
        missing_expected_session_ids=recall_diag["missing_expected_session_ids"],
        error=error,
        pre_opt_total_tokens=int(opt.get("pre_total", 0)),
        post_opt_total_tokens=int(opt.get("post_total", 0)),
        pre_opt_stm_tokens=int(opt.get("pre_stm", 0)),
        pre_opt_mtm_tokens=int(opt.get("pre_mtm", 0)),
        pre_opt_ltm_tokens=int(opt.get("pre_ltm", 0)),
        post_opt_stm_tokens=int(opt.get("post_stm", 0)),
        post_opt_mtm_tokens=int(opt.get("post_mtm", 0)),
        post_opt_ltm_tokens=int(opt.get("post_ltm", 0)),
        retrieved_count=retrieved_count_final,
        retrieval_ms=float(opt.get("retrieval_ms", 0.0)),
        optimizer_ms=float(opt.get("optimizer_ms", 0.0)),
        strategy_savings=dict(opt.get("strategy_savings", {})),
        decomposition_stats={
            **(getattr(adapter, "last_decomposition_stats", None) or {}),
            **(
                {"structured_fallthrough":
                 getattr(adapter, "last_structured_fallthrough", None) or {}}
                if getattr(adapter, "last_structured_fallthrough", None)
                else {}
            ),
        },
        telemetry=dict(telem),
        substring_correct=substring_correct,
        judge_correct=judge_correct,
    )


#: Per LongMemEval ``question_type`` → the TaskMode the question
#: *should* route to. Used by :func:`_classify_failure` to detect
#: ``wrong_route`` — when the adapter's chosen route disagrees with
#: the dataset hint. Mirrors the table in ``task_router.py`` so the
#: two stay in sync; if you add a new category there, mirror it here.
_EXPECTED_ROUTE_BY_TYPE: dict[str, str] = {
    "single-session-user":        "FACT_LOOKUP",
    "single-session-assistant":   "ASSISTANT_MEMORY_LOOKUP",
    "single-session-preference":  "PREFERENCE_PROFILE",
    "knowledge-update":           "KNOWLEDGE_UPDATE",
    "multi-session":              "MULTI_SESSION_AGGREGATE",
    "temporal-reasoning":         "TEMPORAL_REASONING",
}

# Bare-date pattern — used to flag "answer looks like a date when a
# duration was asked for" (the missing_temporal_computation bucket).
_BARE_DATE_RE = re.compile(
    r"^(?:\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?|"
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)$",
    re.IGNORECASE,
)


def _classify_failure(
    *,
    answer: str,
    error: str | None,
    retrieved: Sequence[str],
    expected: Sequence[str],
    question_type: str = "",
    decomposition_stats: dict[str, Any] | None = None,
    expected_answer: str = "",
) -> str:
    """
    Map a wrong answer into a precise diagnostic bucket.

    Buckets (ordered from most-specific to most-generic):

    * ``llm_error`` — adapter raised or returned an ``[error:`` shim.
    * ``other`` — no gold to compare against.
    * ``missing_fact`` — none of the expected sessions were retrieved.
      (Retrieval is the bottleneck here; nothing else could have
      worked.)
    * ``wrong_route`` — adapter chose a route that disagrees with the
      LongMemEval category. The answer might still be plausible but
      it came from the wrong reasoning head.
    * ``wrong_role`` — the answer span came from a turn whose speaker
      doesn't match the question shape (user-only or assistant-only).
    * ``missing_aggregation`` — multi-session question, but a single
      span was returned instead of an aggregation.
    * ``missing_temporal_computation`` — temporal question, but the
      answer looks like a bare date rather than a computed duration.
    * ``stale_fact`` — knowledge-update question where the chosen span
      mentions any of the standard update markers (changed, used to,
      previously) — i.e., probably the wrong half of the update pair.
    * ``wrong_span`` — right session was retrieved, right route, but
      the picked span doesn't match the gold.
    * ``wrong_retrieval`` — catch-all when none of the above fire.
    """
    if error or answer.startswith("[error:"):
        return "llm_error"
    if not expected:
        return "other"
    if not any(sid in retrieved for sid in expected):
        return "missing_fact"

    stats = decomposition_stats or {}
    qtype = (question_type or "").strip().lower()
    answer_clean = (answer or "").strip()
    answer_lower = answer_clean.lower()

    # 1) Wrong route — the adapter's chosen route disagrees with the
    # dataset category. We read the route from any of the keys the
    # structured / claim-first / decomposed-answer heads use.
    actual_route = (
        stats.get("claim_first_route")
        or stats.get("route")
        or stats.get("mode")
        or ""
    )
    expected_route = _EXPECTED_ROUTE_BY_TYPE.get(qtype)
    if expected_route and actual_route:
        actual_norm = str(actual_route).upper().replace("CLAIM_FIRST_", "")
        structured_passthrough = {
            expected_route,
            "CLAIM_FIRST_STRUCTURED",
            "CLAIM_FIRST_STRUCTURED_EXPANDED",
        }
        # The structured/expanded routes encode "claim_first answered
        # without going through a specific head" — only flag a route
        # mismatch when the actual route is a *different* TaskMode head.
        if (
            actual_norm not in structured_passthrough
            and expected_route not in actual_norm
        ):
            return "wrong_route"

    # 2) Wrong role — the answer came from a speaker the question
    # shape forbids.
    matched_claim = stats.get("structured_matched_claim") or {}
    matched_role = str(matched_claim.get("source_role", "")).lower()
    if matched_role:
        if qtype == "single-session-assistant" and matched_role != "assistant":
            return "wrong_role"
        if qtype == "single-session-user" and matched_role == "assistant":
            return "wrong_role"

    # 3) Multi-session that returned a single span without aggregation.
    if qtype == "multi-session" and not any(
        marker in answer_lower
        for marker in (",", " and ", " across ", " total", " altogether")
    ) and len(answer_clean.split()) <= 6:
        return "missing_aggregation"

    # 4) Temporal-reasoning that returned a bare date instead of a
    # computed duration / ordering phrase.
    if qtype == "temporal-reasoning" and _BARE_DATE_RE.match(answer_clean):
        return "missing_temporal_computation"

    # 5) Knowledge-update where the chosen span looks like the stale
    # half of the update pair.
    if qtype == "knowledge-update":
        stale_markers = (
            "used to", "previously", "before",
            "formerly", "old name", "changed from",
        )
        matched_text = str(matched_claim.get("text", "")).lower()
        if any(m in matched_text for m in stale_markers):
            return "stale_fact"

    # 6) Right session retrieved but wrong span chosen.
    if any(sid in retrieved for sid in expected):
        return "wrong_span"

    return "wrong_retrieval"


def _recall_diagnostics(
    *, expected: Sequence[str], retrieved: Sequence[str]
) -> dict[str, Any]:
    expected_list = list(expected)
    retrieved_set = set(retrieved)
    missing = [sid for sid in expected_list if sid not in retrieved_set]
    hit_count = len(expected_list) - len(missing)
    return {
        "expected_session_count": len(expected_list),
        "retrieved_expected_session_count": hit_count,
        "partial_recall": hit_count / len(expected_list) if expected_list else 0.0,
        "missing_expected_session_ids": missing,
    }


def _failure_breakdown(rows: list[RowResult]) -> dict[str, int]:
    # Seed every bucket so the report shows zeros explicitly when a
    # category has no failures of that type — easier to read than
    # "missing key" at the call site.
    out: dict[str, int] = {
        "llm_error": 0,
        "missing_fact": 0,
        "wrong_route": 0,
        "wrong_role": 0,
        "wrong_span": 0,
        "stale_fact": 0,
        "missing_aggregation": 0,
        "missing_temporal_computation": 0,
        "wrong_retrieval": 0,
        "other": 0,
    }
    for r in rows:
        if r.failure_category:
            out[r.failure_category] = out.get(r.failure_category, 0) + 1
    return out


def _extract_retrieved_session_ids(adapter: Any) -> list[str]:
    """
    Pull the session ids the last retrieval surfaced.

    The :class:`ContinuumAdapter` exposes the most recent
    :class:`ContextBundle` on ``adapter.last_ctx``; we walk its
    ``items`` for any ``metadata['session_id']``. Adapters without
    this hook simply return ``[]`` — the failure classifier handles
    the empty case as ``missing_fact``.
    """
    ctx = getattr(adapter, "last_ctx", None)
    if ctx is None:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for it in getattr(ctx, "items", []) or []:
        metadata = getattr(it, "metadata", {}) or {}
        raw_ids: list[Any] = []
        sid = metadata.get("session_id")
        if sid:
            raw_ids.append(sid)
        session_ids = metadata.get("session_ids")
        if isinstance(session_ids, list):
            raw_ids.extend(session_ids)
        for raw_id in raw_ids:
            sid_s = str(raw_id)
            if sid_s in seen:
                continue
            seen.add(sid_s)
            ids.append(sid_s)
    return ids


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist(
    results: BaselineResults,
    out_dir: Path,
    now: Callable[[], dt.datetime],
    *,
    output_file: Path | str | None = None,
) -> None:
    if output_file is not None:
        json_path = Path(output_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = json_path.with_suffix(".failures.csv")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = now().strftime("%Y-%m-%d")
        json_path = out_dir / f"baseline_{stamp}.json"
        csv_path = out_dir / "baseline_failures.csv"

    json_path.write_text(json.dumps(results.to_dict(), indent=2, default=str))
    log.info("metrics written to %s", json_path)

    failures = [r for r in results.rows if not r.correct]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "question_id",
            "failure_category",
            "expected_answer",
            "model_answer",
            "expected_session_ids",
            "retrieved_session_ids",
            "partial_recall",
            "missing_expected_session_ids",
            "error",
            "latency_ms",
        ])
        for r in failures:
            writer.writerow([
                r.question_id,
                r.failure_category or "",
                r.expected_answer,
                r.answer,
                ";".join(r.expected_session_ids),
                ";".join(r.retrieved_session_ids),
                f"{r.partial_recall:.3f}",
                ";".join(r.missing_expected_session_ids),
                r.error or "",
                f"{r.latency_ms:.1f}",
            ])
    log.info("%d failures written to %s", len(failures), csv_path)


def _print_summary(results: BaselineResults) -> None:
    print("=" * 60)
    print(f"LongMemEval baseline — {results.dataset}")
    print(f"  answerer:        {results.answerer}")
    print(f"  questions:       {len(results.rows)}")
    judged = results.judged_accuracy
    if judged is not None:
        print(f"  judged_accuracy: {judged:.1%}  (primary — LLM judge)")
        print(f"  substring_acc:   {results.substring_accuracy:.1%}  (reference only)")
    else:
        print(f"  substring_acc:   {results.substring_accuracy:.1%}  (primary — no judge wired)")
    print(f"  recall:          {results.recall:.1%}")
    print(f"  avg tokens:      {results.avg_tokens:.0f}")
    print(f"  latency p50/p95: {results.latency_p50:.0f}ms / {results.latency_p95:.0f}ms")
    print(f"  total cost:      ${results.total_cost_usd:.2f}")
    by_type = results.by_question_type
    if by_type:
        print("  by question type:")
        for question_type, metrics in by_type.items():
            print(
                f"    {question_type:<24} "
                f"n={int(metrics['n_questions']):<3} "
                f"acc={metrics['accuracy']:.1%} "
                f"recall={metrics['recall']:.1%}"
            )
    breakdown = _failure_breakdown(results.rows)
    print("  failure types:")
    for cat, n in breakdown.items():
        if n:
            print(f"    {cat:<18} {n}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    if isinstance(value, Awaitable):
        return await value
    return value


def _default_token_counter(text: str) -> int:
    """Tiny tiktoken-or-whitespace counter. Avoids a hard tiktoken dep."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # pragma: no cover - tiktoken absent
        return len(text.split())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, round((len(s) - 1) * q))
    return float(s[idx])


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Continuum baseline (no optimizer) on LongMemEval."
    )
    p.add_argument("--dataset", default="longmemeval-s")
    p.add_argument("--answerer", default="gpt-4o-mini")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument(
        "--out", type=Path, default=None,
        help="Explicit JSON output path (overrides --output's dated naming).",
    )
    p.add_argument(
        "--limit", "--n", type=int, default=None, dest="limit",
        help="cap on rows (handy for smoke testing)",
    )
    # --judge defaults ON now that it's the primary metric. Pass --no-judge
    # to fall back to substring-only (e.g. for offline / no-API-key smokes).
    p.add_argument(
        "--judge", dest="judge", action="store_true", default=True,
        help="Run the gpt-4o-mini LLM judge inline. Default: on.",
    )
    p.add_argument(
        "--no-judge", dest="judge", action="store_false",
        help="Disable the inline LLM judge (substring scorer only).",
    )
    p.add_argument(
        "--judge-model", default="gpt-4o-mini",
        help="Model used by the inline LLM judge (default: gpt-4o-mini).",
    )
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    """
    CLI entry — wires the real Continuum session + the upstream
    LongMemEval loader.

    Most production runs go through ``evals.longmemeval.bootstrap_ollama``
    which carries the full adapter / retrieval / decompose wiring. This
    thin entry covers the **promoted smoke** path:

        python -m evals.longmemeval.baseline --n 25 --judge \\
            --out results/smoke_judge.json

    Requires ``OPENAI_API_KEY`` when ``--judge`` is on (default).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    judge_obj: Any | None = None
    if args.judge:
        try:
            from evals.longmemeval.bootstrap_ollama import OpenAILLM
            from evals.longmemeval.judge import LLMJudgeScorer
        except ImportError as exc:
            raise SystemExit(
                f"--judge requires openai + bootstrap_ollama deps: {exc}"
            ) from exc
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "--judge needs OPENAI_API_KEY in the environment. "
                "Pass --no-judge to run the substring scorer only."
            )
        judge_obj = LLMJudgeScorer(
            llm=OpenAILLM(model=args.judge_model, rpm=60)
        )
    raise NotImplementedError(
        "Wire your Continuum adapter + LongMemEval dataset loader and call "
        "evals.longmemeval.baseline.run_baseline(..., judge=<judge>) "
        "directly, or use evals.longmemeval.bootstrap_ollama which already "
        "carries the full wiring. judge_obj=%r limit=%s out=%s"
        % (judge_obj, args.limit, args.out)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())


__all__ = [
    "EvalRow",
    "RowResult",
    "BaselineResults",
    "run_baseline",
    "default_scorer",
]


# Re-export for asyncio.run shorthand in user scripts.
def run_baseline_sync(**kwargs: Any) -> BaselineResults:
    """Thin sync wrapper for users in synchronous bootstraps."""
    return asyncio.run(run_baseline(**kwargs))
