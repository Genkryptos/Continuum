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

from evals.longmemeval.answer_post import is_idk
from evals.longmemeval.evidence import (
    GoldTurn,
    gold_turn_recall,
    supersession_verdict,
)

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
    #: The question's reference date ("now") — temporal-reasoning needs it to
    #: answer "how many weeks ago / since" questions. Empty when unknown.
    question_date: str = ""
    #: The individual turns LongMemEval flags ``has_answer``. Session ids say
    #: *which conversation* holds the answer; these say *which turns*, which is
    #: the granularity the failures live at — see
    #: :mod:`evals.longmemeval.evidence`. Empty for rows loaded from a source
    #: that does not carry the flag; every metric derived from it degrades to
    #: "not measured" rather than to a wrong number.
    gold_turns: list[GoldTurn] = dataclasses.field(default_factory=list)


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

    # ── turn-level evidence (evals.longmemeval.evidence) ───────────────────
    # ``partial_recall`` above is session-level and reads 1.000 on
    # knowledge-update while that category scores 64%. These measure the
    # granularity the failures actually live at. All default to the
    # "not measured" shape (0 / None) so a loader that cannot supply gold
    # turns produces absent metrics rather than misleading ones.
    gold_turn_count: int = 0
    retrieved_gold_turn_count: int = 0
    gold_turn_recall: float = 0.0
    missing_gold_turns: list[str] = dataclasses.field(default_factory=list)
    #: ``"stale"`` when the answer took a value only the *superseded* gold
    #: turn carries, ``"fresh"`` when it took the current one, ``None`` when
    #: the gold pair cannot decide. See :func:`evidence.supersession_verdict`.
    supersession_verdict: str | None = None
    #: The reader declined rather than answering — ``is_idk`` on the final
    #: answer. Orthogonal to ``failure_category``, deliberately: the category
    #: says why the row was wrong (and keeps naming the retrieval miss when
    #: there was one), while this says what the reader *did*. A model that
    #: abstains and one that answers wrongly both score 0, and on the
    #: answerer ladder (PREREGISTRATION.md §4.3) those are opposite
    #: findings — the prompt sanctions "I don't know", so a stronger model
    #: obeying it literally loses to a weaker one that guesses. Collapsing
    #: them would read as accuracy falling while model strength rises.
    abstained: bool = False

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
    # Context-budget accounting — the x-axis of the cost/accuracy ablation
    # (docs/LOW_BUDGET_PLAN.md). Distinct from the ``pre_opt_*`` fields
    # above, which come from the optimizer chain and stay 0 in direct mode.
    # ``context_tokens`` is a tokenizer count of the assembled context;
    # ``prompt_tokens_total`` is what the provider actually billed for the
    # whole row (context + instructions + any extra calls).
    context_chars: int = 0
    context_tokens: int = 0
    items_offered: int = 0
    items_admitted: int = 0
    items_dropped: int = 0
    budget_bound_by: str = ""
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0

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
    #: Which stage of the scoring ladder decided this row, when the caller's
    #: ``row_scorer`` reports it (``"exact"``, ``"normalized"``, ``"numeric"``,
    #: ``"unit"``, ``"rule_semantic"``, ``"llm"``, ``"wrong"``, ``"skipped"``).
    #: ``HybridScorer`` short-circuits on a deterministic hit, so the LLM
    #: judge only ever sees rows the ladder called wrong — it can overturn
    #: wrong→correct but never catches a deterministic false positive. This
    #: breakdown is the only way to see that asymmetry in a result file.
    score_method: str | None = None


@dataclasses.dataclass
class BaselineResults:
    """Aggregate metrics + per-row records — the JSON payload."""

    dataset: str
    answerer: str
    rows: list[RowResult]
    started_at: str
    finished_at: str
    # Provenance for admissibility (evals/PREREGISTRATION.md §3.3): code sha,
    # answerer, judge, judge-prompt hash, flags. ``None`` when the caller
    # supplied none — such a result is inadmissible by §2.1.
    versions: dict[str, Any] | None = None

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
    def abstained_count(self) -> int:
        """Rows where the reader declined to answer."""
        return sum(1 for r in self.rows if r.abstained)

    @property
    def abstention_rate(self) -> float:
        """Share of all rows the reader declined."""
        if not self.rows:
            return 0.0
        return self.abstained_count / len(self.rows)

    @property
    def score_method_breakdown(self) -> dict[str, dict[str, int]]:
        """
        ``{method: {"n": int, "correct": int}}`` over rows whose scorer
        reported a stage. Empty when the scorer does not report one.

        Read it to answer "what actually decided this run?". A run where
        the ``llm`` stage decides most rows is an LLM-judged run whatever
        the field holding the number is called.
        """
        out: dict[str, dict[str, int]] = {}
        for r in self.rows:
            if r.score_method is None:
                continue
            bucket = out.setdefault(r.score_method, {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += 1 if r.correct else 0
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))

    @property
    def primary_metric_source(self) -> str:
        """
        Which mechanism produced the primary ``accuracy``.

        ``"llm_judge"`` when a dedicated judge was wired via ``judge=``;
        ``"scoring_ladder+llm"`` when the injected ``row_scorer`` resolved
        any row through its LLM stage (the number is judged even though
        ``judged_accuracy`` is ``None``); ``"scoring_ladder"`` when it was
        decided deterministically throughout; ``"substring"`` for the
        default reference scorer.
        """
        if self.judged_row_count:
            return "llm_judge"
        methods = self.score_method_breakdown
        if not methods:
            return "substring"
        return "scoring_ladder+llm" if "llm" in methods else "scoring_ladder"

    @property
    def abstained_correct_count(self) -> int:
        """
        Abstentions the scorer **accepted**.

        LongMemEval ships its own negative rows, where the gold answer *is*
        that the user never mentioned the thing. Declining those is the
        right answer, so folding them into a single abstention count
        reports a correct behaviour as though it were a failure mode.
        """
        return sum(1 for r in self.rows if r.abstained and r.correct)

    @property
    def abstained_wrong_count(self) -> int:
        """Abstentions on rows that had a real answer to give."""
        return sum(1 for r in self.rows if r.abstained and not r.correct)

    @property
    def n_evidence_complete(self) -> int:
        """Rows whose gold turns all reached the reader."""
        return sum(
            1 for r in self.rows
            if r.gold_turn_count and r.gold_turn_recall >= 1.0
        )

    @property
    def abstention_rate_with_evidence(self) -> float | None:
        """
        Abstention rate over rows where **every gold turn was in context**.

        This is the one that isolates reader behaviour. The plain
        ``abstention_rate`` mixes two populations: rows where declining was
        correct because retrieval dropped the answer-bearing turn, and rows
        where the answer was present and the reader declined anyway. Only
        the second is a property of the model, so only the second is
        comparable across the §4.3 ladder's rungs. ``None`` when no row
        carried gold turns (a loader that cannot supply them).
        """
        eligible = [
            r for r in self.rows
            if r.gold_turn_count and r.gold_turn_recall >= 1.0
        ]
        if not eligible:
            return None
        return sum(1 for r in eligible if r.abstained) / len(eligible)

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
    def gold_turn_recall(self) -> float:
        """
        Average fraction of answer-bearing *turns* that reached the reader.

        Averaged only over rows that have gold turns — 21 of the 500
        LongMemEval-S rows carry none, and counting them as 0.0 would
        understate the metric by 4 points for no reason.
        """
        per_row = [r.gold_turn_recall for r in self.rows if r.gold_turn_count]
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

    # ── Context-budget metrics (docs/LOW_BUDGET_PLAN.md) ──────────────────
    # These are the ablation's x-axis. Unlike ``avg_context_tokens_pre_opt``
    # they are populated in direct mode, which is the only mode the headline
    # runs use.

    @property
    def avg_context_tokens(self) -> float:
        vals = [r.context_tokens for r in self.rows if r.context_tokens]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def context_tokens_p50(self) -> float:
        vals = [float(r.context_tokens) for r in self.rows if r.context_tokens]
        return _percentile(vals, 0.50) if vals else 0.0

    @property
    def context_tokens_p95(self) -> float:
        vals = [float(r.context_tokens) for r in self.rows if r.context_tokens]
        return _percentile(vals, 0.95) if vals else 0.0

    @property
    def avg_context_chars(self) -> float:
        vals = [r.context_chars for r in self.rows if r.context_chars]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def measured_chars_per_token(self) -> float:
        """Run-wide ratio. Reported so nobody has to assume 4.0 again."""
        chars = sum(r.context_chars for r in self.rows)
        toks = sum(r.context_tokens for r in self.rows)
        return round(chars / toks, 3) if toks else 0.0

    @property
    def avg_prompt_tokens_total(self) -> float:
        """Billed input tokens per row — context plus prompt scaffolding."""
        vals = [r.prompt_tokens_total for r in self.rows if r.prompt_tokens_total]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def pct_rows_budget_bound(self) -> float:
        """
        Share of rows where the budget actually dropped a retrieved item.

        0% means the budget never bound and the run is not a real point on
        the ablation curve — the retriever simply never offered enough to
        fill it. Check this before reading any accuracy delta as a budget
        effect.
        """
        if not self.rows:
            return 0.0
        bound = sum(1 for r in self.rows if r.items_dropped > 0)
        return 100.0 * bound / len(self.rows)

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
            # Per-category context cost. This is the router's design input:
            # the aggregate curve says where accuracy breaks, but only the
            # per-category split says which questions are paying for it.
            ctx_vals = [r.context_tokens for r in rows if r.context_tokens]
            # Turn-level evidence, averaged over the rows that have gold
            # turns. Reported beside ``recall`` because the pair is the
            # diagnostic: session recall 1.00 with turn recall 1.00 and a
            # wrong answer is a reader failure; session recall 1.00 with
            # turn recall < 1.00 is still a retrieval failure.
            turn_vals = [
                r.gold_turn_recall for r in rows if r.gold_turn_count
            ]
            # Per-category abstention. §4.1 already forbids an aggregate
            # accuracy without this breakdown; the same argument applies
            # here, since the categories differ in how much inference the
            # label needs beyond what the gold turn literally says.
            evidenced = [
                r for r in rows
                if r.gold_turn_count and r.gold_turn_recall >= 1.0
            ]
            stale = sum(1 for r in rows if r.supersession_verdict == "stale")
            fresh = sum(1 for r in rows if r.supersession_verdict == "fresh")
            out[question_type] = {
                "n_questions": len(rows),
                "accuracy": correct / len(rows) if rows else 0.0,
                "recall": statistics.fmean(recall_vals) if recall_vals else 0.0,
                "gold_turn_recall": (
                    statistics.fmean(turn_vals) if turn_vals else 0.0
                ),
                "n_gold_turn_measured": len(turn_vals),
                "abstained": sum(1 for r in rows if r.abstained),
                "abstained_wrong": sum(
                    1 for r in rows if r.abstained and not r.correct
                ),
                "abstention_rate": (
                    sum(1 for r in rows if r.abstained) / len(rows)
                    if rows else 0.0
                ),
                "abstention_rate_with_evidence": (
                    sum(1 for r in evidenced if r.abstained) / len(evidenced)
                    if evidenced else 0.0
                ),
                "n_evidence_complete": len(evidenced),
                "stale_answers": stale,
                "fresh_answers": fresh,
                "avg_context_tokens": (
                    statistics.fmean(ctx_vals) if ctx_vals else 0.0
                ),
                "pct_budget_bound": (
                    100.0 * sum(1 for r in rows if r.items_dropped > 0) / len(rows)
                    if rows else 0.0
                ),
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
                # Session recall's blind spots, measured. See
                # :mod:`evals.longmemeval.evidence`.
                "gold_turn_recall": self.gold_turn_recall,
                "n_gold_turn_measured": sum(
                    1 for r in self.rows if r.gold_turn_count
                ),
                # Abstention is reported beside accuracy, not folded into
                # it. Two readers at the same accuracy are not equivalent if
                # one got there by declining and the other by guessing wrong,
                # and the ladder compares exactly that.
                "abstained_count": self.abstained_count,
                # Split, because declining is correct on LongMemEval's own
                # negative rows and a failure everywhere else.
                "abstained_wrong_count": self.abstained_wrong_count,
                "abstained_correct_count": self.abstained_correct_count,
                "abstention_rate": self.abstention_rate,
                "abstention_rate_with_evidence": (
                    self.abstention_rate_with_evidence
                ),
                "n_evidence_complete": self.n_evidence_complete,
                "stale_answers": sum(
                    1 for r in self.rows if r.supersession_verdict == "stale"
                ),
                "fresh_answers": sum(
                    1 for r in self.rows if r.supersession_verdict == "fresh"
                ),
                "latency_p50_ms": self.latency_p50,
                "latency_p95_ms": self.latency_p95,
                "total_cost_usd": self.total_cost_usd,
                "failure_breakdown": _failure_breakdown(self.rows),
                # What decided the primary number above.
                "primary_metric_source": self.primary_metric_source,
                "score_method_breakdown": self.score_method_breakdown,
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
                # Context budget — the ablation's x-axis. Populated in
                # direct mode (the optimizer fields above are not).
                "avg_context_tokens": self.avg_context_tokens,
                "context_tokens_p50": self.context_tokens_p50,
                "context_tokens_p95": self.context_tokens_p95,
                "avg_context_chars": self.avg_context_chars,
                "measured_chars_per_token": self.measured_chars_per_token,
                "avg_prompt_tokens_total": self.avg_prompt_tokens_total,
                "pct_rows_budget_bound": self.pct_rows_budget_bound,
                "by_question_type": self.by_question_type,
            },
            "versions": self.versions,
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
    # OpenRouter model ids (the v1 answerer + judge). APPROXIMATE output
    # prices per 1K tokens — OpenRouter routes to multiple providers whose
    # prices drift, so treat the resulting cost as an estimate, not a bill.
    # Kills the historic $0.00 artifact; verify against your OpenRouter
    # dashboard for exact accounting.
    "openai/gpt-oss-120b": 0.00045,
    "meta-llama/llama-3.3-70b-instruct": 0.00040,
    "meta-llama/llama-3.1-8b-instruct": 0.00006,
    # Meta Model API (dev.meta.ai pay-as-you-go Standard, Oct 2026), per 1K
    # output tokens. Both muse families are reasoning models and the hidden
    # reasoning block is billed as output, so these multiply the *full*
    # `usage.completion_tokens` — which is why a muse row costs far more than
    # its visible answer length suggests. Input is not priced here (this table
    # is output-only): add $0.00125/1K for spark and $0.00400/1K for wonder on
    # the prompt side when reconciling against the credit grant.
    "muse-spark-1.3": 0.00425,
    "muse-spark-1.2": 0.00425,
    "muse-spark-1.1": 0.00425,
    "muse-wonder-1.0": 0.02000,
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
    versions: dict[str, Any] | None = None,
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
    versions:
        Provenance block written verbatim into the result JSON
        (``evals/PREREGISTRATION.md`` §3.3).
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
        versions=versions,
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
        if hasattr(adapter, "dataset_question_date"):
            adapter.dataset_question_date = row.question_date
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
    score_method: str | None = None
    if row_scorer is not None:
        # A row_scorer may return the bare verdict, or ``(verdict, method)``
        # naming the ladder stage that decided it. The second form is what
        # makes an accuracy figure legible: without it a number produced by
        # an LLM judge is indistinguishable from one produced by substring
        # matching, which is how a fully judged run came to be reported as
        # "no judge wired".
        scored = await _maybe_await(row_scorer(row, answer))
        if isinstance(scored, tuple):
            verdict, score_method = scored[0], str(scored[1])
        else:
            verdict = scored
        substring_correct = bool(answer) and bool(verdict)
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
    # Turn-level evidence. Session recall says the right conversation was
    # retrieved; these say whether the answer-bearing *turns* survived, and
    # whether an answer that had both values took the retired one.
    turn_diag = gold_turn_recall(
        row.gold_turns, _extract_retrieved_contents(adapter)
    )
    verdict = supersession_verdict(
        row.gold_turns, answer, row.expected_answer
    )
    # Computed before classification so the classifier can use it, and
    # recorded whether or not the row was wrong — an abstention the judge
    # then *rescued* (a hedged answer the rubric accepted) is still an
    # abstention for ladder purposes.
    abstained = bool(is_idk(str(answer)))
    failure_category = (
        None if correct
        else _classify_failure(
            abstained=abstained,
            answer=answer, error=error,
            retrieved=retrieved, expected=row.answer_session_ids,
            question_type=row.question_type,
            decomposition_stats=(
                getattr(adapter, "last_decomposition_stats", None) or {}
            ),
            expected_answer=row.expected_answer,
            supersession_verdict=verdict,
            gold_turn_recall=(
                turn_diag["gold_turn_recall"] if turn_diag["gold_turn_count"]
                else None
            ),
        )
    )
    # Optional optimizer telemetry — the adapter publishes this dict on
    # itself after answer_question() if a chain ran. Absent in baseline.
    opt = getattr(adapter, "last_optimizer_stats", None) or {}
    # Per-row LLM telemetry (counter wraps every LLM client call). When
    # present, it's the source of truth for retrieved_count / latency /
    # tokens — the previous logs had zeros here while retrieved_session_ids
    # was non-empty. Falls back to optimizer stats when telemetry absent.
    telem = dict(getattr(adapter, "last_telemetry", None) or {})
    # Merge in per-row store metrics (currently only populated by
    # ContinuumLTMHaystackStore). Adds ``ltm_backend``, ``supersessions``,
    # ``ltm_facts_live``, ``promoter_path`` keys to the row's telemetry
    # block so the result JSON records the LTM lift signal end-to-end.
    store_metrics = getattr(adapter, "last_store_metrics", None)
    if store_metrics:
        telem["store_metrics"] = dict(store_metrics)
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
        gold_turn_count=turn_diag["gold_turn_count"],
        retrieved_gold_turn_count=turn_diag["retrieved_gold_turn_count"],
        gold_turn_recall=turn_diag["gold_turn_recall"],
        missing_gold_turns=turn_diag["missing_gold_turns"],
        supersession_verdict=verdict,
        abstained=abstained,
        error=error,
        pre_opt_total_tokens=int(opt.get("pre_total", 0)),
        post_opt_total_tokens=int(opt.get("post_total", 0)),
        pre_opt_stm_tokens=int(opt.get("pre_stm", 0)),
        pre_opt_mtm_tokens=int(opt.get("pre_mtm", 0)),
        pre_opt_ltm_tokens=int(opt.get("pre_ltm", 0)),
        post_opt_stm_tokens=int(opt.get("post_stm", 0)),
        post_opt_mtm_tokens=int(opt.get("post_mtm", 0)),
        post_opt_ltm_tokens=int(opt.get("post_ltm", 0)),
        context_chars=int(telem.get("context_chars", 0)),
        context_tokens=int(telem.get("context_tokens", 0)),
        items_offered=int(telem.get("items_offered", 0)),
        items_admitted=int(telem.get("items_admitted", 0)),
        items_dropped=int(telem.get("items_dropped", 0)),
        budget_bound_by=str(telem.get("budget_bound_by", "") or ""),
        prompt_tokens_total=int(telem.get("prompt_tokens_total", 0)),
        completion_tokens_total=int(telem.get("completion_tokens_total", 0)),
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
        score_method=score_method,
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
    abstained: bool = False,
    retrieved: Sequence[str],
    expected: Sequence[str],
    question_type: str = "",
    decomposition_stats: dict[str, Any] | None = None,
    expected_answer: str = "",
    supersession_verdict: str | None = None,
    gold_turn_recall: float | None = None,
) -> str:
    """
    Map a wrong answer into a precise diagnostic bucket.

    Buckets (ordered from most-specific to most-generic):

    * ``llm_error`` — adapter raised or returned an ``[error:`` shim.
    * ``other`` — no gold to compare against.
    * ``missing_fact`` — none of the expected sessions were retrieved.
      (Retrieval is the bottleneck here; nothing else could have
      worked.)
    * ``stale_fact`` — the answer carries a value only the *superseded*
      gold turn holds. Decided from the gold pair itself
      (:func:`evidence.supersession_verdict`), not from marker words.
    * ``missing_gold_turn`` — the right session was retrieved but an
      answer-bearing *turn* inside it did not reach the reader. This is
      the half of ``wrong_span`` that is still a retrieval failure, and
      separating it is the whole point: ``wrong_span`` was 130 of the
      control run's 131 errors and told us nothing.
    * ``wrong_route`` — adapter chose a route that disagrees with the
      LongMemEval category. The answer might still be plausible but
      it came from the wrong reasoning head.
    * ``wrong_role`` — the answer span came from a turn whose speaker
      doesn't match the question shape (user-only or assistant-only).
    * ``missing_aggregation`` — multi-session question, but a single
      span was returned instead of an aggregation.
    * ``missing_temporal_computation`` — temporal question, but the
      answer looks like a bare date rather than a computed duration.
    * ``abstained`` — the reader declined ("I don't know") with every
      gold turn in front of it. Carved out of ``wrong_span`` for the same
      reason ``missing_gold_turn`` was: the two have opposite fixes. A
      wrong span means the reader misread present evidence; an abstention
      means it judged the evidence insufficient, which is sometimes
      correct — the label can need an inference the gold turn does not
      state. Deliberately ranked *below* the retrieval buckets: when the
      gold turn never arrived, declining is the right move and the
      retrieval miss is the finding, so that row stays
      ``missing_gold_turn``.
    * ``wrong_span`` — right session retrieved, every gold turn present,
      right route, and the answer is still wrong: the evidence was in
      front of the reader.
    * ``wrong_retrieval`` — catch-all when none of the above fire.

    ``stale_fact`` previously keyed off marker words ("used to",
    "previously") appearing in the matched claim. That test read **0 on
    every run ever recorded**, including 28 knowledge-update failures of
    which 14 are provably stale answers. It is now decided from the gold
    evidence and the marker-word path is gone.
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

    # 5) An answer-bearing turn never reached the reader, even though its
    # session was retrieved. Still a retrieval failure, and invisible to
    # session-level recall, which scores a session as recalled when *any*
    # of its turns survives.
    #
    # This is checked BEFORE ``stale_fact`` deliberately. If the superseding
    # turn never arrived, an answer carrying the old value is the only answer
    # the reader could have given — blaming supersession there would charge
    # the resolution layer for a retrieval miss. ``stale_fact`` is reserved
    # for the case that indicts resolution: both values present, retired one
    # chosen.
    if gold_turn_recall is not None and gold_turn_recall < 1.0:
        return "missing_gold_turn"

    # 6) The answer took the superseded value with the current one also in
    # context. Decided from the gold pair (an earlier turn and a later one
    # that disagree, with the expected answer confirming the later), so it
    # fires on any category — the control run has one in temporal-reasoning
    # as well as 14 in knowledge-update.
    if supersession_verdict == "stale":
        return "stale_fact"

    # 7) The reader had the evidence and declined anyway. Reached only
    # after the retrieval buckets above, so an abstention caused by a
    # retrieval miss is still reported as that miss.
    if abstained and any(sid in retrieved for sid in expected):
        return "abstained"

    # 8) Right session retrieved, gold turns present, answer still wrong.
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
        "abstained": 0,
        "stale_fact": 0,
        "missing_gold_turn": 0,
        "missing_aggregation": 0,
        "missing_temporal_computation": 0,
        "wrong_retrieval": 0,
        "other": 0,
    }
    for r in rows:
        if r.failure_category:
            out[r.failure_category] = out.get(r.failure_category, 0) + 1
    return out


def _extract_retrieved_contents(adapter: Any) -> list[str]:
    """
    Text of every item the last retrieval surfaced.

    Same hook as :func:`_extract_retrieved_session_ids` — ``adapter.last_ctx``
    — read for content rather than session ids, so that
    :func:`evidence.gold_turn_recall` can ask whether the answer-bearing
    *turns* reached the reader. Adapters without the hook return ``[]``,
    which reports gold-turn recall as 0 with a non-zero ``gold_turn_count``;
    callers distinguish that from "no gold turns" via the count.
    """
    ctx = getattr(adapter, "last_ctx", None)
    if ctx is None:
        return []
    out: list[str] = []
    for it in getattr(ctx, "items", []) or []:
        content = getattr(it, "content", None)
        if content:
            out.append(str(content))
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
            # These two together are the triage: abstained=1 with
            # gold_turn_recall=1.000 is the reader declining on evidence it
            # had; abstained=1 with a lower recall is a retrieval miss the
            # reader handled correctly.
            "abstained",
            "gold_turn_recall",
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
                "1" if r.abstained else "0",
                f"{r.gold_turn_recall:.3f}" if r.gold_turn_count else "",
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
    source = results.primary_metric_source
    if judged is not None:
        print(f"  judged_accuracy: {judged:.1%}  (primary — LLM judge)")
        print(f"  substring_acc:   {results.substring_accuracy:.1%}  (reference only)")
    else:
        # Name the mechanism rather than asserting a judge was absent: an
        # injected row_scorer can resolve rows through its own LLM stage,
        # in which case this number is judged and saying otherwise is
        # simply false.
        print(
            f"  accuracy:        {results.substring_accuracy:.1%}  "
            f"(primary — {source})"
        )
    methods = results.score_method_breakdown
    if methods:
        print("  decided by:")
        for method, counts in methods.items():
            print(
                f"    {method:<24} n={counts['n']:<4} "
                f"correct={counts['correct']}"
            )
    print(f"  recall:          {results.recall:.1%}")
    # Printed directly under accuracy because it is the number that says
    # whether the accuracy above is comparable to another reader's.
    ab_ev = results.abstention_rate_with_evidence
    if ab_ev is None:
        print(
            f"  abstained:       {results.abstention_rate:.1%} "
            f"({results.abstained_count}/{len(results.rows)})"
        )
    else:
        print(
            f"  abstained:       {results.abstention_rate:.1%} "
            f"({results.abstained_count}/{len(results.rows)})  "
            f"— {ab_ev:.1%} with full evidence "
            f"({results.n_evidence_complete} rows)"
        )
    if results.abstained_count:
        print(
            f"    of which {results.abstained_wrong_count} wrong, "
            f"{results.abstained_correct_count} correct "
            f"(the dataset's own negative rows)"
        )
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
                f"recall={metrics['recall']:.1%} "
                f"abstain={metrics['abstention_rate']:.1%}"
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
    # ── Span-selector fallback (answer_post._pick_answer_from_assistant_claim).
    p.add_argument(
        "--span-fallback", dest="span_fallback",
        action="store_true", default=True,
        help=(
            "Enable the SmallLLM span-selector fallback inside the "
            "assistant-claim picker. Fires when the regex output is a "
            "single nationality adjective, fails a count-shape check, "
            "or sits inside a structured claim much longer than itself. "
            "Default: on."
        ),
    )
    p.add_argument(
        "--no-span-fallback", dest="span_fallback", action="store_false",
        help="Disable the SmallLLM span-selector fallback.",
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
    # ── span-selector fallback wiring ──────────────────────────────────────
    # Parked as a process-wide default so deeply-nested callers in
    # reasoning_heads (head_assistant_memory, FACT_LOOKUP body fallback)
    # pick it up without us having to thread a SmallLLM through every
    # adapter layer. Cheap to construct; the cache is on disk.
    if args.span_fallback:
        try:
            from continuum.extraction.small_llm import SmallLLM
            from evals.longmemeval.answer_post import (
                reset_span_fallback_stats,
                set_default_span_fallback_llm,
            )
            reset_span_fallback_stats()
            set_default_span_fallback_llm(SmallLLM())
            log.info("span fallback enabled (SmallLLM default)")
        except ImportError as exc:
            log.warning("span fallback unavailable: %s", exc)
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
