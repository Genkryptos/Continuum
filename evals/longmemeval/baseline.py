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
    error: str | None = None  # populated on adapter exceptions

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "answerer": self.answerer,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": {
                "n_questions": len(self.rows),
                "accuracy": self.accuracy,
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
            },
            "rows": [dataclasses.asdict(r) for r in self.rows],
        }


# Adapter contract — the baseline only depends on this shape, not on
# ``ContinuumAdapter`` directly, so tests can swap in fakes trivially.
AdapterFactory = Callable[[], Any]
DatasetLoader = Callable[[str], Iterable[EvalRow]]
Scorer = Callable[[str, str], bool]






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
    a = str(answer).strip().lower()
    e = str(expected).strip().lower()
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
    return [t for t in s.replace(",", " ").split() if t]


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
    scorer: Scorer | None = None,
    limit: int | None = None,
    price_per_1k_out: dict[str, float] | None = None,
    now: Callable[[], dt.datetime] | None = None,
    token_counter: Callable[[str], int] | None = None,
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
        :func:`default_scorer`.
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

    finished = clock()
    results = BaselineResults(
        dataset=dataset,
        answerer=answerer,
        rows=rows,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
    )

    _persist(results, out_root, clock)
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
) -> RowResult:
    """Run a single question. All exceptions captured into the result."""
    error: str | None = None
    answer = ""
    retrieved: list[str] = []
    t0 = time.perf_counter()
    try:
        await _maybe_await(adapter.process_conversation(row.messages))
        answer = await _maybe_await(adapter.answer_question(row.question))
    except Exception as exc:
        log.exception("row %s failed", row.question_id)
        error = repr(exc)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Retrieved session ids come from the adapter via an optional hook;
    # any debug surface set by our retriever path makes them available.
    retrieved = _extract_retrieved_session_ids(adapter)

    answer_tokens = count_tokens(answer)
    correct = bool(answer) and scorer(answer, row.expected_answer)
    failure_category = (
        None if correct
        else _classify_failure(
            answer=answer, error=error,
            retrieved=retrieved, expected=row.answer_session_ids,
        )
    )
    cost = (answer_tokens / 1000.0) * prices.get(answerer, 0.0)

    # Optional optimizer telemetry — the adapter publishes this dict on
    # itself after answer_question() if a chain ran. Absent in baseline.
    opt = getattr(adapter, "last_optimizer_stats", None) or {}
    return RowResult(
        question_id=row.question_id,
        correct=correct,
        latency_ms=latency_ms,
        answer_tokens=answer_tokens,
        cost_usd=cost,
        retrieved_session_ids=retrieved,
        expected_session_ids=list(row.answer_session_ids),
        answer=str(answer),
        expected_answer=row.expected_answer,
        failure_category=failure_category,
        error=error,
        pre_opt_total_tokens=int(opt.get("pre_total", 0)),
        post_opt_total_tokens=int(opt.get("post_total", 0)),
        pre_opt_stm_tokens=int(opt.get("pre_stm", 0)),
        pre_opt_mtm_tokens=int(opt.get("pre_mtm", 0)),
        pre_opt_ltm_tokens=int(opt.get("pre_ltm", 0)),
        post_opt_stm_tokens=int(opt.get("post_stm", 0)),
        post_opt_mtm_tokens=int(opt.get("post_mtm", 0)),
        post_opt_ltm_tokens=int(opt.get("post_ltm", 0)),
        retrieved_count=int(opt.get("retrieved_count", 0)),
        retrieval_ms=float(opt.get("retrieval_ms", 0.0)),
        optimizer_ms=float(opt.get("optimizer_ms", 0.0)),
        strategy_savings=dict(opt.get("strategy_savings", {})),
    )


def _classify_failure(
    *,
    answer: str,
    error: str | None,
    retrieved: Sequence[str],
    expected: Sequence[str],
) -> str:
    """Map a wrong answer into one of the four diagnostic buckets."""
    if error or answer.startswith("[error:"):
        return "llm_error"
    if not expected:
        return "other"  # nothing to compare against
    if not any(sid in retrieved for sid in expected):
        return "missing_fact"
    return "wrong_retrieval"


def _failure_breakdown(rows: list[RowResult]) -> dict[str, int]:
    out: dict[str, int] = {
        "llm_error": 0,
        "missing_fact": 0,
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
    for it in getattr(ctx, "items", []) or []:
        sid = (
            getattr(it, "metadata", {}) or {}
        ).get("session_id") if hasattr(it, "metadata") else None
        if sid:
            ids.append(str(sid))
    return ids


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist(
    results: BaselineResults,
    out_dir: Path,
    now: Callable[[], dt.datetime],
) -> None:
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
                r.error or "",
                f"{r.latency_ms:.1f}",
            ])
    log.info("%d failures written to %s", len(failures), csv_path)


def _print_summary(results: BaselineResults) -> None:
    print("=" * 60)
    print(f"LongMemEval baseline — {results.dataset}")
    print(f"  answerer:        {results.answerer}")
    print(f"  questions:       {len(results.rows)}")
    print(f"  accuracy:        {results.accuracy:.1%}")
    print(f"  recall:          {results.recall:.1%}")
    print(f"  avg tokens:      {results.avg_tokens:.0f}")
    print(f"  latency p50/p95: {results.latency_p50:.0f}ms / {results.latency_p95:.0f}ms")
    print(f"  total cost:      ${results.total_cost_usd:.2f}")
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
        "--limit", type=int, default=None,
        help="cap on rows (handy for smoke testing)",
    )
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    """
    CLI entry — wires the real Continuum session + the upstream
    LongMemEval loader.

    The actual loader / adapter wiring depends on user config (DSNs,
    LLM keys), so the CLI raises ``NotImplementedError`` and points
    at :func:`run_baseline` which the user calls from their own
    bootstrap script. The tests cover ``run_baseline`` directly.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _parse_args(argv)
    raise NotImplementedError(
        "Wire your Continuum session + LongMemEval dataset loader and "
        "call evals.longmemeval.baseline.run_baseline(...) directly. "
        "See the module docstring for the contract."
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
