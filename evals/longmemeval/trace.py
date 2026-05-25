"""
evals/longmemeval/trace.py
==========================
Per-row JSONL trace writer.

The :class:`TraceWriter` is a tiny file-handle wrapper: ``write(row)``
appends one JSON line. The spec's "Phase A4 — make debugging reliable"
requirement reduces to *"every row gets one well-typed JSON line in a
known location"*, which is what this module provides.

The trace schema mirrors the spec's "Each row must include..." block
exactly. Missing fields are written as ``null`` rather than omitted so
downstream consumers (jq, pandas) see a stable schema.

Filenames are timestamped so concurrent runs don't clobber:
``logs/eval_trace_<YYYYMMDDTHHMMSS>_<run_id_short>.jsonl``.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import IO, Any

from evals.longmemeval.baseline import RowResult


@dataclass
class TraceRow:
    """One row of trace output. Field names match the spec verbatim."""

    # Core
    run_id: str
    timestamp: str
    dataset: str
    question_id: str
    question: str
    question_type: str | None = None
    expected_answer: str | None = None
    final_answer: str | None = None
    correct: bool | None = None
    failure_category: str | None = None

    # Retrieval
    retrieved_session_ids: list[str] = field(default_factory=list)
    expected_session_ids: list[str] = field(default_factory=list)
    expected_session_count: int = 0
    retrieved_expected_session_count: int = 0
    partial_recall: float = 0.0
    missing_expected_session_ids: list[str] = field(default_factory=list)
    retrieved_count_actual: int = 0
    retrieval_ms_actual: float = 0.0
    raw_hits_count: int = 0
    wiki_hits_count: int = 0
    session_hits_count: int = 0
    source_expansion_hits_count: int = 0
    selected_evidence_count: int = 0
    selected_evidence_session_ids: list[str] = field(default_factory=list)

    # Context / tokens
    pre_compression_context_tokens: int = 0
    post_compression_context_tokens: int = 0
    answer_prompt_tokens: int = 0
    answer_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    context_budget: int = 0
    context_truncated: bool = False

    # Answer synthesis
    candidate_count: int = 0
    candidates_by_type: dict[str, int] = field(default_factory=dict)
    selected_candidate: dict[str, Any] | None = None
    draft_answer: str | None = None
    validator_passed: bool = True
    validator_reason: str = ""
    regeneration_attempted: bool = False
    regeneration_reason: str = ""
    final_answer_cleaned: str | None = None
    abstain_attempted: bool = False
    abstain_allowed: bool = False
    abstain_reason: str = ""

    # Timing
    candidate_extraction_ms: float = 0.0
    answerer_ms: float = 0.0
    validator_ms: float = 0.0
    judge_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Cost / cache
    estimated_cost_usd: float = 0.0
    retrieval_cache_hit: bool = False
    answer_cache_hit: bool = False
    judge_cache_hit: bool = False

    # Free-form extras for fields the schema doesn't cover yet.
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceWriter:
    """Append-only JSONL writer; no-op when disabled.

    Used as a context manager so the file handle is closed deterministically::

        with TraceWriter.from_run("results/foo", enabled=True) as tw:
            for row in rows:
                tw.write(build_trace_row(...))
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle: IO[str] | None = None

    @classmethod
    def from_run(
        cls,
        out_dir: Path | str,
        *,
        run_id: str | None = None,
        enabled: bool = True,
    ) -> TraceWriter:
        if not enabled:
            return cls(None)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rid = (run_id or uuid.uuid4().hex[:8])
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = out_dir / f"eval_trace_{ts}_{rid}.jsonl"
        return cls(path)

    # ── context manager ────────────────────────────────────────────────
    def __enter__(self) -> TraceWriter:
        if self.path is not None:
            self._handle = self.path.open("a", buffering=1, encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            with contextlib.suppress(Exception):
                self._handle.flush()
                self._handle.close()
            self._handle = None

    # ── write API ─────────────────────────────────────────────────────
    def write(self, row: TraceRow | dict[str, Any]) -> None:
        """Append one row. No-op if disabled or handle is closed."""
        if self._handle is None:
            return
        payload = row.to_dict() if isinstance(row, TraceRow) else dict(row)
        try:
            self._handle.write(json.dumps(payload, default=str, ensure_ascii=False))
            self._handle.write("\n")
        except Exception:  # pragma: no cover — never fail the run on trace I/O
            pass


# ─── Adapter — turn a RowResult into a TraceRow ────────────────────────────


def trace_row_from_result(
    result: RowResult,
    *,
    run_id: str,
    dataset: str,
    question_text: str = "",
) -> TraceRow:
    """
    Compose a :class:`TraceRow` from the runner's :class:`RowResult`
    plus the adapter's per-row telemetry.

    Reads from:

    * ``result.telemetry``           — the LLM-call counter snapshot
    * ``result.decomposition_stats`` — the adapter's synth state
    * ``result``                     — top-level row fields

    Any field the source doesn't cover defaults to the schema's default
    so jq/pandas consumers see a stable shape.
    """
    telem: dict[str, Any] = dict(result.telemetry or {})
    decomp: dict[str, Any] = dict(result.decomposition_stats or {})
    candidates_preview = list(decomp.get("candidates_preview") or [])
    selected = candidates_preview[0] if candidates_preview else None

    row = TraceRow(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        dataset=dataset,
        question_id=result.question_id,
        question=question_text,
        question_type=(
            decomp.get("question_type") or result.question_type
        ),
        expected_answer=result.expected_answer,
        final_answer=result.answer,
        correct=result.correct,
        failure_category=result.failure_category,

        retrieved_session_ids=list(result.retrieved_session_ids),
        expected_session_ids=list(result.expected_session_ids),
        expected_session_count=result.expected_session_count,
        retrieved_expected_session_count=result.retrieved_expected_session_count,
        partial_recall=result.partial_recall,
        missing_expected_session_ids=list(result.missing_expected_session_ids),
        retrieved_count_actual=(
            int(telem.get("selected_evidence_count", 0))
            or result.retrieved_count
        ),
        retrieval_ms_actual=(
            float(result.retrieval_ms)
            or float(telem.get("retrieval_ms", 0.0))
        ),
        raw_hits_count=int(telem.get("raw_hits_count", 0)),
        wiki_hits_count=int(telem.get("wiki_hits_count", 0)),
        session_hits_count=int(telem.get("session_hits_count", 0)),
        source_expansion_hits_count=int(telem.get("source_expansion_hits_count", 0)),
        selected_evidence_count=int(telem.get("selected_evidence_count", 0)),
        selected_evidence_session_ids=list(result.retrieved_session_ids),

        pre_compression_context_tokens=int(telem.get("pre_compression_context_tokens", 0)),
        post_compression_context_tokens=int(telem.get("post_compression_context_tokens", 0)),
        answer_prompt_tokens=int(telem.get("answer_prompt_tokens", 0)),
        answer_completion_tokens=int(telem.get("answer_completion_tokens", 0)),
        total_prompt_tokens=int(telem.get("prompt_tokens_total", 0)),
        context_budget=int(telem.get("context_budget", 0)),
        context_truncated=bool(telem.get("context_truncated", False)),

        candidate_count=int(decomp.get("n_candidates_total", 0)),
        candidates_by_type=_candidates_by_type(candidates_preview),
        selected_candidate=selected,
        draft_answer=decomp.get("draft_answer"),
        validator_passed=bool(decomp.get("validator_passed", True)),
        validator_reason=str(decomp.get("validator_reason", "") or ""),
        regeneration_attempted=bool(decomp.get("regeneration_attempted", False)),
        regeneration_reason=str(decomp.get("regeneration_reason", "") or ""),
        final_answer_cleaned=result.answer,
        abstain_attempted=bool(decomp.get("abstain_attempted", False)),
        abstain_allowed=bool(decomp.get("abstain_allowed", False)),
        abstain_reason=str(decomp.get("abstain_reason", "") or ""),

        total_latency_ms=result.latency_ms,
        estimated_cost_usd=result.cost_usd,
        extras={
            "count_override_reason": decomp.get("count_override_reason", ""),
            "mode": decomp.get("mode", ""),
        },
    )
    return row


def _candidates_by_type(preview: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in preview:
        ctype = str(c.get("candidate_type", "unknown"))
        out[ctype] = out.get(ctype, 0) + 1
    return out


# ─── Helpers ───────────────────────────────────────────────────────────────


def validate_trace_row(payload: dict[str, Any]) -> tuple[bool, str]:
    """
    Return ``(ok, reason)`` for ``payload`` against the :class:`TraceRow` schema.

    Used by the test suite + any caller that wants to confirm a JSONL
    line round-trips cleanly through the schema before shipping.
    """
    if not isinstance(payload, dict):
        return False, "row is not a JSON object"
    field_names = {f.name for f in fields(TraceRow)}
    # `question` is part of TraceRow's required set even though the
    # adapter often passes `""` when the eval row doesn't carry it —
    # keep the required set aligned with the dataclass positional args.
    missing_required = (
        {"run_id", "timestamp", "dataset", "question_id", "question"}
        - payload.keys()
    )
    if missing_required:
        return False, f"missing required: {sorted(missing_required)}"
    # Unknown keys are tolerated (they're persisted in `extras` by convention),
    # but at least confirm the type-correct subset re-instantiates.
    try:
        # `is_dataclass(TraceRow)` keeps this strict — bad value types
        # surface as TypeError from the dataclass constructor.
        TraceRow(**{k: v for k, v in payload.items() if k in field_names})
    except TypeError as exc:
        return False, f"type mismatch: {exc!s}"
    if not is_dataclass(TraceRow):  # defensive — should always be True
        return False, "TraceRow lost its dataclass marker"
    return True, ""


__all__ = [
    "TraceRow",
    "TraceWriter",
    "trace_row_from_result",
    "validate_trace_row",
]
