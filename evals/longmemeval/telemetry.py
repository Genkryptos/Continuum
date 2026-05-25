"""
evals/longmemeval/telemetry.py
==============================
Per-row LLM telemetry counter.

The previous trace reported ``retrieved_count = 0`` and
``retrieval_ms = 0`` even when ``retrieved_session_ids`` was
non-empty, because the wiki+decompose-answer path never published
the equivalent of ``last_optimizer_stats``. This module is the
shared bookkeeping layer.

Each LLM client wraps every ``complete()`` call with
:func:`record_llm_call`, which charges the active row's counter for
input + output tokens (read from the provider's ``usage`` block when
available, estimated otherwise) and tallies cost. The adapter pulls
the snapshot at the end of ``answer_question`` and publishes it on
``last_telemetry`` for the baseline runner to attach to
:class:`RowResult`.

Design notes
------------
* Thread-local + asyncio-task-local — every concurrent question
  gets its own counter, even though they share the same global LLM
  client.
* Provider-agnostic — works for any client that calls
  ``record_llm_call`` after each HTTP response.
* Cost figures are best-effort: clients pass token counts when the
  API surfaces them (OpenAI does in ``response.usage``), otherwise
  the counter estimates from prompt length.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

# ─── gpt-4o-mini pricing (USD per token, as of 2026-05) ────────────────────
_DEFAULT_PRICING = {
    "gpt-4o-mini":     {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o":          {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
}


@dataclass
class TelemetryCounter:
    """One row's running totals."""

    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    #: Per-call-type counters — names match the spec's required fields.
    answer_prompt_tokens: int = 0
    answer_completion_tokens: int = 0
    #: Wiki retrieval breakdown (filled by the retriever).
    wiki_hits_count: int = 0
    raw_hits_count: int = 0
    session_hits_count: int = 0
    selected_evidence_count: int = 0
    pre_compression_context_tokens: int = 0
    post_compression_context_tokens: int = 0
    #: Validator state (filled by the adapter).
    validator_passed: bool = True
    validator_reason: str = ""
    regeneration_attempted: bool = False
    abstain_reason: str = ""
    #: Misc.
    extras: dict[str, Any] = field(default_factory=dict)

    def add_call(
        self, *,
        prompt_tokens: int, completion_tokens: int,
        cost_usd: float, is_answer_call: bool,
    ) -> None:
        self.n_llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd
        if is_answer_call:
            self.answer_prompt_tokens += prompt_tokens
            self.answer_completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, Any]:
        """Flat dict suitable for :class:`baseline.RowResult.extras`."""
        return {
            "n_llm_calls":                       self.n_llm_calls,
            "prompt_tokens_total":               self.prompt_tokens,
            "completion_tokens_total":           self.completion_tokens,
            "cost_usd":                          round(self.cost_usd, 6),
            "answer_prompt_tokens":              self.answer_prompt_tokens,
            "answer_completion_tokens":          self.answer_completion_tokens,
            "wiki_hits_count":                   self.wiki_hits_count,
            "raw_hits_count":                    self.raw_hits_count,
            "session_hits_count":                self.session_hits_count,
            "selected_evidence_count":           self.selected_evidence_count,
            "pre_compression_context_tokens":    self.pre_compression_context_tokens,
            "post_compression_context_tokens":   self.post_compression_context_tokens,
            "validator_passed":                  self.validator_passed,
            "validator_reason":                  self.validator_reason,
            "regeneration_attempted":            self.regeneration_attempted,
            "abstain_reason":                    self.abstain_reason,
            **self.extras,
        }


# ─── Per-row counter, accessible from any LLM client ───────────────────────
# Defined after TelemetryCounter so the runtime subscript resolves cleanly.
_active_counter: contextvars.ContextVar[TelemetryCounter | None] = (
    contextvars.ContextVar("telemetry_counter", default=None)
)


# ───────────────────────────────────────────────────────────────────────────
# Context-manager helpers — start / end / get / record
# ───────────────────────────────────────────────────────────────────────────


def start_row_telemetry() -> TelemetryCounter:
    """Install a fresh counter for the current row."""
    counter = TelemetryCounter()
    _active_counter.set(counter)
    return counter


def end_row_telemetry() -> TelemetryCounter | None:
    """Detach the row's counter and return it (or None if none was set)."""
    counter = _active_counter.get()
    _active_counter.set(None)
    return counter


def current_counter() -> TelemetryCounter | None:
    """Get the row's live counter without detaching it."""
    return _active_counter.get()


def _cost_for(model: str, in_tok: int, out_tok: int) -> float:
    table = _DEFAULT_PRICING.get(model.lower())
    if table is None:
        # Unknown provider — leave cost at zero rather than guess wrong.
        return 0.0
    return in_tok * table["in"] + out_tok * table["out"]


def record_llm_call(
    *,
    model: str,
    prompt: str,
    response: Any = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    is_answer_call: bool = False,
) -> None:
    """
    Charge the row's counter for one LLM call.

    Token counts are read from ``response.usage`` (or the ``usage`` key
    of a dict response) when available; otherwise we estimate
    ``prompt_tokens`` from character count and assume the completion
    fits its ``max_tokens`` cap. Estimates are explicitly worse than
    measured numbers — once the provider clients are updated to pass
    the real ``usage`` block this is a no-op upgrade.
    """
    counter = _active_counter.get()
    if counter is None:
        return

    if prompt_tokens is None or completion_tokens is None:
        usage = _extract_usage(response)
        if usage is not None:
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens or 0))
            completion_tokens = int(usage.get("completion_tokens", completion_tokens or 0))
        else:
            # Last-resort estimate: ~4 chars per token.
            prompt_tokens = prompt_tokens or max(0, len(prompt) // 4)
            completion_tokens = completion_tokens or 0

    cost = _cost_for(model, prompt_tokens, completion_tokens)
    counter.add_call(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        cost_usd=cost,
        is_answer_call=is_answer_call,
    )


def _extract_usage(response: Any) -> dict[str, int] | None:
    """Best-effort extraction of an OpenAI-shaped ``usage`` payload."""
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return usage
    # SDK object with .usage attribute (OpenAI v1+).
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return None
    if isinstance(usage_obj, dict):
        return usage_obj
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage_obj, key, None)
        if val is not None:
            out[key] = int(val)
    return out or None


__all__ = [
    "TelemetryCounter",
    "current_counter",
    "end_row_telemetry",
    "record_llm_call",
    "start_row_telemetry",
]
