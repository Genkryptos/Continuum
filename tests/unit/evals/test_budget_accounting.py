"""
tests/unit/evals/test_budget_accounting.py
==========================================
Context-budget accounting — the x-axis of the cost/accuracy ablation
(``docs/LOW_BUDGET_PLAN.md``).

Two properties are load-bearing:

* **Whole-item admission.** The budget must never hand the reader a turn
  cut mid-sentence. The old `"\\n".join(lines)[:max_chars]` did exactly
  that; at 8k budgets it would score as "the budget was too small" when
  the real cause was a corrupted tail item.
* **Monotonicity.** Raising the budget may only *add* items, never swap
  one for another. Without this, two points on the ablation curve are
  not comparable and the curve means nothing.

Pricing is tested too: an unpriced reader silently reports cost $0, which
is how every gpt-oss-120b run in ``results/`` came to claim it was free.
"""

from __future__ import annotations

import pytest

from evals.longmemeval.budget import (
    BudgetStats,
    admit_lines,
    count_tokens,
    tokens_measured,
)
from evals.longmemeval.telemetry import (
    _cost_for,
    _pricing_for,
    start_row_telemetry,
    unpriced_models,
)

pytestmark = pytest.mark.unit


def _turns(n: int = 20) -> list[str]:
    """Realistic-ish LongMemEval turn lines, all distinct."""
    return [
        f"[user] In session {i} I mentioned moving to Berlin "
        f"and starting at a fintech firm in March 202{i % 10}."
        for i in range(n)
    ]


# ── whole-item admission ───────────────────────────────────────────────


def test_admits_only_whole_lines() -> None:
    lines = _turns()
    ctx, stats = admit_lines(lines, max_chars=400)

    assert stats.context_chars <= 400
    assert stats.items_dropped > 0, "fixture must actually exceed the budget"
    # Every line that made it is byte-identical to an original.
    assert all(line in lines for line in ctx.split("\n"))


def test_old_byte_slice_would_have_cut_mid_turn() -> None:
    """Regression guard for the behaviour we replaced."""
    lines = _turns()
    old = "\n".join(lines)[:400]
    assert old.split("\n")[-1] not in lines, (
        "fixture no longer demonstrates the mid-turn cut — pick a budget "
        "that lands inside a line"
    )
    new, _ = admit_lines(lines, max_chars=400)
    assert new.split("\n")[-1] in lines


def test_budget_is_monotone_in_the_cap() -> None:
    lines = _turns()
    seen = -1
    for cap in (100, 250, 500, 1_000, 5_000, 1_000_000):
        _, stats = admit_lines(lines, max_chars=cap)
        assert stats.items_admitted >= seen, "raising the budget dropped an item"
        seen = stats.items_admitted
    assert seen == len(lines), "the largest cap should admit everything"


def test_unbounded_when_no_cap_given() -> None:
    lines = _turns()
    ctx, stats = admit_lines(lines)
    assert stats.items_admitted == len(lines)
    assert stats.items_dropped == 0
    assert stats.budget_bound_by == ""
    assert ctx == "\n".join(lines)


def test_token_cap_is_respected() -> None:
    _, stats = admit_lines(_turns(), max_tokens=80)
    assert stats.context_tokens <= 80
    assert stats.budget_bound_by == "tokens"


def test_tighter_of_the_two_caps_wins() -> None:
    lines = _turns()
    _, by_tokens = admit_lines(lines, max_chars=10**6, max_tokens=40)
    _, by_chars = admit_lines(lines, max_chars=150, max_tokens=10**6)
    assert by_tokens.budget_bound_by == "tokens"
    assert by_chars.budget_bound_by == "chars"


def test_empty_input_does_not_crash() -> None:
    ctx, stats = admit_lines([], max_chars=1_000)
    assert ctx == ""
    assert stats.items_admitted == 0
    assert stats.context_tokens == 0
    assert stats.chars_per_token == 0.0


def test_single_line_over_budget_admits_nothing() -> None:
    """
    Better to send no context than a fragment — the reader's "I don't
    have that information" is an honest signal; half a turn is not.
    """
    ctx, stats = admit_lines(["x" * 5_000], max_chars=100)
    assert ctx == ""
    assert stats.items_admitted == 0
    assert stats.items_dropped == 1


# ── token counting ─────────────────────────────────────────────────────


def test_context_tokens_is_an_exact_count_of_the_final_string() -> None:
    ctx, stats = admit_lines(_turns(), max_chars=600)
    assert stats.context_tokens == count_tokens(ctx)


@pytest.mark.skipif(not tokens_measured(), reason="tiktoken not installed")
def test_measured_ratio_is_not_the_assumed_four() -> None:
    """
    The whole reason this module exists: 4.0 chars/token was an
    assumption, and the measured ratio on real turns is meaningfully
    different. Assert only that we *measure* it, not a specific value.
    """
    _, stats = admit_lines(_turns())
    assert stats.tokens_measured is True
    assert 2.0 < stats.chars_per_token < 8.0


def test_stats_serialise_with_the_ratio_included() -> None:
    payload = BudgetStats(context_chars=390, context_tokens=100).as_dict()
    assert payload["chars_per_token"] == 3.9
    assert payload["context_tokens"] == 100


# ── pricing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-oss-120b",
        "openai/gpt-oss-120b",
        "OpenAI/GPT-OSS-120B",
        "openai/gpt-oss-120b:free",
        "accounts/fireworks/models/gpt-oss-120b",
        "meta-llama/llama-3.3-70b-instruct",
        "gpt-4o-mini",
    ],
)
def test_readers_and_judges_are_priced(model_id: str) -> None:
    """
    An unpriced model costs $0, which is indistinguishable from a free
    one. Every model used by a headline run must resolve.
    """
    assert _pricing_for(model_id) is not None, f"{model_id} has no pricing entry"
    assert _cost_for(model_id, 16_000, 400) > 0


def test_unknown_model_costs_zero_but_is_reported() -> None:
    assert _cost_for("nonexistent/model-x", 1_000, 100) == 0.0
    assert "nonexistent/model-x" in unpriced_models()


def test_gpt_oss_120b_row_cost_is_sub_cent() -> None:
    """
    Anchors the cost estimate in docs/LOW_BUDGET_PLAN.md. If this fails,
    the pricing table moved and the plan's dollar figures need redoing.
    """
    row = _cost_for("openai/gpt-oss-120b", 16_000, 400)
    assert 0.001 < row < 0.005
    assert 0.5 < row * 500 < 2.5, "a full 500-row run should be about $1"


def test_counter_records_cost_for_the_open_weight_reader() -> None:
    """End-to-end: the counter must not silently zero the reader's cost."""
    from evals.longmemeval.telemetry import end_row_telemetry, record_llm_call

    start_row_telemetry()
    record_llm_call(
        model="openai/gpt-oss-120b",
        prompt="x" * 400,
        prompt_tokens=16_000,
        completion_tokens=400,
    )
    counter = end_row_telemetry()
    assert counter is not None
    snap = counter.snapshot()
    assert snap["prompt_tokens_total"] == 16_000
    assert snap["cost_usd"] > 0, "priced model still reported $0"
