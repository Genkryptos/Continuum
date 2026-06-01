"""
tests/unit/test_stm_trim.py
===========================
Unit tests for :class:`continuum.optimizer.strategies.stm_trim.StmTrim`.

Covers
------
* No-op when STM ≤ keep_last
* Keeps exactly the last N STM turns (by created_at)
* Older STM turns collapse into a single MTM summary
* "concat" summarisation runs without a summarizer
* "llm" summarisation delegates to the injected collaborator
* LLM failure falls back to concat (chain robustness)
* Prompt order preserved: LTM → MTM → STM
* Tier breakdown / messages / tokens_used rebuilt
* Pydantic config validates ranges
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from continuum.optimizer.protocol import Optimizer
from continuum.optimizer.strategies.stm_trim import StmTrim, StmTrimConfig

_BASE = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=2_000,
        stm_reserved=200,
        mtm_reserved=200,
        ltm_reserved=200,
        response_reserved=100,
    )


def _stm(text: str, offset_minutes: int, role: str = "user") -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.STM,
        created_at=_BASE + timedelta(minutes=offset_minutes),
        metadata={"role": role},
    )


def _mtm(text: str) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.MTM,
        created_at=_BASE,
    )


def _ltm(text: str) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.LTM,
        created_at=_BASE,
    )


def _bundle(items: list[MemoryItem]) -> ContextBundle:
    return ContextBundle(
        items=items,
        messages=[{"role": "system", "content": i.content} for i in items],
        tokens_used=sum(len(i.content.split()) for i in items),
        budget=_budget(),
    )


# ---------------------------------------------------------------------------
# Protocol satisfaction + config
# ---------------------------------------------------------------------------


def test_stm_trim_satisfies_optimizer_protocol() -> None:
    assert isinstance(StmTrim(), Optimizer)


def test_stm_trim_config_rejects_invalid_method() -> None:
    with pytest.raises(ValidationError):
        StmTrimConfig(summarization_method="magic")  # type: ignore[arg-type]


def test_stm_trim_config_default_keeps_10() -> None:
    cfg = StmTrimConfig()
    assert cfg.keep_last == 10
    assert cfg.summarization_method == "concat"


def test_stm_trim_constructor_kwargs_override_config() -> None:
    cfg = StmTrimConfig(keep_last=5)
    trim = StmTrim(keep_last=3, config=cfg)
    assert trim.keep_last == 3


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_stm_under_limit() -> None:
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(5)]
    ctx = _bundle(items)
    out = await StmTrim(keep_last=10).apply(ctx, _budget())
    assert out is ctx
    assert [it.id for it in out.items] == [it.id for it in items]


@pytest.mark.asyncio
async def test_keeps_last_n_turns() -> None:
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(8)]
    ctx = _bundle(items)
    out = await StmTrim(keep_last=3).apply(ctx, _budget())

    stm_after = [it for it in out.items if it.tier == MemoryTier.STM]
    assert len(stm_after) == 3
    # Newest three turns by created_at = offsets 5, 6, 7.
    assert [it.content for it in stm_after] == ["turn 5", "turn 6", "turn 7"]


@pytest.mark.asyncio
async def test_older_turns_become_one_mtm_summary() -> None:
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(8)]
    ctx = _bundle(items)
    out = await StmTrim(keep_last=3).apply(ctx, _budget())

    mtm_after = [it for it in out.items if it.tier == MemoryTier.MTM]
    assert len(mtm_after) == 1
    summary = mtm_after[0]
    assert summary.metadata["kind"] == "stm_trim_summary"
    assert summary.metadata["compacted_count"] == 5
    # The five oldest contents must appear inside the summary text.
    for older_idx in range(5):
        assert f"turn {older_idx}" in summary.content


@pytest.mark.asyncio
async def test_prompt_order_ltm_mtm_stm_preserved() -> None:
    ltm = _ltm("durable fact")
    mtm = _mtm("existing summary")
    stm = [_stm(f"turn {i}", offset_minutes=i) for i in range(6)]
    ctx = _bundle([ltm, mtm, *stm])

    out = await StmTrim(keep_last=2).apply(ctx, _budget())

    tiers = [it.tier for it in out.items]
    # LTM → MTM (new summary + existing) → STM
    assert tiers == [
        MemoryTier.LTM,
        MemoryTier.MTM,
        MemoryTier.MTM,
        MemoryTier.STM,
        MemoryTier.STM,
    ]
    # The new summary sits at the *front* of MTM.
    assert out.items[1].metadata.get("kind") == "stm_trim_summary"
    assert out.items[2].content == "existing summary"


@pytest.mark.asyncio
async def test_breakdown_and_tokens_rebuilt() -> None:
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(6)]
    ctx = _bundle(items)
    out = await StmTrim(keep_last=2).apply(ctx, _budget())

    # Three rows total: 1 new MTM summary + 2 kept STM turns.
    assert sum(out.tier_breakdown.values()) == out.tokens_used
    assert out.tier_breakdown["stm"] > 0
    assert out.tier_breakdown["mtm"] > 0
    assert "stm_trim" in out.debug_info


@pytest.mark.asyncio
async def test_concat_truncates_to_max_chars() -> None:
    big = "x" * 500
    items = [_stm(big, offset_minutes=i) for i in range(10)]
    ctx = _bundle(items)
    out = await StmTrim(
        keep_last=2,
        max_summary_chars=200,
    ).apply(ctx, _budget())
    summary = next(it for it in out.items if it.tier == MemoryTier.MTM)
    assert len(summary.content) <= 200


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


class _FakeSummarizer:
    def __init__(self, output: str = "LLM SUMMARY", raise_: bool = False) -> None:
        self.output = output
        self.raise_ = raise_
        self.calls: list[list[MemoryItem]] = []

    async def summarize(self, items: list[MemoryItem]) -> str:
        self.calls.append(items)
        if self.raise_:
            raise RuntimeError("model unavailable")
        return self.output


@pytest.mark.asyncio
async def test_llm_method_delegates_to_summarizer() -> None:
    summariser = _FakeSummarizer(output="LLM SUMMARY of older turns")
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(6)]
    out = await StmTrim(
        keep_last=2,
        summarization_method="llm",
        summarizer=summariser,
    ).apply(_bundle(items), _budget())

    assert len(summariser.calls) == 1
    assert len(summariser.calls[0]) == 4  # older turns passed in
    summary = next(it for it in out.items if it.tier == MemoryTier.MTM)
    assert summary.content == "LLM SUMMARY of older turns"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_concat() -> None:
    summariser = _FakeSummarizer(raise_=True)
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(5)]
    out = await StmTrim(
        keep_last=2,
        summarization_method="llm",
        summarizer=summariser,
    ).apply(_bundle(items), _budget())

    summary = next(it for it in out.items if it.tier == MemoryTier.MTM)
    # Fallback summary contains concatenated older turn texts.
    for older_idx in range(3):
        assert f"turn {older_idx}" in summary.content


# ---------------------------------------------------------------------------
# cost_estimate
# ---------------------------------------------------------------------------


def test_cost_estimate_no_trim_returns_current_tokens() -> None:
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(3)]
    ctx = _bundle(items)
    cost = StmTrim(keep_last=10).cost_estimate(ctx)
    assert cost.latency_ms == 0.0
    assert cost.tokens > 0


def test_cost_estimate_llm_method_reports_latency() -> None:
    items = [_stm(f"turn {i}", offset_minutes=i) for i in range(20)]
    cost = StmTrim(
        keep_last=2,
        summarization_method="llm",
    ).cost_estimate(_bundle(items))
    assert cost.latency_ms > 0
