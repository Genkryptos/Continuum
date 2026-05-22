"""
tests/unit/test_score_prune.py
==============================
Unit tests for
:class:`continuum.optimizer.strategies.score_prune.ScoreAwareBudgetPrune`.

Covers
------
* Protocol + Pydantic config validation
* No-op when bundle already fits
* LTM trimmed lowest-score first; under-budget stop condition
* STM preserved by default
* MTM preserve-count honoured; oldest dropped first beyond it
* preserve_stm=False allows STM removal as last resort
* too_large_stm_strategy="keep" leaves over-budget bundle alone
* too_large_stm_strategy="truncate" cuts oversized STM rows
* too_large_stm_strategy="summarise" delegates to injected summariser,
  falls back to truncate on failure
* debug_info + tier_breakdown rebuilt
* Custom score_key honoured
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
from continuum.optimizer.base import estimate_tokens_text
from continuum.optimizer.protocol import Optimizer
from continuum.optimizer.strategies.score_prune import (
    ScoreAwareBudgetPrune,
    ScorePruneConfig,
)

_BASE = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _budget(total: int = 100, response: int = 10) -> TokenBudget:
    return TokenBudget(
        total=total,
        stm_reserved=10,
        mtm_reserved=10,
        ltm_reserved=10,
        response_reserved=response,
    )


def _ltm(
    text: str, *, importance: float, confidence: float = 1.0, offset: int = 0
) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.LTM,
        importance=importance,
        confidence=confidence,
        created_at=_BASE + timedelta(seconds=offset),
    )


def _mtm(text: str, *, offset: int) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.MTM,
        created_at=_BASE + timedelta(seconds=offset),
    )


def _stm(text: str, *, offset: int) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.STM,
        created_at=_BASE + timedelta(seconds=offset),
        metadata={"role": "user"},
    )


def _bundle(items: list[MemoryItem], budget: TokenBudget) -> ContextBundle:
    return ContextBundle(
        items=items,
        messages=[{"role": "system", "content": i.content} for i in items],
        tokens_used=sum(estimate_tokens_text(i.content) for i in items),
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Protocol + config
# ---------------------------------------------------------------------------


def test_protocol_satisfaction() -> None:
    assert isinstance(ScoreAwareBudgetPrune(), Optimizer)


def test_config_defaults() -> None:
    cfg = ScorePruneConfig()
    assert cfg.preserve_stm is True
    assert cfg.preserve_mtm_count == 2
    assert cfg.too_large_stm_strategy == "keep"


def test_config_rejects_invalid_strategy() -> None:
    with pytest.raises(ValidationError):
        ScorePruneConfig(too_large_stm_strategy="nuke")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_under_budget() -> None:
    items = [_ltm("short", importance=0.5)]
    budget = _budget(total=1_000, response=100)
    ctx = _bundle(items, budget)
    out = await ScoreAwareBudgetPrune().apply(ctx, budget)
    assert out is ctx


# ---------------------------------------------------------------------------
# LTM pruning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drops_lowest_scored_ltm_first() -> None:
    low = _ltm("low score fact " * 5, importance=0.1, offset=0)
    mid = _ltm("mid score fact " * 5, importance=0.5, offset=1)
    high = _ltm("high score fact " * 5, importance=0.9, offset=2)

    # Each item is ≈ 15 tokens. Pick a budget that only fits two of them.
    budget = _budget(total=40, response=5)  # context_available = 35
    ctx = _bundle([high, mid, low], budget)

    out = await ScoreAwareBudgetPrune().apply(ctx, budget)
    survivors = {it.id for it in out.items}
    assert low.id not in survivors
    assert high.id in survivors  # never drop the highest


@pytest.mark.asyncio
async def test_stops_once_under_budget() -> None:
    """Only as many items as needed are removed."""
    items = [
        _ltm(f"fact-{i} " * 10, importance=0.1 + 0.05 * i, offset=i)
        for i in range(8)
    ]
    # Total ≈ 8 × 12 = 96 tokens; budget 50 means we should keep ~4 items.
    budget = _budget(total=55, response=5)
    ctx = _bundle(items, budget)
    out = await ScoreAwareBudgetPrune().apply(ctx, budget)

    # Kept items are the higher-scored half.
    survivors = list(out.items)
    # The highest-importance item must still be in.
    highest = max(items, key=lambda it: it.importance)
    assert highest.id in {it.id for it in survivors}
    # And we should have removed *at least one* but not all.
    assert 0 < len(survivors) < len(items)
    # And we are under budget.
    after = sum(estimate_tokens_text(it.content) for it in survivors)
    assert after <= budget.context_available


@pytest.mark.asyncio
async def test_custom_score_key_used() -> None:
    """A custom score_key inverts the default ordering."""
    big = _ltm("content " * 20, importance=0.1, offset=0)
    small = _ltm("content " * 20, importance=0.9, offset=1)

    # Each is ≈ 20 tokens; budget comfortably allows ~one item.
    budget = _budget(total=35, response=5)  # avail = 30
    ctx = _bundle([big, small], budget)

    # Custom score: prefer LOW importance (inverts the default). The
    # high-importance "small" item now has the lowest custom score and
    # gets removed first.
    out = await ScoreAwareBudgetPrune(
        score_key=lambda it: -float(it.importance),
    ).apply(ctx, budget)
    survivors = {it.id for it in out.items}
    assert small.id not in survivors
    assert big.id in survivors


# ---------------------------------------------------------------------------
# STM preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stm_preserved_by_default() -> None:
    stm = [_stm("stm turn " * 5, offset=i) for i in range(3)]
    ltm = [_ltm("ltm fact " * 5, importance=0.5, offset=i) for i in range(3)]

    budget = _budget(total=30, response=5)
    ctx = _bundle([*ltm, *stm], budget)

    out = await ScoreAwareBudgetPrune().apply(ctx, budget)
    stm_after = [it for it in out.items if it.tier == MemoryTier.STM]
    # All STM items survive.
    assert {it.id for it in stm_after} == {it.id for it in stm}


@pytest.mark.asyncio
async def test_preserve_stm_false_allows_dropping() -> None:
    stm = [_stm("stm turn " * 10, offset=i) for i in range(5)]
    budget = _budget(total=20, response=5)  # forces some drops
    ctx = _bundle(stm, budget)

    out = await ScoreAwareBudgetPrune(
        preserve_stm=False,
    ).apply(ctx, budget)
    # Oldest STM dropped first.
    survivors = list(out.items)
    assert len(survivors) < len(stm)
    surviving_offsets = [
        (it.created_at - _BASE).total_seconds() for it in survivors
    ]
    # All remaining offsets are larger than the dropped ones.
    if survivors:
        dropped_offsets = [
            i for i in range(5)
            if not any(
                (s.created_at - _BASE).total_seconds() == i for s in survivors
            )
        ]
        assert all(d < min(surviving_offsets) for d in dropped_offsets)


# ---------------------------------------------------------------------------
# MTM preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mtm_preserve_count_honoured() -> None:
    mtm = [_mtm(f"mtm-{i} " * 10, offset=i) for i in range(5)]
    budget = _budget(total=30, response=5)  # forces aggressive prune
    ctx = _bundle(mtm, budget)

    out = await ScoreAwareBudgetPrune(
        preserve_mtm_count=2,
    ).apply(ctx, budget)
    mtm_after = [it for it in out.items if it.tier == MemoryTier.MTM]
    # At least 2 MTM survived (the 2 newest).
    assert len(mtm_after) >= 2
    surviving_offsets = sorted(
        (it.created_at - _BASE).total_seconds() for it in mtm_after
    )
    # The freshest 2 (offsets 3, 4) are in.
    assert 4.0 in surviving_offsets
    assert 3.0 in surviving_offsets


# ---------------------------------------------------------------------------
# too_large_stm_strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keep_strategy_returns_over_budget_lossless() -> None:
    huge = _stm("word " * 200, offset=0)  # ≈ 200 tokens
    budget = _budget(total=50, response=5)
    ctx = _bundle([huge], budget)

    out = await ScoreAwareBudgetPrune(
        too_large_stm_strategy="keep",
    ).apply(ctx, budget)

    # STM untouched.
    assert out.items[0].id == huge.id
    assert out.items[0].content == huge.content


@pytest.mark.asyncio
async def test_truncate_strategy_cuts_oversized_stm() -> None:
    huge = _stm("word " * 400, offset=0)  # ≈ 400 tokens
    other = _ltm("other " * 5, importance=0.5)
    budget = _budget(total=50, response=5)  # avail = 45
    ctx = _bundle([huge, other], budget)

    out = await ScoreAwareBudgetPrune(
        too_large_stm_strategy="truncate",
    ).apply(ctx, budget)
    after = sum(estimate_tokens_text(it.content) for it in out.items)
    assert after <= budget.context_available + 5  # slack for ellipsis


@pytest.mark.asyncio
async def test_summarise_strategy_delegates_then_truncates_on_failure() -> None:
    huge = _stm("word " * 400, offset=0)
    budget = _budget(total=40, response=5)

    class _Summ:
        def __init__(self) -> None:
            self.calls = 0
            self.fail = False

        async def summarize(self, text: str) -> str:
            self.calls += 1
            if self.fail:
                raise RuntimeError("nope")
            return "tiny summary"

    summ = _Summ()
    ctx = _bundle([huge], budget)
    out = await ScoreAwareBudgetPrune(
        too_large_stm_strategy="summarise",
        summarizer=summ,
    ).apply(ctx, budget)
    assert summ.calls == 1
    assert out.items[0].content == "tiny summary"

    # Failure path
    summ.fail = True
    out2 = await ScoreAwareBudgetPrune(
        too_large_stm_strategy="summarise",
        summarizer=summ,
    ).apply(_bundle([huge], budget), budget)
    # Falls back to truncate.
    assert out2.items[0].content.endswith("…")


# ---------------------------------------------------------------------------
# Bundle plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_info_and_breakdown() -> None:
    items = [
        _ltm(f"ltm-{i} " * 5, importance=0.1 + 0.1 * i, offset=i)
        for i in range(6)
    ]
    budget = _budget(total=30, response=5)
    ctx = _bundle(items, budget)
    out = await ScoreAwareBudgetPrune().apply(ctx, budget)

    info = out.debug_info["score_prune"]
    assert info["removed"] > 0
    assert info["after_tokens"] <= info["before_tokens"]
    assert info["limit"] == budget.context_available
    assert sum(out.tier_breakdown.values()) == out.tokens_used
