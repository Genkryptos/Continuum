"""
tests/unit/test_optimizer_chain.py
===================================
Unit tests for :mod:`continuum.optimizer` — the Protocol, the chain
composition, and the token helpers. Every test uses fakes; no LLM /
network / tiktoken-mocking required (tiktoken is a hard dep of the repo
already, so it is exercised directly).

Coverage
--------
* Protocol satisfaction (runtime_checkable isinstance)
* :func:`estimate_tokens_text` matches tiktoken within 10 %
* :func:`estimate_tokens` sums across the bundle items
* Chain applies strategies in order
* Chain short-circuits once the bundle fits the budget
* A failing strategy is swallowed; the chain keeps going
* :meth:`OptimizerChain.cost_estimate` aggregates strategies' projections
"""
from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from continuum.optimizer import (
    BaseOptimizer,
    Cost,
    Optimizer,
    OptimizerChain,
    estimate_tokens,
    estimate_tokens_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(content: str, tier: MemoryTier = MemoryTier.LTM) -> MemoryItem:
    return MemoryItem(
        id=uuid.uuid4(),
        content=content,
        tier=tier,
        confidence=1.0,
        importance=0.5,
    )


def _bundle(*contents: str) -> ContextBundle:
    items = [_item(c) for c in contents]
    return ContextBundle(
        items=items,
        messages=[{"role": "system", "content": c} for c in contents],
        tokens_used=sum(len(c.split()) for c in contents),
        budget=TokenBudget(
            total=200,
            stm_reserved=20,
            mtm_reserved=20,
            ltm_reserved=20,
            response_reserved=10,
        ),
    )


class _RecordingOptimizer(BaseOptimizer):
    """Logs each invocation and optionally drops the last item."""

    def __init__(
        self,
        name: str,
        log: list[str],
        *,
        drop_one: bool = False,
        raise_: bool = False,
    ) -> None:
        self.name = name
        self.log = log
        self.drop_one = drop_one
        self.raise_ = raise_

    async def apply(
        self, ctx: ContextBundle, budget: TokenBudget
    ) -> ContextBundle:
        self.log.append(self.name)
        if self.raise_:
            raise RuntimeError(f"{self.name} boom")
        if self.drop_one and ctx.items:
            new_items = ctx.items[:-1]
            return replace(
                ctx,
                items=new_items,
                messages=ctx.messages[: len(new_items)],
            )
        return ctx


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_recording_optimizer_satisfies_protocol() -> None:
    opt = _RecordingOptimizer("x", [])
    assert isinstance(opt, Optimizer)


def test_chain_satisfies_protocol() -> None:
    chain = OptimizerChain([_RecordingOptimizer("a", [])])
    assert isinstance(chain, Optimizer)


# ---------------------------------------------------------------------------
# Token estimators
# ---------------------------------------------------------------------------


def test_estimate_tokens_text_within_10pct_of_tiktoken() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    sample = (
        "The quick brown fox jumps over the lazy dog — "
        "Continuum keeps your context tight."
    ) * 4
    truth = len(enc.encode(sample))
    estimate = estimate_tokens_text(sample)
    assert estimate == truth, (
        f"estimate {estimate} ≠ tiktoken {truth}; the helper must agree "
        "with tiktoken exactly when tiktoken is installed."
    )


def test_estimate_tokens_text_handles_empty() -> None:
    assert estimate_tokens_text("") == 0
    assert estimate_tokens_text(None) == 0  # type: ignore[arg-type]


def test_estimate_tokens_sums_items() -> None:
    ctx = _bundle("hello world", "another short fact", "one more line of text")
    expected = sum(estimate_tokens_text(it.content) for it in ctx.items)
    assert estimate_tokens(ctx) == expected
    # Cost estimate accuracy: within 10 % of the sum of per-string estimates.
    assert abs(estimate_tokens(ctx) - expected) <= max(1, expected // 10)


# ---------------------------------------------------------------------------
# Chain — ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_applies_strategies_in_order() -> None:
    log: list[str] = []
    a = _RecordingOptimizer("a", log, drop_one=True)
    b = _RecordingOptimizer("b", log, drop_one=True)
    c = _RecordingOptimizer("c", log, drop_one=True)

    ctx = _bundle("alpha", "beta", "gamma", "delta", "epsilon")
    # Force a tight budget so the chain can't short-circuit.
    tight = TokenBudget(
        total=2,
        stm_reserved=0, mtm_reserved=0, ltm_reserved=0, response_reserved=0,
    )

    chain = OptimizerChain([a, b, c])
    out = await chain.apply(ctx, tight)

    assert log == ["a", "b", "c"]
    assert len(out.items) == len(ctx.items) - 3  # one drop per strategy


# ---------------------------------------------------------------------------
# Chain — short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_short_circuits_when_under_budget() -> None:
    log: list[str] = []
    a = _RecordingOptimizer("a", log, drop_one=True)
    b = _RecordingOptimizer("b", log, drop_one=True)

    ctx = _bundle("short")  # tiny bundle, easily under budget
    # Generous budget — every step should be skipped.
    roomy = TokenBudget(
        total=10_000,
        stm_reserved=0, mtm_reserved=0, ltm_reserved=0, response_reserved=0,
    )

    chain = OptimizerChain([a, b])
    out = await chain.apply(ctx, roomy)

    assert log == []  # nothing fired
    assert out is ctx  # untouched reference


@pytest.mark.asyncio
async def test_chain_short_circuits_midway() -> None:
    """First strategy drops items below budget → strategies 2/3 don't run."""
    log: list[str] = []
    # Each item ≈ 1 token; budget = 2 means after a single drop we're under.
    big_ctx = _bundle("alpha", "beta", "gamma", "delta")
    budget = TokenBudget(
        total=2,
        stm_reserved=0, mtm_reserved=0, ltm_reserved=0, response_reserved=0,
    )
    # First strategy clears all but one item; subsequent strategies must skip.
    clear_all = _RecordingOptimizer("clear", log)

    async def _clear(ctx, b):
        log.append("clear")
        return replace(ctx, items=ctx.items[:1], messages=ctx.messages[:1])

    clear_all.apply = _clear  # type: ignore[method-assign]

    b = _RecordingOptimizer("b", log)
    c = _RecordingOptimizer("c", log)

    chain = OptimizerChain([clear_all, b, c])
    await chain.apply(big_ctx, budget)
    assert log == ["clear"]  # only the first strategy fired


# ---------------------------------------------------------------------------
# Chain — error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_swallows_strategy_errors() -> None:
    log: list[str] = []
    a = _RecordingOptimizer("a", log, raise_=True)
    b = _RecordingOptimizer("b", log, drop_one=True)

    ctx = _bundle("alpha", "beta", "gamma")
    tight = TokenBudget(
        total=1,
        stm_reserved=0, mtm_reserved=0, ltm_reserved=0, response_reserved=0,
    )
    chain = OptimizerChain([a, b])
    out = await chain.apply(ctx, tight)

    # a raised but the chain continued; b ran successfully.
    assert log == ["a", "b"]
    assert len(out.items) == len(ctx.items) - 1


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def test_cost_estimate_aggregates_strategies() -> None:
    class _Half(BaseOptimizer):
        async def apply(self, ctx, budget):  # pragma: no cover - not exercised
            return ctx

        def cost_estimate(self, ctx):
            return Cost(tokens=estimate_tokens(ctx) // 2, latency_ms=50.0)

    class _Quarter(BaseOptimizer):
        async def apply(self, ctx, budget):  # pragma: no cover
            return ctx

        def cost_estimate(self, ctx):
            return Cost(tokens=estimate_tokens(ctx) // 4, latency_ms=20.0)

    ctx = _bundle("alpha beta gamma", "delta epsilon zeta", "eta theta")
    chain = OptimizerChain([_Half(), _Quarter()])
    cost = chain.cost_estimate(ctx)

    # Lower bound is the cheaper projection; latency is the sum.
    assert cost.tokens == estimate_tokens(ctx) // 4
    assert cost.latency_ms == pytest.approx(70.0)


def test_base_optimizer_default_cost_is_current_token_count() -> None:
    class _Noop(BaseOptimizer):
        async def apply(self, ctx, budget):  # pragma: no cover
            return ctx

    ctx = _bundle("hello there", "general kenobi")
    expected = estimate_tokens(ctx)
    cost = _Noop().cost_estimate(ctx)
    assert cost.tokens == expected
    assert cost.latency_ms == 0.0
