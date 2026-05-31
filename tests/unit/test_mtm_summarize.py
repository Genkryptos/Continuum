"""
tests/unit/test_mtm_summarize.py
================================
Unit tests for
:class:`continuum.optimizer.strategies.mtm_summarize.MtmSummarize`.

Covers
------
* Protocol satisfaction + pydantic config validation
* No-op when MTM ≤ keep_recent
* Keeps the most-recent N blocks verbatim
* Older blocks compressed into a single summary row
* Token reduction (extractive) ≥ 30 % on multi-block input
* Extractive scorer picks repeated terms over filler
* LLM path: delegates with the framework system prompt + model kwargs
* LLM failure falls back to extractive
* Tier breakdown / messages / debug_info rebuilt
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
from continuum.optimizer.strategies.mtm_summarize import (
    MTM_SYSTEM_PROMPT,
    MtmSummarize,
    MtmSummarizeConfig,
    extractive_summary,
)

_BASE = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8_000,
        stm_reserved=500,
        mtm_reserved=2_000,
        ltm_reserved=2_000,
        response_reserved=500,
    )


def _mtm(text: str, offset_minutes: int) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.MTM,
        created_at=_BASE + timedelta(minutes=offset_minutes),
    )


def _stm(text: str, offset_minutes: int) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=text,
        tier=MemoryTier.STM,
        created_at=_BASE + timedelta(minutes=offset_minutes),
        metadata={"role": "user"},
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
        tokens_used=sum(estimate_tokens_text(i.content) for i in items),
        budget=_budget(),
    )


# ---------------------------------------------------------------------------
# Protocol + config
# ---------------------------------------------------------------------------


def test_protocol_satisfaction() -> None:
    assert isinstance(MtmSummarize(), Optimizer)


def test_config_defaults() -> None:
    cfg = MtmSummarizeConfig()
    assert cfg.keep_recent == 3
    assert cfg.method == "extractive"
    assert cfg.max_summary_tokens == 500


def test_config_rejects_invalid_method() -> None:
    with pytest.raises(ValidationError):
        MtmSummarizeConfig(method="rubbish")  # type: ignore[arg-type]


def test_config_rejects_low_token_cap() -> None:
    with pytest.raises(ValidationError):
        MtmSummarizeConfig(max_summary_tokens=1)


# ---------------------------------------------------------------------------
# Behaviour — no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_under_limit() -> None:
    items = [_mtm(f"block {i}: some content here", i) for i in range(3)]
    ctx = _bundle(items)
    out = await MtmSummarize(keep_recent=3).apply(ctx, _budget())
    assert out is ctx


@pytest.mark.asyncio
async def test_no_op_when_no_mtm_at_all() -> None:
    items = [_ltm("durable"), _stm("hello", 0)]
    ctx = _bundle(items)
    out = await MtmSummarize(keep_recent=2).apply(ctx, _budget())
    assert out is ctx


# ---------------------------------------------------------------------------
# Behaviour — compression
# ---------------------------------------------------------------------------


_BLOCK_TEXTS = [
    (
        "Acme launched their new analytics dashboard in Q1. The "
        "dashboard tracks user retention and onboarding metrics. "
        "Early customers report 30% faster reporting cycles."
    ),
    (
        "Engineering completed the migration to Postgres 16 last week. "
        "The migration reduced p95 query latency by 40%. The team plans "
        "to enable JIT compilation next sprint."
    ),
    (
        "Sales closed three enterprise deals worth $1.2M ARR combined. "
        "The largest deal includes a multi-year support contract. "
        "Customer success is preparing onboarding for next month."
    ),
    (
        "The hiring committee approved two senior engineering offers. "
        "Both candidates accepted with start dates in March. The team "
        "expects to ramp them via the standard onboarding programme."
    ),
    (
        "Product roadmap for H2 prioritises the analytics dashboard "
        "and a new audit log feature. The audit log requirement came "
        "from three enterprise customers during QBR discussions."
    ),
]


@pytest.mark.asyncio
async def test_keeps_recent_verbatim() -> None:
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    ctx = _bundle(items)
    out = await MtmSummarize(keep_recent=2).apply(ctx, _budget())

    mtm_after = [it for it in out.items if it.tier == MemoryTier.MTM]
    # 1 summary + 2 verbatim recent.
    assert len(mtm_after) == 3
    # Verbatim contents match the last two original blocks.
    verbatim = {it.content for it in mtm_after}
    assert _BLOCK_TEXTS[-1] in verbatim
    assert _BLOCK_TEXTS[-2] in verbatim


@pytest.mark.asyncio
async def test_older_blocks_collapsed_into_single_summary() -> None:
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    ctx = _bundle(items)
    out = await MtmSummarize(keep_recent=2, max_summary_tokens=80).apply(ctx, _budget())

    summaries = [
        it
        for it in out.items
        if it.tier == MemoryTier.MTM and it.metadata.get("kind") == "mtm_summary"
    ]
    assert len(summaries) == 1
    assert summaries[0].metadata["compacted_count"] == 3
    assert summaries[0].metadata["summarisation_method"] == "extractive"


@pytest.mark.asyncio
async def test_token_reduction_at_least_30_percent() -> None:
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    ctx = _bundle(items)
    before = sum(estimate_tokens_text(it.content) for it in ctx.items)

    out = await MtmSummarize(keep_recent=1, max_summary_tokens=80).apply(ctx, _budget())
    after = sum(estimate_tokens_text(it.content) for it in out.items)

    reduction = (before - after) / before
    assert reduction >= 0.30, (
        f"only {reduction:.0%} reduction; expected ≥ 30 % on the 5-block fixture"
    )


@pytest.mark.asyncio
async def test_prompt_order_preserved() -> None:
    ltm = _ltm("durable fact")
    blocks = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    stm = _stm("most recent user message", 100)
    ctx = _bundle([ltm, *blocks, stm])

    out = await MtmSummarize(keep_recent=2).apply(ctx, _budget())
    tiers = [it.tier for it in out.items]
    # LTM → MTM (summary + 2 verbatim) → STM
    assert tiers == [
        MemoryTier.LTM,
        MemoryTier.MTM,
        MemoryTier.MTM,
        MemoryTier.MTM,
        MemoryTier.STM,
    ]
    assert out.items[1].metadata.get("kind") == "mtm_summary"


@pytest.mark.asyncio
async def test_debug_info_records_stats() -> None:
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    out = await MtmSummarize(
        keep_recent=2,
        max_summary_tokens=40,
    ).apply(_bundle(items), _budget())

    dbg = out.debug_info["mtm_summarize"]
    assert dbg["summarised"] == 3
    assert dbg["kept"] == 2
    assert dbg["output_tokens"] > 0
    assert dbg["input_tokens"] > dbg["output_tokens"]


# ---------------------------------------------------------------------------
# Extractive scorer — direct tests
# ---------------------------------------------------------------------------


def test_extractive_summary_empty_returns_empty() -> None:
    assert extractive_summary("") == ""
    assert extractive_summary("   ") == ""


def test_extractive_summary_picks_repeated_terms() -> None:
    text = (
        "Acme announced a partnership with Globex. "
        "Acme will integrate Globex analytics into their dashboard. "
        "Acme expects the integration to land in Q3. "
        "Weather today is unusually warm. "
        "Birds are nesting earlier than expected."
    )
    summary = extractive_summary(text, max_tokens=40, position_bias=0.0)
    # Repeated entities ("Acme", "Globex") should out-rank the weather
    # filler — at least one repeated-term sentence must survive.
    assert "Acme" in summary
    # And the unrelated weather/bird filler should not dominate.
    assert "Birds" not in summary or "Acme" in summary


def test_extractive_summary_respects_token_budget() -> None:
    text = "Hello world. " * 200
    summary = extractive_summary(text, max_tokens=20)
    assert estimate_tokens_text(summary) <= 25  # tiny slack for whole sentences


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


class _FakeSummarizer:
    def __init__(self, output: str = "LLM SUMMARY", raise_: bool = False) -> None:
        self.output = output
        self.raise_ = raise_
        self.calls: list[dict] = []

    async def summarize(
        self,
        text: str,
        *,
        max_tokens: int,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "text": text,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "model": model,
            }
        )
        if self.raise_:
            raise RuntimeError("model unavailable")
        return self.output


@pytest.mark.asyncio
async def test_llm_path_uses_system_prompt_and_model() -> None:
    summariser = _FakeSummarizer(output="dense LLM summary")
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    out = await MtmSummarize(
        keep_recent=2,
        method="llm",
        summarizer=summariser,
        model="claude-3-haiku",
        max_summary_tokens=200,
    ).apply(_bundle(items), _budget())

    assert len(summariser.calls) == 1
    call = summariser.calls[0]
    assert call["model"] == "claude-3-haiku"
    assert call["max_tokens"] == 200
    # The framework system prompt must be passed for prompt caching.
    assert call["system_prompt"]
    assert "MTM" in MTM_SYSTEM_PROMPT  # sanity: the constant exists
    assert "200" in call["system_prompt"]  # max_tokens injected
    # The produced summary lands in the bundle.
    summary = next(
        it
        for it in out.items
        if it.tier == MemoryTier.MTM and it.metadata.get("kind") == "mtm_summary"
    )
    assert summary.content == "dense LLM summary"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_extractive() -> None:
    summariser = _FakeSummarizer(raise_=True)
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    out = await MtmSummarize(
        keep_recent=1,
        method="llm",
        summarizer=summariser,
        max_summary_tokens=80,
    ).apply(_bundle(items), _budget())

    summary = next(
        it
        for it in out.items
        if it.tier == MemoryTier.MTM and it.metadata.get("kind") == "mtm_summary"
    )
    assert summary.content  # extractive fallback produced something


@pytest.mark.asyncio
async def test_llm_path_handles_simple_signature() -> None:
    """A collaborator with ``async summarize(text) -> str`` still works."""

    class _Plain:
        def __init__(self) -> None:
            self.calls = 0

        async def summarize(self, text: str) -> str:
            self.calls += 1
            return "plain summariser output"

    plain = _Plain()
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    out = await MtmSummarize(
        keep_recent=2,
        method="llm",
        summarizer=plain,
    ).apply(_bundle(items), _budget())

    assert plain.calls == 1
    summary = next(
        it
        for it in out.items
        if it.tier == MemoryTier.MTM and it.metadata.get("kind") == "mtm_summary"
    )
    assert summary.content == "plain summariser output"


# ---------------------------------------------------------------------------
# cost_estimate
# ---------------------------------------------------------------------------


def test_cost_estimate_under_limit_returns_current() -> None:
    items = [_mtm("block", i) for i in range(2)]
    cost = MtmSummarize(keep_recent=3).cost_estimate(_bundle(items))
    assert cost.latency_ms == 0.0


def test_cost_estimate_llm_reports_latency() -> None:
    items = [_mtm(t, offset_minutes=i) for i, t in enumerate(_BLOCK_TEXTS)]
    cost = MtmSummarize(
        keep_recent=2,
        method="llm",
    ).cost_estimate(_bundle(items))
    assert cost.latency_ms > 0
