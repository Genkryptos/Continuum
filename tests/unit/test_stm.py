"""
tests/unit/test_stm.py
=======================
Unit tests for STM-tier behaviour, demonstrating the project's pytest-asyncio
patterns.

Key conventions shown here
--------------------------
* ``pytestmark = pytest.mark.unit`` — applies the "unit" marker to every
  test in the module; lets ``make test-fast`` run this file without DB.

* ``asyncio_mode = auto`` (from pytest.ini) — every ``async def test_*``
  function runs automatically in an event loop.  No ``@pytest.mark.asyncio``
  decorator required.

* Sync tests co-exist freely with async ones in the same file.

* ``@pytest.mark.parametrize`` works identically for both sync and async tests.

* Fixtures from ``tests/conftest.py`` (``sample_memories``) are injected by
  name, exactly as in synchronous pytest.

Module layout
-------------
InMemorySTM          — minimal async STM stub used only in this test module
_make_item()         — factory helper
Tests: InMemorySTM   — async operations (append, window, flush, concurrency)
Tests: MemoryItem    — dataclass invariants
Tests: BiTemporalRange — validity-window logic
Tests: ScoreBreakdown  — scoring formula correctness
Tests: TokenBudget     — budget arithmetic
Tests: sample_memories — fixture smoke-tests
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from continuum.core.types import (
    BiTemporalRange,
    MemoryItem,
    MemoryTier,
    ProcessingState,
    ScoreBreakdown,
    ScoredItem,
    TokenBudget,
)
from continuum.stores.stm import InMemorySTM as ProductionInMemorySTM

# Every test in this module is a "unit" test — no DB or network required.
pytestmark = pytest.mark.unit


# =============================================================================
# Minimal in-memory async STM
# =============================================================================
# This stub satisfies the STMProtocol interface for pure-unit testing.
# In production, ``continuum.stores.memory.InMemorySTM`` will be the
# concrete implementation; keeping the stub local avoids an import cycle.


class InMemorySTM:
    """Trivial async STM backed by a bounded deque."""

    def __init__(self, max_items: int = 20) -> None:
        self._window: deque[MemoryItem] = deque(maxlen=max_items)

    async def append(self, item: MemoryItem) -> None:
        """Append one item, evicting the oldest when at capacity."""
        self._window.append(item)

    async def window(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> Sequence[MemoryItem]:
        """Return all items belonging to *session_id*, oldest-first."""
        return [m for m in self._window if m.session_id == session_id]

    async def flush_to(self, session_id: str, target: object) -> int:
        """
        Remove *session_id* items from the window and return how many were
        flushed.  ``target`` is the MTM sink (unused in this stub).
        """
        kept: deque[MemoryItem] = deque(maxlen=self._window.maxlen)
        flushed = 0
        for item in self._window:
            if item.session_id == session_id:
                flushed += 1
            else:
                kept.append(item)
        self._window = kept
        return flushed

    async def clear(self, session_id: str) -> None:
        """Discard all items for *session_id* without flushing downstream."""
        self._window = deque(
            (m for m in self._window if m.session_id != session_id),
            maxlen=self._window.maxlen,
        )

    def __len__(self) -> int:
        return len(self._window)


# =============================================================================
# Helper factory
# =============================================================================


def _make_item(
    content: str = "hello",
    session_id: str = "sess-1",
    tier: MemoryTier = MemoryTier.STM,
    importance: float = 0.5,
    created_at: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        content=content,
        tier=tier,
        session_id=session_id,
        importance=importance,
        agent_id="agent-test",
        created_at=created_at or datetime.now(UTC),
    )


# =============================================================================
# InMemorySTM — async operation tests
# =============================================================================


async def test_append_and_window_returns_only_session_items() -> None:
    stm = InMemorySTM()
    item_a = _make_item("msg-a", session_id="sess-1")
    item_b = _make_item("msg-b", session_id="sess-2")

    await stm.append(item_a)
    await stm.append(item_b)

    result = await stm.window("sess-1")

    assert len(result) == 1
    assert result[0].content == "msg-a"


async def test_window_returns_empty_list_for_unknown_session() -> None:
    stm = InMemorySTM()
    await stm.append(_make_item("x", session_id="sess-known"))

    result = await stm.window("sess-unknown")

    assert result == []


async def test_maxlen_enforced_oldest_item_evicted() -> None:
    stm = InMemorySTM(max_items=3)
    for i in range(5):
        await stm.append(_make_item(f"msg-{i}", session_id="s"))

    assert len(stm) == 3
    # Oldest two (msg-0, msg-1) have been evicted; msg-2/3/4 remain.
    window = await stm.window("s")
    contents = [m.content for m in window]
    assert "msg-0" not in contents
    assert "msg-4" in contents


async def test_flush_removes_session_items_and_returns_count() -> None:
    stm = InMemorySTM()
    for i in range(4):
        await stm.append(_make_item(f"msg-{i}", session_id="alpha"))
    await stm.append(_make_item("other", session_id="beta"))

    count = await stm.flush_to("alpha", target=None)

    assert count == 4
    assert len(stm) == 1  # only "beta" survives


async def test_flush_leaves_other_sessions_intact() -> None:
    stm = InMemorySTM()
    await stm.append(_make_item("a1", session_id="alpha"))
    await stm.append(_make_item("b1", session_id="beta"))
    await stm.append(_make_item("a2", session_id="alpha"))

    await stm.flush_to("alpha", target=None)

    remaining = await stm.window("beta")
    assert [m.content for m in remaining] == ["b1"]


async def test_clear_removes_session_without_count() -> None:
    stm = InMemorySTM()
    await stm.append(_make_item("x", session_id="target"))
    await stm.append(_make_item("y", session_id="keep"))

    await stm.clear("target")

    assert len(stm) == 1
    assert (await stm.window("target")) == []
    assert len(await stm.window("keep")) == 1


async def test_concurrent_appends_do_not_corrupt_state() -> None:
    """
    Five concurrent producers each appending 10 items must not lose data
    or raise errors.  The deque is single-threaded (asyncio), so structural
    integrity is guaranteed as long as we don't block between yield points.
    """
    stm = InMemorySTM(max_items=50)

    async def producer(n: int) -> None:
        for i in range(10):
            await stm.append(_make_item(f"p{n}-msg{i}", session_id="shared"))
            await asyncio.sleep(0)  # explicit yield — lets other tasks run

    await asyncio.gather(*(producer(n) for n in range(5)))

    # 50 produced, 50 fit in the deque — all should be present.
    result = await stm.window("shared")
    assert len(result) == 50


async def test_append_then_window_preserves_insertion_order() -> None:
    stm = InMemorySTM()
    contents = ["first", "second", "third"]
    for c in contents:
        await stm.append(_make_item(c))

    result = await stm.window("sess-1")
    assert [m.content for m in result] == contents


# =============================================================================
# MemoryItem — dataclass invariants
# =============================================================================


def test_memory_item_default_id_is_unique() -> None:
    a, b = MemoryItem(), MemoryItem()
    assert a.id != b.id


def test_memory_item_id_is_valid_uuid() -> None:
    item = MemoryItem()
    # Should not raise:
    parsed = uuid.UUID(item.id)
    assert str(parsed) == item.id


def test_memory_item_created_at_is_utc_aware() -> None:
    item = MemoryItem()
    assert item.created_at.tzinfo is not None


def test_memory_item_default_processing_state() -> None:
    item = MemoryItem()
    assert item.processing_state == ProcessingState.UNPROCESSED


def test_memory_item_default_tier_is_stm() -> None:
    item = MemoryItem()
    assert item.tier == MemoryTier.STM


def test_memory_item_related_ids_default_empty() -> None:
    item = MemoryItem()
    assert item.related_ids == []


def test_memory_item_metadata_default_empty() -> None:
    item = MemoryItem()
    assert item.metadata == {}


@pytest.mark.parametrize(
    "tier, importance, confidence",
    [
        (MemoryTier.LTM, 0.9, 1.0),
        (MemoryTier.MTM, 0.5, 0.8),
        (MemoryTier.STM, 0.1, 0.5),
    ],
)
def test_memory_item_accepts_all_tiers(
    tier: MemoryTier, importance: float, confidence: float
) -> None:
    item = MemoryItem(tier=tier, importance=importance, confidence=confidence)
    assert item.tier == tier
    assert item.importance == importance
    assert item.confidence == confidence


def test_memory_item_with_valid_range() -> None:
    vr = BiTemporalRange(
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2025, 1, 1, tzinfo=UTC),
    )
    item = MemoryItem(tier=MemoryTier.LTM, valid_range=vr)
    assert item.valid_range is vr
    assert not item.valid_range.is_currently_valid()


def test_memory_item_related_ids_stored_correctly() -> None:
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    item = MemoryItem(related_ids=ids)
    assert item.related_ids == ids


# =============================================================================
# BiTemporalRange — validity window logic
# =============================================================================


def test_bitemporal_open_ended_is_currently_valid() -> None:
    rng = BiTemporalRange(valid_from=datetime.now(UTC) - timedelta(days=1))
    assert rng.is_currently_valid()


def test_bitemporal_future_start_not_valid_yet() -> None:
    rng = BiTemporalRange(valid_from=datetime.now(UTC) + timedelta(days=1))
    assert not rng.is_currently_valid()


def test_bitemporal_closed_past_window_not_valid() -> None:
    rng = BiTemporalRange(
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert not rng.is_currently_valid()


def test_bitemporal_frozen_immutability() -> None:
    """BiTemporalRange is frozen=True; mutation must raise TypeError."""
    rng = BiTemporalRange(valid_from=datetime.now(UTC))
    with pytest.raises((AttributeError, TypeError)):
        rng.valid_from = datetime.now(UTC)  # type: ignore[misc]


@pytest.mark.parametrize(
    "from_delta, to_delta, point_delta, expected",
    [
        (-2, -1, -1.5, True),  # point inside a closed window
        (-2, None, 0.0, True),  # open-ended window; point is "now"
        (-2, -1, 0.0, False),  # point is after the window closed
        (1, None, 0.0, False),  # window hasn't opened yet
        (-1, None, -1.5, False),  # point is before valid_from
    ],
)
def test_bitemporal_is_valid_at(
    from_delta: float,
    to_delta: float | None,
    point_delta: float,
    expected: bool,
) -> None:
    now = datetime.now(UTC)
    valid_from = now + timedelta(days=from_delta)
    valid_to = (now + timedelta(days=to_delta)) if to_delta is not None else None
    point = now + timedelta(days=point_delta)

    rng = BiTemporalRange(valid_from=valid_from, valid_to=valid_to)
    assert rng.is_valid_at(point) is expected


# =============================================================================
# ScoreBreakdown — scoring formula correctness
# =============================================================================
# canonical formula: composite = 0.45·rel + 0.25·imp + 0.20·rec + 0.10·conf


def test_score_all_ones_gives_composite_one() -> None:
    score = ScoreBreakdown.compute(1.0, 1.0, 1.0, 1.0)
    assert score.composite == pytest.approx(1.0)


def test_score_all_zeros_gives_composite_zero() -> None:
    score = ScoreBreakdown.compute(0.0, 0.0, 0.0, 0.0)
    assert score.composite == pytest.approx(0.0)


@pytest.mark.parametrize(
    "rel, imp, rec, conf, expected",
    [
        # Isolate each weight to verify the formula coefficients.
        (1.0, 0.0, 0.0, 0.0, 0.45),
        (0.0, 1.0, 0.0, 0.0, 0.25),
        (0.0, 0.0, 1.0, 0.0, 0.20),
        (0.0, 0.0, 0.0, 1.0, 0.10),
        # Mixed case: 0.45×0.8 + 0.25×0.6 + 0.20×0.9 + 0.10×0.7
        (0.8, 0.6, 0.9, 0.7, 0.45 * 0.8 + 0.25 * 0.6 + 0.20 * 0.9 + 0.10 * 0.7),
    ],
)
def test_score_formula_coefficients(
    rel: float, imp: float, rec: float, conf: float, expected: float
) -> None:
    score = ScoreBreakdown.compute(rel, imp, rec, conf)
    assert score.composite == pytest.approx(expected, abs=1e-9)


def test_score_composite_clamped_when_inputs_exceed_one() -> None:
    # Passing values > 1 must not produce a composite > 1.
    score = ScoreBreakdown.compute(2.0, 2.0, 2.0, 2.0)
    assert score.composite == pytest.approx(1.0)


def test_score_breakdown_fields_preserved() -> None:
    score = ScoreBreakdown.compute(0.7, 0.5, 0.8, 0.6)
    assert score.relevance == pytest.approx(0.7)
    assert score.importance == pytest.approx(0.5)
    assert score.recency == pytest.approx(0.8)
    assert score.confidence == pytest.approx(0.6)


def test_score_breakdown_is_frozen() -> None:
    score = ScoreBreakdown.compute(0.5, 0.5, 0.5, 0.5)
    with pytest.raises((AttributeError, TypeError)):
        score.composite = 999.0  # type: ignore[misc]


def test_scored_item_score_property_delegates() -> None:
    breakdown = ScoreBreakdown.compute(0.8, 0.6, 0.9, 0.7)
    si = ScoredItem(item=MemoryItem(), scores=breakdown)
    assert si.score == breakdown.composite


# =============================================================================
# TokenBudget — budget arithmetic
# =============================================================================


def test_token_budget_context_available() -> None:
    budget = TokenBudget(
        total=8192,
        stm_reserved=512,
        mtm_reserved=1024,
        ltm_reserved=512,
        response_reserved=1024,
    )
    assert budget.context_available == 8192 - 1024


def test_token_budget_memory_reserved() -> None:
    budget = TokenBudget(
        total=8192,
        stm_reserved=512,
        mtm_reserved=1024,
        ltm_reserved=512,
        response_reserved=1024,
    )
    assert budget.memory_reserved == 512 + 1024 + 512


def test_token_budget_slack_is_context_minus_reserved() -> None:
    budget = TokenBudget(
        total=8192,
        stm_reserved=512,
        mtm_reserved=512,
        ltm_reserved=512,
        response_reserved=1024,
    )
    # context_available = 7168, memory_reserved = 1536 → slack = 5632
    assert budget.slack == 7168 - 1536


def test_token_budget_slack_never_negative() -> None:
    # Reserves exceed the context window — slack must be clamped at 0.
    budget = TokenBudget(
        total=100,
        stm_reserved=200,
        mtm_reserved=200,
        ltm_reserved=200,
        response_reserved=50,
    )
    assert budget.slack == 0


def test_token_budget_context_available_never_negative() -> None:
    budget = TokenBudget(
        total=100,
        stm_reserved=0,
        mtm_reserved=0,
        ltm_reserved=0,
        response_reserved=200,  # bigger than total
    )
    assert budget.context_available == 0


@pytest.mark.parametrize(
    "total, response, expected_ctx",
    [
        (8192, 1024, 7168),
        (4096, 512, 3584),
        (2048, 2048, 0),  # entire window reserved for response
        (1000, 1500, 0),  # over-reserved → clamp at 0
    ],
)
def test_token_budget_context_available_parametric(
    total: int, response: int, expected_ctx: int
) -> None:
    budget = TokenBudget(
        total=total,
        stm_reserved=0,
        mtm_reserved=0,
        ltm_reserved=0,
        response_reserved=response,
    )
    assert budget.context_available == expected_ctx


# =============================================================================
# sample_memories fixture — integration with conftest
# =============================================================================


def test_sample_memories_count(sample_memories: list[MemoryItem]) -> None:
    assert len(sample_memories) == 100


def test_sample_memories_all_unique_ids(sample_memories: list[MemoryItem]) -> None:
    ids = {m.id for m in sample_memories}
    assert len(ids) == 100


def test_sample_memories_span_all_tiers(sample_memories: list[MemoryItem]) -> None:
    tiers = {m.tier for m in sample_memories}
    assert tiers == {MemoryTier.STM, MemoryTier.MTM, MemoryTier.LTM}


def test_sample_memories_have_valid_created_at(sample_memories: list[MemoryItem]) -> None:
    for item in sample_memories:
        assert item.created_at.tzinfo is not None, f"Item {item.id} lacks tz-aware timestamp"


def test_sample_memories_mtm_are_unprocessed(sample_memories: list[MemoryItem]) -> None:
    mtm_items = [m for m in sample_memories if m.tier == MemoryTier.MTM]
    assert all(m.processing_state == ProcessingState.UNPROCESSED for m in mtm_items)


def test_sample_memories_spread_across_five_sessions(sample_memories: list[MemoryItem]) -> None:
    sessions = {m.session_id for m in sample_memories}
    assert sessions == {"sess-00", "sess-01", "sess-02", "sess-03", "sess-04"}


async def test_stm_loads_all_stm_tier_items_from_fixture(
    sample_memories: list[MemoryItem],
) -> None:
    """Async fixture test: load all STM items and verify per-session retrieval."""
    stm = InMemorySTM(max_items=200)
    stm_items = [m for m in sample_memories if m.tier == MemoryTier.STM]

    for item in stm_items:
        await stm.append(item)

    for session_id in {"sess-00", "sess-01", "sess-02"}:
        window = await stm.window(session_id)
        assert all(m.session_id == session_id for m in window), (
            f"Window for {session_id} contains items from another session"
        )


async def test_production_inmemory_stm_preserves_user_id_and_agent_id_on_flush() -> None:
    class Target:
        def __init__(self) -> None:
            self.items: list[MemoryItem] = []

        async def add_summary(self, item: MemoryItem) -> str:
            self.items.append(item)
            return item.id

    stm = ProductionInMemorySTM()
    target = Target()
    await stm.append(
        MemoryItem(
            content="remember my favorite color is blue",
            session_id="chat-a",
            user_id="u1",
            agent_id="agent-1",
            metadata={"role": "user"},
        )
    )

    recent = await stm.get_recent("chat-a", n=1)
    assert recent[0].user_id == "u1"
    assert recent[0].agent_id == "agent-1"

    flushed = await stm.flush_to("chat-a", target)

    assert flushed == 1
    assert target.items[0].user_id == "u1"
    assert target.items[0].agent_id == "agent-1"
