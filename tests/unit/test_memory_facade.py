"""
tests/unit/test_memory_facade.py
================================
The public `continuum.Memory` facade. Hermetic — a fake ContinuumSession
(fake stm/ltm/search) so the facade's delegation, supersession filtering, and
timeline ordering are validated without a DB, embedder, or LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from continuum import Memory, MemoryItem, MemoryTier
from continuum.core.types import BiTemporalRange

pytestmark = pytest.mark.unit


class _FakeSTM:
    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    async def append(self, item: MemoryItem) -> None:
        self.items.append(item)


class _FakeLTM:
    def __init__(self) -> None:
        self.upserts: list[MemoryItem] = []

    async def upsert(self, item: MemoryItem) -> Any:
        self.upserts.append(item)
        return item.id


class _FakeSession:
    """Minimal ContinuumSession stand-in for the facade."""

    def __init__(self, search_result: list[MemoryItem] | None = None) -> None:
        self.stm = _FakeSTM()
        self.ltm = _FakeLTM()
        self.session_id = "default"
        self._search_result = search_result or []

    async def search(self, query: str, k: int = 10) -> list[MemoryItem]:
        return list(self._search_result[:k])


def _item(
    content: str, *, valid_from: datetime | None = None, valid_to: datetime | None = None
) -> MemoryItem:
    vr = None
    if valid_from is not None or valid_to is not None:
        vr = BiTemporalRange(valid_from=valid_from or datetime.now(UTC), valid_to=valid_to)
    return MemoryItem(content=content, tier=MemoryTier.LTM, valid_range=vr)


# ── construction + exports ────────────────────────────────────────────────────


def test_public_exports() -> None:
    import continuum

    assert hasattr(continuum, "Memory")
    assert {"Memory", "MemoryItem", "MemoryTier", "Query"} <= set(continuum.__all__)


def test_in_memory_factory_constructs() -> None:
    mem = Memory.in_memory()
    assert mem.session is not None
    assert mem.session.ltm is not None  # in-memory LTM attached


# ── write path ────────────────────────────────────────────────────────────────


async def test_add_writes_stm_and_ltm() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("I moved to Boston")
    assert [i.content for i in s.stm.items] == ["I moved to Boston"]
    assert [i.content for i in s.ltm.upserts] == ["I moved to Boston"]
    assert s.ltm.upserts[0].tier is MemoryTier.LTM


async def test_remember_is_add() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.remember("favorite rice is Japanese short-grain")
    assert len(s.stm.items) == 1


# ── recall ────────────────────────────────────────────────────────────────────


async def test_recall_delegates_to_search() -> None:
    hits = [_item("Boston"), _item("NYC")]
    mem = Memory(_FakeSession(hits))  # type: ignore[arg-type]
    out = await mem.recall("where do I live?", k=2)
    assert [h.content for h in out] == ["Boston", "NYC"]


# ── current() — supersession-resolved ─────────────────────────────────────────


async def test_current_prefers_live_over_superseded() -> None:
    old = _item(
        "Boston",
        valid_from=datetime(2023, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 6, 1, tzinfo=UTC),
    )
    new = _item("NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))  # open (current)
    mem = Memory(_FakeSession([old, new]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "NYC"


async def test_current_none_when_empty() -> None:
    mem = Memory(_FakeSession([]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") is None


async def test_current_falls_back_to_latest_when_all_superseded() -> None:
    a = _item(
        "Boston",
        valid_from=datetime(2023, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 3, 1, tzinfo=UTC),
    )
    b = _item(
        "Chicago",
        valid_from=datetime(2023, 3, 1, tzinfo=UTC),
        valid_to=datetime(2023, 6, 1, tzinfo=UTC),
    )
    mem = Memory(_FakeSession([a, b]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "Chicago"  # latest of the closed ones


# ── timeline() — bi-temporal history, ordered ─────────────────────────────────


async def test_timeline_orders_oldest_to_newest() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    b = _item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([b, a]))  # type: ignore[arg-type]  # search returns unordered
    hist = await mem.timeline("residence")
    assert [h.content for h in hist] == ["residence: Boston", "residence: NYC"]


async def test_timeline_filters_by_since() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    b = _item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([a, b]))  # type: ignore[arg-type]
    hist = await mem.timeline("residence", since=datetime(2023, 3, 1, tzinfo=UTC))
    assert [h.content for h in hist] == ["residence: NYC"]


async def test_timeline_only_matching_entity() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    noise = _item("bought a tank", valid_from=datetime(2023, 2, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([a, noise]))  # type: ignore[arg-type]
    hist = await mem.timeline("residence")
    assert [h.content for h in hist] == ["residence: Boston"]


# ── sync wrapper ──────────────────────────────────────────────────────────────


def test_recall_sync_outside_loop() -> None:
    hits = [_item("Boston")]
    mem = Memory(_FakeSession(hits))  # type: ignore[arg-type]
    out = mem.recall_sync("where?", k=1)
    assert [h.content for h in out] == ["Boston"]


def test_sync_wrapper_refuses_inside_running_loop() -> None:
    import asyncio

    async def _inner() -> None:
        mem = Memory(_FakeSession([]))  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="cannot be called from inside"):
            mem.recall_sync("x")

    asyncio.run(_inner())


_ = timedelta  # keep import referenced for future date-math tests
