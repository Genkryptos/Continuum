"""
tests/unit/test_triggers.py
===========================
Unit tests for ``continuum.promotion.triggers.TriggerManager`` and its
optional ``ContinuumSession`` integration. All collaborators are fakes —
no DB / LLM / network.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.config import ContinuumConfig, TriggerConfig
from continuum.core.types import (
    MemoryItem,
    MemoryTier,
    ScoreBreakdown,
    ScoredItem,
    SummaryBlock,
)
from continuum.extraction.entity_extractor import Entity
from continuum.promotion import TriggerManager

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _ent(text: str) -> Entity:
    return Entity(text=text, type="PERSON", start=0, end=len(text),
                  confidence=0.9)


class FakeLTM:
    """
    search_hybrid → a ScoredItem only for query texts in `known`.

    A value of ``cos`` means the top hit's content == the query (exact
    match). A ``(content, cos)`` tuple lets the hit's stored text differ
    from the query (to exercise the cosine-threshold path).
    """

    def __init__(
        self, known: dict[str, float | tuple[str, float]] | None = None
    ) -> None:
        self.known = known or {}
        self.raises = False

    async def search_hybrid(self, q: Any, k: int) -> list[ScoredItem]:
        if self.raises:
            raise RuntimeError("ltm down")
        spec = self.known.get(q.text)
        if spec is None:
            return []
        content, cos = (
            (q.text, spec) if isinstance(spec, int | float) else spec
        )
        return [
            ScoredItem(
                item=MemoryItem(id=str(uuid.uuid4()), content=content,
                                tier=MemoryTier.LTM),
                scores=ScoreBreakdown(relevance=cos, importance=0.0,
                                      recency=0.0, confidence=1.0,
                                      composite=cos),
            )
        ]


class FakeMTM:
    def __init__(self, n: int, *, raises: bool = False) -> None:
        self._n = n
        self.raises = raises
        self.scanned = 0

    async def scan_unprocessed(self, **_: Any) -> AsyncIterator[SummaryBlock]:
        if self.raises:
            raise RuntimeError("mtm down")
        for _ in range(self._n):
            self.scanned += 1
            yield SummaryBlock(text="b", id=uuid.uuid4())


class FakePromoter:
    def __init__(self) -> None:
        self.calls = 0
        self.scopes: list[Any] = []

    async def promote(self, scope: Any = None) -> Any:
        self.calls += 1
        self.scopes.append(scope)
        return SimpleNamespace(promoted_count=1)


class FakeBG:
    """Records scheduled jobs; ``run()`` executes queued nowait factories."""

    def __init__(self, *, nowait_ok: bool = True) -> None:
        self.nowait_ok = nowait_ok
        self.jobs: list[Any] = []
        self.periodic: list[tuple[Any, int, str]] = []

    def schedule_nowait(self, factory: Any, *_a: Any, **_k: Any) -> bool:
        if not self.nowait_ok:
            return False
        self.jobs.append(factory)
        return True

    async def schedule_periodic(
        self, factory: Any, *, interval_seconds: int, name: str = "",
        **_k: Any,
    ) -> Any:
        self.periodic.append((factory, interval_seconds, name))
        return SimpleNamespace(cancel=lambda: None)

    async def run(self) -> None:
        jobs, self.jobs = self.jobs, []
        for f in jobs:
            await f()


def _tm(**kw: Any) -> tuple[TriggerManager, FakeBG, FakePromoter]:
    bg = kw.pop("bg", None) or FakeBG()
    prom = kw.pop("promoter", None) or FakePromoter()
    cfg = kw.pop("config", None)
    tm = TriggerManager(
        cfg,
        mtm=kw.pop("mtm", FakeMTM(0)),
        ltm=kw.pop("ltm", FakeLTM()),
        promoter=prom,
        background=bg,
        **kw,
    )
    return tm, bg, prom


# ---------------------------------------------------------------------------
# (a) New-entity trigger
# ---------------------------------------------------------------------------


class TestNewEntity:
    async def test_unknown_entity_triggers(self) -> None:
        tm, _, _ = _tm(ltm=FakeLTM(known={"Alice": 0.99}))
        # Alice is in LTM, Bob is not → trigger.
        assert await tm.check_new_entity([_ent("Alice"), _ent("Bob")]) is True

    async def test_all_known_does_not_trigger(self) -> None:
        tm, _, _ = _tm(ltm=FakeLTM(known={"Alice": 0.99, "Bob": 0.95}))
        assert await tm.check_new_entity([_ent("Alice"), _ent("Bob")]) is False

    async def test_low_similarity_counts_as_new(self) -> None:
        # Top hit has different text and only 0.40 similarity → still "new".
        tm, _, _ = _tm(
            ltm=FakeLTM(known={"Alicia": ("Alice Smith", 0.40)}),
            match_threshold=0.85,
        )
        assert await tm.check_new_entity([_ent("Alicia")]) is True

    async def test_disabled_by_config(self) -> None:
        tm, _, _ = _tm(
            config=TriggerConfig(on_new_entity=False), ltm=FakeLTM()
        )
        assert await tm.check_new_entity([_ent("Whoever")]) is False

    async def test_lookup_error_assumes_present(self) -> None:
        ltm = FakeLTM()
        ltm.raises = True
        tm, _, _ = _tm(ltm=ltm)
        assert await tm.check_new_entity([_ent("X")]) is False  # no spurious

    async def test_entity_lookup_override(self) -> None:
        async def lookup(text: str) -> bool:
            return text == "Known"

        tm, _, _ = _tm(entity_lookup=lookup)
        assert await tm.check_new_entity([_ent("Known")]) is False
        assert await tm.check_new_entity([_ent("Fresh")]) is True


# ---------------------------------------------------------------------------
# (b) Block-accumulation trigger
# ---------------------------------------------------------------------------


class TestBlockAccumulation:
    async def test_triggers_at_threshold_with_early_exit(self) -> None:
        mtm = FakeMTM(100)
        tm, _, _ = _tm(config=TriggerConfig(block_threshold=20), mtm=mtm)
        assert await tm.check_block_accumulation() is True
        assert mtm.scanned == 20  # early-exit — never scans all 100

    async def test_below_threshold_no_trigger(self) -> None:
        tm, _, _ = _tm(config=TriggerConfig(block_threshold=20),
                       mtm=FakeMTM(19))
        assert await tm.check_block_accumulation() is False

    async def test_scan_error_no_trigger(self) -> None:
        tm, _, _ = _tm(mtm=FakeMTM(0, raises=True))
        assert await tm.check_block_accumulation() is False


# ---------------------------------------------------------------------------
# (d) Periodic trigger
# ---------------------------------------------------------------------------


class TestPeriodic:
    async def test_schedules_with_config_interval(self) -> None:
        tm, bg, prom = _tm(
            config=TriggerConfig(periodic_interval_seconds=21600)
        )
        await tm.schedule_periodic()
        assert len(bg.periodic) == 1
        factory, interval, name = bg.periodic[0]
        assert interval == 21600
        assert name == "promotion-periodic"
        # The scheduled factory runs a promotion when fired.
        await factory()
        assert prom.calls == 1

    async def test_explicit_interval_override(self) -> None:
        tm, bg, _ = _tm()
        await tm.schedule_periodic(interval_seconds=5)
        assert bg.periodic[0][1] == 5

    async def test_real_background_queue_fires(self) -> None:
        from continuum.core.background import BackgroundQueue

        bg = BackgroundQueue(backoff_initial=0.0, backoff_max=0.0)
        prom = FakePromoter()
        tm = TriggerManager(
            None, mtm=FakeMTM(0), ltm=FakeLTM(),
            promoter=prom, background=bg,
        )
        async with bg:
            await tm.schedule_periodic(interval_seconds=0.02)
            await asyncio.sleep(0.09)          # ~4 intervals
        assert prom.calls >= 2                  # actually fired on schedule


# ---------------------------------------------------------------------------
# after_turn + queue / in-flight guard
# ---------------------------------------------------------------------------


class TestAfterTurnAndFire:
    async def test_after_turn_queues_on_new_entity(self) -> None:
        tm, bg, prom = _tm(ltm=FakeLTM(known={}))  # nothing known → new
        queued = await tm.after_turn([_ent("Fresh")])
        assert queued is True
        assert len(bg.jobs) == 1
        await bg.run()
        assert prom.calls == 1

    async def test_after_turn_queues_on_block_accumulation(self) -> None:
        tm, bg, _ = _tm(
            config=TriggerConfig(on_new_entity=True, block_threshold=2),
            mtm=FakeMTM(5),
        )
        assert await tm.after_turn(entities=None) is True
        assert len(bg.jobs) == 1

    async def test_after_turn_no_trigger(self) -> None:
        tm, bg, _ = _tm(
            config=TriggerConfig(block_threshold=999),
            ltm=FakeLTM(known={"Alice": 0.99}),
            mtm=FakeMTM(1),
        )
        assert await tm.after_turn([_ent("Alice")]) is False
        assert bg.jobs == []

    async def test_inflight_guard_dedups(self) -> None:
        tm, bg, _ = _tm()
        assert tm.fire("a") is True             # queued
        assert tm.fire("b") is False            # already in flight
        assert len(bg.jobs) == 1
        await bg.run()                          # job finishes → re-arm
        assert tm.fire("c") is True             # can queue again

    async def test_fire_requeues_if_queue_rejected(self) -> None:
        tm, _, _ = _tm(bg=FakeBG(nowait_ok=False))
        assert tm.fire() is False               # queue closed/full
        # guard was released so a later attempt can still arm
        tm2, _, _ = _tm()
        assert tm2.fire() is True

    async def test_force_now_runs_immediately(self) -> None:
        tm, bg, prom = _tm()
        rep = await tm.force_now("session:abc")
        assert prom.calls == 1
        assert prom.scopes == ["session:abc"]
        assert rep.promoted_count == 1
        assert bg.jobs == []                     # bypassed the queue

    async def test_force_now_ignores_inflight_guard(self) -> None:
        tm, _, prom = _tm()
        tm.fire("queued")                        # sets in-flight
        await tm.force_now()                     # still runs
        assert prom.calls == 1


# ---------------------------------------------------------------------------
# ContinuumSession integration
# ---------------------------------------------------------------------------


class TestSessionIntegration:
    async def test_process_turn_queues_after_turn(self) -> None:
        from continuum.core.session import ContinuumSession

        seen = asyncio.Event()

        class SpyTM:
            calls = 0

            async def after_turn(self, entities: Any = None) -> bool:
                SpyTM.calls += 1
                seen.set()
                return False

        async with ContinuumSession(
            ContinuumConfig(), trigger_manager=SpyTM()
        ) as s:
            reply = await asyncio.wait_for(s.process_turn("hi"), timeout=2.0)
            assert "hi" in reply                 # response returned…
            await asyncio.wait_for(seen.wait(), timeout=2.0)  # …then trigger
            await s.drain()
        assert SpyTM.calls >= 1

    async def test_no_trigger_manager_is_noop(self) -> None:
        from continuum.core.session import ContinuumSession

        async with ContinuumSession(ContinuumConfig()) as s:
            assert s.trigger_manager is None
            assert "x" in await s.process_turn("x")   # unchanged behaviour
