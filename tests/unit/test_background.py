"""
tests/unit/test_background.py
=============================
Unit tests for ``continuum.core.background.BackgroundQueue``.

Retry tests zero out the backoff so they run instantly. Priority ordering
is asserted at ``max_concurrent=1`` (strict ordering regime — see the module
docstring). Timing tests use small sleeps with generous wait_for bounds.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from continuum.core.background import (
    BackgroundQueue,
    DeadLetter,
    make_access_log_write,
    make_incremental_index,
    make_promotion_sweep,
)

pytestmark = pytest.mark.unit


def _q(**kw: Any) -> BackgroundQueue:
    """Queue with instant backoff unless overridden."""
    kw.setdefault("backoff_initial", 0.0)
    kw.setdefault("backoff_max", 0.0)
    return BackgroundQueue(**kw)


# ---------------------------------------------------------------------------
# Schedule & execute
# ---------------------------------------------------------------------------


class TestScheduleExecute:
    async def test_scheduled_task_runs(self) -> None:
        ran = asyncio.Event()

        async def job() -> None:
            ran.set()

        q = _q()
        async with q:
            q.schedule_nowait(job)
            await asyncio.wait_for(ran.wait(), timeout=2.0)
        assert ran.is_set()
        assert q.stats()["completed"] == 1

    async def test_schedule_async_form(self) -> None:
        seen: list[int] = []

        def make(n: int) -> Any:
            async def _t() -> None:
                seen.append(n)

            return _t

        q = _q()
        async with q:
            await q.schedule(make(1))
            await q.schedule(make(2))
            await q.drain()
        assert sorted(seen) == [1, 2]

    async def test_bare_coroutine_accepted_once(self) -> None:
        ran = asyncio.Event()

        async def _c() -> None:
            ran.set()

        q = _q()
        async with q:
            q.schedule_nowait(_c())  # a coroutine object, not a factory
            await asyncio.wait_for(ran.wait(), timeout=2.0)
        assert q.stats()["completed"] == 1

    async def test_schedule_nowait_rejected_after_stop(self) -> None:
        q = _q()
        await q.start()
        await q.stop()

        async def job() -> None:
            pass

        assert q.schedule_nowait(job) is False

    async def test_schedule_nowait_full_returns_false(self) -> None:
        q = _q(max_queue_size=1)

        async def job() -> None:
            pass

        # Not started → nothing drains; second enqueue overflows.
        assert q.schedule_nowait(job) is True
        assert q.schedule_nowait(job) is False

    async def test_bad_payload_rejected(self) -> None:
        q = _q()
        assert q.schedule_nowait(42) is False  # not callable / coroutine

    async def test_schedule_periodic_rejects_bare_coroutine(self) -> None:
        q = _q()

        async def _c() -> None:
            pass

        with pytest.raises(TypeError, match="zero-arg callable"):
            await q.schedule_periodic(_c(), interval_seconds=0.01)


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriority:
    async def test_higher_priority_runs_first(self) -> None:
        order: list[str] = []

        def make(tag: str) -> Any:
            async def _t() -> None:
                order.append(tag)

            return _t

        # concurrency=1 → strict ordering. Pre-queue before start so the
        # dispatcher sees the whole priority-ordered backlog at once.
        q = _q(max_concurrent=1)
        q.schedule_nowait(make("low"), priority=0)
        q.schedule_nowait(make("high"), priority=100)
        q.schedule_nowait(make("mid"), priority=50)

        async with q:
            await q.drain()

        assert order == ["high", "mid", "low"]

    async def test_equal_priority_is_fifo(self) -> None:
        order: list[int] = []

        def make(n: int) -> Any:
            async def _t() -> None:
                order.append(n)

            return _t

        q = _q(max_concurrent=1)
        for n in range(5):
            q.schedule_nowait(make(n), priority=10)
        async with q:
            await q.drain()
        assert order == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Delayed & periodic
# ---------------------------------------------------------------------------


class TestDelayedPeriodic:
    async def test_delayed_runs_after_delay(self) -> None:
        ran = asyncio.Event()

        async def job() -> None:
            ran.set()

        q = _q()
        async with q:
            await q.schedule_delayed(job, delay_seconds=0.1)
            assert not ran.is_set()  # not yet
            await asyncio.wait_for(ran.wait(), timeout=2.0)
        assert ran.is_set()

    async def test_periodic_runs_repeatedly(self) -> None:
        calls = 0

        async def job() -> None:
            nonlocal calls
            calls += 1

        q = _q()
        async with q:
            handle = await q.schedule_periodic(job, interval_seconds=0.05)
            await asyncio.sleep(0.28)  # ~5 intervals
            handle.cancel()
            await q.drain()

        assert calls >= 3  # fired several times on schedule

    async def test_stop_cancels_periodic(self) -> None:
        calls = 0

        async def job() -> None:
            nonlocal calls
            calls += 1

        q = _q()
        await q.start()
        await q.schedule_periodic(job, interval_seconds=0.05)
        await asyncio.sleep(0.12)
        await q.stop()
        frozen = calls
        await asyncio.sleep(0.15)
        assert calls == frozen  # no further firings after stop


# ---------------------------------------------------------------------------
# Retry & dead-letter
# ---------------------------------------------------------------------------


class TestRetry:
    async def test_failing_task_is_retried_then_succeeds(self) -> None:
        attempts = 0

        async def flaky() -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient")

        q = _q(max_retries=3)
        async with q:
            q.schedule_nowait(flaky)
            await q.drain()

        assert attempts == 3  # 2 failures + 1 success
        assert q.stats()["completed"] == 1
        assert q.stats()["failed"] == 0

    async def test_permanent_failure_goes_to_dead_letter(self) -> None:
        async def always_fail() -> None:
            raise ValueError("nope")

        q = _q(max_retries=2)
        async with q:
            q.schedule_nowait(always_fail, name="doomed")
            await q.drain()

        assert q.stats()["failed"] == 1
        dl = q.dead_letters
        assert len(dl) == 1
        assert isinstance(dl[0], DeadLetter)
        assert dl[0].name == "doomed"
        assert dl[0].attempts == 3  # 1 + max_retries
        assert "ValueError" in dl[0].error

    async def test_failing_task_does_not_crash_queue(self) -> None:
        good_ran = asyncio.Event()

        async def bad() -> None:
            raise RuntimeError("boom")

        async def good() -> None:
            good_ran.set()

        q = _q(max_retries=0)
        async with q:
            q.schedule_nowait(bad)
            q.schedule_nowait(good)
            await asyncio.wait_for(good_ran.wait(), timeout=2.0)
            await q.drain()
            # Dispatcher survived the failing task (checked while running).
            assert q.health()["ok"] is True

        assert good_ran.is_set()
        assert q.stats()["failed"] == 1
        assert q.stats()["completed"] == 1


# ---------------------------------------------------------------------------
# Stats & health
# ---------------------------------------------------------------------------


class TestStatsHealth:
    async def test_stats_accurate(self) -> None:
        async def job() -> None:
            await asyncio.sleep(0)

        q = _q()
        async with q:
            for _ in range(7):
                q.schedule_nowait(job)
            await q.drain()
            s = q.stats()

        assert s["completed"] == 7
        assert s["failed"] == 0
        assert s["pending"] == 0
        assert s["running"] == 0

    async def test_health_lifecycle(self) -> None:
        q = _q()
        assert q.health()["ok"] is False  # not started
        await q.start()
        assert q.health()["ok"] is True
        assert q.health()["detail"] == "processing"
        await q.stop()
        assert q.health()["ok"] is False  # stopped

    async def test_stats_keys_present(self) -> None:
        q = _q()
        assert set(q.stats()) == {"pending", "running", "completed", "failed", "dead_letter"}

    async def test_stop_drains_queued_tasks(self) -> None:
        done: list[int] = []

        def make(n: int) -> Any:
            async def _t() -> None:
                done.append(n)

            return _t

        q = _q(max_concurrent=2)
        await q.start()
        for n in range(6):
            q.schedule_nowait(make(n))
        await q.stop(drain=True)  # must finish all 6 before returning
        assert sorted(done) == [0, 1, 2, 3, 4, 5]

    async def test_stop_idempotent_and_safe_without_start(self) -> None:
        q = _q()
        await q.stop()  # never started → no-op
        await q.start()
        await q.stop()
        await q.stop()  # double stop → safe


# ---------------------------------------------------------------------------
# Built-in task factories
# ---------------------------------------------------------------------------


class TestBuiltins:
    async def test_access_log_write_batches(self) -> None:
        written: list[list[dict[str, Any]]] = []

        async def sink(rows: Any) -> None:
            written.append(list(rows))

        entries = [{"node_id": 1}, {"node_id": 2}]
        q = _q()
        async with q:
            q.schedule_nowait(make_access_log_write(sink, entries))
            await q.drain()

        assert written == [[{"node_id": 1}, {"node_id": 2}]]

    async def test_access_log_write_noop_when_empty(self) -> None:
        called = False

        async def sink(rows: Any) -> None:
            nonlocal called
            called = True

        q = _q()
        async with q:
            q.schedule_nowait(make_access_log_write(sink, []))
            await q.drain()
        assert called is False
        assert q.stats()["completed"] == 1

    async def test_promotion_sweep_invokes_promoter(self) -> None:
        class FakePromoter:
            def __init__(self) -> None:
                self.scope: str | None = None

            async def promote(self, scope: str) -> Any:
                self.scope = scope
                return type("R", (), {"promoted_count": 2, "skipped_count": 1})()

        promoter = FakePromoter()
        q = _q()
        async with q:
            q.schedule_nowait(make_promotion_sweep(promoter, "session:abc"))
            await q.drain()
        assert promoter.scope == "session:abc"
        assert q.stats()["completed"] == 1

    async def test_incremental_index_invokes_indexer(self) -> None:
        indexed: list[list[str]] = []

        async def indexer(paths: Any) -> None:
            indexed.append(list(paths))

        q = _q()
        async with q:
            q.schedule_nowait(make_incremental_index(indexer, ["a.py", "b.py"]))
            await q.drain()
        assert indexed == [["a.py", "b.py"]]


# ---------------------------------------------------------------------------
# Session integration (process_turn returns immediately; bg writes happen)
# ---------------------------------------------------------------------------


class TestSessionIntegration:
    async def test_process_turn_does_not_block_on_bookkeeping(self) -> None:
        from continuum.core.config import ContinuumConfig
        from continuum.core.session import ContinuumSession

        release = asyncio.Event()
        logged = asyncio.Event()

        class SlowSTM:
            async def append(self, item: Any) -> None: ...
            async def get_recent(self, sid: str, n: int = 10) -> list[Any]:
                return []

        async with ContinuumSession(ContinuumConfig(), stm=SlowSTM()) as s:

            async def slow_log(*_a: Any, **_k: Any) -> None:
                await release.wait()  # block until the test releases it
                logged.set()

            s._log_access = slow_log  # type: ignore[method-assign]

            # Returns even though the bg log job is still blocked.
            reply = await asyncio.wait_for(s.process_turn("hi"), timeout=2.0)
            assert "hi" in reply
            assert not logged.is_set()  # bookkeeping hasn't run yet

            release.set()  # let the bg job proceed
            await asyncio.wait_for(logged.wait(), timeout=2.0)
            assert s.background.stats()["completed"] >= 1
