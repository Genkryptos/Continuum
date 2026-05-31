"""
tests/unit/test_idle_flush.py
=============================
Unit tests for ``continuum.promotion.idle_flush.IdleStmFlush`` — the
watcher that flushes partial STM → MTM after a configurable idle gap.

The tests run a real background task but use short ``idle_seconds``
values (50–200 ms) so the suite finishes in well under a second.
Time-sensitive assertions use a small slack buffer to stay green under
CI scheduling jitter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from continuum.promotion.idle_flush import IdleStmFlush


class _FakeSTM:
    """STMProtocol stand-in that records flush_to calls."""

    def __init__(self, *, items_per_flush: int = 3, raise_: bool = False) -> None:
        self.items_per_flush = items_per_flush
        self.raise_ = raise_
        self.calls: list[tuple[str, Any]] = []
        # If non-None on a given call, the value is returned; otherwise
        # we return items_per_flush. Reset by tests as needed.
        self.next_return: int | None = None

    async def flush_to(self, session_id: str, target: Any) -> int:
        self.calls.append((session_id, target))
        if self.raise_:
            raise RuntimeError("stm boom")
        if self.next_return is not None:
            n = self.next_return
            self.next_return = None
            return n
        return self.items_per_flush


class _FakeMTM:
    """Just an opaque sentinel — flush_to forwards the reference."""


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_zero_idle_seconds() -> None:
    with pytest.raises(ValueError):
        IdleStmFlush(stm=_FakeSTM(), mtm=_FakeMTM(), idle_seconds=0)


def test_check_interval_defaults_to_half_idle() -> None:
    f = IdleStmFlush(stm=_FakeSTM(), mtm=_FakeMTM(), idle_seconds=4)
    assert f.check_interval_seconds == pytest.approx(2.0)


def test_check_interval_floored_at_half_second() -> None:
    # idle=0.1 → naive half is 0.05; should be clamped to 0.5.
    f = IdleStmFlush(stm=_FakeSTM(), mtm=_FakeMTM(), idle_seconds=0.1)
    assert f.check_interval_seconds == 0.5


def test_explicit_check_interval_overrides_default() -> None:
    f = IdleStmFlush(
        stm=_FakeSTM(),
        mtm=_FakeMTM(),
        idle_seconds=4,
        check_interval_seconds=0.05,
    )
    assert f.check_interval_seconds == 0.05


# ---------------------------------------------------------------------------
# Idle firing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flushes_after_idle_gap() -> None:
    stm, mtm = _FakeSTM(), _FakeMTM()
    flush = IdleStmFlush(
        stm=stm,
        mtm=mtm,
        idle_seconds=0.1,
        check_interval_seconds=0.04,
    )
    await flush.start("session-a")
    # Don't touch — let the idle watcher fire at least once.
    await asyncio.sleep(0.25)
    await flush.stop()

    assert len(stm.calls) >= 1
    assert stm.calls[0] == ("session-a", mtm)


@pytest.mark.asyncio
async def test_touch_resets_idle_timer() -> None:
    """Continuous activity (touch every < idle gap) should suppress flushes."""
    stm, mtm = _FakeSTM(), _FakeMTM()
    flush = IdleStmFlush(
        stm=stm,
        mtm=mtm,
        idle_seconds=0.2,
        check_interval_seconds=0.04,
    )
    await flush.start("hot-session")
    # Hammer touch() at 25 ms intervals for 250 ms — well below 200 ms idle
    # threshold (touches reset the clock before the watcher fires).
    deadline = asyncio.get_event_loop().time() + 0.25
    while asyncio.get_event_loop().time() < deadline:
        flush.touch()
        await asyncio.sleep(0.025)
    # No flush during the activity loop is expected. Stop() WILL perform a
    # final best-effort flush; record before stopping.
    fired_during = len(stm.calls)
    await flush.stop()
    assert fired_during == 0


@pytest.mark.asyncio
async def test_on_flush_callback_fires_after_non_zero_flush() -> None:
    stm, mtm = _FakeSTM(items_per_flush=2), _FakeMTM()
    calls: list[int] = []

    async def on_flush() -> None:
        calls.append(1)

    flush = IdleStmFlush(
        stm=stm,
        mtm=mtm,
        idle_seconds=0.1,
        check_interval_seconds=0.04,
        on_flush=on_flush,
    )
    await flush.start("with-cb")
    await asyncio.sleep(0.25)
    await flush.stop()
    assert sum(calls) >= 1


@pytest.mark.asyncio
async def test_on_flush_skipped_when_nothing_to_flush() -> None:
    stm = _FakeSTM(items_per_flush=0)
    calls: list[int] = []

    async def on_flush() -> None:
        calls.append(1)

    flush = IdleStmFlush(
        stm=stm,
        mtm=_FakeMTM(),
        idle_seconds=0.1,
        check_interval_seconds=0.04,
        on_flush=on_flush,
    )
    await flush.start("empty")
    await asyncio.sleep(0.25)
    await flush.stop()
    # STM was probed at least once but the callback didn't fire (n=0).
    assert calls == []


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stm_failure_is_swallowed() -> None:
    """A broken STM mustn't crash the watcher loop."""
    stm = _FakeSTM(raise_=True)
    flush = IdleStmFlush(
        stm=stm,
        mtm=_FakeMTM(),
        idle_seconds=0.1,
        check_interval_seconds=0.04,
    )
    await flush.start("bad-stm")
    await asyncio.sleep(0.25)
    await flush.stop()
    # We called flush_to, it raised, watcher kept going (didn't crash).
    assert len(stm.calls) >= 1


@pytest.mark.asyncio
async def test_on_flush_callback_failure_swallowed() -> None:
    stm = _FakeSTM()

    async def boom() -> None:
        raise RuntimeError("cb broken")

    flush = IdleStmFlush(
        stm=stm,
        mtm=_FakeMTM(),
        idle_seconds=0.1,
        check_interval_seconds=0.04,
        on_flush=boom,
    )
    await flush.start("cb-fails")
    await asyncio.sleep(0.25)
    await flush.stop()
    # No exception propagated; we still flushed at least once.
    assert len(stm.calls) >= 1


# ---------------------------------------------------------------------------
# Explicit flush_now + final flush on stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_now_forces_immediate_flush() -> None:
    stm, mtm = _FakeSTM(items_per_flush=4), _FakeMTM()
    flush = IdleStmFlush(stm=stm, mtm=mtm, idle_seconds=10)
    # Don't even start the watcher — flush_now works standalone.
    n = await flush.flush_now("force-flush")
    assert n == 4
    assert stm.calls == [("force-flush", mtm)]


@pytest.mark.asyncio
async def test_stop_performs_final_flush() -> None:
    """
    stop() should fire one last flush so partial STM survives the
    teardown even when no idle gap fired.
    """
    stm, mtm = _FakeSTM(items_per_flush=5), _FakeMTM()
    flush = IdleStmFlush(stm=stm, mtm=mtm, idle_seconds=10)
    await flush.start("teardown")
    await asyncio.sleep(0.02)  # not enough to fire on idle
    await flush.stop()

    # The teardown flush happened — exact count depends on whether the
    # watcher ticked at all, but ≥ 1 stm.flush_to call is guaranteed.
    assert len(stm.calls) >= 1
    assert stm.calls[-1] == ("teardown", mtm)


# ---------------------------------------------------------------------------
# Start idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_start_is_noop() -> None:
    stm = _FakeSTM()
    flush = IdleStmFlush(stm=stm, mtm=_FakeMTM(), idle_seconds=10)
    await flush.start("session")
    task = flush._task
    await flush.start("session")  # second call must NOT spawn a new task
    assert flush._task is task
    await flush.stop()
