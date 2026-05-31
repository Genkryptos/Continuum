"""
tests/unit/evals/test_adaptive_throttle.py
==========================================
Unit tests for ``evals.longmemeval.bootstrap_ollama._AdaptiveThrottle``
— the AIMD client-side rate limiter that lets a run self-tune to a
provider's real sustainable rate instead of retry-storming a fixed pace.

All tests are pure-logic (no network, no sleeps beyond a few ms).
"""

from __future__ import annotations

import asyncio

import pytest

from evals.longmemeval.bootstrap_ollama import _AdaptiveThrottle

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_interval_matches_rpm() -> None:
    # 30 rpm → one request every 2 s.
    t = _AdaptiveThrottle(rpm=30)
    assert t.interval == pytest.approx(2.0)


def test_rpm_floor_at_one() -> None:
    # rpm=0 must not divide-by-zero — clamped to 1 → 60 s interval.
    t = _AdaptiveThrottle(rpm=0)
    assert t.interval == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Multiplicative increase on 429
# ---------------------------------------------------------------------------


def test_on_429_widens_interval() -> None:
    t = _AdaptiveThrottle(rpm=60)  # target 1.0 s
    assert t.interval == pytest.approx(1.0)
    t.on_429()
    assert t.interval == pytest.approx(1.5)  # ×1.5
    t.on_429()
    assert t.interval == pytest.approx(2.25)  # ×1.5 again


def test_on_429_capped_at_max_interval() -> None:
    t = _AdaptiveThrottle(rpm=60, max_interval=3.0)
    for _ in range(20):
        t.on_429()
    assert t.interval == pytest.approx(3.0)  # never exceeds the cap


# ---------------------------------------------------------------------------
# Additive decrease on sustained success
# ---------------------------------------------------------------------------


def test_success_streak_narrows_interval_back() -> None:
    t = _AdaptiveThrottle(rpm=60, recover_after=5)  # target 1.0 s
    t.on_429()
    t.on_429()
    widened = t.interval  # 2.25 s
    assert widened > 1.0

    # 4 successes — not enough to trigger recovery yet.
    for _ in range(4):
        t.on_success()
    assert t.interval == pytest.approx(widened)

    # 5th success crosses recover_after → narrow ×0.9.
    t.on_success()
    assert t.interval == pytest.approx(widened * 0.9)


def test_success_never_narrows_below_target() -> None:
    t = _AdaptiveThrottle(rpm=60, recover_after=1)  # target 1.0 s
    # Already at target — successes must not push below it.
    for _ in range(50):
        t.on_success()
    assert t.interval == pytest.approx(1.0)


def test_429_resets_success_streak() -> None:
    t = _AdaptiveThrottle(rpm=60, recover_after=3)
    t.on_429()  # interval 1.5
    t.on_success()
    t.on_success()  # streak = 2, one short of recovery
    t.on_429()  # streak reset, interval 2.25
    t.on_success()
    t.on_success()  # streak only 2 again — no narrow
    assert t.interval == pytest.approx(2.25)


# ---------------------------------------------------------------------------
# await_slot paces requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_await_slot_paces_calls() -> None:
    # 600 rpm → 0.1 s interval. Three calls should span ≥ 0.2 s
    # (first is immediate, then two gaps of 0.1 s).
    t = _AdaptiveThrottle(rpm=600)
    loop = asyncio.get_event_loop()
    start = loop.time()
    await t.await_slot()
    await t.await_slot()
    await t.await_slot()
    elapsed = loop.time() - start
    assert elapsed >= 0.18  # ~0.2 s with scheduling slack


@pytest.mark.asyncio
async def test_await_slot_widened_interval_paces_slower() -> None:
    t = _AdaptiveThrottle(rpm=600)  # 0.1 s base
    t.on_429()  # → 0.15 s
    t.on_429()  # → 0.225 s
    loop = asyncio.get_event_loop()
    start = loop.time()
    await t.await_slot()
    await t.await_slot()  # one gap at the widened pace
    elapsed = loop.time() - start
    assert elapsed >= 0.20  # ~0.225 s gap
