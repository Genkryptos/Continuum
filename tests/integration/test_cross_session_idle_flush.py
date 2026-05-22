"""
tests/integration/test_cross_session_idle_flush.py
==================================================
End-to-end demonstration: partial STM survives a session switch when
``IdleStmFlush`` is wired in.

Scenario
--------
1. Session A: user says "remember my favorite color is blue", then goes
   idle for longer than ``idle_seconds``.
2. The watcher fires `stm.flush_to(MTM)` automatically.
3. Session A closes — its STM is gone, but the flushed content lives in
   a *shared* MTM store.
4. Session B opens (different ``session_id``) and the shared MTM is
   queried — the colour preference is still discoverable.

This test deliberately uses **trivially small** in-memory MTM and STM so
the assertion is about the *plumbing* — that the idle-flush watcher
actually copies STM rows into MTM under the right conditions — not about
the quality of any specific MTM retrieval implementation.

The Continuum framework's full retrieval pipeline (with embeddings +
Postgres + bi-temporal LTM) sits on top of these protocols; if a turn
makes it into MTM here, it would make it into LTM in production by the
same Promoter logic.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from continuum.core.config import ContinuumConfig
from continuum.core.session import ContinuumSession
from continuum.core.types import MemoryItem, MemoryTier, ProcessingState
from continuum.promotion.idle_flush import IdleStmFlush
from continuum.stores.stm import InMemorySTM

pytestmark = [pytest.mark.integration]


class _MinimalMTM:
    """
    Bare-bones MTMProtocol implementation: a list of MemoryItems.

    We don't need vector search for this test — Session B will probe
    the MTM directly to verify the colour preference made it across.
    """

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    async def add_summary(self, summary: MemoryItem) -> str:
        summary.tier = MemoryTier.MTM
        summary.processing_state = ProcessingState.UNPROCESSED
        if not summary.id:
            summary.id = str(uuid.uuid4())
        self.items.append(summary)
        return str(summary.id)

    def find(self, substring: str) -> MemoryItem | None:
        """Linear scan for the test's "did the colour survive?" check."""
        s = substring.lower()
        return next(
            (it for it in self.items if s in (it.content or "").lower()),
            None,
        )


@pytest.mark.asyncio
async def test_partial_stm_survives_session_switch() -> None:
    # ── shared cross-session MTM ────────────────────────────────────────────
    shared_mtm = _MinimalMTM()

    # ── Session A ──────────────────────────────────────────────────────────
    stm_a = InMemorySTM()
    flush_a = IdleStmFlush(
        stm=stm_a, mtm=shared_mtm,
        idle_seconds=0.15, check_interval_seconds=0.05,
    )
    session_a = ContinuumSession(
        config=ContinuumConfig(),
        stm=stm_a,
        mtm=shared_mtm,
        idle_flush=flush_a,
        session_id="user-mayank:chat-a",
    )

    await session_a.start()

    # Push a "remember this" turn directly into STM (no LLM responder
    # needed to demonstrate the plumbing).
    await stm_a.append(MemoryItem(
        id=str(uuid.uuid4()),
        content="my favorite color is blue",
        tier=MemoryTier.STM,
        session_id="user-mayank:chat-a",
        metadata={"role": "user"},
    ))

    # Sanity-check: MTM has NOT received the item yet (idle hasn't fired).
    assert shared_mtm.find("blue") is None

    # Go idle long enough for the watcher to fire.
    await asyncio.sleep(0.30)

    # The colour preference should now live in MTM.
    promoted = shared_mtm.find("blue")
    assert promoted is not None, (
        "idle-flush did not promote the partial STM row to MTM"
    )
    assert promoted.tier == MemoryTier.MTM
    assert promoted.processing_state == ProcessingState.UNPROCESSED

    # Close Session A — STM gets torn down.
    await session_a.aclose()

    # ── Session B (different session_id, same user) ─────────────────────────
    stm_b = InMemorySTM()
    session_b = ContinuumSession(
        config=ContinuumConfig(),
        stm=stm_b,
        mtm=shared_mtm,
        session_id="user-mayank:chat-b",
    )
    await session_b.start()

    # Session B has empty STM but the shared MTM still carries the fact.
    assert stm_b._sessions == {}
    promoted_b = shared_mtm.find("blue")
    assert promoted_b is not None
    assert promoted_b.id == promoted.id  # same row, not a duplicate

    await session_b.aclose()


@pytest.mark.asyncio
async def test_active_session_is_not_prematurely_flushed() -> None:
    """
    Counter-test: while the user is actively typing (touch() resets the
    clock), idle-flush must NOT prematurely promote STM.
    """
    shared_mtm = _MinimalMTM()
    stm = InMemorySTM()
    flush = IdleStmFlush(
        stm=stm, mtm=shared_mtm,
        idle_seconds=0.2, check_interval_seconds=0.05,
    )
    session = ContinuumSession(
        config=ContinuumConfig(),
        stm=stm,
        mtm=shared_mtm,
        idle_flush=flush,
        session_id="hot-conv",
    )
    await session.start()

    await stm.append(MemoryItem(
        content="user is still typing turn 1",
        tier=MemoryTier.STM,
        session_id="hot-conv",
        metadata={"role": "user"},
    ))

    # Simulate continuous activity: touch every 50 ms for 250 ms (the
    # idle window is 200 ms — but each touch resets the clock).
    deadline = asyncio.get_event_loop().time() + 0.25
    while asyncio.get_event_loop().time() < deadline:
        flush.touch()
        await asyncio.sleep(0.05)

    # No premature promotion: the in-progress conversation is preserved
    # in STM and MTM should still be empty.
    fired_count_before_close = len(shared_mtm.items)

    # NB: aclose() runs the final flush. That's expected — close means
    # we explicitly want any partial STM persisted.
    await session.aclose()
    assert fired_count_before_close == 0
