"""
continuum/promotion/idle_flush.py
=================================
``IdleStmFlush`` — proactively flush partial STM into MTM when the user
goes idle, so cross-session retrieval can still see what the user said.

Why this exists
---------------
:class:`continuum.stores.stm.InMemorySTM` only evicts when its token /
message buffer overflows. Real conversations frequently say things
worth remembering ("my favorite colour is blue", "remember I prefer
Python over Java") *long* before the buffer is anywhere near full —
and then the user switches sessions. Without an idle flush, those
turns die with the session: never compacted to MTM, never promoted to
LTM, invisible to the next session's retrieval.

How it works
------------
1. The watcher records a ``last_activity_at`` timestamp.
2. The session calls :meth:`touch` on every ``process_turn`` /
   retrieval / explicit user activity.
3. A background task wakes every ``idle_seconds / 2`` and, if the gap
   since the last touch is ≥ ``idle_seconds``, calls
   ``stm.flush_to(session_id, mtm)``. The flushed items become
   ``UNPROCESSED`` MTM blocks, eligible for normal promoter pickup.
4. An optional ``on_flush`` callback fires after a non-zero flush so
   callers can chain a Promoter run that pushes high-importance items
   straight into LTM.

Failure handling
----------------
* STM-or-MTM exceptions are logged and swallowed — the watcher must
  never crash the host event loop.
* The activity clock is reset after a fired flush so we don't
  immediately try again on the next tick.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)


class IdleStmFlush:
    """
    Background watcher that flushes partial STM → MTM after idle.

    Parameters
    ----------
    stm:
        STMProtocol with ``async flush_to(session_id, target) -> int``.
        Compatible with :class:`InMemorySTM` and :class:`PostgresSTM`.
    mtm:
        MTMProtocol — the flush target.
    idle_seconds:
        Idle gap (in seconds) before a partial STM is flushed.
        Defaults to 10 s — short enough to catch tab-switches, long
        enough to avoid mid-sentence flushes.
    on_flush:
        Optional ``async () -> None`` callback. Fires after a *non-zero*
        flush. Typical use: ``promoter.promote(scope=session_scope)``
        to fast-track high-importance facts into LTM.
    check_interval_seconds:
        How often the watcher wakes. Default ``idle_seconds / 2``;
        smaller values catch idle gaps faster at the cost of more
        wake-ups.
    """

    def __init__(
        self,
        *,
        stm: Any,
        mtm: Any,
        idle_seconds: float = 10.0,
        on_flush: Callable[[], Awaitable[None]] | None = None,
        check_interval_seconds: float | None = None,
    ) -> None:
        if idle_seconds <= 0:
            raise ValueError("idle_seconds must be positive")
        self.stm = stm
        self.mtm = mtm
        self.idle_seconds = float(idle_seconds)
        self.on_flush = on_flush
        self.check_interval_seconds = (
            float(check_interval_seconds)
            if check_interval_seconds is not None
            else max(0.5, idle_seconds / 2.0)
        )
        self._last_activity_at: float = time.monotonic()
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    # ── public API ──────────────────────────────────────────────────────────

    def touch(self) -> None:
        """Record activity. Call from every turn / retrieve / user event."""
        self._last_activity_at = time.monotonic()

    async def start(self, session_id: str) -> None:
        """Spawn the background watcher. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(
            self._watch(session_id), name=f"idle-stm-flush:{session_id}"
        )

    async def stop(self) -> None:
        """
        Stop the watcher. Performs one final flush so partial STM
        survives the shutdown.
        """
        self._stopped = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # Final flush — best-effort.
        try:
            await self._flush_once(self._session_id_of_last_run)
        except Exception:
            log.exception("idle-flush final flush failed (swallowed)")

    async def flush_now(self, session_id: str) -> int:
        """Force an immediate flush regardless of the idle gap."""
        return await self._flush_once(session_id)

    # ── internals ───────────────────────────────────────────────────────────

    _session_id_of_last_run: str = ""

    async def _watch(self, session_id: str) -> None:
        self._session_id_of_last_run = session_id
        while not self._stopped:
            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                return
            elapsed = time.monotonic() - self._last_activity_at
            if elapsed < self.idle_seconds:
                continue
            try:
                n = await self._flush_once(session_id)
                if n:
                    log.debug(
                        "idle-flush: %d items %s → MTM (idle %.1fs)",
                        n,
                        session_id,
                        elapsed,
                    )
            except Exception:
                log.exception("idle-flush iteration failed (swallowed)")
            # Reset the clock — don't try again until there's fresh activity
            self._last_activity_at = time.monotonic()

    async def _flush_once(self, session_id: str) -> int:
        if not session_id:
            return 0
        try:
            n = int(await self.stm.flush_to(session_id, self.mtm))
        except Exception:
            log.exception("idle-flush stm.flush_to failed")
            return 0
        if n > 0 and self.on_flush is not None:
            try:
                await self.on_flush()
            except Exception:
                log.exception("idle-flush on_flush callback failed")
        return n


__all__ = ["IdleStmFlush"]
