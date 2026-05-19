"""
continuum/core/background.py
============================
``BackgroundQueue`` — a priority, retrying, observable async task queue that
keeps slow bookkeeping (access-log writes, promotion sweeps, incremental
re-indexing) off the request path.

Design
------
* **asyncio.TaskGroup** (3.11+) owns every in-flight task: a single
  dispatcher loop pulls the highest-priority item, acquires a concurrency
  slot, and spawns the runner inside the group.  Shutdown is structured —
  the group joins all survivors before the queue reports stopped.
* **Priority** — ``asyncio.PriorityQueue`` keyed by ``(-priority, seq)`` so
  *higher* ``priority`` runs first; ``seq`` is a monotonic FIFO tiebreaker
  (and avoids ever comparing payloads).  Ordering is strict at
  ``max_concurrent=1`` and best-effort above that (standard for worker pools
  — once N tasks run in parallel their *completion* order is not ordered).
* **Resilience** — each task runs under tenacity exponential backoff
  (``max_retries`` extra attempts).  Exhausted tasks go to a bounded
  **dead-letter queue** and increment ``failed``; a failing task never
  crashes the dispatcher or other tasks.
* **Observability** — :meth:`stats` (pending/running/completed/failed/
  dead-letter) and :meth:`health` (is the dispatcher alive and consuming).

Scheduling API
--------------
    await q.schedule(factory, priority=10)          # enqueue (async, backpressured)
    q.schedule_nowait(factory)                      # hot-path, never blocks/raises
    await q.schedule_delayed(factory, delay_seconds=5)
    handle = await q.schedule_periodic(factory, interval_seconds=30)

``factory`` is a **zero-arg callable returning an awaitable** (preferred —
it can be retried/repeated).  A bare coroutine is also accepted for
one-shot :meth:`schedule` (it cannot be retried — a fresh coroutine is
needed per attempt — so a warning is logged if it fails).

Built-in task factories
-----------------------
:func:`make_access_log_write`, :func:`make_promotion_sweep`,
:func:`make_incremental_index` return ready-to-schedule factories for the
three canonical Continuum background jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import traceback
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

#: A task is a zero-arg callable returning an awaitable (re-callable → retryable).
#: ``schedule`` also tolerates a bare coroutine (one-shot, not retryable).
TaskFactory = Callable[[], Awaitable[Any]]


class _Shutdown:
    """Singleton sentinel that tells the dispatcher to stop."""

    __slots__ = ()


_SHUTDOWN = _Shutdown()
#: Sort key that always sorts AFTER any real task ((-priority, seq)).
_SENTINEL_KEY = (math.inf, math.inf)


@dataclass(order=True)
class _Queued:
    """Internal heap entry: ordered by (neg_priority, seq) only."""

    neg_priority: float
    seq: int
    factory: TaskFactory = field(compare=False)
    name: str = field(compare=False)


@dataclass
class DeadLetter:
    """A task that exhausted all retries."""

    name: str
    attempts: int
    error: str
    traceback: str
    failed_at: float = field(default_factory=time.time)


class BackgroundQueue:
    """
    Priority async task queue with retry, dead-lettering and metrics.

    Parameters
    ----------
    max_concurrent:
        Maximum tasks executing simultaneously (default 10).
    max_queue_size:
        Bound on pending items; ``0`` = unbounded.  ``schedule`` awaits when
        full (backpressure); ``schedule_nowait`` drops + returns ``False``.
    max_retries:
        Extra attempts after the first failure (default 3 → up to 4 runs).
    backoff_initial / backoff_max:
        Exponential-jitter backoff bounds, in seconds.
    dead_letter_maxlen:
        Ring-buffer size for permanently-failed tasks.
    name:
        Used in task names / logs.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 10,
        max_queue_size: int = 1000,
        max_retries: int = 3,
        backoff_initial: float = 0.2,
        backoff_max: float = 10.0,
        dead_letter_maxlen: int = 100,
        name: str = "continuum-bg",
    ) -> None:
        self.name = name
        self._max_concurrent = max(1, max_concurrent)
        self._max_retries = max(0, max_retries)
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

        self._pq: asyncio.PriorityQueue[tuple[float, int, Any]] = asyncio.PriorityQueue(
            maxsize=max(0, max_queue_size)
        )
        self._sem = asyncio.Semaphore(self._max_concurrent)
        self._seq = 0

        self._run_task: asyncio.Task[None] | None = None
        self._timers: set[asyncio.Task[None]] = set()
        # Accept (queue) tasks before start(); the dispatcher drains them
        # once running. stop() flips this off so nothing lands post-shutdown.
        self._accepting = True
        self._started = False

        # Metrics
        self._running = 0
        self._completed = 0
        self._failed = 0
        self._dlq: deque[DeadLetter] = deque(maxlen=dead_letter_maxlen)

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Idempotently launch the dispatcher loop."""
        if self._started:
            return
        self._accepting = True
        self._started = True
        self._run_task = asyncio.create_task(self._run(), name=f"{self.name}:dispatcher")
        log.debug("BackgroundQueue '%s' started (concurrency=%d)", self.name, self._max_concurrent)

    async def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        """
        Stop accepting work and shut the dispatcher down.

        ``drain=True`` lets queued + in-flight tasks finish (bounded by
        *timeout*); ``drain=False`` cancels promptly.  Periodic / delayed
        timers are always cancelled first so they cannot re-enqueue.
        """
        if not self._started:
            return
        self._accepting = False
        self._started = False

        # Kill repeating/delayed schedulers so nothing new lands after drain.
        for t in list(self._timers):
            t.cancel()
        for t in list(self._timers):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._timers.clear()

        if self._run_task is None:
            return

        if drain:
            # Sentinel sorts last → all real tasks dispatched first.
            self._seq += 1
            self._pq.put_nowait((_SENTINEL_KEY[0], self._seq, _SHUTDOWN))
            try:
                await asyncio.wait_for(self._run_task, timeout=timeout)
            except TimeoutError:
                log.warning(
                    "BackgroundQueue '%s' drain exceeded %.1fs — cancelling",
                    self.name,
                    timeout,
                )
                self._run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._run_task
        else:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._run_task

        self._run_task = None
        log.debug("BackgroundQueue '%s' stopped", self.name)

    async def __aenter__(self) -> BackgroundQueue:
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.stop()

    # ── scheduling ──────────────────────────────────────────────────────────

    def _normalize(self, schedulable: Any, name: str | None) -> tuple[TaskFactory, str]:
        """Coerce a factory-or-coroutine into a ``(factory, name)`` pair."""
        if asyncio.iscoroutine(schedulable):
            coro = schedulable
            label: str = name or str(getattr(coro, "__qualname__", "coroutine"))

            async def _once() -> Any:
                # A coroutine object can only be awaited once; retrying it
                # would raise "cannot reuse already awaited coroutine".
                return await coro

            return _once, label
        if callable(schedulable):
            label = name or str(
                getattr(schedulable, "__qualname__", schedulable.__class__.__name__)
            )
            return schedulable, label
        raise TypeError(
            "schedule() expects a zero-arg callable returning an awaitable, "
            f"or a coroutine; got {type(schedulable).__name__}"
        )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def schedule(
        self,
        schedulable: Any,
        priority: int = 0,
        *,
        name: str | None = None,
    ) -> None:
        """
        Enqueue a task (async; awaits if the queue is full — backpressure).

        Higher *priority* runs sooner.  Raises ``RuntimeError`` if the queue
        is not accepting work (stopped).
        """
        factory, label = self._normalize(schedulable, name)
        if not self._accepting:
            raise RuntimeError(f"BackgroundQueue '{self.name}' is not accepting tasks")
        seq = self._next_seq()
        await self._pq.put((-float(priority), seq, _Queued(-float(priority), seq, factory, label)))

    def schedule_nowait(
        self,
        schedulable: Any,
        priority: int = 0,
        *,
        name: str | None = None,
    ) -> bool:
        """
        Non-blocking enqueue for hot paths.  Never raises, never blocks.

        Returns ``False`` (and logs) if the queue is closed or full so the
        caller's latency-critical path degrades instead of stalling.
        """
        try:
            factory, label = self._normalize(schedulable, name)
        except TypeError:
            log.exception("schedule_nowait rejected a bad payload")
            return False
        if not self._accepting:
            log.debug("BackgroundQueue '%s' closed — dropping %s", self.name, label)
            return False
        seq = self._next_seq()
        try:
            self._pq.put_nowait(
                (-float(priority), seq, _Queued(-float(priority), seq, factory, label))
            )
            return True
        except asyncio.QueueFull:
            log.warning("BackgroundQueue '%s' full — dropping %s", self.name, label)
            return False

    async def schedule_delayed(
        self,
        schedulable: Any,
        delay_seconds: float,
        priority: int = 0,
        *,
        name: str | None = None,
    ) -> asyncio.Task[None]:
        """Schedule *schedulable* once, after *delay_seconds*."""
        factory, label = self._normalize(schedulable, name)

        async def _delayed() -> None:
            try:
                await asyncio.sleep(delay_seconds)
                self.schedule_nowait(factory, priority, name=label)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("delayed scheduler '%s' failed", label)

        return self._spawn_timer(_delayed(), f"{self.name}:delay:{label}")

    async def schedule_periodic(
        self,
        schedulable: Any,
        interval_seconds: float,
        priority: int = 0,
        *,
        name: str | None = None,
    ) -> asyncio.Task[None]:
        """
        Re-schedule *schedulable* every *interval_seconds* until cancelled.

        Requires a factory (a bare coroutine cannot be re-run); the returned
        task can be cancelled individually, and :meth:`stop` cancels it too.
        """
        if asyncio.iscoroutine(schedulable):
            schedulable.close()
            raise TypeError(
                "schedule_periodic requires a zero-arg callable "
                "(a coroutine cannot be awaited more than once)"
            )
        factory, label = self._normalize(schedulable, name)

        async def _periodic() -> None:
            try:
                while True:
                    await asyncio.sleep(interval_seconds)
                    self.schedule_nowait(factory, priority, name=label)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("periodic scheduler '%s' crashed", label)

        return self._spawn_timer(_periodic(), f"{self.name}:every:{label}")

    def _spawn_timer(self, coro: Coroutine[Any, Any, None], task_name: str) -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.create_task(coro, name=task_name)
        self._timers.add(task)
        task.add_done_callback(self._timers.discard)
        return task

    # ── dispatcher / execution ──────────────────────────────────────────────

    async def _run(self) -> None:
        """Pull highest-priority items and run them inside one TaskGroup."""
        try:
            async with asyncio.TaskGroup() as tg:
                while True:
                    _, _, item = await self._pq.get()
                    if item is _SHUTDOWN:
                        self._pq.task_done()
                        break
                    # Backpressure: don't dispatch past the concurrency cap.
                    await self._sem.acquire()
                    tg.create_task(
                        self._execute(item),
                        name=f"{self.name}:{item.name}",
                    )
        except* Exception as eg:  # pragma: no cover - _execute swallows its own
            log.error("BackgroundQueue '%s' dispatcher error: %r", self.name, eg)

    async def _execute(self, item: _Queued) -> None:
        """Run one task with retry; never propagate (would cancel siblings)."""
        self._running += 1
        try:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._max_retries + 1),
                    wait=wait_exponential_jitter(
                        initial=self._backoff_initial, max=self._backoff_max
                    ),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        await item.factory()
                self._completed += 1
            except Exception as exc:
                self._failed += 1
                self._dlq.append(
                    DeadLetter(
                        name=item.name,
                        attempts=self._max_retries + 1,
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                )
                log.error(
                    "background task '%s' failed permanently after %d attempts: %r",
                    item.name,
                    self._max_retries + 1,
                    exc,
                )
        finally:
            self._running -= 1
            self._sem.release()
            self._pq.task_done()

    # ── flush / introspection ───────────────────────────────────────────────

    async def drain(self) -> None:
        """Block until every queued task has finished (test/flush aid)."""
        await self._pq.join()

    def stats(self) -> dict[str, int]:
        """Counts: ``pending``, ``running``, ``completed``, ``failed``,
        ``dead_letter``."""
        return {
            "pending": self._pq.qsize(),
            "running": self._running,
            "completed": self._completed,
            "failed": self._failed,
            "dead_letter": len(self._dlq),
        }

    def health(self) -> dict[str, Any]:
        """
        ``{"ok": bool, "running": bool, "detail": str}``.

        ``ok`` is True while the dispatcher task is alive and consuming.
        If the dispatcher died with an exception, ``ok`` is False and
        ``detail`` carries the reason.
        """
        if not self._started or self._run_task is None:
            return {"ok": False, "running": False, "detail": "not started"}
        if self._run_task.done():
            exc = self._run_task.exception() if not self._run_task.cancelled() else None
            return {
                "ok": False,
                "running": False,
                "detail": f"dispatcher stopped: {exc!r}" if exc else "dispatcher cancelled",
            }
        return {
            "ok": True,
            "running": self._running > 0 or self._pq.qsize() > 0,
            "detail": "processing",
        }

    @property
    def dead_letters(self) -> list[DeadLetter]:
        """Snapshot of the dead-letter ring buffer (oldest → newest)."""
        return list(self._dlq)


# ============================================================================
# Built-in task factories
# ============================================================================


def make_access_log_write(
    sink: Callable[[Sequence[dict[str, Any]]], Awaitable[Any]],
    entries: Sequence[dict[str, Any]],
) -> TaskFactory:
    """
    Batch-write access-log *entries* via *sink* (``async (rows) -> None``).

    Returns a factory so the write is retried atomically on transient DB
    errors; a no-op when there are no entries.
    """

    async def _task() -> None:
        if not entries:
            return
        await sink(list(entries))

    return _task


def make_promotion_sweep(
    promoter: Any,
    scope: str,
) -> TaskFactory:
    """
    Run ``promoter.promote(scope)`` over unprocessed MTM in *scope*.

    *promoter* is a ``PromoterProtocol`` (duck-typed to avoid an import
    cycle). The :class:`PromotionReport` it returns is logged, not returned —
    background tasks have no caller to hand it to.
    """

    async def _task() -> None:
        report = await promoter.promote(scope)
        log.info(
            "promotion_sweep scope=%s promoted=%s skipped=%s",
            scope,
            getattr(report, "promoted_count", "?"),
            getattr(report, "skipped_count", "?"),
        )

    return _task


def make_incremental_index(
    indexer: Callable[[Sequence[str]], Awaitable[Any]],
    paths: Sequence[str],
) -> TaskFactory:
    """
    Re-index changed *paths* via *indexer* (Continuum-Code hook).

    No-op when *paths* is empty so callers can schedule unconditionally.
    """

    async def _task() -> None:
        if not paths:
            return
        await indexer(list(paths))

    return _task


__all__ = [
    "BackgroundQueue",
    "DeadLetter",
    "TaskFactory",
    "make_access_log_write",
    "make_promotion_sweep",
    "make_incremental_index",
]
