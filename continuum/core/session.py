"""
continuum/core/session.py
=========================
``ContinuumSession`` — the async-first orchestration entry point that ties
STM, MTM, LTM, the Retriever, Promoter and Optimizer into a single object an
agent can drive with three verbs:

    async with ContinuumSession(config) as session:
        response = await session.process_turn("What's the status of project X?")
        report   = await session.checkpoint()           # MTM → LTM promotion
        hits     = await session.search("project X", k=5)

Design principles
-----------------
* **Async-first, sync-friendly.**  Every public coroutine has a thin
  ``*_sync`` wrapper built on ``asyncio.run`` for scripts / notebooks /
  legacy call sites.  The wrappers refuse to run inside an existing event
  loop (that is a programming error) and say so clearly.

* **The response path never blocks on bookkeeping.**  Access-log writes,
  scheduled promotions and incremental indexing are pushed onto a shared
  :class:`continuum.core.background.BackgroundQueue` (priority, retrying,
  ``asyncio.TaskGroup``-supervised) via a non-blocking ``schedule_nowait``.
  ``process_turn`` returns as soon as the reply is ready; inspect the queue
  through ``session.background.stats()`` / ``.health()``.

* **Graceful degradation.**  Retrieval / promotion failures are retried
  with exponential backoff (tenacity); if they still fail the session logs
  the error and continues with a degraded-but-correct result rather than
  raising into the agent's hot path.

* **Components are optional.**  Only ``config`` is required.  A missing STM
  is replaced with an in-memory one; a missing retriever/promoter/optimizer
  simply disables that capability (search falls back to raw STM recency,
  checkpoint becomes a no-op returning an empty report).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from continuum.core.background import BackgroundQueue
from continuum.core.config import ContinuumConfig
from continuum.core.protocols import (
    LTMProtocol,
    MTMProtocol,
    OptimizerProtocol,
    PromoterProtocol,
    RetrieverProtocol,
    STMProtocol,
)
from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    PromotionReport,
    Query,
    TokenBudget,
)

log = logging.getLogger(__name__)

_T = TypeVar("_T")

# A background job is any zero-arg coroutine factory.  We store factories
# (not coroutines) so a job that is dropped on shutdown never emits a
# "coroutine was never awaited" warning.
BackgroundJob = Callable[[], Awaitable[Any]]

# A responder turns (user_message, retrieved_context) into a reply string.
# Inject your real LLM call here; the default is a deterministic stub so
# the framework is usable end-to-end before an LLM client is wired in.
Responder = Callable[[str, ContextBundle | None], Awaitable[str]]


async def _default_responder(user_message: str, ctx: ContextBundle | None) -> str:
    """
    Stub responder used when no LLM is injected.

    It echoes the question and how much context was assembled so that
    ``process_turn`` returns something meaningful (and testable) without a
    network call or API key.
    """
    n = len(ctx.items) if ctx else 0
    used = ctx.tokens_used if ctx else 0
    return (
        f"[continuum-stub] You said: {user_message!r}. "
        f"Assembled {n} memory item(s) using {used} context token(s). "
        f"Inject a real `responder` to generate model replies."
    )


class ContinuumSession:
    """
    Stateful, async-first session over the Continuum memory tiers.

    Parameters
    ----------
    config:
        The only required argument.  Drives token budgets, retry policy and
        component defaults.
    stm / mtm / ltm / retriever / promoter / optimizer:
        Concrete protocol implementations.  Any may be omitted; see the
        module docstring for degradation behaviour.  ``stm`` defaults to an
        in-memory implementation so the session is usable out of the box.
    responder:
        ``async (user_message, context) -> reply``.  Inject your LLM here.
    session_id / agent_id / user_id:
        Routing identifiers stamped onto every ``MemoryItem`` and ``Query``.
    background_workers:
        Number of concurrent background workers (default 2).
    queue_maxsize:
        Bounded queue size; when full, new background jobs are dropped with
        a warning rather than blocking the response path (default 1000).
    """

    def __init__(
        self,
        config: ContinuumConfig,
        stm: STMProtocol | None = None,
        mtm: MTMProtocol | None = None,
        ltm: LTMProtocol | None = None,
        retriever: RetrieverProtocol | None = None,
        promoter: PromoterProtocol | None = None,
        optimizer: OptimizerProtocol | None = None,
        *,
        responder: Responder | None = None,
        trigger_manager: Any | None = None,
        idle_flush: Any | None = None,
        session_id: str = "default",
        agent_id: str | None = None,
        user_id: str | None = None,
        background_workers: int = 2,
        queue_maxsize: int = 1000,
        shutdown_timeout: float = 10.0,
    ) -> None:
        self.config = config

        # STM gets an in-memory default so the session works with just config.
        if stm is None:
            from continuum.stores.stm import InMemorySTM

            stm = InMemorySTM()
        self.stm = stm
        self.mtm = mtm
        self.ltm = ltm
        self.retriever = retriever
        self.promoter = promoter
        self.optimizer = optimizer

        self._responder: Responder = responder or _default_responder
        # Optional: a continuum.promotion.TriggerManager. When set,
        # process_turn queues its after_turn() check off the response path.
        self.trigger_manager = trigger_manager
        # Optional: a continuum.promotion.IdleStmFlush watcher. When set,
        # partial STM is flushed to MTM after a configurable idle gap so
        # cross-session retrieval sees it even when the buffer never
        # fills. The watcher is started in :meth:`start` and stopped in
        # :meth:`aclose`.
        self.idle_flush = idle_flush
        self.session_id = session_id
        self.agent_id = agent_id
        self.user_id = user_id

        # ── Retry policy (read from config where available) ──────────────────
        self._max_attempts = max(1, int(getattr(config.database, "max_retries", 3)))
        self._backoff_initial = 0.25
        self._backoff_max = 5.0

        # ── Background task plumbing ─────────────────────────────────────────
        # All non-blocking bookkeeping (access-log writes, incremental
        # indexing, scheduled promotions) runs through this shared queue so
        # process_turn never waits on it. Exposed as ``session.background``
        # for stats()/health()/manual scheduling.
        self.background = BackgroundQueue(
            max_concurrent=max(1, background_workers),
            max_queue_size=max(1, queue_maxsize),
            max_retries=2,  # bookkeeping is best-effort; a couple of retries
            backoff_initial=0.1,
            backoff_max=2.0,
            name="continuum-session",
        )
        self._shutdown_timeout = shutdown_timeout
        self._started = False

    # ────────────────────────────────────────────────────────────────────────
    # Async context manager / lifecycle
    # ────────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> ContinuumSession:
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def start(self) -> None:
        """
        Idempotently bring the session online.

        Bootstraps any component that exposes ``ensure_schema`` (e.g. the
        Postgres stores create their tables here) and launches the background
        worker pool under an ``asyncio.TaskGroup`` supervisor.
        """
        if self._started:
            return

        # Bootstrap connection pools / schemas where supported.
        for component in (self.stm, self.mtm, self.ltm):
            ensure = getattr(component, "ensure_schema", None)
            if ensure is not None:
                try:
                    await ensure()
                except Exception:
                    log.exception(
                        "ensure_schema failed for %s — continuing degraded",
                        type(component).__name__,
                    )

        await self.background.start()

        # Spin up the idle-flush watcher (if wired) so partial STM gets
        # promoted automatically when the user goes quiet.
        if self.idle_flush is not None:
            try:
                await self.idle_flush.start(self.session_id)
            except Exception:
                log.exception("idle_flush.start failed — continuing without watcher")

        self._started = True
        log.debug("ContinuumSession started")

    async def aclose(self) -> None:
        """
        Drain in-flight background work and shut the session down cleanly.

        Delegates draining to :class:`BackgroundQueue` (queued jobs run
        before the dispatcher exits, bounded by ``shutdown_timeout``), then
        releases components exposing ``aclose``/``close``.
        """
        if not self._started:
            return
        self._started = False

        # Stop the idle-flush watcher first — its stop() performs one
        # final best-effort flush so partial STM still survives the
        # shutdown even when no background queue work is queued.
        if self.idle_flush is not None:
            try:
                await self.idle_flush.stop()
            except Exception:
                log.exception("idle_flush.stop failed (swallowed)")

        await self.background.stop(drain=True, timeout=self._shutdown_timeout)

        # Release component connection pools if they expose a closer.
        for component in (self.stm, self.mtm, self.ltm, self.retriever):
            for closer_name in ("aclose", "close"):
                closer = getattr(component, closer_name, None)
                if closer is None:
                    continue
                try:
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    log.exception(
                        "error closing %s.%s",
                        type(component).__name__,
                        closer_name,
                    )
                break

        log.debug("ContinuumSession closed")

    # ────────────────────────────────────────────────────────────────────────
    # Background task queue (delegated to BackgroundQueue)
    # ────────────────────────────────────────────────────────────────────────

    def _enqueue(self, job: BackgroundJob, *, priority: int = 0) -> bool:
        """
        Schedule *job* on the background queue without ever blocking.

        Returns ``False`` (and logs) if the session is not started or the
        queue is saturated, so the response path degrades instead of
        stalling.
        """
        if not self._started:
            log.debug("dropping background job — session not started")
            return False
        return self.background.schedule_nowait(job, priority=priority)

    async def drain(self) -> None:
        """Await completion of all queued background jobs (test/flush aid)."""
        await self.background.drain()

    # ────────────────────────────────────────────────────────────────────────
    # Retry helper
    # ────────────────────────────────────────────────────────────────────────

    def _retrying(self) -> AsyncRetrying:
        """Build a fresh tenacity controller from the session's retry policy."""
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=self._backoff_initial, max=self._backoff_max),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    async def _with_retry(
        self,
        op: Callable[[], Awaitable[_T]],
        *,
        description: str,
    ) -> _T:
        """Run *op* under the retry policy; re-raise the final error."""
        async for attempt in self._retrying():
            with attempt:
                return await op()
        raise RuntimeError(f"unreachable retry exit for {description}")  # pragma: no cover

    # ────────────────────────────────────────────────────────────────────────
    # Core async API
    # ────────────────────────────────────────────────────────────────────────

    async def process_turn(
        self,
        user_message: str,
        context_budget: int | TokenBudget | None = None,
    ) -> str:
        """
        Run one conversational turn and return the assistant reply.

        Pipeline
        --------
        1. Persist the user message to STM.
        2. Retrieve budgeted context (retriever + optional optimizer).  On
           repeated failure the turn degrades to *no retrieved context*.
        3. Generate the reply via the injected responder.
        4. Persist the assistant reply to STM.
        5. Enqueue access-log + incremental-index work (non-blocking).

        ``context_budget`` accepts a raw token count, a fully-specified
        ``TokenBudget``, or ``None`` for the configured default.
        """
        budget = self._coerce_budget(context_budget)

        # Reset the idle-flush clock — the user is actively interacting,
        # so we shouldn't flush partial STM yet.
        if self.idle_flush is not None:
            self.idle_flush.touch()

        await self._append_to_stm(user_message, role="user")

        ctx = await self._retrieve(user_message, budget)

        reply = await self._responder(user_message, ctx)

        await self._append_to_stm(reply, role="assistant")

        # Bookkeeping must not block the response path.
        self._enqueue(lambda: self._log_access(user_message, reply, ctx))
        if ctx is not None and self.ltm is not None:
            self._enqueue(lambda: self._incremental_index(ctx))
        if self.trigger_manager is not None:
            # after_turn() runs its trigger checks and (if fired) queues a
            # promotion — itself off the response path.
            self._enqueue(self.trigger_manager.after_turn)

        return reply

    async def checkpoint(self) -> PromotionReport:
        """
        Trigger MTM → LTM promotion for this session and return the report.

        Degrades gracefully: with no promoter configured, or after retries
        are exhausted, an empty :class:`PromotionReport` is returned and the
        failure is logged — the caller's flow is never interrupted.
        """
        scope = f"session:{self.session_id}"
        if self.promoter is None:
            log.debug("checkpoint() called with no promoter — no-op")
            return self._empty_report(scope)

        promoter = self.promoter
        try:
            report: PromotionReport = await self._with_retry(
                lambda: promoter.promote(scope),
                description="promoter.promote",
            )
            return report
        except (RetryError, Exception):
            log.exception("checkpoint failed after retries — returning empty report")
            return self._empty_report(scope)

    async def search(self, query: str, k: int = 10) -> list[MemoryItem]:
        """
        Search memory and return up to *k* items, best-first.

        Uses the retriever when available; otherwise (or if retrieval keeps
        failing) falls back to the most-recent STM items so the caller still
        gets a useful, correct — if shallower — result.
        """
        q = Query(
            text=query,
            session_id=self.session_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            top_k=k,
        )
        budget = self._coerce_budget(None)

        if self.retriever is not None:
            retriever = self.retriever  # local bind → no Optional in closure
            try:
                bundle: ContextBundle = await self._with_retry(
                    lambda: retriever.retrieve(q, budget),
                    description="retriever.retrieve",
                )
                return list(bundle.items[:k])
            except (RetryError, Exception):
                log.exception("search retrieval failed — falling back to STM recency")

        # Degraded path: raw STM recency.
        try:
            recent = await self.stm.get_recent(self.session_id, n=k)
            return list(recent)
        except Exception:
            log.exception("STM fallback failed in search — returning empty")
            return []

    # ────────────────────────────────────────────────────────────────────────
    # Sync wrappers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _run_sync(coro: Coroutine[Any, Any, _T], *, label: str) -> _T:
        """
        Execute *coro* on a private event loop via ``asyncio.run``.

        Refuses to run when called from inside a live event loop — that is a
        programming error (use the ``await`` form there instead).  On that
        error path the unstarted coroutine is closed so it never leaks a
        "coroutine was never awaited" warning.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop — good, this is the intended path
        else:
            coro.close()
            raise RuntimeError(
                f"{label}() is a sync wrapper and cannot be called from inside "
                f"a running event loop. Use 'await session.{label}(...)' instead."
            )
        return asyncio.run(coro)

    def process_turn_sync(
        self,
        user_message: str,
        context_budget: int | TokenBudget | None = None,
    ) -> str:
        """Blocking wrapper around :meth:`process_turn`.

        Manages the full session lifecycle (start + close) for one call so it
        is safe to use from plain scripts without ``async with``.
        """

        async def _run() -> str:
            async with self:
                return await self.process_turn(user_message, context_budget)

        return self._run_sync(_run(), label="process_turn")

    def checkpoint_sync(self) -> PromotionReport:
        """Blocking wrapper around :meth:`checkpoint`."""

        async def _run() -> PromotionReport:
            async with self:
                return await self.checkpoint()

        return self._run_sync(_run(), label="checkpoint")

    def search_sync(self, query: str, k: int = 10) -> list[MemoryItem]:
        """Blocking wrapper around :meth:`search`."""

        async def _run() -> list[MemoryItem]:
            async with self:
                return await self.search(query, k)

        return self._run_sync(_run(), label="search")

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────

    def _coerce_budget(self, context_budget: int | TokenBudget | None) -> TokenBudget:
        """Normalise the public budget argument to a full TokenBudget."""
        if isinstance(context_budget, TokenBudget):
            return context_budget

        total = context_budget if isinstance(context_budget, int) else 8192
        response_reserved = min(1024, total // 4)
        usable = max(total - response_reserved, 0)
        # Even three-way split of the usable window across the tiers.
        per_tier = usable // 3
        return TokenBudget(
            total=total,
            stm_reserved=per_tier,
            mtm_reserved=per_tier,
            ltm_reserved=usable - 2 * per_tier,
            response_reserved=response_reserved,
        )

    async def _append_to_stm(self, content: str, *, role: str) -> None:
        """Persist a single message to STM, degrading on failure."""
        item = MemoryItem(
            content=content,
            session_id=self.session_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            metadata={"role": role},
        )
        try:
            await self._with_retry(
                lambda: self.stm.append(item),
                description="stm.append",
            )
        except (RetryError, Exception):
            log.exception("STM append failed (role=%s) — turn continues", role)

    async def _retrieve(self, user_message: str, budget: TokenBudget) -> ContextBundle | None:
        """Retrieve + optimise context; return ``None`` on persistent failure."""
        if self.retriever is None:
            return None
        retriever = self.retriever  # local bind → no Optional in closure

        q = Query(
            text=user_message,
            session_id=self.session_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
        )
        try:
            bundle: ContextBundle = await self._with_retry(
                lambda: retriever.retrieve(q, budget),
                description="retriever.retrieve",
            )
        except (RetryError, Exception):
            log.exception("retrieval failed — turn proceeds with no context")
            return None

        if self.optimizer is not None:
            optimizer = self.optimizer  # local bind → no Optional in closure
            try:
                bundle = await self._with_retry(
                    lambda: optimizer.apply(bundle, budget),
                    description="optimizer.apply",
                )
            except (RetryError, Exception):
                log.exception("optimizer failed — using un-optimised bundle")

        return bundle

    async def _log_access(
        self,
        user_message: str,
        reply: str,
        ctx: ContextBundle | None,
    ) -> None:
        """Background: record an access-log entry (best-effort)."""
        log.debug(
            "access_log session=%s q_len=%d r_len=%d ctx_items=%d",
            self.session_id,
            len(user_message),
            len(reply),
            len(ctx.items) if ctx else 0,
        )

    async def _incremental_index(self, ctx: ContextBundle) -> None:
        """Background: opportunistically refresh LTM neighbours for hot items."""
        if self.ltm is None:
            return
        for item in ctx.items:
            if item.tier.name == "LTM":
                try:
                    await self.ltm.neighbors(item.id, depth=1)
                except Exception:
                    log.debug("incremental index skipped for %s", item.id)

    @staticmethod
    def _empty_report(scope: str) -> PromotionReport:
        return PromotionReport(
            scope=scope,
            promoted_count=0,
            skipped_count=0,
            candidates=[],
            decisions=[],
            elapsed_ms=0.0,
        )


__all__ = ["ContinuumSession", "BackgroundJob", "Responder"]
