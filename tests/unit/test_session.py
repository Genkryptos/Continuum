"""
tests/unit/test_session.py
==========================
Unit tests for ``ContinuumSession``.

All protocol dependencies are replaced with in-process fakes — no DB, no
network, no LLM.  The four behaviours called out in the spec each get a
dedicated section:

* Async usage           — process_turn / checkpoint / search happy paths
* Sync wrapper usage    — *_sync wrappers + running-loop guard
* Background execution   — access-log / indexing jobs run off the hot path
* Error handling         — tenacity retry + graceful degradation
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from continuum.core.config import ContinuumConfig
from continuum.core.session import ContinuumSession
from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    PromotionReport,
    Query,
    TokenBudget,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSTM:
    """Minimal STMProtocol fake recording appends."""

    def __init__(self) -> None:
        self.appended: list[MemoryItem] = []
        self.fail_appends = False

    async def append(self, item: MemoryItem) -> None:
        if self.fail_appends:
            raise RuntimeError("stm down")
        self.appended.append(item)

    async def window(self, session_id: str, max_tokens: int | None = None):
        return list(self.appended)

    async def get_recent(self, session_id: str, n: int = 10):
        return list(self.appended[-n:])

    async def flush_to(self, session_id: str, target: Any) -> int:
        return 0

    async def clear(self, session_id: str) -> None:
        self.appended.clear()

    async def stats(self, session_id: str) -> dict[str, Any]:
        return {"message_count": len(self.appended)}


class FlakyRetriever:
    """
    Fails ``fail_times`` times then succeeds — exercises tenacity retry.

    Set ``fail_times`` huge to simulate a permanently-down retriever.
    """

    def __init__(self, fail_times: int = 0, items: list[MemoryItem] | None = None) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self._items = items or [
            MemoryItem(content="ctx-A", tier=MemoryTier.LTM),
            MemoryItem(content="ctx-B", tier=MemoryTier.MTM),
        ]

    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError(f"retriever flaky (call {self.calls})")
        return ContextBundle(
            items=list(self._items),
            messages=[{"role": "system", "content": "ctx"}],
            tokens_used=42,
            budget=budget,
            tier_breakdown={"ltm": 21, "mtm": 21},
        )


class FakePromoter:
    """Promoter fake; optionally always raises to test degradation."""

    def __init__(self, *, always_fail: bool = False) -> None:
        self.always_fail = always_fail
        self.calls = 0

    async def promote(self, scope: str) -> PromotionReport:
        self.calls += 1
        if self.always_fail:
            raise RuntimeError("promoter exploded")
        return PromotionReport(
            scope=scope,
            promoted_count=2,
            skipped_count=1,
            candidates=[],
            decisions=[],
            elapsed_ms=1.5,
        )


class FakeOptimizer:
    async def apply(self, ctx: ContextBundle, budget: TokenBudget) -> ContextBundle:
        # Drop everything but the first item to prove it ran.
        return ContextBundle(
            items=ctx.items[:1],
            messages=ctx.messages,
            tokens_used=ctx.tokens_used // 2,
            budget=budget,
            tier_breakdown={"optimized": 1},
        )


@pytest.fixture
def config() -> ContinuumConfig:
    return ContinuumConfig()


def _fast_session(session: ContinuumSession) -> ContinuumSession:
    """Zero out backoff so retry tests run instantly."""
    session._backoff_initial = 0.0
    session._backoff_max = 0.0
    return session


# ---------------------------------------------------------------------------
# 1. Async usage
# ---------------------------------------------------------------------------


class TestAsyncUsage:
    async def test_process_turn_returns_reply_and_persists_messages(
        self, config: ContinuumConfig
    ) -> None:
        stm = FakeSTM()
        async with ContinuumSession(config, stm=stm) as session:
            reply = await session.process_turn("hello there")

        assert isinstance(reply, str)
        assert "hello there" in reply
        roles = [m.metadata["role"] for m in stm.appended]
        assert roles == ["user", "assistant"]

    async def test_process_turn_uses_injected_responder_and_context(
        self, config: ContinuumConfig
    ) -> None:
        captured: dict[str, Any] = {}

        async def responder(msg: str, ctx: ContextBundle | None) -> str:
            captured["msg"] = msg
            captured["n_items"] = len(ctx.items) if ctx else 0
            return "custom-reply"

        retriever = FlakyRetriever(fail_times=0)
        async with ContinuumSession(
            config, stm=FakeSTM(), retriever=retriever, responder=responder
        ) as session:
            reply = await session.process_turn("status of project X?")

        assert reply == "custom-reply"
        assert captured["msg"] == "status of project X?"
        assert captured["n_items"] == 2

    async def test_optimizer_is_applied_to_bundle(self, config: ContinuumConfig) -> None:
        seen: dict[str, Any] = {}

        async def responder(msg: str, ctx: ContextBundle | None) -> str:
            seen["items"] = len(ctx.items) if ctx else -1
            return "ok"

        async with ContinuumSession(
            config,
            stm=FakeSTM(),
            retriever=FlakyRetriever(),
            optimizer=FakeOptimizer(),
            responder=responder,
        ) as session:
            await session.process_turn("q")

        assert seen["items"] == 1  # FakeOptimizer trims to a single item

    async def test_checkpoint_returns_promoter_report(self, config: ContinuumConfig) -> None:
        promoter = FakePromoter()
        async with ContinuumSession(
            config, stm=FakeSTM(), promoter=promoter, session_id="s1"
        ) as session:
            report = await session.checkpoint()

        assert report.promoted_count == 2
        assert report.scope == "session:s1"
        assert promoter.calls == 1

    async def test_search_uses_retriever(self, config: ContinuumConfig) -> None:
        async with ContinuumSession(config, stm=FakeSTM(), retriever=FlakyRetriever()) as session:
            hits = await session.search("project X", k=1)

        assert len(hits) == 1
        assert hits[0].content == "ctx-A"

    async def test_example_usage_from_docstring(self, config: ContinuumConfig) -> None:
        # Mirrors the README example: config-only construction works.
        async with ContinuumSession(config) as session:
            response = await session.process_turn("What's the status of project X?")
        assert "What's the status of project X?" in response


# ---------------------------------------------------------------------------
# 2. Sync wrapper usage
# ---------------------------------------------------------------------------


class TestSyncWrappers:
    def test_process_turn_sync(self, config: ContinuumConfig) -> None:
        stm = FakeSTM()
        session = ContinuumSession(config, stm=stm)
        reply = session.process_turn_sync("sync hello")
        assert "sync hello" in reply
        assert len(stm.appended) == 2

    def test_checkpoint_sync(self, config: ContinuumConfig) -> None:
        session = ContinuumSession(config, stm=FakeSTM(), promoter=FakePromoter(), session_id="z")
        report = session.checkpoint_sync()
        assert report.scope == "session:z"
        assert report.promoted_count == 2

    def test_search_sync(self, config: ContinuumConfig) -> None:
        session = ContinuumSession(config, stm=FakeSTM(), retriever=FlakyRetriever())
        hits = session.search_sync("q", k=2)
        assert len(hits) == 2

    async def test_sync_wrapper_rejected_inside_running_loop(self, config: ContinuumConfig) -> None:
        session = ContinuumSession(config, stm=FakeSTM())
        with pytest.raises(RuntimeError, match="cannot be called from inside"):
            session.process_turn_sync("nope")


# ---------------------------------------------------------------------------
# 3. Background task execution
# ---------------------------------------------------------------------------


class TestBackgroundTasks:
    async def test_access_log_job_runs_off_hot_path(self, config: ContinuumConfig) -> None:
        ran = asyncio.Event()

        session = ContinuumSession(config, stm=FakeSTM())

        async def spy_log(*_a: Any, **_k: Any) -> None:
            ran.set()

        session._log_access = spy_log  # type: ignore[method-assign]

        async with session:
            await session.process_turn("trigger bg")
            # Reply already returned; the job runs in the background.
            await asyncio.wait_for(ran.wait(), timeout=2.0)

        assert ran.is_set()

    async def test_incremental_index_enqueued_when_ltm_present(
        self, config: ContinuumConfig
    ) -> None:
        indexed: list[str] = []

        class FakeLTM:
            async def neighbors(self, item_id: str, depth: int = 1):
                indexed.append(item_id)
                return []

        session = ContinuumSession(
            config,
            stm=FakeSTM(),
            ltm=FakeLTM(),
            retriever=FlakyRetriever(),
        )
        async with session:
            await session.process_turn("q")
            await session.drain()  # wait for queued bg jobs

        # FlakyRetriever returns one LTM-tier item → indexed once.
        assert len(indexed) == 1

    async def test_jobs_dropped_when_session_not_started(self, config: ContinuumConfig) -> None:
        session = ContinuumSession(config, stm=FakeSTM())
        # Not started → _enqueue must refuse without raising.
        assert session._enqueue(lambda: asyncio.sleep(0)) is False

    async def test_failing_background_job_does_not_crash_pool(
        self, config: ContinuumConfig
    ) -> None:
        session = ContinuumSession(config, stm=FakeSTM())
        good_ran = asyncio.Event()

        async def bad_job() -> None:
            raise RuntimeError("boom in background")

        async def good_job() -> None:
            good_ran.set()

        async with session:
            session._enqueue(bad_job)
            session._enqueue(good_job)
            await asyncio.wait_for(good_ran.wait(), timeout=2.0)

        # Pool survived the bad job and still processed the good one.
        assert good_ran.is_set()


# ---------------------------------------------------------------------------
# 4. Error handling & graceful degradation
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_retriever_retried_then_succeeds(self, config: ContinuumConfig) -> None:
        retriever = FlakyRetriever(fail_times=2)  # 3rd attempt succeeds
        session = _fast_session(ContinuumSession(config, stm=FakeSTM(), retriever=retriever))
        async with session:
            hits = await session.search("q", k=2)

        assert retriever.calls == 3
        assert len(hits) == 2

    async def test_search_degrades_to_stm_when_retriever_dead(
        self, config: ContinuumConfig
    ) -> None:
        stm = FakeSTM()
        await stm.append(MemoryItem(content="recent-1"))
        await stm.append(MemoryItem(content="recent-2"))
        retriever = FlakyRetriever(fail_times=999)  # never recovers

        session = _fast_session(ContinuumSession(config, stm=stm, retriever=retriever))
        async with session:
            hits = await session.search("q", k=5)

        # Falls back to STM recency rather than raising.
        assert {h.content for h in hits} == {"recent-1", "recent-2"}

    async def test_checkpoint_degrades_to_empty_report(self, config: ContinuumConfig) -> None:
        promoter = FakePromoter(always_fail=True)
        session = _fast_session(
            ContinuumSession(config, stm=FakeSTM(), promoter=promoter, session_id="deg")
        )
        async with session:
            report = await session.checkpoint()

        assert report.promoted_count == 0
        assert report.skipped_count == 0
        assert report.scope == "session:deg"
        assert promoter.calls == session._max_attempts  # retried to exhaustion

    async def test_checkpoint_noop_without_promoter(self, config: ContinuumConfig) -> None:
        async with ContinuumSession(config, stm=FakeSTM(), session_id="np") as session:
            report = await session.checkpoint()
        assert report == ContinuumSession._empty_report("session:np")

    async def test_process_turn_survives_stm_failure(self, config: ContinuumConfig) -> None:
        stm = FakeSTM()
        stm.fail_appends = True
        session = _fast_session(ContinuumSession(config, stm=stm))
        async with session:
            reply = await session.process_turn("still works?")
        # STM is down but the turn still produced a reply (degraded).
        assert "still works?" in reply
        assert stm.appended == []

    async def test_process_turn_continues_when_retriever_fails(
        self, config: ContinuumConfig
    ) -> None:
        retriever = FlakyRetriever(fail_times=999)
        seen: dict[str, Any] = {}

        async def responder(msg: str, ctx: ContextBundle | None) -> str:
            seen["ctx_is_none"] = ctx is None
            return "degraded-ok"

        session = _fast_session(
            ContinuumSession(config, stm=FakeSTM(), retriever=retriever, responder=responder)
        )
        async with session:
            reply = await session.process_turn("q")

        assert reply == "degraded-ok"
        assert seen["ctx_is_none"] is True


# ---------------------------------------------------------------------------
# 5. Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_is_idempotent(self, config: ContinuumConfig) -> None:
        session = ContinuumSession(config, stm=FakeSTM())
        await session.start()
        first_task = session.background._run_task
        await session.start()  # no-op
        assert session.background._run_task is first_task
        assert session.background.health()["ok"] is True
        await session.aclose()

    async def test_aclose_without_start_is_safe(self, config: ContinuumConfig) -> None:
        session = ContinuumSession(config, stm=FakeSTM())
        await session.aclose()  # must not raise

    async def test_ensure_schema_invoked_on_components(self, config: ContinuumConfig) -> None:
        called = asyncio.Event()

        class SchemaSTM(FakeSTM):
            async def ensure_schema(self) -> None:
                called.set()

        async with ContinuumSession(config, stm=SchemaSTM()):
            pass
        assert called.is_set()

    async def test_component_close_called_on_exit(self, config: ContinuumConfig) -> None:
        closed = asyncio.Event()

        class ClosableSTM(FakeSTM):
            async def aclose(self) -> None:
                closed.set()

        async with ContinuumSession(config, stm=ClosableSTM()):
            pass
        assert closed.is_set()
