"""
tests/unit/test_promoter.py
===========================
Unit tests for ``continuum.promotion.promoter.Promoter`` — the full
MTM→LTM orchestrator, driven entirely by injected fakes (no DB / LLM).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from continuum.core.config import PromoterConfig
from continuum.core.protocols import PromoterProtocol
from continuum.core.types import PromotionScope, SummaryBlock
from continuum.extraction.fact_extractor import Fact
from continuum.promotion import Decision, Promoter, PromotionRunReport

pytestmark = pytest.mark.unit

NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _block(
    text: str, *, session: str | None = None, created: datetime | None = None
) -> SummaryBlock:
    return SummaryBlock(
        text=text,
        id=uuid.uuid4(),
        session_id=session,
        created_at=created or NOW,
    )


class FakeMTM:
    def __init__(self, blocks: list[SummaryBlock]) -> None:
        self._blocks = blocks
        self.marked: list[uuid.UUID] = []
        self.scan_raises = False

    async def scan_unprocessed(self, **_: Any) -> AsyncIterator[SummaryBlock]:
        if self.scan_raises:
            raise RuntimeError("mtm down")
        for b in self._blocks:
            yield b

    async def mark_processed(self, ids: list[uuid.UUID]) -> None:
        self.marked.extend(ids)


class FakeLTM:
    def __init__(self) -> None:
        self.upserted: list[Any] = []
        self.updated: list[tuple[Any, dict]] = []
        self.invalidated: list[tuple[Any, Any]] = []
        self.upsert_raises = False

    async def search_hybrid(self, q: Any, k: int) -> list[Any]:
        return []  # decider is faked; neighbours don't drive the assertions

    async def upsert(self, item: Any) -> uuid.UUID:
        if self.upsert_raises:
            raise RuntimeError("ltm write failed")
        nid = uuid.uuid4()
        self.upserted.append((nid, item))
        return nid

    async def update(self, id: Any, patch: dict) -> None:
        self.updated.append((id, patch))

    async def invalidate(self, id: Any, at: Any = None) -> None:
        self.invalidated.append((id, at))


class FakeFacts:
    """Maps block.text → facts; a configured text raises (extraction fail)."""

    def __init__(self, mapping: dict[str, list[Fact]], fail_on: str | None = None) -> None:
        self._map = mapping
        self._fail_on = fail_on

    async def extract_facts(self, block: SummaryBlock, entities: list[Any]) -> list[Fact]:
        if self._fail_on is not None and block.text == self._fail_on:
            raise RuntimeError("fact extraction blew up")
        return self._map.get(block.text, [])


class FakeDecider:
    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions
        self.calls = 0
        self.batch_sizes: list[int] = []

    async def decide_operations_batch(self, pairs: list[tuple[Fact, list[Any]]]) -> list[Decision]:
        self.calls += 1
        self.batch_sizes.append(len(pairs))
        return self._decisions[: len(pairs)]


def _fact(text: str, conf: float = 0.9) -> Fact:
    return Fact(
        text=text,
        confidence=conf,
        entities_mentioned=["E"],
        source_block_id=uuid.uuid4(),
        category="x",
    )


def _dec(op: str, *, target: uuid.UUID | None = None, merged: str | None = None) -> Decision:
    return Decision(
        op=op,
        target_id=target,
        rationale="r",
        merged_text=merged,
        candidate_text="c",
        model="gpt-4o-mini",
        tokens_in=100,
        tokens_out=20,
    )


def _promoter(
    mtm: Any, ltm: Any, facts: Any, decider: Any, metrics: Any = None, **cfg: Any
) -> Promoter:
    return Promoter(
        PromoterConfig(**cfg),
        mtm=mtm,
        ltm=ltm,
        fact_extractor=facts,
        decider=decider,
        metrics_fn=metrics,
        clock=lambda: NOW,
    )


# ---------------------------------------------------------------------------
# Protocol / report
# ---------------------------------------------------------------------------


def test_satisfies_promoter_protocol() -> None:
    p = _promoter(FakeMTM([]), FakeLTM(), FakeFacts({}), FakeDecider([]))
    assert isinstance(p, PromoterProtocol)


def test_report_compat_properties() -> None:
    r = PromotionRunReport(
        scope="session:x",
        added=[uuid.uuid4()],
        updated=[uuid.uuid4()],
        noop_count=3,
    )
    assert r.promoted_count == 2  # added + updated + deleted
    assert r.skipped_count == 3  # noop
    assert r.scope == "session:x"


# ---------------------------------------------------------------------------
# The four operations
# ---------------------------------------------------------------------------


class TestOperations:
    async def test_add_creates_ltm_fact(self) -> None:
        blk = _block("b")
        mtm, ltm = FakeMTM([blk]), FakeLTM()
        facts = FakeFacts({"b": [_fact("Alice works at Acme")]})
        p = _promoter(mtm, ltm, facts, FakeDecider([_dec("ADD")]))

        rep = await p.promote()

        assert len(ltm.upserted) == 1
        assert rep.added == [ltm.upserted[0][0]]
        assert rep.promoted_count == 1
        assert mtm.marked == [blk.id]  # block marked processed
        assert rep.tokens_used == 120
        assert rep.cost_usd == pytest.approx(100 / 1000 * 0.00015 + 20 / 1000 * 0.00060)

    async def test_update_merges(self) -> None:
        tid = uuid.uuid4()
        blk = _block("b")
        ltm = FakeLTM()
        p = _promoter(
            FakeMTM([blk]),
            ltm,
            FakeFacts({"b": [_fact("f")]}),
            FakeDecider([_dec("UPDATE", target=tid, merged="merged text")]),
        )
        rep = await p.promote()
        assert ltm.updated == [(tid, {"content": "merged text"})]
        assert rep.updated == [tid]
        assert rep.added == []

    async def test_delete_invalidates_with_now(self) -> None:
        tid = uuid.uuid4()
        ltm = FakeLTM()
        p = _promoter(
            FakeMTM([_block("b")]),
            ltm,
            FakeFacts({"b": [_fact("f")]}),
            FakeDecider([_dec("DELETE", target=tid)]),
        )
        rep = await p.promote()
        assert ltm.invalidated == [(tid, NOW)]  # clock injected
        assert rep.deleted == [tid]

    async def test_noop_writes_nothing(self) -> None:
        ltm = FakeLTM()
        p = _promoter(
            FakeMTM([_block("b")]),
            ltm,
            FakeFacts({"b": [_fact("f")]}),
            FakeDecider([_dec("NOOP")]),
        )
        rep = await p.promote()
        assert rep.noop_count == 1
        assert not ltm.upserted and not ltm.updated and not ltm.invalidated
        assert rep.promoted_count == 0

    async def test_update_without_target_is_error_not_crash(self) -> None:
        p = _promoter(
            FakeMTM([_block("b")]),
            FakeLTM(),
            FakeFacts({"b": [_fact("f")]}),
            FakeDecider([_dec("UPDATE", target=None)]),
        )
        rep = await p.promote()
        assert rep.updated == []
        assert any("UPDATE without target_id" in e for e in rep.errors)


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    async def test_extraction_failure_skips_block_not_run(self) -> None:
        good, bad = _block("good"), _block("bad")
        mtm = FakeMTM([bad, good])
        facts = FakeFacts({"good": [_fact("g")]}, fail_on="bad")
        p = _promoter(mtm, FakeLTM(), facts, FakeDecider([_dec("ADD")]))

        rep = await p.promote()

        assert rep.promoted_count == 1  # good block still done
        assert good.id in mtm.marked
        assert bad.id not in mtm.marked  # left for retry
        assert any("block" in e and "extract" in e for e in rep.errors)

    async def test_execute_failure_skips_fact_not_run(self) -> None:
        ltm = FakeLTM()
        ltm.upsert_raises = True
        blk = _block("b")
        mtm = FakeMTM([blk])
        p = _promoter(mtm, ltm, FakeFacts({"b": [_fact("f")]}), FakeDecider([_dec("ADD")]))

        rep = await p.promote()

        assert rep.added == []
        assert any("execute ADD" in e for e in rep.errors)
        assert blk.id in mtm.marked  # block still processed

    async def test_catastrophic_scan_failure_returns_report(self) -> None:
        mtm = FakeMTM([])
        mtm.scan_raises = True
        p = _promoter(mtm, FakeLTM(), FakeFacts({}), FakeDecider([]))
        rep = await p.promote()  # must NOT raise
        assert isinstance(rep, PromotionRunReport)
        assert any("aborted" in e for e in rep.errors)

    async def test_mark_processed_failure_recorded(self) -> None:
        class BadMark(FakeMTM):
            async def mark_processed(self, ids: list[uuid.UUID]) -> None:
                raise RuntimeError("db down")

        mtm = BadMark([_block("b")])
        p = _promoter(mtm, FakeLTM(), FakeFacts({"b": [_fact("f")]}), FakeDecider([_dec("ADD")]))
        rep = await p.promote()
        assert rep.promoted_count == 1  # decision still executed
        assert any("mark_processed failed" in e for e in rep.errors)


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


class TestScope:
    async def test_session_scope_filters(self) -> None:
        sid = uuid.uuid4()
        mine = _block("mine", session=str(sid))
        other = _block("other", session=str(uuid.uuid4()))
        mtm = FakeMTM([mine, other])
        facts = FakeFacts({"mine": [_fact("m")], "other": [_fact("o")]})
        p = _promoter(mtm, FakeLTM(), facts, FakeDecider([_dec("ADD")]))

        rep = await p.promote(PromotionScope(session_id=sid))

        assert rep.blocks_processed == 1
        assert mtm.marked == [mine.id]
        assert rep.scope == f"session:{sid}"

    async def test_string_scope_accepted(self) -> None:
        sid = uuid.uuid4()
        mine = _block("mine", session=str(sid))
        mtm = FakeMTM([mine])
        p = _promoter(mtm, FakeLTM(), FakeFacts({"mine": [_fact("m")]}), FakeDecider([_dec("ADD")]))
        rep = await p.promote(f"session:{sid}")  # legacy str scope
        assert rep.promoted_count == 1

    async def test_since_filter(self) -> None:
        old = _block("old", created=NOW - timedelta(days=2))
        new = _block("new", created=NOW)
        mtm = FakeMTM([old, new])
        facts = FakeFacts({"old": [_fact("o")], "new": [_fact("n")]})
        p = _promoter(mtm, FakeLTM(), facts, FakeDecider([_dec("ADD")]))

        rep = await p.promote(PromotionScope(since=NOW - timedelta(hours=1)))
        assert rep.blocks_processed == 1  # only "new"
        assert mtm.marked == [new.id]


# ---------------------------------------------------------------------------
# Confidence gate, batching, metrics
# ---------------------------------------------------------------------------


class TestMisc:
    async def test_low_confidence_facts_dropped(self) -> None:
        blk = _block("b")
        facts = FakeFacts({"b": [_fact("keep", 0.8), _fact("drop", 0.3)]})
        decider = FakeDecider([_dec("ADD"), _dec("ADD")])
        p = _promoter(FakeMTM([blk]), FakeLTM(), facts, decider, confidence_threshold=0.6)
        rep = await p.promote()
        assert decider.batch_sizes == [1]  # only the 0.8 fact
        assert rep.promoted_count == 1

    async def test_single_batched_decider_call(self) -> None:
        b1, b2 = _block("b1"), _block("b2")
        facts = FakeFacts({"b1": [_fact("f1")], "b2": [_fact("f2")]})
        decider = FakeDecider([_dec("ADD"), _dec("ADD")])
        p = _promoter(FakeMTM([b1, b2]), FakeLTM(), facts, decider)
        rep = await p.promote()
        assert decider.calls == 1  # one batched call
        assert decider.batch_sizes == [2]
        assert rep.promoted_count == 2

    async def test_metrics_emitted(self) -> None:
        seen: list[tuple[str, float]] = []
        p = _promoter(
            FakeMTM([_block("b")]),
            FakeLTM(),
            FakeFacts({"b": [_fact("f")]}),
            FakeDecider([_dec("ADD")]),
            metrics=lambda n, v: seen.append((n, v)),
        )
        await p.promote()
        names = {n for n, _ in seen}
        assert {"promoter.processed", "promoter.added", "promoter.errors"} <= names

    async def test_llm_entity_enhancer_chained(self) -> None:
        seen: dict[str, Any] = {}

        class GLiNER:
            async def extract(self, text: str) -> tuple[list[Any], list[Any]]:
                return (["g"], [])

        class LLMx:
            async def extract(self, text: str, ents: list[Any]) -> tuple[list[Any], list[Any]]:
                seen["chained_from"] = ents
                return (["g", "llm"], [])

        class CaptureFacts:
            async def extract_facts(self, block: Any, entities: list[Any]) -> list[Fact]:
                seen["entities"] = entities
                return [_fact("f")]

        p = Promoter(
            PromoterConfig(),
            mtm=FakeMTM([_block("b")]),
            ltm=FakeLTM(),
            fact_extractor=CaptureFacts(),
            decider=FakeDecider([_dec("ADD")]),
            entity_extractor=GLiNER(),
            llm_entity_extractor=LLMx(),
        )
        await p.promote()
        assert seen["chained_from"] == ["g"]  # GLiNER → LLM
        assert seen["entities"] == ["g", "llm"]  # merged → facts
