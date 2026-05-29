"""
tests/unit/evals/test_continuum_ltm_store.py
============================================
Unit tests for :class:`evals.longmemeval.continuum_ltm_store.ContinuumLTMHaystackStore`
and the supporting :class:`continuum.stores.in_memory.ltm.InMemoryLTM`.

Each test is hermetic — stubbed FactExtractor and Mem0Promoter, no
embedder downloads, no network. The full STM → facts → LTM →
supersession path is exercised on a tiny scripted input.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from continuum.core.types import (
    MemoryItem,
    MemoryTier,
    Query,
    ScoredItem,
    SummaryBlock,
)
from continuum.extraction.fact_extractor import Fact
from continuum.promotion.mem0_promoter import Decision
from continuum.stores.in_memory.ltm import InMemoryLTM
from evals.longmemeval.continuum_ltm_store import ContinuumLTMHaystackStore

pytestmark = pytest.mark.unit


# ── fakes ──────────────────────────────────────────────────────────────────


class _CannedFactExtractor:
    """Returns a scripted list of facts regardless of block content."""

    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts
        self.calls: list[SummaryBlock] = []

    async def extract_facts(
        self, block: SummaryBlock, entities: list[Any],
    ) -> list[Fact]:
        self.calls.append(block)
        # Re-stamp source_block_id so each invocation sees the actual block.
        return [
            Fact(
                text=f.text,
                confidence=f.confidence,
                entities_mentioned=list(f.entities_mentioned),
                source_block_id=block.id,
                category=f.category,
            )
            for f in self._facts
        ]


class _CannedPromoter:
    """Returns scripted decisions in the order the facts arrive."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions
        self.calls: list[list[tuple[Fact, list[ScoredItem]]]] = []

    async def decide_operations_batch(
        self, items: list[tuple[Fact, list[ScoredItem]]],
    ) -> list[Decision]:
        self.calls.append(list(items))
        # Always return one decision per input, in order. Test code
        # pre-loads exactly enough decisions for the scripted facts.
        out: list[Decision] = []
        for i, (fact, _) in enumerate(items):
            dec = self._decisions[i]
            # Re-stamp candidate_text so the audit log is accurate.
            out.append(Decision(
                op=dec.op, target_id=dec.target_id, rationale=dec.rationale,
                merged_text=dec.merged_text, candidate_text=fact.text,
            ))
        return out


class _NullEmbedder:
    """No-op embedder — sets the embedding to a tiny constant vector."""

    def encode(self, texts: list[str]) -> Any:
        return [[0.1, 0.2, 0.3] for _ in texts]


# ── helpers ────────────────────────────────────────────────────────────────


def _turn(session_id: str, role: str, content: str) -> MemoryItem:
    return MemoryItem(
        content=content,
        tier=MemoryTier.STM,
        metadata={
            "role": role,
            "session_id": session_id,
            "user_id": "u1",
            "date": "2026-01-01T00:00:00",
        },
    )


def _fact(text: str, **kw: Any) -> Fact:
    return Fact(
        text=text,
        confidence=kw.get("confidence", 0.8),
        entities_mentioned=kw.get("entities", []),
        source_block_id=uuid.uuid4(),
        category=kw.get("category"),
    )


# ── ContinuumLTMHaystackStore — LLM path ───────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_triggers_promotion_and_writes_ltm_facts_into_items() -> None:
    ltm = InMemoryLTM()
    extractor = _CannedFactExtractor([
        _fact("User lives in Paris", entities=["Paris"]),
        _fact("User drives a Tesla", entities=["Tesla"]),
    ])
    promoter = _CannedPromoter([
        Decision(op="ADD", target_id=None, rationale="new fact"),
        Decision(op="ADD", target_id=None, rationale="new fact"),
    ])
    store = ContinuumLTMHaystackStore(
        ltm=ltm, embedder=_NullEmbedder(),
        promoter=promoter, fact_extractor=extractor,
        llm_available=True, ltm_backend_label="in_memory",
    )

    await store.append(_turn("s1", "user", "I moved to Paris last year."))
    await store.append(_turn("s1", "user", "Just picked up a new Tesla."))
    await store.finalize()

    # Raw turns are still in items (FlatHaystackStore compat).
    contents = [it.content for it in store.items]
    assert "I moved to Paris last year." in contents
    # LTM facts were extracted, upserted, and pushed back onto items.
    fact_items = [it for it in store.items if it.metadata.get("source") == "ltm_fact"]
    assert {f.content for f in fact_items} == {
        "User lives in Paris", "User drives a Tesla",
    }
    # Promoter was called with both facts in one batch.
    assert promoter.calls and len(promoter.calls[0]) == 2
    # Metrics surface the path taken.
    m = store.metrics()
    assert m["promoter_path"] == "llm"
    assert m["promoter_added"] == 2
    assert m["supersessions"] == 0
    assert m["ltm_facts_live"] == 2
    assert m["ltm_backend"] == "in_memory"


@pytest.mark.asyncio
async def test_supersession_on_contradiction_removes_stale_fact_from_items() -> None:
    ltm = InMemoryLTM()
    # Pre-seed the LTM with the stale fact so the promoter has a
    # target_id to point DELETE at. In a real run this would be a
    # row from an earlier session inside the same row.
    stale_id = await ltm.upsert(MemoryItem(
        content="User lives in Berlin",
        tier=MemoryTier.LTM,
        metadata={"kind": "fact", "source": "ltm_fact"},
    ))

    extractor = _CannedFactExtractor([
        _fact("User lives in Paris", entities=["Paris"]),
    ])
    promoter = _CannedPromoter([
        Decision(op="DELETE", target_id=stale_id, rationale="contradicts Berlin"),
    ])
    store = ContinuumLTMHaystackStore(
        ltm=ltm, embedder=_NullEmbedder(),
        promoter=promoter, fact_extractor=extractor,
        llm_available=True, ltm_backend_label="in_memory",
    )

    await store.append(_turn("s2", "user", "I moved to Paris."))
    await store.finalize()

    # The Berlin row was invalidated and shouldn't appear in items.
    fact_items = {it.content for it in store.items if it.metadata.get("source") == "ltm_fact"}
    assert "User lives in Berlin" not in fact_items
    # The new Paris row IS present (we upserted it ahead of the decide step).
    assert "User lives in Paris" in fact_items
    m = store.metrics()
    assert m["supersessions"] == 1
    assert m["promoter_deleted"] == 1


@pytest.mark.asyncio
async def test_no_finalize_double_fires() -> None:
    """Calling finalize() twice is a no-op the second time."""
    ltm = InMemoryLTM()
    extractor = _CannedFactExtractor([_fact("X")])
    promoter = _CannedPromoter([Decision(op="ADD", target_id=None, rationale="")])
    store = ContinuumLTMHaystackStore(
        ltm=ltm, embedder=_NullEmbedder(),
        promoter=promoter, fact_extractor=extractor,
        llm_available=True,
    )
    await store.append(_turn("s", "user", "stuff"))
    await store.finalize()
    initial_items = list(store.items)
    await store.finalize()
    assert store.items == initial_items
    # Promoter only called once even across two finalize invocations.
    assert len(promoter.calls) == 1


# ── ContinuumLTMHaystackStore — heuristic path ─────────────────────────────


@pytest.mark.asyncio
async def test_heuristic_path_emits_delete_on_update_marker() -> None:
    """No LLM available → regex over 'no longer / now' markers fires DELETE."""
    ltm = InMemoryLTM()
    stale_id = await ltm.upsert(MemoryItem(
        content="User lives in Berlin",
        tier=MemoryTier.LTM,
        metadata={"kind": "fact"},
    ))
    store = ContinuumLTMHaystackStore(
        ltm=ltm, embedder=_NullEmbedder(),
        promoter=None, fact_extractor=None,
        llm_available=False, ltm_backend_label="in_memory",
    )
    await store.append(_turn(
        "s3", "user", "I no longer live in Berlin; I moved to Paris.",
    ))
    await store.finalize()
    m = store.metrics()
    assert m["promoter_path"] == "heuristic"
    assert m["supersessions"] == 1
    # Berlin row was invalidated.
    assert stale_id in ltm._invalidated_at  # noqa: SLF001 — test invariant
    live = {row.content async for row in ltm.iter_live()}
    assert "User lives in Berlin" not in live


@pytest.mark.asyncio
async def test_heuristic_path_noop_when_no_update_marker() -> None:
    ltm = InMemoryLTM()
    store = ContinuumLTMHaystackStore(
        ltm=ltm, embedder=_NullEmbedder(),
        promoter=None, fact_extractor=None,
        llm_available=False,
    )
    await store.append(_turn("s4", "user", "I really like pasta."))
    await store.finalize()
    m = store.metrics()
    assert m["supersessions"] == 0
    # The user statement was still upserted as a fact and pushed back
    # into items so the retriever sees it.
    assert any(
        it.metadata.get("source") == "ltm_fact" for it in store.items
    )


# ── InMemoryLTM — direct tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inmemory_search_hybrid_excludes_invalidated_rows() -> None:
    ltm = InMemoryLTM()
    a = await ltm.upsert(MemoryItem(content="Boston is the capital", tier=MemoryTier.LTM))
    b = await ltm.upsert(MemoryItem(content="Paris is the capital", tier=MemoryTier.LTM))
    await ltm.invalidate(a, datetime.now(UTC))
    hits = await ltm.search_hybrid(Query(text="capital"), k=5)
    contents = [h.item.content for h in hits]
    assert "Paris is the capital" in contents
    assert "Boston is the capital" not in contents
    assert ltm.live_count == 1
    assert ltm.total_count == 2


@pytest.mark.asyncio
async def test_inmemory_iter_live_skips_invalidated() -> None:
    ltm = InMemoryLTM()
    a = await ltm.upsert(MemoryItem(content="A", tier=MemoryTier.LTM))
    b = await ltm.upsert(MemoryItem(content="B", tier=MemoryTier.LTM))
    await ltm.invalidate(a)
    live = [row.content async for row in ltm.iter_live()]
    assert live == ["B"]
    # Total count unchanged — the row is kept for audit, just hidden.
    assert ltm.total_count == 2


@pytest.mark.asyncio
async def test_inmemory_update_patches_content_only_live_rows() -> None:
    ltm = InMemoryLTM()
    a = await ltm.upsert(MemoryItem(content="old", tier=MemoryTier.LTM))
    await ltm.update(a, {"content": "new"})
    live = [row.content async for row in ltm.iter_live()]
    assert live == ["new"]


@pytest.mark.asyncio
async def test_inmemory_invalidate_unknown_id_is_noop() -> None:
    ltm = InMemoryLTM()
    await ltm.invalidate(uuid.uuid4())  # should not raise
    assert ltm.total_count == 0
