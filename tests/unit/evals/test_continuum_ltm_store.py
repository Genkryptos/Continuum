"""
tests/unit/evals/test_continuum_ltm_store.py
============================================
Unit tests for :class:`evals.longmemeval.continuum_ltm_store.ContinuumLTMHaystackStore`
and the supporting :class:`continuum.stores.in_memory.ltm.InMemoryLTM`.

Each test is hermetic — stubbed FactExtractor and ContinuumPromoter, no
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
from continuum.promotion.continuum_promoter import Decision
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
        self,
        block: SummaryBlock,
        entities: list[Any],
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
        self,
        items: list[tuple[Fact, list[ScoredItem]]],
    ) -> list[Decision]:
        self.calls.append(list(items))
        # Always return one decision per input, in order. Test code
        # pre-loads exactly enough decisions for the scripted facts.
        out: list[Decision] = []
        for i, (fact, _) in enumerate(items):
            dec = self._decisions[i]
            # Re-stamp candidate_text so the audit log is accurate.
            out.append(
                Decision(
                    op=dec.op,
                    target_id=dec.target_id,
                    rationale=dec.rationale,
                    merged_text=dec.merged_text,
                    candidate_text=fact.text,
                )
            )
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
    extractor = _CannedFactExtractor(
        [
            _fact("User lives in Paris", entities=["Paris"]),
            _fact("User drives a Tesla", entities=["Tesla"]),
        ]
    )
    promoter = _CannedPromoter(
        [
            Decision(op="ADD", target_id=None, rationale="new fact"),
            Decision(op="ADD", target_id=None, rationale="new fact"),
        ]
    )
    store = ContinuumLTMHaystackStore(
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=promoter,
        fact_extractor=extractor,
        llm_available=True,
        ltm_backend_label="in_memory",
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
        "User lives in Paris",
        "User drives a Tesla",
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
    stale_id = await ltm.upsert(
        MemoryItem(
            content="User lives in Berlin",
            tier=MemoryTier.LTM,
            metadata={"kind": "fact", "source": "ltm_fact"},
        )
    )

    extractor = _CannedFactExtractor(
        [
            _fact("User lives in Paris", entities=["Paris"]),
        ]
    )
    promoter = _CannedPromoter(
        [
            Decision(op="DELETE", target_id=stale_id, rationale="contradicts Berlin"),
        ]
    )
    store = ContinuumLTMHaystackStore(
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=promoter,
        fact_extractor=extractor,
        llm_available=True,
        ltm_backend_label="in_memory",
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
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=promoter,
        fact_extractor=extractor,
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
    stale_id = await ltm.upsert(
        MemoryItem(
            content="User lives in Berlin",
            tier=MemoryTier.LTM,
            metadata={"kind": "fact"},
        )
    )
    store = ContinuumLTMHaystackStore(
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=None,
        fact_extractor=None,
        llm_available=False,
        ltm_backend_label="in_memory",
    )
    await store.append(
        _turn(
            "s3",
            "user",
            "I no longer live in Berlin; I moved to Paris.",
        )
    )
    await store.finalize()
    m = store.metrics()
    assert m["promoter_path"] == "heuristic"
    assert m["supersessions"] == 1
    # Berlin row was invalidated.
    assert stale_id in ltm._invalidated_at
    live = {row.content async for row in ltm.iter_live()}
    assert "User lives in Berlin" not in live


@pytest.mark.asyncio
async def test_heuristic_path_noop_when_no_update_marker() -> None:
    ltm = InMemoryLTM()
    store = ContinuumLTMHaystackStore(
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=None,
        fact_extractor=None,
        llm_available=False,
    )
    await store.append(_turn("s4", "user", "I really like pasta."))
    await store.finalize()
    m = store.metrics()
    assert m["supersessions"] == 0
    # The user statement was still upserted as a fact and pushed back
    # into items so the retriever sees it.
    assert any(it.metadata.get("source") == "ltm_fact" for it in store.items)


# ── InMemoryLTM — direct tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inmemory_search_hybrid_excludes_invalidated_rows() -> None:
    ltm = InMemoryLTM()
    a = await ltm.upsert(MemoryItem(content="Boston is the capital", tier=MemoryTier.LTM))
    await ltm.upsert(MemoryItem(content="Paris is the capital", tier=MemoryTier.LTM))
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
    await ltm.upsert(MemoryItem(content="B", tier=MemoryTier.LTM))
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


# ── A5: supersession_mode arms (docs/ACCURACY_AND_BENCHMARK_PLAN.md §A5) ───


@pytest.mark.asyncio
async def test_inmemory_update_honours_valid_to() -> None:
    """
    Closing valid-time via ``update`` must actually land on the in-memory
    backend. It previously dropped ``valid_to`` silently while Postgres
    honoured it, which would make arm B a no-op on the eval's default backend.
    """
    ltm = InMemoryLTM()
    a = await ltm.upsert(MemoryItem(content="lives in Berlin", tier=MemoryTier.LTM))
    at = datetime.now(UTC)
    await ltm.update(a, {"valid_to": at})

    rows = [row async for row in ltm.iter_live()]
    assert len(rows) == 1, "closing valid-time must NOT hide the row"
    assert rows[0].valid_range is not None
    assert rows[0].valid_range.valid_to == at
    assert rows[0].valid_range.valid_from is not None, "valid_from must survive"
    assert rows[0].content == "lives in Berlin", "content must not be touched"


@pytest.mark.asyncio
async def test_inmemory_update_without_valid_keys_preserves_range() -> None:
    """A content-only patch must leave the bi-temporal range alone."""
    ltm = InMemoryLTM()
    a = await ltm.upsert(MemoryItem(content="old", tier=MemoryTier.LTM))
    (before_row,) = [r async for r in ltm.iter_live()]
    before = before_row.valid_range
    await ltm.update(a, {"content": "new"})
    (after_row,) = [r async for r in ltm.iter_live()]
    after = after_row.valid_range
    assert (before.valid_from, before.valid_to) == (after.valid_from, after.valid_to)


def test_supersession_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="supersession_mode"):
        ContinuumLTMHaystackStore(ltm=InMemoryLTM(), supersession_mode="nonsense")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expect_retained"),
    [("invalidate", 0), ("bitemporal", 1)],
)
async def test_supersession_arms_hide_stale_fact_but_differ_on_retention(
    mode: str, expect_retained: int
) -> None:
    """
    Both arms must hide the superseded fact from retrieval — that is the
    control property that makes the A/B interpretable. They differ in whether
    the fact is *retained* and therefore still answerable as-of.
    """
    ltm = InMemoryLTM()
    stale_id = await ltm.upsert(
        MemoryItem(
            content="User lives in Berlin",
            tier=MemoryTier.LTM,
            metadata={"kind": "fact", "source": "ltm_fact"},
        )
    )
    store = ContinuumLTMHaystackStore(
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=_CannedPromoter(
            [Decision(op="DELETE", target_id=stale_id, rationale="contradicts Berlin")]
        ),
        fact_extractor=_CannedFactExtractor([_fact("User lives in Paris", entities=["Paris"])]),
        llm_available=True,
        ltm_backend_label="in_memory",
        supersession_mode=mode,
    )

    await store.append(_turn("s2", "user", "I moved to Paris."))
    await store.finalize()

    fact_items = {it.content for it in store.items if it.metadata.get("source") == "ltm_fact"}
    assert "User lives in Berlin" not in fact_items, "stale fact must not reach retrieval"
    assert "User lives in Paris" in fact_items

    m = store.metrics()
    assert m["supersession_mode"] == mode
    assert m["supersessions"] == 1
    assert m["superseded_retained"] == expect_retained

    # The retention difference is the whole point: arm B keeps the row live
    # (reachable by an as-of query), arm A stamps invalidated_at and loses it.
    live = [row.content async for row in ltm.iter_live()]
    assert ("User lives in Berlin" in live) is (mode == "bitemporal")


@pytest.mark.asyncio
async def test_dead_llm_extractor_raises_instead_of_silently_downgrading() -> None:
    """
    ``FactExtractor`` logs and swallows its own errors, and ``_extract_facts``
    then falls back to regex mining. Without this guard the run keeps going,
    reports a clean accuracy number, and measures the *heuristic* promoter
    rather than the LLM one under test.
    """
    store = ContinuumLTMHaystackStore(
        ltm=InMemoryLTM(),
        embedder=_NullEmbedder(),
        promoter=_CannedPromoter([]),
        fact_extractor=_CannedFactExtractor([]),  # yields nothing → falls back
        llm_available=True,
        ltm_backend_label="in_memory",
    )
    await store.append(_turn("s1", "user", "I moved to Paris."))

    with pytest.raises(RuntimeError, match="failed on all"):
        await store.finalize()


@pytest.mark.asyncio
async def test_heuristic_path_requested_explicitly_does_not_raise() -> None:
    """The guard is about a *silent* downgrade. Asking for the heuristic path
    (``--no-llm-promoter`` → ``llm_available=False``) is legitimate."""
    store = ContinuumLTMHaystackStore(
        ltm=InMemoryLTM(),
        embedder=_NullEmbedder(),
        fact_extractor=_CannedFactExtractor([]),
        llm_available=False,
        ltm_backend_label="in_memory",
    )
    await store.append(_turn("s1", "user", "I moved to Paris."))
    await store.finalize()  # must not raise
    assert store.metrics()["llm_fallbacks"] == 0
    assert store.metrics()["promoter_path"] == "heuristic"


@pytest.mark.asyncio
async def test_working_llm_extractor_records_no_fallbacks() -> None:
    """A healthy LLM path must leave the fallback counter at zero, so the
    metric is a usable signal rather than always-on noise."""
    store = ContinuumLTMHaystackStore(
        ltm=InMemoryLTM(),
        embedder=_NullEmbedder(),
        promoter=_CannedPromoter([Decision(op="ADD", target_id=None, rationale="new")]),
        fact_extractor=_CannedFactExtractor([_fact("User lives in Paris")]),
        llm_available=True,
        ltm_backend_label="in_memory",
    )
    await store.append(_turn("s1", "user", "I moved to Paris."))
    await store.finalize()
    m = store.metrics()
    assert m["llm_fallbacks"] == 0
    assert m["promoter_path"] == "llm"
    assert m["blocks_seen"] == 1


@pytest.mark.asyncio
async def test_no_sessions_does_not_trip_the_guard() -> None:
    """Zero blocks means nothing was asked of the extractor — not a failure."""
    store = ContinuumLTMHaystackStore(
        ltm=InMemoryLTM(),
        embedder=_NullEmbedder(),
        promoter=_CannedPromoter([]),
        fact_extractor=_CannedFactExtractor([]),
        llm_available=True,
        ltm_backend_label="in_memory",
    )
    await store.finalize()  # no append() at all — must not raise
    assert store.metrics()["blocks_seen"] == 0


@pytest.mark.asyncio
async def test_ingest_cache_replays_without_calling_the_llm(tmp_path) -> None:
    """
    The cache must reproduce the same supersession outcome with **zero**
    extractor calls on the second pass. That is the whole point: the second
    arm of A5 costs nothing.
    """
    from evals.longmemeval import ingest_cache as ic

    cache = ic.IngestCache(tmp_path / "cache")

    class _SupersedeEarlierFact:
        """Mirrors the real flow: both facts come from this ingest, and the
        later one supersedes the earlier. Targets are resolved at decide time,
        so they are this run's UUIDs — exactly what the cache must remap."""

        def __init__(self, ltm):
            self._ltm = ltm

        async def decide_operations_batch(self, items):
            # Sessions are now decided as separate batches, so key off what is
            # actually in the store rather than position within a batch.
            out = []
            for fact, _ in items:
                target = None
                if fact.text == "User lives in Paris":
                    async for row in self._ltm.iter_live():
                        if row.content == "User lived in Berlin":
                            target = row.id
                            break
                out.append(
                    Decision(
                        op="DELETE" if target else "ADD",
                        target_id=target,
                        rationale="later fact supersedes earlier",
                    )
                )
            return out

    class _OneFactPerSession:
        """Distinct fact per session — the store now decides a whole session
        against prior state, so a supersession must cross a session boundary."""

        def __init__(self) -> None:
            self.calls = []

        async def extract_facts(self, block, entities):
            self.calls.append(block)
            text = (block.text or "").lower()
            fact_text = "User lives in Paris" if "paris" in text else "User lived in Berlin"
            return [
                Fact(
                    text=fact_text,
                    confidence=0.9,
                    entities_mentioned=[],
                    source_block_id=block.id,
                    category="personal",
                )
            ]

    async def run(mode: str):
        ltm = InMemoryLTM()
        extractor = _OneFactPerSession()
        store = ContinuumLTMHaystackStore(
            ltm=ltm,
            embedder=_NullEmbedder(),
            promoter=_SupersedeEarlierFact(ltm),
            fact_extractor=extractor,
            llm_available=True,
            ltm_backend_label="in_memory",
            supersession_mode=mode,
            cache=cache,
            cache_fingerprint="fp-test",
        )
        await store.append(_turn("s1", "user", "I lived in Berlin."))
        await store.append(_turn("s2", "user", "I moved to Paris."))
        await store.finalize()
        return store, len(extractor.calls)

    first, calls_first = await run("invalidate")
    assert first.metrics()["ingest_cache"] == "miss"
    assert calls_first == 2, "first pass must extract both sessions"

    # Second pass — same haystack, the *other* arm. Must replay.
    second, calls_second = await run("bitemporal")
    assert second.metrics()["ingest_cache"] == "hit"
    assert calls_second == 0, "cached pass must make no extractor calls"

    # And the replayed decision must still produce a real supersession.
    assert second.metrics()["supersessions"] == 1
    assert second.metrics()["superseded_retained"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 4])
async def test_concurrent_extraction_preserves_fact_order(concurrency: int) -> None:
    """
    Fact order determines decision indices, which the ingest cache stores as
    its target mapping. If concurrency reordered facts, cached entries would
    silently retarget supersessions at the wrong rows — so order must be
    identical whether extraction runs sequentially or in parallel.
    """
    import asyncio as _asyncio

    class _SlowPerSessionExtractor:
        """Returns a fact naming its session, with a delay that is *inverted*
        against session order — so a naive gather would surface them backwards."""

        def __init__(self) -> None:
            self.calls = []

        async def extract_facts(self, block, entities):
            self.calls.append(block)
            idx = int((block.text or "s0").split("-")[-1])
            await _asyncio.sleep(0.01 * (5 - idx))  # later sessions finish first
            return [
                Fact(
                    text=f"fact from session {idx}",
                    confidence=0.9,
                    entities_mentioned=[],
                    source_block_id=block.id,
                    category="personal",
                )
            ]

    store = ContinuumLTMHaystackStore(
        ltm=InMemoryLTM(),
        embedder=_NullEmbedder(),
        promoter=_CannedPromoter(
            [Decision(op="ADD", target_id=None, rationale="new") for _ in range(5)]
        ),
        fact_extractor=_SlowPerSessionExtractor(),
        llm_available=True,
        ltm_backend_label="in_memory",
        concurrency=concurrency,
    )
    for i in range(5):
        await store.append(_turn(f"session-{i}", "user", f"session-{i}"))
    await store.finalize()

    got = [it.content for it in store.items if it.metadata.get("source") == "ltm_fact"]
    assert got == [f"fact from session {i}" for i in range(5)], (
        f"fact order diverged at concurrency={concurrency}"
    )
    assert store.metrics()["ingest_concurrency"] == concurrency


def test_neighbours_are_rescored_to_true_cosine_not_rrf() -> None:
    """
    ``ContinuumPromoter`` short-circuits on ``scores.relevance`` treated as cosine:
    ``< add_threshold`` (0.5) → ADD *without calling the LLM*. But
    ``search_hybrid`` puts RRF there — ~1/(60+rank) ≈ 0.016 — so every candidate
    fell under the bar and the promoter was never consulted. Measured in a real
    run: 912 facts → 912 ADDs, zero NOOP/UPDATE/DELETE.

    A unit mismatch, not a threshold to tune.
    """
    from continuum.core.types import ScoreBreakdown
    from evals.longmemeval.continuum_ltm_store import _rescore_as_cosine

    def si(content: str, emb: list[float], rrf: float) -> ScoredItem:
        return ScoredItem(
            item=MemoryItem(content=content, tier=MemoryTier.LTM, embedding=emb),
            scores=ScoreBreakdown(
                relevance=rrf, importance=0.5, recency=0.5, confidence=0.9, composite=rrf
            ),
        )

    qvec = [1.0, 0.0, 0.0]
    neighbours = [
        si("identical direction", [1.0, 0.0, 0.0], 0.0164),  # cosine 1.0
        si("orthogonal", [0.0, 1.0, 0.0], 0.0161),  # cosine 0.0
    ]
    assert all(n.scores.relevance < 0.5 for n in neighbours), "precondition: RRF is tiny"

    rescored = _rescore_as_cosine(qvec, neighbours)
    assert rescored[0].scores.relevance == pytest.approx(1.0)
    assert rescored[1].scores.relevance == pytest.approx(0.0)
    # The aligned neighbour must now clear the promoter's ADD short-circuit.
    assert rescored[0].scores.relevance >= 0.5

    # No embedding on the query → leave scores untouched rather than zeroing
    # everything, which would short-circuit just as hard in the other direction.
    assert _rescore_as_cosine(None, neighbours) is neighbours


@pytest.mark.asyncio
async def test_promoter_never_sees_the_candidates_own_row_as_a_neighbour() -> None:
    """
    The store must decide BEFORE upserting, per session.

    Upserting everything first puts each candidate's own just-inserted row into
    its neighbour set; the promoter then sees its own identical text already in
    memory and returns NOOP/ADD instead of superseding the older contradicting
    fact. Measured against the real promoter:

        neighbours=[Berlin]              -> DELETE
        neighbours=[Paris(self), Berlin] -> NOOP

    That single inversion made every --use-ltm run report supersessions=0.
    """
    seen: list[tuple[str, list[str]]] = []

    class _RecordingPromoter:
        async def decide_operations_batch(self, items):
            for fact, nbrs in items:
                seen.append((fact.text, [n.item.content for n in nbrs]))
            return [Decision(op="ADD", target_id=None, rationale="n/a") for _ in items]

    class _PerSessionExtractor:
        """One distinct fact per session, so 'own row' is unambiguous."""

        def __init__(self) -> None:
            self.calls = []

        async def extract_facts(self, block, entities):
            self.calls.append(block)
            text = (block.text or "").lower()
            fact_text = "User lives in Paris" if "paris" in text else "User lives in Berlin"
            return [
                Fact(
                    text=fact_text,
                    confidence=0.9,
                    entities_mentioned=[],
                    source_block_id=block.id,
                    category="personal",
                )
            ]

    store = ContinuumLTMHaystackStore(
        ltm=InMemoryLTM(),
        embedder=_NullEmbedder(),
        promoter=_RecordingPromoter(),
        fact_extractor=_PerSessionExtractor(),
        llm_available=True,
        ltm_backend_label="in_memory",
    )
    # Two sessions: the earlier establishes a fact, the later contradicts it.
    await store.append(_turn("s1", "user", "I live in Berlin."))
    await store.append(_turn("s2", "user", "I moved to Paris."))
    await store.finalize()

    assert len(seen) == 2, f"expected one decision per session, got {seen}"
    berlin, paris = seen[0], seen[1]
    assert berlin[0] == "User lives in Berlin"
    assert berlin[1] == [], f"first session should see no priors, saw {berlin[1]}"

    # The crux: the later candidate sees the EARLIER fact — and never itself.
    assert paris[0] == "User lives in Paris"
    assert "User lives in Berlin" in paris[1], (
        f"later session must see the earlier fact as prior context, saw {paris[1]}"
    )
    assert "User lives in Paris" not in paris[1], (
        f"candidate was handed its own row as a neighbour: {paris[1]}"
    )


@pytest.mark.asyncio
async def test_ingest_cache_refuses_rows_it_cannot_replay(tmp_path) -> None:
    """
    A decision targeting a row this ingest did not create has no stable
    identity across runs. Caching it would silently degrade the supersession
    to a NOOP on replay, so such rows must stay uncached and keep recomputing
    — a cache that changes results is worse than no cache.
    """
    from evals.longmemeval import ingest_cache as ic

    cache = ic.IngestCache(tmp_path / "cache")

    async def run():
        ltm = InMemoryLTM()
        # Pre-existing row — NOT produced by this ingest.
        stale_id = await ltm.upsert(
            MemoryItem(
                content="User lives in Berlin",
                tier=MemoryTier.LTM,
                metadata={"kind": "fact", "source": "ltm_fact"},
            )
        )
        store = ContinuumLTMHaystackStore(
            ltm=ltm,
            embedder=_NullEmbedder(),
            promoter=_CannedPromoter(
                [Decision(op="DELETE", target_id=stale_id, rationale="contradicts")]
            ),
            fact_extractor=_CannedFactExtractor([_fact("User lives in Paris")]),
            llm_available=True,
            ltm_backend_label="in_memory",
            supersession_mode="bitemporal",
            cache=cache,
            cache_fingerprint="fp-test",
        )
        await store.append(_turn("s2", "user", "I moved to Paris."))
        await store.finalize()
        return store

    first = await run()
    assert first.metrics()["ingest_cache"] == "uncacheable"
    assert first.metrics()["supersessions"] == 1

    # Second run recomputes and still supersedes — no silent NOOP.
    second = await run()
    assert second.metrics()["ingest_cache"] == "uncacheable"
    assert second.metrics()["supersessions"] == 1


@pytest.mark.asyncio
async def test_ingest_cache_misses_when_config_fingerprint_changes(tmp_path) -> None:
    """A different extractor model must not replay facts produced by the old
    one — otherwise a config change silently reuses stale output."""
    from evals.longmemeval import ingest_cache as ic

    cache = ic.IngestCache(tmp_path / "cache")

    async def run(fingerprint: str) -> str:
        store = ContinuumLTMHaystackStore(
            ltm=InMemoryLTM(),
            embedder=_NullEmbedder(),
            promoter=_CannedPromoter([Decision(op="ADD", target_id=None, rationale="new")]),
            fact_extractor=_CannedFactExtractor([_fact("User lives in Paris")]),
            llm_available=True,
            ltm_backend_label="in_memory",
            cache=cache,
            cache_fingerprint=fingerprint,
        )
        await store.append(_turn("s1", "user", "I moved to Paris."))
        await store.finalize()
        return store.metrics()["ingest_cache"]

    assert await run("fp-a") == "miss"
    assert await run("fp-a") == "hit"
    assert await run("fp-b") == "miss", "changing model config must invalidate"


@pytest.mark.asyncio
async def test_bitemporal_update_supersedes_instead_of_overwriting() -> None:
    """
    Arm A's ``UPDATE`` overwrites the target's text, destroying the old value.
    Arm B must instead close the old fact's valid-time and leave its content
    intact, so history survives.
    """
    ltm = InMemoryLTM()
    old_id = await ltm.upsert(
        MemoryItem(
            content="User works at Acme",
            tier=MemoryTier.LTM,
            metadata={"kind": "fact", "source": "ltm_fact"},
        )
    )
    store = ContinuumLTMHaystackStore(
        ltm=ltm,
        embedder=_NullEmbedder(),
        promoter=_CannedPromoter([Decision(op="UPDATE", target_id=old_id, rationale="job change")]),
        fact_extractor=_CannedFactExtractor([_fact("User works at Globex")]),
        llm_available=True,
        ltm_backend_label="in_memory",
        supersession_mode="bitemporal",
    )

    await store.append(_turn("s2", "user", "I switched to Globex."))
    await store.finalize()

    rows = {row.content: row async for row in ltm.iter_live()}
    assert "User works at Acme" in rows, "the prior value must survive an UPDATE"
    assert rows["User works at Acme"].valid_range.valid_to is not None
    assert "User works at Globex" in rows
    assert rows["User works at Globex"].valid_range.valid_to is None
    # An UPDATE retires a prior fact, so it must register as a supersession.
    # Some models (llama-3.3-70b) emit UPDATE where others (gpt-4o-mini) emit
    # DELETE for the same contradiction; counting only DELETE would make the
    # P1c gate read 0 on a run that is superseding throughout.
    assert store.metrics()["supersessions"] == 1


@pytest.mark.asyncio
async def test_replay_facts_only_reuses_facts_but_reruns_decisions(tmp_path) -> None:
    """
    ``replay_facts_only`` is what makes a promotion-logic change measurable
    without paying to re-extract.

    ``ic.config_fingerprint`` hashes models and sampling settings, not decision
    rules — so adding a deterministic short-circuit leaves the fingerprint
    unchanged and a warm cache would replay the OLD decisions, silently never
    exercising the new logic. This flag keeps the expensive half of the cache
    (extraction) and throws away the cheap half (decisions).
    """
    from evals.longmemeval import ingest_cache as ic

    cache = ic.IngestCache(tmp_path / "cache")

    class _Extractor:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def extract_facts(self, block, entities):
            self.calls.append(block)
            text = (block.text or "").lower()
            return [
                Fact(
                    text="User lives in Paris" if "paris" in text else "User lived in Berlin",
                    confidence=0.9,
                    entities_mentioned=[],
                    source_block_id=block.id,
                    category="personal",
                )
            ]

    class _Promoter:
        """Counts how often it is consulted; always ADDs."""

        def __init__(self) -> None:
            self.calls = 0

        async def decide_operations_batch(self, items):
            self.calls += 1
            return [Decision(op="ADD", target_id=None, rationale="new") for _ in items]

    async def run(*, replay_facts_only: bool):
        extractor, promoter = _Extractor(), _Promoter()
        store = ContinuumLTMHaystackStore(
            ltm=InMemoryLTM(),
            embedder=_NullEmbedder(),
            promoter=promoter,
            fact_extractor=extractor,
            llm_available=True,
            ltm_backend_label="in_memory",
            cache=cache,
            cache_fingerprint="fp-replay",
            replay_facts_only=replay_facts_only,
        )
        await store.append(_turn("s1", "user", "I lived in Berlin."))
        await store.append(_turn("s2", "user", "I moved to Paris."))
        await store.finalize()
        return store, len(extractor.calls), promoter.calls

    first, extracts, decides = await run(replay_facts_only=False)
    assert first.metrics()["ingest_cache"] == "miss"
    assert extracts == 2 and decides > 0

    # Plain cache hit: neither extractor nor promoter runs.
    plain, extracts, decides = await run(replay_facts_only=False)
    assert plain.metrics()["ingest_cache"] == "hit"
    assert extracts == 0, "cached facts must not be re-extracted"
    assert decides == 0, "cached decisions must be replayed"

    # Replay mode: facts still free, decisions recomputed.
    replay, extracts, decides = await run(replay_facts_only=True)
    assert replay.metrics()["ingest_cache"] == "hit"
    assert extracts == 0, "the expensive half must still come from cache"
    assert decides > 0, "decisions must be recomputed, not replayed"


@pytest.mark.asyncio
async def test_replay_preserves_session_grouping(tmp_path) -> None:
    """
    Recomputed decisions must see the same per-session batching as a fresh
    ingest.

    Decisions are grouped by session so each candidate is compared only against
    genuinely *prior* facts. The cached payload carries session ids; if the
    replay path drops them, every fact in the row collapses into one batch
    decided against an empty store, every neighbour set is empty, and the run
    reports all-ADD / zero supersessions — which reads exactly like "this
    haystack had no contradictions". Measured on the real rig: 4 supersessions
    replaying cached decisions, 0 recomputing them.
    """
    from evals.longmemeval import ingest_cache as ic

    cache = ic.IngestCache(tmp_path / "cache")

    class _SupersedeBerlin:
        def __init__(self, ltm):
            self._ltm = ltm
            self.batches: list[int] = []

        async def decide_operations_batch(self, items):
            self.batches.append(len(items))
            out = []
            for fact, _ in items:
                target = None
                if fact.text == "User lives in Paris":
                    async for row in self._ltm.iter_live():
                        if row.content == "User lived in Berlin":
                            target = row.id
                            break
                out.append(
                    Decision(
                        op="DELETE" if target else "ADD",
                        target_id=target,
                        rationale="later supersedes earlier",
                    )
                )
            return out

    class _OnePerSession:
        async def extract_facts(self, block, entities):
            text = (block.text or "").lower()
            return [
                Fact(
                    text="User lives in Paris" if "paris" in text else "User lived in Berlin",
                    confidence=0.9,
                    entities_mentioned=[],
                    source_block_id=block.id,
                    category="personal",
                )
            ]

    async def run(*, replay_facts_only: bool):
        ltm = InMemoryLTM()
        promoter = _SupersedeBerlin(ltm)
        store = ContinuumLTMHaystackStore(
            ltm=ltm,
            embedder=_NullEmbedder(),
            promoter=promoter,
            fact_extractor=_OnePerSession(),
            llm_available=True,
            ltm_backend_label="in_memory",
            cache=cache,
            cache_fingerprint="fp-session",
            replay_facts_only=replay_facts_only,
        )
        await store.append(_turn("s1", "user", "I lived in Berlin."))
        await store.append(_turn("s2", "user", "I moved to Paris."))
        await store.finalize()
        return store, promoter.batches

    fresh, fresh_batches = await run(replay_facts_only=False)
    assert fresh.metrics()["supersessions"] == 1
    assert len(fresh_batches) == 2, "one decision batch per session"

    replayed, replay_batches = await run(replay_facts_only=True)
    assert replayed.metrics()["ingest_cache"] == "hit"
    assert len(replay_batches) == 2, "replay must keep per-session batching"
    assert replayed.metrics()["supersessions"] == 1, (
        "recomputed decisions must reproduce the fresh-ingest supersession"
    )
