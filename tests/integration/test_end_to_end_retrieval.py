"""
tests/integration/test_end_to_end_retrieval.py
==============================================
End-to-end test against a real PostgreSQL + pgvector + pg_trgm database.

Wires every concrete component together — PostgresSTM / PostgresMTM /
PostgresLTM, the Mem0-style Promoter (with deterministic fakes for the LLM
extractor/decider so the run is reproducible offline), the deterministic
Scorer, and the Retriever pipeline (with a deterministic fake reranker so
the BGE cross-encoder is not required) — and asserts the full
``query → ContextBundle`` flow.

Honest scope (deliberate, documented)
-------------------------------------
* The **fact extractor**, **Mem0 decider**, and **cross-encoder reranker**
  are replaced with tiny deterministic fakes. Their interfaces are unit-
  tested elsewhere; the role of this test is the **pipeline + DB**
  integration, not the LLM/model components.
* Migrations 002 (halfvec/HNSW) and 003 (pg_trgm) are applied idempotently
  at the top of the test — ``conftest.postgres_db`` only applies 001.

Skips
-----
* ``postgres_db`` skips if Postgres / psycopg2 is unavailable.
* This module ``importorskip``s **psycopg3** and **numpy**.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("psycopg", reason="psycopg3 required for the async stores")
np = pytest.importorskip("numpy", reason="numpy required for embeddings")
import psycopg2  # noqa: E402  (only reachable once psycopg present)

from continuum.core.config import PromoterConfig, RetrieverConfig  # noqa: E402
from continuum.core.types import (  # noqa: E402
    BiTemporalRange,
    MemoryItem,
    MemoryTier,
    Query,
    ScoredItem,
    TokenBudget,
)
from continuum.extraction.fact_extractor import Fact  # noqa: E402
from continuum.promotion import Decision, Promoter  # noqa: E402
from continuum.retrieval import Retriever  # noqa: E402
from continuum.stores.postgres import PostgresLTM, PostgresMTM  # noqa: E402
from continuum.stores.stm.postgres_stm import PostgresSTM  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
MIG_002 = REPO_ROOT / "migrations" / "002_pgvector_upgrade.sql"
MIG_003 = REPO_ROOT / "migrations" / "003_lexical_search.sql"


# ---------------------------------------------------------------------------
# Per-test isolation: wipe shared tables
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean(postgres_db: str) -> Iterator[None]:
    conn = psycopg2.connect(postgres_db)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE memory_nodes, memory_edges, memory_access_log, "
            "memory_promotions, memory_episodes RESTART IDENTITY CASCADE"
        )
        cur.execute("SELECT to_regclass('public.stm_messages')")
        if cur.fetchone()[0] is not None:
            cur.execute("TRUNCATE stm_messages")
    conn.close()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply(dsn: str, path: Path) -> None:
    """Apply an idempotent self-transactional migration via psycopg2."""
    conn = psycopg2.connect(dsn)
    conn.autocommit = True  # the file owns its BEGIN/COMMIT
    try:
        with conn.cursor() as cur:
            cur.execute(path.read_text())
    finally:
        conn.close()


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024)
    return (v / np.linalg.norm(v)).tolist()


def _near(base: list[float], seed: int, eps: float = 0.01) -> list[float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(base) + eps * rng.standard_normal(1024)
    return (a / np.linalg.norm(a)).tolist()


def _count(dsn: str, sql: str, params: tuple = ()) -> int:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        conn.close()


def _insert_edge(
    dsn: str,
    src: uuid.UUID,
    dst: uuid.UUID,
    predicate: str,
    weight: float = 0.9,
) -> None:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory_edges "
                "(id, source_id, target_id, predicate, weight) "
                "VALUES (gen_random_uuid(), %s, %s, %s, %s)",
                (str(src), str(dst), predicate, weight),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Deterministic fakes for the LLM-driven stages
# ---------------------------------------------------------------------------


class FakeFacts:
    """One promoted fact per MTM block, deterministic from block.text."""

    async def extract_facts(
        self, block: Any, entities: list[Any]
    ) -> list[Fact]:
        return [
            Fact(
                text=f"Promoted: {block.text}",
                confidence=0.9,
                entities_mentioned=[],
                source_block_id=block.id,
                category="promoted",
            )
        ]


class FakeDecider:
    """Always ADD — exercises Promoter→ltm.upsert against the real DB."""

    async def decide_operations_batch(
        self, pairs: list[tuple[Fact, list[ScoredItem]]]
    ) -> list[Decision]:
        return [
            Decision(
                op="ADD", target_id=None, rationale="new",
                candidate_text=p[0].text, model="gpt-4o-mini",
                tokens_in=10, tokens_out=2,
            )
            for p in pairs
        ]


class FakeReranker:
    """
    Deterministic stand-in for the BGE cross-encoder.

    Ranks by lexical evidence (the BGE model is not loaded in CI):
    exact target sentence > contains "Alice works" > contains "Alice" >
    contains "Acme" > everything else.
    """

    async def rerank(
        self, query: str, items: list[ScoredItem]
    ) -> list[ScoredItem]:
        from continuum.core.types import ScoreBreakdown

        def rr(content: str) -> float:
            c = content.lower()
            if c == "alice works at acme corp":
                return 0.99
            if "alice works" in c:
                return 0.90
            if "alice" in c:
                return 0.80
            if "acme" in c:
                return 0.60
            return 0.10

        out: list[tuple[float, ScoredItem]] = []
        for it in items:
            r = rr(it.item.content)
            new_scores = ScoreBreakdown(
                relevance=r, importance=it.scores.importance,
                recency=it.scores.recency, confidence=it.scores.confidence,
                composite=r,
            )
            out.append((r, ScoredItem(item=it.item, scores=new_scores)))
        out.sort(key=lambda p: p[0], reverse=True)
        return [si for _, si in out]


# ---------------------------------------------------------------------------
# The end-to-end test
# ---------------------------------------------------------------------------


async def test_query_to_context_assembly(postgres_db: str) -> None:
    # ── 0. Reach the Phase 0.1 DB state (idempotent) ─────────────────────────
    _apply(postgres_db, MIG_002)        # → embedding halfvec(1024) + HNSW
    _apply(postgres_db, MIG_003)        # → pg_trgm GIN

    session_id = f"e2e-{uuid.uuid4().hex[:8]}"
    query_text = "What does Alice do?"
    query_vec = _unit_vec(seed=1)        # "alice topic" direction

    stm = PostgresSTM(dsn=postgres_db, max_tokens=100_000)
    mtm = PostgresMTM(dsn=postgres_db)
    ltm = PostgresLTM(dsn=postgres_db, embedding_type="halfvec")

    # ── 1a. STM: 20 turns ────────────────────────────────────────────────────
    async with stm:
        for i in range(20):
            await stm.append(
                MemoryItem(
                    id=str(uuid.uuid4()),
                    content=f"turn {i:02d}: chatting about projects",
                    tier=MemoryTier.STM,
                    session_id=session_id,
                    metadata={"role": "user" if i % 2 == 0 else "assistant"},
                )
            )

        # ── 1b. MTM: 10 summary blocks (some processed, some not) ─────────────
        block_ids: list[uuid.UUID] = []
        for i in range(10):
            bid = await mtm.add_summary(
                MemoryItem(
                    content=f"Project update summary number {i}.",
                    tier=MemoryTier.MTM,
                    session_id=session_id,
                    metadata={"role": "summary", "tokens": 5},
                )
            )
            block_ids.append(bid)
        # 4 already-processed (no facts will be promoted from these)
        await mtm.mark_processed(block_ids[:4])

        # ── 1c. LTM: 100 facts (incl. Alice entity, Acme entity, key fact) ────
        ltm_alice_entity = MemoryItem(
            content="Alice",
            tier=MemoryTier.LTM,
            confidence=0.95,
            importance=0.8,
            embedding=_near(query_vec, seed=10),
            metadata={"kind": "entity", "label": "PERSON"},
            valid_range=BiTemporalRange(valid_from=datetime.now(UTC)),
        )
        alice_id = await ltm.upsert(ltm_alice_entity)

        ltm_acme_entity = MemoryItem(
            content="Acme Corp",
            tier=MemoryTier.LTM,
            confidence=0.95,
            importance=0.8,
            embedding=_unit_vec(seed=2),      # orthogonal-ish to query
            metadata={"kind": "entity", "label": "ORG"},
        )
        acme_id = await ltm.upsert(ltm_acme_entity)

        ltm_alice_fact = MemoryItem(
            content="Alice works at Acme Corp",
            tier=MemoryTier.LTM,
            confidence=0.95,
            importance=0.9,
            embedding=_near(query_vec, seed=11),
            metadata={"kind": "fact"},
        )
        await ltm.upsert(ltm_alice_fact)

        # Filler facts — random, mostly far from the query direction.
        for i in range(97):
            await ltm.upsert(
                MemoryItem(
                    content=f"Unrelated fact about topic {i}",
                    tier=MemoryTier.LTM,
                    confidence=0.7,
                    importance=0.4,
                    embedding=_unit_vec(seed=1000 + i),
                    metadata={"kind": "fact"},
                )
            )

        # Knowledge-graph edge: Alice WORKS_AT Acme.
        _insert_edge(postgres_db, alice_id, acme_id, "WORKS_AT", 0.95)

        # ── 2. Promotion: drive 6 unprocessed MTM blocks → LTM ────────────────
        ltm_before = _count(
            postgres_db,
            "SELECT count(*) FROM memory_nodes "
            "WHERE layer='LTM' AND invalidated_at IS NULL",
        )
        unproc_before = _count(
            postgres_db,
            "SELECT count(*) FROM memory_nodes mn "
            "LEFT JOIN memory_promotions mp ON mp.target_id = mn.id "
            "WHERE mn.layer='MTM' AND mn.invalidated_at IS NULL "
            "AND mp.id IS NULL",
        )
        assert unproc_before == 6, f"expected 6 unprocessed, got {unproc_before}"

        promoter = Promoter(
            PromoterConfig(confidence_threshold=0.6),
            mtm=mtm, ltm=ltm,
            fact_extractor=FakeFacts(), decider=FakeDecider(),
        )
        report = await promoter.promote()

        assert report.blocks_processed == 6
        assert len(report.added) == 6                         # one ADD per block
        assert report.errors == []
        # LTM grew by exactly the promoted-block count.
        ltm_after = _count(
            postgres_db,
            "SELECT count(*) FROM memory_nodes "
            "WHERE layer='LTM' AND invalidated_at IS NULL",
        )
        assert ltm_after == ltm_before + 6
        # All MTM blocks are now processed.
        unproc_after = _count(
            postgres_db,
            "SELECT count(*) FROM memory_nodes mn "
            "LEFT JOIN memory_promotions mp ON mp.target_id = mn.id "
            "WHERE mn.layer='MTM' AND mn.invalidated_at IS NULL "
            "AND mp.id IS NULL",
        )
        assert unproc_after == 0

        # ── 3 + 4. Run the retriever pipeline against the populated DB ───────
        retriever = Retriever(
            # k1=3 keeps the recall pool small (~Alice fact + Alice entity)
            # so the Acme Corp node genuinely arrives via the WORKS_AT edge
            # rather than already being a hybrid hit — the point of this
            # E2E assertion is that the graph stage adds new candidates.
            RetrieverConfig(
                k1=3, stm_turns=10, ltm_top_k=10, graph_expand_n=10,
            ),
            ltm=ltm, stm=stm, mtm=mtm,
            reranker=FakeReranker(),
            session_id=session_id,
        )
        budget = TokenBudget(
            total=8000, stm_reserved=1000, mtm_reserved=2000,
            ltm_reserved=2000, response_reserved=1000,
        )
        bundle = await retriever.retrieve(
            Query(text=query_text, embedding=query_vec, session_id=session_id),
            budget,
        )

    # ── 5. Assertions ────────────────────────────────────────────────────────
    contents = [it.content for it in bundle.items]

    # The reranker confirms the target fact is the top item overall.
    assert bundle.items[0].content == "Alice works at Acme Corp", contents[:5]

    # LTM portion contains Alice-related facts.
    assert any("Alice" in c for c in contents)
    # Graph expansion pulled the Acme Corp entity in (via the Alice edge).
    assert "Acme Corp" in contents
    acme_item = next(it for it in bundle.items if it.content == "Acme Corp")
    assert acme_item.metadata.get("via_graph") == str(alice_id)
    assert bundle.debug_info["graph_added"] >= 1

    # STM/MTM populated as configured.
    assert bundle.debug_info["stm"] == 10                 # config.stm_turns
    assert bundle.debug_info["mtm"] >= 1                   # at least one summary
    assert bundle.debug_info["ltm_final"] <= 10           # config.ltm_top_k cap

    # Tier accounting + budget invariants.
    assert set(bundle.tier_breakdown) == {"stm", "mtm", "ltm"}
    assert bundle.tier_breakdown["stm"] > 0
    assert bundle.tier_breakdown["mtm"] > 0
    assert bundle.tier_breakdown["ltm"] > 0
    assert bundle.tier_breakdown["mtm"] <= budget.mtm_reserved
    assert bundle.tokens_used <= budget.total
    assert bundle.tokens_used == sum(bundle.tier_breakdown.values())
    assert len(bundle.messages) == len(bundle.items)
