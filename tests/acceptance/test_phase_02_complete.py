"""
tests/acceptance/test_phase_02_complete.py
==========================================
**Phase 0.2 — LTM + Promotion + Retrieval — quality gate.**

One end-to-end test asserting every Phase 0.2 deliverable is wired together
and working against a real PostgreSQL + pgvector + pg_trgm. If it fails,
Phase 0.3 must not begin.

Checks
------
1. ``memory_nodes`` has the bi-temporal columns ``valid_from`` / ``valid_to``
   / ``invalidated_at``.
2. ``PostgresLTM`` CRUD: ``upsert`` → ``update`` → ``invalidate`` keeps the
   row (history preserved, not deleted).
3. ``PostgresLTM.search_hybrid`` (dense + sparse + RRF) returns the correct
   top-k for a controlled corpus.
4. Entity extraction: ``EntityExtractor`` (GLiNER) + ``LLMEntityExtractor``
   merge correctly.
5. Fact extraction: ``FactExtractor`` distils atomic facts from a
   ``SummaryBlock`` (LLM via injected ``completion_fn``).
6. ``Promoter`` orchestrates real MTM→LTM promotion with ``Mem0Promoter``
   decisions and a Postgres audit sink — ``memory_promotions`` is populated.
7. ``TriggerManager``: new-entity and block-accumulation triggers fire.
8. ``Retriever`` pipeline assembles STM + MTM + LTM into a ``ContextBundle``.
9. Test coverage ≥ 75 % (real ``pytest --cov`` subprocess).
10. Type coverage 100 % (``mypy --strict continuum`` exit 0).

Honest scope
------------
Per the project's acceptance-test convention (see also
``test_phase_01_complete.py``), the heavy ML models (GLiNER, the BGE cross-
encoder) and any LLM calls are replaced with deterministic in-process fakes
— their own unit tests cover behaviour. The role of this gate is the
**integration + DB plumbing**, and it skips cleanly when the full
environment (Postgres + psycopg3 + numpy + mypy + pytest-cov) is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("psycopg", reason="psycopg3 required for Phase 0.2 gate")
np = pytest.importorskip("numpy", reason="numpy required for Phase 0.2 gate")
import psycopg2  # noqa: E402

from continuum.core.config import (  # noqa: E402
    FactExtractionConfig,
    LLMExtractionConfig,
    PromoterConfig,
    RetrieverConfig,
    TriggerConfig,
)
from continuum.core.protocols import LTMProtocol  # noqa: E402
from continuum.core.types import (  # noqa: E402
    MemoryItem,
    MemoryTier,
    Query,
    SummaryBlock,
    TokenBudget,
)
from continuum.extraction import (  # noqa: E402
    Entity,
    EntityExtractor,
    Fact,
    FactExtractor,
    LLMEntityExtractor,
)
from continuum.promotion import (  # noqa: E402
    Mem0Promoter,
    Promoter,
    TriggerManager,
    make_postgres_audit_sink,
)
from continuum.retrieval import Retriever  # noqa: E402
from continuum.stores.postgres import PostgresLTM, PostgresMTM  # noqa: E402
from continuum.stores.stm.postgres_stm import PostgresSTM  # noqa: E402

pytestmark = [pytest.mark.acceptance, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
MIG_002 = REPO_ROOT / "migrations" / "002_pgvector_upgrade.sql"
MIG_003 = REPO_ROOT / "migrations" / "003_lexical_search.sql"
_RECURSION_ENV = "CONTINUUM_ACCEPTANCE_RUNNING"
_MIN_COVERAGE_PCT = 75.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_sql_file(dsn: str, path: Path) -> None:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(path.read_text())
    finally:
        conn.close()


def _scalar(dsn: str, sql: str, params: tuple = ()) -> Any:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None
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
# Deterministic fakes (LLM / model components)
# ---------------------------------------------------------------------------


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _tool_response(args_list: list[dict]) -> SimpleNamespace:
    calls = [
        SimpleNamespace(function=SimpleNamespace(name="memory_operation", arguments=json.dumps(a)))
        for a in args_list
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=80, completion_tokens=20),
    )


class FakeGLiNER:
    """Deterministic GLiNER stand-in returning one PERSON + one ORG span."""

    def predict_entities(
        self, text: str, labels: list[str], *, threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        return [
            {"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.95},
            {"text": "Acme Corp", "label": "org", "start": 15, "end": 24, "score": 0.88},
        ]


class FakeFactExtractor:
    """One promoted fact per MTM block (deterministic, no LLM)."""

    async def extract_facts(self, block: Any, entities: list[Any]) -> list[Fact]:
        return [
            Fact(
                text=f"Promoted fact about {block.text}",
                confidence=0.9,
                entities_mentioned=[],
                source_block_id=block.id,
                category="promoted",
            )
        ]


class FakeReranker:
    """Deterministic stand-in for the BGE cross-encoder."""

    async def rerank(self, query: str, items: list[Any]) -> list[Any]:
        from continuum.core.types import ScoreBreakdown, ScoredItem

        def rank(content: str) -> float:
            c = content.lower()
            if "alpha" in c:
                return 0.99
            if "beta" in c:
                return 0.5
            return 0.1

        out: list[tuple[float, ScoredItem]] = []
        for it in items:
            r = rank(it.item.content)
            sb = ScoreBreakdown(
                relevance=r,
                importance=it.scores.importance,
                recency=it.scores.recency,
                confidence=it.scores.confidence,
                composite=r,
            )
            out.append((r, ScoredItem(item=it.item, scores=sb)))
        out.sort(key=lambda p: p[0], reverse=True)
        return [si for _, si in out]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


async def test_ltm_and_promotion_complete(postgres_db: str) -> None:
    if os.environ.get(_RECURSION_ENV):
        pytest.skip("nested acceptance invocation — skipping to avoid recursion")
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy not installed — cannot enforce the type-coverage gate")
    if importlib.util.find_spec("pytest_cov") is None:
        pytest.skip("pytest-cov not installed — cannot enforce the coverage gate")

    _apply_sql_file(postgres_db, MIG_002)
    _apply_sql_file(postgres_db, MIG_003)

    # ── 1. Bi-temporal columns present in memory_nodes ───────────────────────
    cols = _scalar(
        postgres_db,
        """
        SELECT array_agg(column_name::text) FROM information_schema.columns
        WHERE table_name='memory_nodes'
          AND column_name IN ('valid_from','valid_to','invalidated_at')
        """,
    )
    assert cols is not None
    assert {"valid_from", "valid_to", "invalidated_at"} <= set(cols), cols

    # ── 2. LTM CRUD + bi-temporal invalidate (row NOT deleted) ───────────────
    ltm = PostgresLTM(dsn=postgres_db, embedding_type="halfvec")
    alpha_vec = _unit_vec(seed=1)
    alpha_item = MemoryItem(
        content="alpha doc text",
        tier=MemoryTier.LTM,
        confidence=0.9,
        importance=0.7,
        embedding=_near(alpha_vec, seed=10),
        metadata={"kind": "fact"},
    )
    aid = await ltm.upsert(alpha_item)
    assert isinstance(aid, uuid.UUID)
    assert isinstance(ltm, LTMProtocol)  # name-based runtime check

    # update bumps updated_at, leaves the row live.
    await ltm.update(aid, {"importance": 0.95, "content": "alpha doc revised"})
    assert (
        _scalar(postgres_db, "SELECT importance FROM memory_nodes WHERE id=%s", (str(aid),)) == 0.95
    )

    # invalidate does NOT delete the row — bi-temporal close only.
    pre = _scalar(postgres_db, "SELECT count(*) FROM memory_nodes WHERE id=%s", (str(aid),))
    when = datetime(2030, 1, 1, tzinfo=UTC)
    await ltm.invalidate(aid, when)
    post = _scalar(postgres_db, "SELECT count(*) FROM memory_nodes WHERE id=%s", (str(aid),))
    inv = _scalar(postgres_db, "SELECT invalidated_at FROM memory_nodes WHERE id=%s", (str(aid),))
    assert pre == post == 1  # row preserved
    assert inv is not None and inv.astimezone(UTC) == when

    # ── 3. Hybrid search returns the right top-k ─────────────────────────────
    # Three live LTM rows (alpha is now invalidated; insert a fresh alpha).
    for content, seed in (
        ("alpha doc text", 11),
        ("beta doc text", 2),
        ("gamma doc text", 3),
    ):
        await ltm.upsert(
            MemoryItem(
                content=content,
                tier=MemoryTier.LTM,
                confidence=0.9,
                importance=0.7,
                embedding=_near(alpha_vec, seed=seed)
                if content.startswith("alpha")
                else _unit_vec(seed=seed),
                metadata={"kind": "fact"},
            )
        )
    hits = await ltm.search_hybrid(
        Query(text="alpha", embedding=alpha_vec, tiers=[MemoryTier.LTM]),
        k=3,
    )
    assert hits, "hybrid search returned nothing"
    assert hits[0].item.content == "alpha doc text"  # dense + sparse agree

    # ── 4. Entity extraction (GLiNER fake) + LLM merge ───────────────────────
    gliner = EntityExtractor(model_factory=lambda _m, _d: FakeGLiNER())
    ents, _rels = await gliner.extract("Alice works at Acme Corp")
    names = {e.text for e in ents}
    assert "Alice" in names and "Acme Corp" in names

    async def llm_x_cf(**_kw: Any) -> Any:
        return _llm_response(
            json.dumps(
                {
                    "entities": [{"text": "Engineer", "type": "CONCEPT", "confidence": 0.85}],
                    "relations": [
                        {
                            "subject": "Alice",
                            "predicate": "EMPLOYED_BY",
                            "object": "Acme Corp",
                            "confidence": 0.9,
                        },
                    ],
                }
            )
        )

    llm_x = LLMEntityExtractor(LLMExtractionConfig(), completion_fn=llm_x_cf)
    merged_ents, rels = await llm_x.extract("Alice works at Acme Corp", ents)
    merged_names = {e.text for e in merged_ents}
    assert merged_names >= {"Alice", "Acme Corp", "Engineer"}  # LLM enhanced
    assert any(r.predicate == "EMPLOYED_BY" for r in rels)

    # ── 5. Fact extraction from a SummaryBlock ───────────────────────────────
    async def facts_cf(**_kw: Any) -> Any:
        return _llm_response(
            json.dumps(
                {
                    "facts": [
                        {
                            "text": "Alice works at Acme Corp",
                            "confidence": 0.95,
                            "entities": ["Alice", "Acme Corp"],
                            "type": "employment",
                        }
                    ]
                }
            )
        )

    fx = FactExtractor(FactExtractionConfig(), completion_fn=facts_cf)
    block = SummaryBlock(
        text="Alice mentioned she works at Acme.",
        id=uuid.uuid4(),
    )
    facts = await fx.extract_facts(block, merged_ents)
    assert facts and isinstance(facts[0], Fact)
    assert facts[0].source_block_id == block.id
    assert facts[0].text == "Alice works at Acme Corp"

    # ── 6. Promoter end-to-end with Mem0 + Postgres audit sink ───────────────
    mtm = PostgresMTM(dsn=postgres_db)
    # Seed 2 unprocessed MTM blocks.
    seeded_block_ids: list[uuid.UUID] = []
    for i in range(2):
        bid = await mtm.add_summary(
            MemoryItem(
                content=f"summary block {i}",
                tier=MemoryTier.MTM,
                session_id="acc-2",
                metadata={"role": "summary"},
            )
        )
        seeded_block_ids.append(bid)

    async def mem0_cf(**_kw: Any) -> Any:
        # One ADD tool-call per candidate (parallel function calling).
        return _tool_response([{"operation": "ADD", "rationale": "novel fact"} for _ in range(2)])

    audit_sink = make_postgres_audit_sink(dsn=postgres_db)
    decider = Mem0Promoter(
        PromoterConfig(confidence_threshold=0.6, add_threshold=0.5),
        completion_fn=mem0_cf,
        audit_sink=audit_sink,
    )

    ltm_before = _scalar(
        postgres_db,
        "SELECT count(*) FROM memory_nodes WHERE layer='LTM' AND invalidated_at IS NULL",
    )
    promotions_before = _scalar(postgres_db, "SELECT count(*) FROM memory_promotions")

    promoter = Promoter(
        PromoterConfig(confidence_threshold=0.6),
        mtm=mtm,
        ltm=ltm,
        fact_extractor=FakeFactExtractor(),
        decider=decider,
    )
    report = await promoter.promote()
    assert report.blocks_processed == 2
    assert len(report.added) == 2
    assert report.errors == []

    ltm_after = _scalar(
        postgres_db,
        "SELECT count(*) FROM memory_nodes WHERE layer='LTM' AND invalidated_at IS NULL",
    )
    promotions_after = _scalar(postgres_db, "SELECT count(*) FROM memory_promotions")
    assert ltm_after - ltm_before == 2
    # Audit log: 2 Mem0 ADD rows + 2 mark_processed NOOP rows = +4.
    assert promotions_after - promotions_before >= 4
    # Audit log captures every decision (short-circuited *and* LLM).
    add_rows = _scalar(
        postgres_db,
        "SELECT count(*) FROM memory_promotions WHERE op='ADD'",
    )
    assert add_rows >= 2

    # ── 7. Triggers: new-entity + block accumulation ─────────────────────────
    from continuum.core.background import BackgroundQueue

    bg = BackgroundQueue(backoff_initial=0.0, backoff_max=0.0)
    tm = TriggerManager(
        TriggerConfig(block_threshold=1),  # any unprocessed → trigger
        mtm=mtm,
        ltm=ltm,
        promoter=promoter,
        background=bg,
    )
    # No "Bob" entity in LTM → new-entity trigger fires.
    assert await tm.check_new_entity([Entity("Bob", "PERSON", 0, 3, 0.95)]) is True
    # Add an unprocessed block so accumulation ≥ 1 → trigger fires.
    await mtm.add_summary(
        MemoryItem(
            content="fresh summary",
            tier=MemoryTier.MTM,
            session_id="trig-1",
            metadata={"role": "summary"},
        )
    )
    assert await tm.check_block_accumulation() is True

    # after_turn queues a promotion through the BackgroundQueue (non-blocking).
    async with bg:
        queued = await tm.after_turn([Entity("Bob", "PERSON", 0, 3, 0.95)])
        assert queued is True
        await bg.drain()

    # ── 8. Retrieval pipeline assembles STM + MTM + LTM ──────────────────────
    stm = PostgresSTM(dsn=postgres_db, max_tokens=100_000)
    async with stm:
        for i in range(5):
            await stm.append(
                MemoryItem(
                    content=f"turn {i}",
                    tier=MemoryTier.STM,
                    session_id="acc-2",
                    metadata={"role": "user" if i % 2 == 0 else "assistant"},
                )
            )
        retriever = Retriever(
            RetrieverConfig(k1=5, stm_turns=5, ltm_top_k=5),
            ltm=ltm,
            stm=stm,
            mtm=mtm,
            reranker=FakeReranker(),
            session_id="acc-2",
        )
        budget = TokenBudget(
            total=4000,
            stm_reserved=500,
            mtm_reserved=1000,
            ltm_reserved=1000,
            response_reserved=500,
        )
        bundle = await retriever.retrieve(
            Query(text="alpha", embedding=alpha_vec, session_id="acc-2"),
            budget,
        )
    assert set(bundle.tier_breakdown) == {"stm", "mtm", "ltm"}
    assert bundle.tier_breakdown["stm"] > 0
    assert bundle.tier_breakdown["ltm"] > 0
    assert bundle.tokens_used <= budget.total
    assert bundle.items[0].content == "alpha doc text"  # reranked top

    # ── 9. Test coverage ≥ 75 % (real pytest --cov subprocess) ───────────────
    env = {**os.environ, _RECURSION_ENV: "1"}
    t0 = time.perf_counter()
    cov = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit",
            "-m",
            "unit",
            "--cov=continuum",
            "--cov-report=term",
            "--cov-fail-under=0",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert cov.returncode == 0, (cov.stdout + cov.stderr)[-3000:]
    total_line = next(
        (ln for ln in (cov.stdout + cov.stderr).splitlines() if ln.strip().startswith("TOTAL")),
        None,
    )
    assert total_line, "no coverage TOTAL line found"
    pct = float(re.search(r"(\d+(?:\.\d+)?)%", total_line).group(1))  # type: ignore[union-attr]
    assert pct >= _MIN_COVERAGE_PCT, (
        f"coverage {pct:.1f}% < {_MIN_COVERAGE_PCT}% gate ({total_line.strip()})"
    )
    cov_dt = time.perf_counter() - t0

    # ── 10. Type coverage 100 % (mypy --strict continuum) ────────────────────
    t0 = time.perf_counter()
    mp = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "continuum"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert mp.returncode == 0, (
        f"mypy --strict continuum failed (type coverage < 100%):\n{(mp.stdout + mp.stderr)[-3000:]}"
    )
    mp_dt = time.perf_counter() - t0

    # Diagnostics — surfaces nicely on -v output.
    print(
        f"\n[phase-0.2] coverage {pct:.1f}% in {cov_dt:.1f}s | "
        f"mypy --strict in {mp_dt:.1f}s | "
        f"ltm_promoted={len(report.added)} | bundle_items={len(bundle.items)}"
    )
