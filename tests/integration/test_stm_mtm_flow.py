"""
tests/integration/test_stm_mtm_flow.py
======================================
End-to-end STM ↔ MTM flow against a **real** PostgreSQL database.

Fixtures
--------
``postgres_db`` (session, conftest.py) creates an isolated DB with migration
001 applied — ``memory_nodes`` (``embedding vector(1024)``),
``memory_promotions``, etc. ``PostgresSTM`` lazily creates its own
``stm_messages`` table via ``ensure_schema``.

Isolation
---------
The autouse ``_clean`` fixture TRUNCATEs every shared table before each test,
so each scenario starts from empty state and they can run in any order.

Skips
-----
* ``postgres_db`` skips when Postgres / psycopg2 is unavailable.
* This module ``importorskip``s **psycopg3** (the stores' async driver), so
  the suite degrades gracefully where only psycopg2 is installed.

Honest scope notes
------------------
* The framework has no LLM summarizer wired into eviction yet, so
  ``flush_to`` performs a *structural* STM→MTM transfer (one MTM block per
  evicted turn). Semantic summarization is the Promoter's job (later tier).
* Topic-shift detection is not yet a framework component; scenario 2 drives
  the policy in-test (cosine of consecutive embeddings) and calls the real
  ``flush_to`` — STM does not persist embeddings, so they live in the test.
* ``ContinuumSession.checkpoint()`` is MTM→LTM; the STM→MTM end-of-session
  primitive is ``stm.flush_to``, which scenario 4 exercises directly.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

# The async stores require psycopg3; skip the whole module without it.
pytest.importorskip("psycopg", reason="psycopg3 required for PostgresSTM/MTM")

from continuum.core.types import MemoryItem, MemoryTier, SummaryBlock
from continuum.stores.postgres.mtm import PostgresMTM
from continuum.stores.stm.postgres_stm import PostgresSTM

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Isolation: wipe shared tables before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean(postgres_db: str) -> Iterator[None]:
    import psycopg2

    conn = psycopg2.connect(postgres_db)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE memory_nodes, memory_edges, memory_access_log, "
            "memory_promotions RESTART IDENTITY CASCADE"
        )
        cur.execute("SELECT to_regclass('public.stm_messages')")
        if cur.fetchone()[0] is not None:
            cur.execute("TRUNCATE stm_messages")
    conn.close()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(seed: int) -> list[float]:
    """A deterministic 1024-d unit vector (matches memory_nodes vector(1024))."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024)
    return (v / np.linalg.norm(v)).tolist()


def _near(vec: list[float], seed: int, eps: float = 0.01) -> list[float]:
    """A unit vector very close to *vec* (cosine ≈ 0.999)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(vec) + eps * rng.standard_normal(1024)
    return (a / np.linalg.norm(a)).tolist()


def _cos(a: list[float], b: list[float]) -> float:
    av, bv = np.asarray(a), np.asarray(b)
    return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))


def _turn(session: str, i: int, topic: str = "general") -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=f"turn {i:03d} about {topic} topic lorem ipsum dolor",
        tier=MemoryTier.STM,
        session_id=session,
        agent_id="it-agent",
        metadata={"role": "user" if i % 2 == 0 else "assistant"},
    )


def _count_mtm(postgres_db: str, session: str | None = None) -> int:
    import psycopg2

    conn = psycopg2.connect(postgres_db)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            if session is None:
                cur.execute("SELECT count(*) FROM memory_nodes WHERE layer = 'MTM'")
            else:
                cur.execute(
                    "SELECT count(*) FROM memory_nodes "
                    "WHERE layer = 'MTM' AND tags->>'session_id' = %s",
                    (session,),
                )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. STM exceeds budget → flush_to(MTM), keep recent N
# ---------------------------------------------------------------------------


async def test_stm_flush_to_mtm(postgres_db: str) -> None:
    session = f"flush-{uuid.uuid4().hex[:8]}"
    keep_recent = 10

    stm = PostgresSTM(dsn=postgres_db, max_tokens=120, reserved_for_response=20)
    mtm = PostgresMTM(dsn=postgres_db)

    async with stm:
        for i in range(50):
            await stm.append(_turn(session, i))

        # STM is well over 70% of its usable budget (eff = 120 - 20 = 100;
        # 50 turns × ~8 tokens ≫ 100).
        stats = await stm.stats(session)
        assert stats["utilization"] >= 0.7

        # Eviction policy: snapshot the recent window, flush everything to
        # MTM, then restore the recent window to STM.
        kept = list(await stm.get_recent(session, keep_recent))
        moved = await stm.flush_to(session, mtm)
        assert moved == 50
        for item in kept:
            await stm.append(item)

        # STM now holds exactly the recent N …
        post = await stm.stats(session)
        assert post["message_count"] == keep_recent
        kept_contents = [m.content for m in await stm.get_recent(session, 99)]
        assert kept_contents == [k.content for k in kept]

    # … and MTM received every evicted turn as a block.
    assert _count_mtm(postgres_db, session) == 50
    unproc = [b async for b in mtm.scan_unprocessed()]
    assert len(unproc) == 50
    assert all(b.session_id == session for b in unproc)


# ---------------------------------------------------------------------------
# 2. Topic shift (cosine < 0.4) triggers flush
# ---------------------------------------------------------------------------


async def test_topic_shift_triggers_flush(postgres_db: str) -> None:
    session = f"topic-{uuid.uuid4().hex[:8]}"
    stm = PostgresSTM(dsn=postgres_db, max_tokens=100_000)
    mtm = PostgresMTM(dsn=postgres_db)

    base_a = _unit_vec(seed=1)
    base_b = _unit_vec(seed=999)
    assert _cos(base_a, base_b) < 0.4  # genuinely different topics

    flushes = 0
    async with stm:
        # Topic A — 5 coherent turns.
        centroid: list[float] | None = None
        for i in range(5):
            emb = _near(base_a, seed=10 + i)
            await stm.append(_turn(session, i, topic="alpha"))
            centroid = emb if centroid is None else (np.mean([centroid, emb], axis=0)).tolist()

        # First Topic-B turn: detect the shift BEFORE appending it.
        b_emb = _near(base_b, seed=50)
        assert centroid is not None
        if _cos(centroid, b_emb) < 0.4:  # ← topic-shift policy
            flushes += await_flush(stm, session, mtm)

        # Now continue the conversation on Topic B.
        for i in range(5, 10):
            await stm.append(_turn(session, i, topic="beta"))

        # The 5 Topic-A turns were flushed; STM holds only Topic-B turns.
        remaining = await stm.get_recent(session, 99)
        assert len(remaining) == 5
        assert all("beta" in m.content for m in remaining)

    assert flushes == 1
    a_blocks = [b async for b in mtm.scan_unprocessed()]
    assert len(a_blocks) == 5
    assert all("alpha" in b.text for b in a_blocks)


async def await_flush(stm: PostgresSTM, session: str, mtm: PostgresMTM) -> int:
    """Return 1 if a non-empty flush occurred (test policy helper)."""
    moved = await stm.flush_to(session, mtm)
    return 1 if moved > 0 else 0


# ---------------------------------------------------------------------------
# 3. MTM dedupe-on-write (cosine > 0.92)
# ---------------------------------------------------------------------------


async def test_mtm_deduplication(postgres_db: str) -> None:
    session = f"dedup-{uuid.uuid4().hex[:8]}"
    mtm = PostgresMTM(
        dsn=postgres_db,
        embedding_type="vector",  # migration 001 column is vector(1024)
        dedupe_threshold=0.92,
    )

    v1 = _unit_vec(seed=7)
    v2 = _near(v1, seed=8, eps=0.01)  # cosine ≈ 0.999  (> 0.92)
    v3 = _unit_vec(seed=4242)  # ~orthogonal     (< 0.92)
    assert _cos(v1, v2) > 0.92
    assert _cos(v1, v3) < 0.92

    id1 = await mtm.add_summary(
        SummaryBlock(text="DB connection pooling notes", embedding=v1, session_id=session)
    )
    id2 = await mtm.add_summary(
        SummaryBlock(
            text="DB connection pooling notes (rephrased)", embedding=v2, session_id=session
        )
    )
    id3 = await mtm.add_summary(
        SummaryBlock(text="Totally unrelated summary", embedding=v3, session_id=session)
    )

    assert id2 == id1  # near-duplicate folded into the first block
    assert id3 != id1  # distinct content stored separately
    assert _count_mtm(postgres_db, session) == 2  # only two rows persisted


# ---------------------------------------------------------------------------
# 4. Session-end flush: all STM → MTM, STM cleared
# ---------------------------------------------------------------------------


async def test_session_end_flush(postgres_db: str) -> None:
    session = f"end-{uuid.uuid4().hex[:8]}"
    stm = PostgresSTM(dsn=postgres_db, max_tokens=4096)
    mtm = PostgresMTM(dsn=postgres_db)

    async with stm:
        for i in range(12):
            await stm.append(_turn(session, i))
        assert (await stm.stats(session))["message_count"] == 12

        # End-of-session hook: drain STM into MTM.
        moved = await stm.flush_to(session, mtm)
        assert moved == 12

        # STM is now empty for this session.
        assert (await stm.stats(session))["message_count"] == 0
        assert list(await stm.window(session)) == []

    # Every turn is now an MTM block awaiting promotion.
    assert _count_mtm(postgres_db, session) == 12
    blocks = [b async for b in mtm.scan_unprocessed()]
    assert len(blocks) == 12


# ---------------------------------------------------------------------------
# 5. MTM.recent respects the token budget, newest-first
# ---------------------------------------------------------------------------


async def test_mtm_recent_respects_token_budget(postgres_db: str) -> None:
    session = f"recent-{uuid.uuid4().hex[:8]}"
    mtm = PostgresMTM(dsn=postgres_db)

    base = datetime(2024, 1, 1, tzinfo=UTC)
    sizes = [50, 120, 200, 80, 300]  # cycled, varied block sizes
    for i in range(100):
        await mtm.add_summary(
            SummaryBlock(
                text=f"block {i:03d}",
                tokens=sizes[i % len(sizes)],
                session_id=session,
                created_at=base + timedelta(minutes=i),  # i↑ ⇒ newer
            )
        )

    budget = 2000
    out = await mtm.recent(budget, session_id=session)

    assert out, "expected at least one recent block"
    # Budget respected (the single-oversized-newest exception cannot apply
    # here — max block is 300 ≪ 2000).
    total = sum(b.tokens for b in out)
    assert total <= budget

    # Newest-first and strictly older down the list.
    times = [b.created_at for b in out]
    assert times == sorted(times, reverse=True)
    assert out[0].created_at == base + timedelta(minutes=99)  # the newest

    # The window is maximal: adding the next-older block would overflow.
    assert len(out) < 100
    next_older_tokens = sizes[(99 - len(out)) % len(sizes)]
    assert total + next_older_tokens > budget
