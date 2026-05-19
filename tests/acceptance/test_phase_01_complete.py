"""
tests/acceptance/test_phase_01_complete.py
==========================================
**Phase 0.1 — Foundation Hardening — quality gate.**

A single end-to-end acceptance test that asserts every Phase 0.1 deliverable
is in place and working *together*. If this test fails, Phase 0.2 must not
begin.

Checks
------
1. STM and MTM implement their Protocols (``isinstance`` against
   ``@runtime_checkable`` STMProtocol / MTMProtocol).
2. All store / session APIs are ``async`` (coroutine or async-generator).
3. ``async with ContinuumSession(config) as s: await s.process_turn(...)``.
4. The background queue accepts work without blocking the response path.
5. pgvector ≥ 0.8 **and** the HNSW ``halfvec`` index exists.
6. ``pg_trgm`` extension + trigram GIN index enabled.
7. Hybrid search (dense ⊕ sparse ⊕ RRF) returns ranked results.
8. Test coverage ≥ 60 % (real ``pytest --cov`` subprocess, parsed).
9. Type coverage 100 % (``mypy --strict continuum`` exits 0).

To reach the Phase 0.1 *target DB state* the test idempotently applies
migrations 002 (halfvec + HNSW) and 003 (pg_trgm) on top of the 001 schema
that ``conftest.postgres_db`` already created.

Skips (not failures) when the gate cannot run honestly
------------------------------------------------------
* ``postgres_db`` fixture skips → Postgres unavailable.
* psycopg3 / numpy missing → async stores can't run.
* ``mypy`` / ``pytest-cov`` missing → toolchain incomplete.
This mirrors the rest of the suite: the gate is *enforced where the full
environment exists* (CI), and skipped — never falsely failed — elsewhere.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# psycopg3 (async stores) and numpy (vector fixtures) are hard requirements
# for the DB half of the gate.
pytest.importorskip("psycopg", reason="psycopg3 required for Phase 0.1 gate")
np = pytest.importorskip("numpy", reason="numpy required for Phase 0.1 gate")
import psycopg2  # noqa: E402  (only reachable once psycopg present)

from continuum.core.protocols import MTMProtocol, STMProtocol  # noqa: E402
from continuum.core.session import ContinuumSession  # noqa: E402
from continuum.core.types import ScoredItem  # noqa: E402
from continuum.db.pgvector_upgrade import (  # noqa: E402
    MIN_PGVECTOR,
    meets_minimum,
    parse_pgvector_version,
    to_halfvec_literal,
)
from continuum.stores.postgres import PostgresMTM, PostgresRetriever  # noqa: E402
from continuum.stores.stm import InMemorySTM  # noqa: E402
from continuum.stores.stm.postgres_stm import PostgresSTM  # noqa: E402

pytestmark = [pytest.mark.acceptance, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
MIG_002 = REPO_ROOT / "migrations" / "002_pgvector_upgrade.sql"
MIG_003 = REPO_ROOT / "migrations" / "003_lexical_search.sql"
_RECURSION_ENV = "CONTINUUM_ACCEPTANCE_RUNNING"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_sql_file(dsn: str, path: Path) -> None:
    """Run a (self-transactional, idempotent) migration file via psycopg2."""
    conn = psycopg2.connect(dsn)
    conn.autocommit = True  # the file owns its BEGIN/COMMIT
    try:
        with conn.cursor() as cur:
            cur.execute(path.read_text())
    finally:
        conn.close()


def _scalar(dsn: str, sql: str, params: tuple = ()) -> object:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def _is_async_callable(obj: object, name: str) -> bool:
    """True if ``obj.name`` is a coroutine *or* async-generator function."""
    fn = getattr(obj, name, None)
    return fn is not None and (
        inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


async def test_foundation_hardening_complete(
    postgres_db: str,
    continuum_config,
) -> None:
    # Guard: never let an inner `pytest --cov` subprocess recurse into us.
    if os.environ.get(_RECURSION_ENV):
        pytest.skip("nested acceptance invocation — skipping to avoid recursion")

    # The subprocess checks (#8, #9) need the full dev toolchain.
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy not installed — cannot enforce the type-coverage gate")
    if importlib.util.find_spec("pytest_cov") is None:
        pytest.skip("pytest-cov not installed — cannot enforce the coverage gate")

    # Bring the test DB to the Phase 0.1 target state (001 already applied by
    # the fixture; 002/003 are idempotent + version-gated).
    _apply_sql_file(postgres_db, MIG_002)
    _apply_sql_file(postgres_db, MIG_003)

    # ── 1. STM & MTM implement their Protocols ───────────────────────────────
    stm_mem = InMemorySTM()
    stm_pg = PostgresSTM(dsn=postgres_db)
    mtm_pg = PostgresMTM(dsn=postgres_db)
    assert isinstance(stm_mem, STMProtocol), "InMemorySTM ⊀ STMProtocol"
    assert isinstance(stm_pg, STMProtocol), "PostgresSTM ⊀ STMProtocol"
    assert isinstance(mtm_pg, MTMProtocol), "PostgresMTM ⊀ MTMProtocol"

    # ── 2. All APIs have async versions ──────────────────────────────────────
    for name in ("append", "window", "get_recent", "flush_to", "clear", "stats"):
        assert _is_async_callable(stm_pg, name), f"STM.{name} not async"
    for name in ("add_summary", "recent", "scan_unprocessed", "mark_processed"):
        assert _is_async_callable(mtm_pg, name), f"MTM.{name} not async"
    for name in ("process_turn", "checkpoint", "search"):
        assert _is_async_callable(
            ContinuumSession, name
        ), f"ContinuumSession.{name} not async"
        # …and a sync wrapper exists for every async verb.
        assert callable(getattr(ContinuumSession, f"{name}_sync", None)), (
            f"missing sync wrapper for {name}"
        )

    # ── 3. Async context manager + process_turn ──────────────────────────────
    async with ContinuumSession(continuum_config) as session:
        reply = await session.process_turn("test")
        assert isinstance(reply, str) and "test" in reply

        # ── 4. Background queue runs work without blocking the hot path ───────
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def _slow_job() -> None:
            started.set()
            await release.wait()
            finished.set()

        t0 = time.perf_counter()
        accepted = session.background.schedule_nowait(_slow_job)
        enqueue_dt = time.perf_counter() - t0
        assert accepted is True
        assert enqueue_dt < 0.05, "scheduling blocked the caller"

        await asyncio.wait_for(started.wait(), timeout=2.0)
        # The reply path returns even though the bg job is still blocked.
        reply2 = await asyncio.wait_for(session.process_turn("again"), timeout=2.0)
        assert "again" in reply2
        assert not finished.is_set()

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=2.0)
        await session.drain()
        assert session.background.stats()["completed"] >= 1
        assert session.background.health()["ok"] is True

    # ── 5. pgvector ≥ 0.8 + HNSW halfvec index ───────────────────────────────
    ver_raw = _scalar(
        postgres_db, "SELECT extversion FROM pg_extension WHERE extname='vector'"
    )
    assert ver_raw is not None, "pgvector extension not installed"
    assert meets_minimum(parse_pgvector_version(str(ver_raw)), MIN_PGVECTOR), (
        f"pgvector {ver_raw} < {'.'.join(map(str, MIN_PGVECTOR))}"
    )
    col_type = _scalar(
        postgres_db,
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname='memory_nodes' AND a.attname='embedding'
          AND a.attnum > 0 AND NOT a.attisdropped
        """,
    )
    assert col_type == "halfvec(1024)", f"embedding is {col_type}, expected halfvec(1024)"
    hnsw_def = _scalar(
        postgres_db,
        "SELECT indexdef FROM pg_indexes "
        "WHERE indexname='memory_nodes_embedding_hnsw_idx'",
    )
    assert hnsw_def is not None and "hnsw" in str(hnsw_def).lower(), (
        "HNSW index missing on memory_nodes.embedding"
    )

    # ── 6. pg_trgm extension + trigram GIN index ─────────────────────────────
    assert _scalar(
        postgres_db, "SELECT 1 FROM pg_extension WHERE extname='pg_trgm'"
    ) == 1, "pg_trgm extension not enabled"
    trgm_idx = _scalar(
        postgres_db,
        "SELECT indexdef FROM pg_indexes "
        "WHERE indexname='memory_nodes_text_trgm_live_idx'",
    )
    assert trgm_idx is not None and "gin_trgm_ops" in str(trgm_idx), (
        "trigram GIN index missing"
    )

    # ── 7. Hybrid search (dense ⊕ sparse ⊕ RRF) ──────────────────────────────
    def _vec(seed: int) -> list[float]:
        v = np.random.default_rng(seed).standard_normal(1024)
        return (v / np.linalg.norm(v)).tolist()

    target_vec = _vec(1)
    rows = [
        ("connection pooling and database tuning notes", target_vec),
        ("unrelated cooking recipe about pasta", _vec(2)),
        ("kubernetes deployment rollout strategy", _vec(3)),
    ]
    ins = psycopg2.connect(postgres_db)
    ins.autocommit = True
    try:
        with ins.cursor() as cur:
            for text, emb in rows:
                cur.execute(
                    "INSERT INTO memory_nodes (id, layer, \"text\", embedding) "
                    "VALUES (gen_random_uuid(), 'MTM', %s, %s::halfvec)",
                    (text, to_halfvec_literal(emb)),
                )
    finally:
        ins.close()

    retriever = PostgresRetriever(dsn=postgres_db)
    results = await retriever.hybrid_search(
        query_text="database connection pooling",   # lexical signal
        query_embedding=target_vec,                  # dense signal
        k=3,
    )
    assert results, "hybrid_search returned nothing"
    assert all(isinstance(r, ScoredItem) for r in results)
    assert all(isinstance(r.score, float) for r in results)
    # The doubly-relevant row (lexical + dense) must surface.
    assert any("connection pooling" in r.item.content for r in results)

    # ── 8. Test coverage ≥ 60 % ──────────────────────────────────────────────
    env = {**os.environ, _RECURSION_ENV: "1"}
    cov = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/unit", "-m", "unit",
            "--cov=continuum", "--cov-report=term", "--cov-fail-under=0",
            "-q", "-p", "no:cacheprovider", "--no-header",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = cov.stdout + cov.stderr
    assert cov.returncode == 0, f"unit suite failed under coverage:\n{out[-3000:]}"
    total_line = next(
        (ln for ln in out.splitlines() if ln.strip().startswith("TOTAL")), None
    )
    assert total_line, f"no coverage TOTAL line found:\n{out[-2000:]}"
    m = re.search(r"(\d+(?:\.\d+)?)%", total_line)
    assert m, f"could not parse coverage %: {total_line!r}"
    coverage_pct = float(m.group(1))
    assert coverage_pct >= 60.0, (
        f"coverage {coverage_pct:.1f}% < 60% gate ({total_line.strip()})"
    )

    # ── 9. Type coverage 100 % (mypy --strict exit 0) ────────────────────────
    mp = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "continuum"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert mp.returncode == 0, (
        "mypy --strict continuum failed (type coverage < 100%):\n"
        f"{(mp.stdout + mp.stderr)[-3000:]}"
    )
