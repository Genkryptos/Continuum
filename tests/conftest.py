"""
tests/conftest.py
=================
Global pytest fixtures shared across unit, integration, and benchmark suites.

Fixture catalogue
-----------------
postgres_db        session  — isolated test Postgres DB, migration applied, torn down at exit
continuum_config   session  — ContinuumConfig wired for testing (no YAML, low timeouts)
mock_llm           function — patches litellm.completion; prevents real API calls
sample_memories    function — 100 MemoryItems spanning all tiers, ages, and sessions
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from continuum.core.config import ContinuumConfig, DatabaseConfig
from continuum.core.types import MemoryItem, MemoryTier, ProcessingState

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
MIGRATION_FILE = ROOT / "migrations" / "001_ltm_schema.sql"

# Credentials must match vectorDb/docker-compose.yml
_ADMIN_DSN = "postgresql://myuser:mypassword@localhost:5432/mydb"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# postgres_db — session-scoped isolated test database
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_db() -> Generator[str, None, None]:
    """
    Create a throw-away PostgreSQL database, apply the v1 migration, yield
    the DSN, then drop the database on exit.

    Behaviour when Postgres is unavailable
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The fixture calls ``pytest.skip`` so integration tests are skipped
    gracefully on machines without Docker (CI will catch them via the
    postgres service container defined in ci.yml).

    Isolation guarantee
    ~~~~~~~~~~~~~~~~~~~
    The test database name is ``continuum_test_<10-hex-chars>`` — unique per
    pytest session so parallel runs on the same host don't collide.
    """
    try:
        import psycopg2
        from psycopg2 import sql as pgsql
    except ImportError:
        pytest.skip("psycopg2 not installed — skipping all DB fixtures")

    test_db = f"continuum_test_{uuid.uuid4().hex[:10]}"

    # ── Create the isolated test database ────────────────────────────────────
    try:
        admin_conn = psycopg2.connect(_ADMIN_DSN, connect_timeout=5)
    except Exception as exc:
        pytest.skip(f"Postgres unreachable ({exc}) — skipping DB fixtures")

    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(test_db)))
    admin_conn.close()
    log.info("Created test database: %s", test_db)

    test_dsn = f"postgresql://myuser:mypassword@localhost:5432/{test_db}"

    # ── Apply the FULL migration set, in order ───────────────────────────────
    # Not just 001: the schema the code writes to must be the current one. This
    # fixture used to apply only 001_ltm_schema.sql, so the test database had no
    # `namespace` column (migration 005) while PostgresLTM.upsert writes it —
    # the acceptance suite failed the moment CI ran against Postgres. Driving
    # the real runner means the test schema can never again drift from the code.
    import io

    from continuum.db.migrate import apply_migrations

    if not MIGRATION_FILE.exists():
        pytest.fail(f"Migration file not found: {MIGRATION_FILE}")
    apply_migrations(test_dsn, out=io.StringIO())
    log.info("Migrations applied to %s", test_db)

    yield test_dsn

    # ── Teardown — drop the test database ────────────────────────────────────
    try:
        drop_conn = psycopg2.connect(_ADMIN_DSN, connect_timeout=5)
        drop_conn.autocommit = True
        with drop_conn.cursor() as cur:
            # Terminate active connections first so DROP DATABASE succeeds.
            cur.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM   pg_stat_activity
                WHERE  datname = %s
                  AND  pid <> pg_backend_pid()
                """,
                (test_db,),
            )
            cur.execute(pgsql.SQL("DROP DATABASE IF EXISTS {}").format(pgsql.Identifier(test_db)))
        drop_conn.close()
        log.info("Dropped test database: %s", test_db)
    except Exception:
        log.warning("Could not drop test database %s — manual cleanup may be needed", test_db)


# ---------------------------------------------------------------------------
# continuum_config — lightweight test configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def continuum_config() -> ContinuumConfig:
    """
    A ContinuumConfig suitable for unit and integration tests:
    - DEBUG logging so test output is verbose
    - Small pool (2 connections) to avoid resource exhaustion
    - Short timeout so hangs fail fast
    - All other values are framework defaults
    """
    return ContinuumConfig(
        log_level="DEBUG",
        environment="development",
        database=DatabaseConfig(
            dsn=_ADMIN_DSN,
            pool_size=2,
            timeout=5.0,
        ),
    )


# ---------------------------------------------------------------------------
# mock_llm — patched litellm.completion
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """
    Patch ``litellm.completion`` for the duration of one test.

    The stub returns a deterministic JSON body that the Promoter interprets as
    a confident promotion decision.  Tests that need a different response can
    update ``mock_llm.return_value`` directly.

    Usage::

        def test_promoter_accepts_stable_fact(mock_llm):
            # mock_llm is the patched MagicMock; adjust per-test if needed:
            mock_llm.return_value.choices[0].message.content = '{"promote": false, ...}'
            ...
    """
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '{"promote": true, "confidence": 0.92,'
                    ' "reason": "Stable factual claim verified across sessions."}'
                )
            )
        )
    ]
    response.usage = MagicMock(prompt_tokens=120, completion_tokens=40, total_tokens=160)

    with patch("litellm.completion", return_value=response) as mock:
        yield mock


# ---------------------------------------------------------------------------
# sample_memories — 100 varied MemoryItems
# ---------------------------------------------------------------------------

# Content templates cycling across tiers.  Each tuple is
# (content_prefix, tier, importance).
_TEMPLATES: list[tuple[str, MemoryTier, float]] = [
    ("User prefers dark mode in the UI.", MemoryTier.LTM, 0.80),
    ("Meeting scheduled for next Tuesday at 3 PM.", MemoryTier.MTM, 0.60),
    ("Hello, how can I help you today?", MemoryTier.STM, 0.30),
    ("Project deadline is end of Q2.", MemoryTier.LTM, 0.90),
    ("Python is the preferred language.", MemoryTier.LTM, 0.85),
    ("User asked about pricing plans.", MemoryTier.MTM, 0.50),
    ("Search returned 42 items.", MemoryTier.STM, 0.20),
    ("Authentication uses JWT with 1h expiry.", MemoryTier.LTM, 0.95),
    ("User is based in New York.", MemoryTier.MTM, 0.70),
    ("Retrying failed request, attempt 2.", MemoryTier.STM, 0.10),
]


@pytest.fixture
def sample_memories() -> list[MemoryItem]:
    """
    100 MemoryItems spanning all three tiers with varied content, importance,
    confidence, age, and session assignment.

    Design choices
    --------------
    * Items are spread across 5 sessions (sess-00 … sess-04) to test
      session-scoped retrieval.
    * Ages run from 0 h to ~7 days so recency scoring is exercised across the
      full decay curve.
    * ``confidence`` cycles through [0.5, 0.6, 0.7, 0.8, 0.9] to give the
      scorer meaningful signal.
    * MTM items are UNPROCESSED (waiting for promotion); all others are
      PROCESSED.
    """
    now = datetime.now(UTC)
    items: list[MemoryItem] = []

    for i in range(100):
        template_content, tier, importance = _TEMPLATES[i % len(_TEMPLATES)]
        age_hours = (i * 3.7) % (7 * 24)  # spread over one week
        created_at = now - timedelta(hours=age_hours)
        confidence = 0.5 + (i % 5) * 0.1  # 0.5 → 0.9

        items.append(
            MemoryItem(
                id=str(uuid.uuid4()),
                content=f"[{i:03d}] {template_content}",
                tier=tier,
                importance=importance,
                confidence=confidence,
                created_at=created_at,
                session_id=f"sess-{i % 5:02d}",
                agent_id="test-agent",
                user_id="test-user",
                processing_state=(
                    ProcessingState.UNPROCESSED
                    if tier == MemoryTier.MTM
                    else ProcessingState.PROCESSED
                ),
                metadata={"index": i, "template_slot": i % len(_TEMPLATES)},
            )
        )

    return items
