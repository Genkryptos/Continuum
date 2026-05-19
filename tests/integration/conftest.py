"""
tests/integration/conftest.py
==============================
Integration-test fixtures that wrap the session-scoped ``postgres_db``
fixture with module-scoped database connections and a test-targeted config.

All fixtures here are automatically available to every test in
tests/integration/ without explicit import — pytest discovers conftest.py
files by directory traversal.

Skipping
--------
If ``postgres_db`` skips (Postgres unreachable), these fixtures propagate
the skip automatically, so individual integration tests don't need
``@pytest.mark.skipif`` guards.
"""

from __future__ import annotations

import pytest

from continuum.core.config import ContinuumConfig, DatabaseConfig

# ---------------------------------------------------------------------------
# db_conn — raw psycopg2 connection for SQL assertions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_conn(postgres_db: str):
    """
    A ``psycopg2`` connection to the isolated test database.

    Module-scoped so the same connection is reused within one test module
    (avoids per-test connection overhead while still being reset between
    test files).

    Usage::

        def test_memory_nodes_table_exists(db_conn):
            with db_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM memory_nodes LIMIT 1")
    """
    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed")

    conn = psycopg2.connect(postgres_db)
    conn.autocommit = False  # tests can commit / rollback explicitly
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# integration_config — ContinuumConfig pointing at the test DB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integration_config(postgres_db: str) -> ContinuumConfig:
    """
    A fully wired ContinuumConfig whose ``database.dsn`` points at the
    isolated test database created by ``postgres_db``.

    Use this wherever the production code instantiates its own DB pool
    (e.g. MTMRepository, LTM persistence layer).
    """
    return ContinuumConfig(
        log_level="DEBUG",
        environment="development",
        database=DatabaseConfig(
            dsn=postgres_db,
            pool_size=2,
            timeout=10.0,
        ),
    )


# ---------------------------------------------------------------------------
# db_transaction — auto-rollback per test (keeps tests hermetic)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_transaction(db_conn):
    """
    Wraps each test in a savepoint that is rolled back on teardown, keeping
    the test database clean without recreating it between tests.

    Usage::

        def test_insert_then_rollback(db_transaction):
            cur = db_transaction
            cur.execute("INSERT INTO memory_nodes (id, ...) VALUES (...)")
            # assertion here
            # teardown automatically rolls back the savepoint
    """
    with db_conn.cursor() as cur:
        cur.execute("SAVEPOINT test_savepoint")
    yield db_conn.cursor()
    with db_conn.cursor() as cur:
        cur.execute("ROLLBACK TO SAVEPOINT test_savepoint")
