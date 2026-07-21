"""
tests/e2e/test_forget_end_to_end.py
===================================
Store-level forgetting against a real Postgres (Phase 2.2).

Mocks can prove the SQL we *meant* to write; only the database proves what it
does. What matters here is the blast radius: a fact that is still true survives
even when nobody reads it, another namespace is never touched, and a forgotten
row is *retired*, not deleted — `invalidated_at` is set and the row stays on
disk, so a bad policy is recoverable by hand.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.e2e

STALE = "I work at Infosys"  # superseded + cold  → forgettable
LIVE = "My daughter is named Mira"  # still true + cold → must survive
FRESH = "I am learning to sail"  # still true + warm → must survive
OTHER = "Bob works at Nimbus"  # different namespace → untouchable


@pytest.fixture
def aged_store(e2e_dsn: str):
    """A store whose rows have been aged on disk, as a months-old store would be."""
    psycopg = pytest.importorskip("psycopg")

    from continuum.memory import Memory

    async def _build() -> tuple[Memory, dict[str, uuid.UUID]]:
        # The migrated database is session-scoped; each test wants a clean store,
        # otherwise a second run's duplicate rows quietly inflate every count.
        with psycopg.connect(e2e_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE memory_nodes CASCADE")

        alice = Memory.from_postgres(e2e_dsn, embeddings=False, namespace="alice")
        bob = Memory.from_postgres(e2e_dsn, embeddings=False, namespace="bob")
        await alice.start()
        await bob.start()
        for text in (STALE, LIVE, FRESH):
            await alice.add(text)
        await bob.add(OTHER)
        await bob.aclose()

        ids: dict[str, uuid.UUID] = {}
        long_ago = datetime.now(UTC) - timedelta(days=400)
        with psycopg.connect(e2e_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute('SELECT id, "text" FROM memory_nodes')
            by_text = {row[1]: row[0] for row in cur.fetchall()}
            ids = {t: by_text[t] for t in (STALE, LIVE, FRESH, OTHER)}
            # Age everything: nothing has been read in over a year.
            cur.execute(
                "UPDATE memory_nodes SET created_at = %s, last_access = NULL WHERE id = ANY(%s)",
                (long_ago, [ids[STALE], ids[LIVE], ids[OTHER]]),
            )
            # …and only STALE has been superseded (its valid window is closed).
            cur.execute(
                "UPDATE memory_nodes SET valid_to = %s WHERE id = %s",
                (datetime.now(UTC) - timedelta(days=200), ids[STALE]),
            )
        return alice, ids

    return _build


def _live_ids(dsn: str) -> set[uuid.UUID]:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM memory_nodes WHERE invalidated_at IS NULL")
        return {r[0] for r in cur.fetchall()}


def _all_ids(dsn: str) -> set[uuid.UUID]:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM memory_nodes")
        return {r[0] for r in cur.fetchall()}


async def test_dry_run_changes_nothing(e2e_dsn: str, aged_store) -> None:
    mem, _ids = await aged_store()
    try:
        before = _live_ids(e2e_dsn)
        report = await mem.forget(unused_for=timedelta(days=90))

        assert report.dry_run is True
        assert report.matched == 1 and report.pruned == 0
        assert report.samples == (STALE,)
        assert _live_ids(e2e_dsn) == before  # the store is untouched
    finally:
        await mem.aclose()


async def test_forget_retires_only_the_stale_cold_fact(e2e_dsn: str, aged_store) -> None:
    mem, ids = await aged_store()
    try:
        report = await mem.forget(unused_for=timedelta(days=90), dry_run=False)
        assert report.pruned == 1

        live = _live_ids(e2e_dsn)
        assert ids[STALE] not in live  # superseded + cold → forgotten
        assert ids[LIVE] in live  # still true, though nobody read it → kept
        assert ids[FRESH] in live  # recent → kept
        assert ids[OTHER] in live  # another namespace → never in scope

        # Retired, not destroyed: the row is still on disk and recoverable.
        assert _all_ids(e2e_dsn) >= {ids[STALE], ids[LIVE], ids[FRESH], ids[OTHER]}
    finally:
        await mem.aclose()


async def test_a_forgotten_fact_is_gone_from_the_next_session(e2e_dsn: str, aged_store) -> None:
    # The contract, stated precisely. `forget` prunes LONG-TERM memory: the
    # session that just wrote the fact still has it in its short-term buffer,
    # and that is not a half-forgotten row — STM is session-scoped and
    # transient. What must hold is that a later session sees nothing, in
    # `recall` *and* in `timeline` together, so history cannot contradict the
    # current answer.
    from continuum.memory import Memory

    mem, _ = await aged_store()
    try:
        assert any(STALE in (h.content or "") for h in await mem.recall("Infosys", k=20))
        await mem.forget(unused_for=timedelta(days=90), dry_run=False)
    finally:
        await mem.aclose()

    later = Memory.from_postgres(
        e2e_dsn, embeddings=False, namespace="alice", session_id="a-later-session"
    )
    await later.start()
    try:
        assert not any(STALE in (h.content or "") for h in await later.recall("Infosys", k=20))
        assert not any(STALE in (h.content or "") for h in await later.timeline("Infosys"))
        # …and the facts that were kept are still there for that later session.
        assert any(LIVE in (h.content or "") for h in await later.recall("daughter Mira", k=20))
    finally:
        await later.aclose()


async def test_superseded_only_false_reaches_facts_that_are_still_true(
    e2e_dsn: str, aged_store
) -> None:
    # The escape hatch, and proof the default is what protects LIVE: same policy,
    # same cold rows, only `superseded_only` differs.
    mem, ids = await aged_store()
    try:
        report = await mem.forget(
            unused_for=timedelta(days=90), superseded_only=False, dry_run=False
        )
        assert report.pruned == 2  # STALE *and* LIVE

        live = _live_ids(e2e_dsn)
        assert ids[LIVE] not in live
        assert ids[FRESH] in live  # still warm — the age filter still applies
        assert ids[OTHER] in live  # namespace scoping is not negotiable
    finally:
        await mem.aclose()


async def test_limit_caps_the_blast_radius(e2e_dsn: str, aged_store) -> None:
    mem, _ = await aged_store()
    try:
        report = await mem.forget(
            unused_for=timedelta(days=90), superseded_only=False, limit=1, dry_run=False
        )
        assert report.matched == 1 and report.pruned == 1
    finally:
        await mem.aclose()


# ── restating a fact reinforces it instead of piling up copies ────────────────


async def test_restating_a_fact_does_not_create_copies(e2e_dsn: str) -> None:
    """People restate things, and automatic capture sees the same sentence again.

    Each restatement used to cost a row and an embedding forever. Retrieval
    dedups at read, so the copies were invisible — never free.
    """
    import psycopg

    from continuum.memory import Memory

    with psycopg.connect(e2e_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE memory_nodes CASCADE")

    mem = Memory.from_postgres(e2e_dsn, embeddings=False, namespace="repeat")
    await mem.start()
    try:
        for _ in range(10):
            await mem.add("I cycle to work every day.")
        await mem.add("My bike is a blue Brompton.")

        with psycopg.connect(e2e_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*), count(DISTINCT text) FROM memory_nodes")
            rows, distinct = cur.fetchone()
            assert (rows, distinct) == (2, 2), f"{rows} rows for 2 distinct facts"

            # Restating is evidence, not noise — the repeats are recorded.
            cur.execute("SELECT access_count FROM memory_nodes WHERE text LIKE 'I cycle%'")
            assert cur.fetchone()[0] == 9

        # …and both facts are still retrievable.
        found = {h.content for h in await mem.recall("how do I get to work?", k=8)}
        assert any("cycle" in c for c in found) and any("Brompton" in c for c in found)
    finally:
        await mem.aclose()


async def test_a_restatement_fills_in_what_the_first_telling_lacked(e2e_dsn: str) -> None:
    import psycopg

    from continuum.memory import Memory

    with psycopg.connect(e2e_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE memory_nodes CASCADE")

    mem = Memory.from_postgres(e2e_dsn, embeddings=False, namespace="fillin")
    await mem.start()
    try:
        await mem.add("I work at Nimbus.")  # undated, untagged
        await mem.add(
            "I work at Nimbus.",
            occurred_at=datetime(2025, 4, 1, tzinfo=UTC),
            attribute="employer",
        )
        with psycopg.connect(e2e_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), valid_from, tags->>'attribute' FROM memory_nodes GROUP BY 2,3"
            )
            rows = cur.fetchall()
        assert len(rows) == 1
        count, valid_from, attribute = rows[0]
        assert count == 1  # one row, not two
        assert valid_from is not None and valid_from.year == 2025  # the date landed
        assert attribute == "employer"
    finally:
        await mem.aclose()
