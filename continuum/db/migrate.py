"""
continuum.db.migrate
====================
Apply the SQL migrations under ``migrations/`` to the configured Postgres,
in order, idempotently. This is the LTM/MTM half of "is my DB ready?" — the
``stm_messages`` table self-bootstraps via ``PostgresSTM.ensure_schema()``,
but the ``memory_nodes`` graph, the ``vector`` / ``pg_trgm`` extensions, the
HNSW index, and the policy tables all live in ``migrations/*.sql`` and are
**not** run by the app at startup (production DBs shouldn't have apps silently
running DDL). This runner is the explicit, opt-in step::

    python -m continuum.db.migrate            # apply all pending migrations
    python -m continuum.db.migrate --dry-run  # list pending, apply nothing
    python -m continuum.db.migrate --dsn postgresql://…   # override the DSN

Each migration file is idempotent (``IF NOT EXISTS`` / ``ON CONFLICT DO
NOTHING``) and records itself into the ``schema_migrations`` table, so re-runs
are safe. Files are executed **whole** (never split on ``;``) because they
contain ``DO $$ … $$`` blocks — splitting would corrupt them.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# migrations/ lives at the repo root, two levels up from this file.
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "migrations"
_VERSION_RE = re.compile(r"^(\d+)")


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


def discover_migrations(directory: Path) -> list[Migration]:
    """Return ``*.sql`` migrations sorted by their leading numeric version."""
    out: list[Migration] = []
    for p in sorted(directory.glob("*.sql")):
        m = _VERSION_RE.match(p.name)
        if m:
            out.append(Migration(version=m.group(1), path=p))
    return out


def pending(all_migrations: list[Migration], applied: set[str]) -> list[Migration]:
    """The migrations whose version is not yet recorded as applied."""
    return [m for m in all_migrations if m.version not in applied]


# ── DB-touching helpers (psycopg3, lazy import) ───────────────────────────────


def _resolve_dsn(dsn: str | None) -> str:
    if dsn:
        return dsn
    from continuum.core.config import ContinuumConfig

    return str(ContinuumConfig.load().database.dsn)


def _applied_versions(conn: Any) -> set[str]:
    """Versions already in ``schema_migrations`` ([] before 001 creates it)."""
    import psycopg

    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {str(r[0]) for r in rows}
    except psycopg.Error:
        # Table doesn't exist yet (fresh DB) — nothing applied.
        conn.rollback()
        return set()


def apply_migrations(
    dsn: str | None = None,
    *,
    directory: Path | None = None,
    dry_run: bool = False,
    out: Any = sys.stdout,
) -> int:
    """
    Apply pending migrations. Returns the number applied (0 if already
    up to date or ``dry_run``). Raises on the first failing migration
    (that file is rolled back; earlier files stay committed).
    """
    directory = directory or _DEFAULT_DIR
    all_migrations = discover_migrations(directory)
    if not all_migrations:
        print(f"no migrations found in {directory}", file=out)
        return 0

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "psycopg is required to run migrations.\n"
            "  pip install 'psycopg[binary,pool]'"
        ) from exc

    resolved = _resolve_dsn(dsn)
    applied_count = 0
    with psycopg.connect(resolved, connect_timeout=10) as conn:
        already = _applied_versions(conn)
        todo = pending(all_migrations, already)

        if not todo:
            print(f"database is up to date — {len(all_migrations)} migration(s) applied", file=out)
            return 0

        if dry_run:
            print(f"pending migrations ({len(todo)}):", file=out)
            for m in todo:
                print(f"  • {m.name}", file=out)
            print("(dry run — nothing applied)", file=out)
            return 0

        for m in todo:
            print(f"applying {m.name} …", file=out)
            sql = m.path.read_text()
            try:
                # Whole-file execute (simple protocol, no params) so DO $$ … $$
                # blocks survive; one transaction per file for atomicity.
                conn.execute(sql)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"  ✗ {m.name} failed: {str(exc).splitlines()[0]}", file=out)
                raise
            applied_count += 1
            print(f"  ✓ {m.name}", file=out)

    print(f"done — applied {applied_count} migration(s)", file=out)
    return applied_count


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m continuum.db.migrate",
        description="Apply Continuum's SQL migrations to the configured Postgres.",
    )
    p.add_argument("--dsn", default=None,
                   help="Postgres DSN (default: ContinuumConfig.load().database.dsn)")
    p.add_argument("--dry-run", action="store_true",
                   help="list pending migrations without applying them")
    p.add_argument("--migrations-dir", type=Path, default=None,
                   help=f"directory of *.sql migrations (default: {_DEFAULT_DIR})")
    args = p.parse_args(argv)

    try:
        apply_migrations(args.dsn, directory=args.migrations_dir, dry_run=args.dry_run)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"migration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["discover_migrations", "pending", "apply_migrations", "Migration", "main"]
