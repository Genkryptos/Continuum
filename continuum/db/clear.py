"""
continuum.db.clear
=================
Delete **all memory records** from the configured Postgres while keeping the
schema and the migration history intact. This is the "start fresh without
re-migrating" button — distinct from ``make db-reset`` (which destroys the
whole Docker volume) and ``make db-migrate`` (which creates the schema)::

    python -m continuum.db.clear            # prompts for confirmation
    python -m continuum.db.clear --yes      # non-interactive (CI / scripts)
    python -m continuum.db.clear --dsn …    # override the DSN

It ``TRUNCATE``s every table in the ``public`` schema **except**
``schema_migrations`` (so the DB still knows which migrations are applied),
with ``RESTART IDENTITY CASCADE``. Schema-agnostic: it discovers the tables at
runtime, so new tables from future migrations are cleared automatically.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

# Never truncated — clearing this would make the DB forget which migrations ran.
_PRESERVE = {"schema_migrations"}


def _resolve_dsn(dsn: str | None) -> str:
    if dsn:
        return dsn
    from continuum.core.config import ContinuumConfig

    return str(ContinuumConfig.load().database.dsn)


def discover_data_tables(conn: Any) -> list[str]:
    """All ``public`` tables except the preserved ones, sorted."""
    rows = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    ).fetchall()
    return [str(r[0]) for r in rows if str(r[0]) not in _PRESERVE]


def _row_counts(conn: Any, tables: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tables:
        try:
            row = conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()
            counts[t] = int(row[0]) if row else 0
        except Exception:
            counts[t] = -1  # unreadable — still truncated below
    return counts


def clear_all(
    dsn: str | None = None,
    *,
    yes: bool = False,
    out: Any = sys.stdout,
) -> int:
    """
    Truncate all data tables. Returns the number of tables cleared.
    Refuses to run without confirmation (interactive prompt or ``yes=True``).
    """
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "psycopg is required to clear the database.\n"
            "  pip install 'psycopg[binary,pool]'"
        ) from exc

    resolved = _resolve_dsn(dsn)
    with psycopg.connect(resolved, connect_timeout=10) as conn:
        tables = discover_data_tables(conn)
        if not tables:
            print("no data tables found — nothing to clear", file=out)
            return 0

        counts = _row_counts(conn, tables)
        total = sum(c for c in counts.values() if c > 0)
        print(f"about to clear {len(tables)} table(s), ~{total} row(s):", file=out)
        for t in tables:
            n = counts[t]
            print(f"  • {t}: {'?' if n < 0 else n} row(s)", file=out)

        if not yes:
            if not sys.stdin.isatty():
                print(
                    "refusing to clear without confirmation — re-run with --yes",
                    file=out,
                )
                return 0
            reply = input("type 'yes' to permanently delete these records: ").strip()
            if reply.lower() != "yes":
                print("aborted — nothing was deleted", file=out)
                return 0

        # One statement so FK dependencies are handled together; RESTART
        # IDENTITY resets any serial counters.
        quoted = ", ".join(f'"{t}"' for t in tables)
        conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE")
        conn.commit()

    print(f"cleared {len(tables)} table(s) — schema + migration history kept", file=out)
    return len(tables)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m continuum.db.clear",
        description="Delete all memory records (keeps schema + migration history).",
    )
    p.add_argument("--dsn", default=None,
                   help="Postgres DSN (default: ContinuumConfig.load().database.dsn)")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt (non-interactive)")
    args = p.parse_args(argv)

    try:
        clear_all(args.dsn, yes=args.yes)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"clear error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["discover_data_tables", "clear_all", "main"]
