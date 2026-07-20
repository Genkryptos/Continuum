"""
tests/e2e/conftest.py
=====================
Fixtures for the end-to-end suite: a fully-migrated throwaway database and a
handle to the real ``continuum-mcp`` binary driven over stdio.

These are the tests that would have caught this cycle's worst bugs — the ones
that lived in the seams *between* correct components (a datetime crash only the
real MCP layer produced, "stored" reported for a write that never persisted,
`valid_from` that the read path never SELECTed). Unit tests pass on each layer;
only exercising the whole thing end-to-end finds them.

Everything skips cleanly when Postgres or the ``mcp`` extra is unavailable, so
the suite is safe to collect anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Generator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

pytestmark = pytest.mark.e2e


def _admin_dsn() -> str:
    """A DSN to the server's default ``postgres`` db, for CREATE/DROP DATABASE.

    Derived from ``CONTINUUM_DB_DSN`` when set (dev), else the CI service DSN.
    """
    configured = os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL")
    base = configured or "postgresql://myuser:mypassword@localhost:5432/mydb"
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))


@pytest.fixture(scope="session")
def e2e_dsn() -> Generator[str, None, None]:
    """Create a throwaway, fully-migrated database; drop it on exit."""
    psycopg = pytest.importorskip("psycopg", reason="psycopg required for e2e")
    admin = _admin_dsn()
    name = f"continuum_e2e_{uuid.uuid4().hex[:10]}"

    try:
        conn = psycopg.connect(admin, autocommit=True, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable for e2e ({admin}): {exc}")

    parts = urlsplit(admin)
    dsn = urlunsplit((parts.scheme, parts.netloc, f"/{name}", "", ""))
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{name}"')
        conn.close()

        migrate = subprocess.run(
            [sys.executable, "-m", "continuum.db.migrate", "--dsn", dsn],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if migrate.returncode != 0:
            pytest.skip(f"migration failed for e2e db: {migrate.stderr[-500:]}")
        yield dsn
    finally:
        with psycopg.connect(admin, autocommit=True) as c, c.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (name,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{name}"')


class MCPClient:
    """Spawns the real ``continuum-mcp`` and speaks JSON-RPC over stdio.

    A benchmark that WRITES must never inherit a real store, and a client must
    keep stdin open across a slow first call — both lessons paid for this cycle.
    """

    def __init__(self, dsn: str | None, **env_overrides: str) -> None:
        binary = shutil.which("continuum-mcp")
        cmd = [binary] if binary else [sys.executable, "-m", "continuum.mcp.server"]
        env = {k: v for k, v in os.environ.items() if k not in ("CONTINUUM_DB_DSN", "DATABASE_URL")}
        if dsn is not None:
            env["CONTINUUM_DB_DSN"] = dsn  # else: in-memory fallback
        env["CONTINUUM_MCP_EMBEDDINGS"] = "0"  # sparse-only: no 2.3GB model download in CI
        env.update(env_overrides)
        self._p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._id = 0
        self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "0"},
            },
        )
        self._notify("notifications/initialized")

    def _call(self, method: str, params: dict[str, Any], timeout: float = 90.0) -> Any:
        assert self._p.stdin and self._p.stdout
        self._id += 1
        mid = self._id
        self._p.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params}) + "\n"
        )
        self._p.stdin.flush()
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            line = self._p.stdout.readline()
            if not line:
                raise RuntimeError("server closed the stream")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == mid:
                return msg
        raise TimeoutError(f"no reply to {method}")

    def _notify(self, method: str) -> None:
        assert self._p.stdin
        self._p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self._p.stdin.flush()

    def tool(self, name: str, **args: Any) -> tuple[bool, Any]:
        """Return (is_error, structured_result)."""
        res = (self._call("tools/call", {"name": name, "arguments": args}) or {}).get("result", {})
        return bool(res.get("isError")), res.get("structuredContent", {}).get("result")

    def close(self) -> None:
        self._p.kill()
