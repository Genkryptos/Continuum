"""
tests/unit/test_mcp_server.py
=============================
The Continuum MCP server. Tests the tool LOGIC directly (plain functions over a
fake Memory) and that build_server registers the four tools. Hermetic — no DB,
no network. Skips cleanly if the optional `mcp` extra isn't installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from continuum.mcp.server import (
    _current,
    _default_memory,
    _recall,
    _remember,
    _timeline,
    build_server,
    main,
)

pytestmark = pytest.mark.unit


class _FakeMemory:
    """Stands in for continuum.Memory — records writes, scripts reads."""

    def __init__(self) -> None:
        self.added: list[tuple[str, Any]] = []
        self._recall: list[Any] = []
        self._current: str | None = None
        self._timeline: list[Any] = []
        self.attributes: list[Any] = []
        self.starts = 0

    async def start(self) -> None:
        self.starts += 1

    async def add(
        self, text: str, *, occurred_at: Any = None, attribute: Any = None, split: bool = False
    ) -> None:
        self.added.append((text, occurred_at))
        self.attributes.append(attribute)

    async def recall(self, query: str, *, k: int = 8) -> list[Any]:
        return list(self._recall[:k])

    async def current(self, subject: str, attribute: str, *, as_of: Any = None) -> str | None:
        return self._current

    async def timeline(self, entity: str, *, since: Any = None, until: Any = None) -> list[Any]:
        return list(self._timeline)


class _FakeItem:
    def __init__(self, content: str) -> None:
        self.content = content
        self.id = "id-" + content
        self.session_id = "default"
        self.created_at = None


# ── tool logic ────────────────────────────────────────────────────────────────


async def test_recall_returns_json_safe_dicts() -> None:
    m = _FakeMemory()
    m._recall = [_FakeItem("Boston"), _FakeItem("NYC")]
    out = await _recall(m, "where?", k=5)  # type: ignore[arg-type]
    assert [d["content"] for d in out] == ["Boston", "NYC"]
    assert out[0]["id"] == "id-Boston" and out[0]["created_at"] is None


async def test_remember_stores_and_acks() -> None:
    m = _FakeMemory()
    assert await _remember(m, "I moved to NYC") == "stored"  # type: ignore[arg-type]
    assert m.added == [("I moved to NYC", None)]


async def test_remember_parses_iso_date() -> None:
    m = _FakeMemory()
    await _remember(m, "trip", "2023-05-10")  # type: ignore[arg-type]
    _, when = m.added[0]
    assert when is not None and when.year == 2023 and when.month == 5


async def test_remember_ignores_bad_date() -> None:
    m = _FakeMemory()
    await _remember(m, "trip", "not-a-date")  # type: ignore[arg-type]
    assert m.added[0][1] is None  # bad date → stored without occurred_at


async def test_remember_warns_when_the_store_is_not_durable() -> None:
    # The silent-failure case: a client session started before CONTINUUM_DB_DSN
    # was configured keeps writing to the in-memory fallback and reporting
    # success. The ack must distinguish a real write from a vanishing one.
    class _Ephemeral(_FakeMemory):
        is_durable = False

    out = await _remember(_Ephemeral(), "x")  # type: ignore[arg-type]
    assert "NOT durable" in out
    assert "CONTINUUM_DB_DSN" in out  # tells you how to fix it


async def test_remember_is_terse_when_durable() -> None:
    class _Durable(_FakeMemory):
        is_durable = True

    assert await _remember(_Durable(), "x") == "stored"  # type: ignore[arg-type]


async def test_remember_forwards_attribute() -> None:
    m = _FakeMemory()
    captured: dict[str, Any] = {}

    async def _spy(
        text: str, *, occurred_at: Any = None, attribute: Any = None, split: bool = False
    ) -> None:
        captured.update(text=text, attribute=attribute)

    m.add = _spy  # type: ignore[method-assign]
    await _remember(m, "I moved to NYC", None, "residence")  # type: ignore[arg-type]
    assert captured == {"text": "I moved to NYC", "attribute": "residence"}


async def test_current_forwards_as_of() -> None:
    m = _FakeMemory()
    captured: dict[str, Any] = {}

    async def _spy(subject: str, attribute: str, *, as_of: Any = None) -> str:
        captured["as_of"] = as_of
        return "Boston"

    m.current = _spy  # type: ignore[method-assign]
    out = await _current(m, "user", "residence", "2026-03-01")  # type: ignore[arg-type]
    assert out == "Boston"
    assert captured["as_of"] is not None and captured["as_of"].year == 2026


async def test_current_returns_value_or_not_found() -> None:
    m = _FakeMemory()
    m._current = "NYC"
    assert await _current(m, "user", "residence") == "NYC"  # type: ignore[arg-type]
    m._current = None
    assert await _current(m, "user", "residence") == "not found"  # type: ignore[arg-type]


async def test_timeline_returns_dicts() -> None:
    m = _FakeMemory()
    m._timeline = [_FakeItem("Boston"), _FakeItem("NYC")]
    out = await _timeline(m, "residence")  # type: ignore[arg-type]
    assert [d["content"] for d in out] == ["Boston", "NYC"]


async def test_timeline_parses_and_tolerates_bad_bounds() -> None:
    # good `since` is parsed to a datetime; malformed `until` is dropped to None.
    m = _FakeMemory()

    captured: dict[str, Any] = {}

    async def _spy(entity: str, *, since: Any = None, until: Any = None) -> list[Any]:
        captured["since"], captured["until"] = since, until
        return []

    m.timeline = _spy  # type: ignore[method-assign]
    await _timeline(m, "residence", since="2023-05-10", until="not-a-date")  # type: ignore[arg-type]
    assert captured["since"] is not None and captured["since"].year == 2023
    assert captured["until"] is None  # bad ISO string tolerated, not raised


def test_item_dict_serialises_created_at() -> None:
    from datetime import UTC, datetime

    from continuum.mcp.server import _item_dict

    class _WithTime:
        content = "x"
        id = "1"
        session_id = "s"
        created_at = datetime(2023, 1, 2, tzinfo=UTC)

    d = _item_dict(_WithTime())
    assert d["created_at"] == "2023-01-02T00:00:00+00:00"


# ── registered tool wrappers (invoked through the MCP server) ──────────────────


async def test_registered_tools_invoke_logic() -> None:
    pytest.importorskip("mcp.server.fastmcp")  # optional [mcp] extra
    m = _FakeMemory()
    m._recall = [_FakeItem("Boston")]
    m._current = "NYC"
    m._timeline = [_FakeItem("Boston"), _FakeItem("NYC")]
    server = build_server(memory=m)  # type: ignore[arg-type]

    # FastMCP.call_tool returns (content, structured) — assert on the structured half.
    _, recall_out = await server.call_tool("recall", {"query": "where?", "k": 3})
    assert recall_out["result"][0]["content"] == "Boston"

    _, remember_out = await server.call_tool("remember", {"text": "I moved to NYC"})
    assert remember_out["result"] == "stored"
    assert m.added == [("I moved to NYC", None)]

    _, current_out = await server.call_tool(
        "current", {"subject": "user", "attribute": "residence"}
    )
    assert current_out["result"] == "NYC"

    _, timeline_out = await server.call_tool("timeline", {"entity": "residence"})
    assert [d["content"] for d in timeline_out["result"]] == ["Boston", "NYC"]


# ── default memory + entry point ──────────────────────────────────────────────


def test_default_memory_without_dsn_is_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTINUUM_DB_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mem = _default_memory()
    assert mem.session is not None and mem.session.ltm is not None


def test_default_memory_uses_postgres_when_dsn_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum.mcp import server as srv

    captured: dict[str, Any] = {}

    def _fake_from_postgres(dsn: str, *, embeddings: bool = True) -> str:
        captured["dsn"], captured["embeddings"] = dsn, embeddings
        return "PG_MEMORY"

    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://x:y@localhost:5432/db")
    monkeypatch.delenv("CONTINUUM_MCP_EMBEDDINGS", raising=False)
    monkeypatch.setattr(srv.Memory, "from_postgres", _fake_from_postgres)
    assert srv._default_memory() == "PG_MEMORY"
    assert captured == {"dsn": "postgresql://x:y@localhost:5432/db", "embeddings": True}


def test_default_memory_embeddings_env_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum.mcp import server as srv

    captured: dict[str, Any] = {}

    def _fake_from_postgres(dsn: str, *, embeddings: bool = True) -> str:
        captured["embeddings"] = embeddings
        return "PG_MEMORY"

    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://x/y")
    monkeypatch.setenv("CONTINUUM_MCP_EMBEDDINGS", "0")
    monkeypatch.setattr(srv.Memory, "from_postgres", _fake_from_postgres)
    srv._default_memory()
    assert captured["embeddings"] is False  # CONTINUUM_MCP_EMBEDDINGS=0 disables the embedder


def test_default_memory_falls_back_when_postgres_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from continuum.mcp import server as srv

    def _boom(dsn: str, *, embeddings: bool = True) -> object:
        raise RuntimeError("no migrated DB")

    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://x/y")
    monkeypatch.setattr(srv.Memory, "from_postgres", _boom)
    mem = srv._default_memory()  # must not raise
    assert mem.session.ltm is not None  # degraded to an in-memory Memory


def test_main_builds_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    class _FakeServer:
        def run(self, transport: str = "stdio") -> None:
            ran.append(transport)

    monkeypatch.setattr("continuum.mcp.server.build_server", lambda **_kw: _FakeServer())
    main([])  # explicit empty argv — don't parse pytest's sys.argv
    assert ran == ["stdio"]  # default entry point wires build_server().run() over stdio


@pytest.mark.parametrize(
    ("flag", "expected_transport"),
    [("--http", "streamable-http"), ("--sse", "sse")],
)
def test_main_http_transports(
    monkeypatch: pytest.MonkeyPatch, flag: str, expected_transport: str
) -> None:
    captured: dict[str, Any] = {}

    class _FakeServer:
        def run(self, transport: str = "stdio") -> None:
            captured["transport"] = transport

    def _fake_build(
        memory: Any = None, *, host: str | None = None, port: int | None = None
    ) -> _FakeServer:
        captured["host"], captured["port"] = host, port
        return _FakeServer()

    monkeypatch.setattr("continuum.mcp.server.build_server", _fake_build)
    main([flag, "--host", "0.0.0.0", "--port", "9001"])
    assert captured["transport"] == expected_transport
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9001


async def test_tools_start_the_backend_exactly_once() -> None:
    # Regression: the server never called session.start(), so the Postgres
    # connection pool was never opened and the first tool call hung. It must be
    # started lazily on first use — once, no matter how many tools are called.
    pytest.importorskip("mcp.server.fastmcp")  # optional [mcp] extra
    m = _FakeMemory()
    server = build_server(memory=m)  # type: ignore[arg-type]
    assert m.starts == 0  # not started at build time — spawn stays instant

    await server.call_tool("remember", {"text": "a"})
    assert m.starts == 1
    await server.call_tool("recall", {"query": "a", "k": 1})
    await server.call_tool("current", {"subject": "user", "attribute": "x"})
    assert m.starts == 1  # idempotent — started once, not per call


# ── backend availability / auto-start ─────────────────────────────────────────


async def test_port_open_detects_a_listener() -> None:
    import asyncio
    import contextlib

    from continuum.mcp.server import _port_open

    assert await _port_open("postgresql://u@127.0.0.1:5599/none", timeout=1.0) is False

    async def _drop(reader: Any, writer: Any) -> None:
        writer.close()  # close server-side too, else wait_closed() blocks on it

    server = await asyncio.start_server(_drop, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await _port_open(f"postgresql://u@127.0.0.1:{port}/db", timeout=2.0) is True
    finally:
        server.close()
        # Bounded: on 3.12 wait_closed() waits for lingering connections.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(server.wait_closed(), 2.0)


async def test_dsn_reachable_is_not_fooled_by_a_bare_listener() -> None:
    """The whole point of the DB-level probe.

    A port check says "up" whenever *anything* owns the port — including a
    different PostgreSQL major version that does not have this database. Acting
    on that false positive is worse than an outage: the caller proceeds against
    the wrong store. `_dsn_reachable` must reject a non-Postgres listener.
    """
    import asyncio
    import contextlib

    from continuum.mcp.server import _dsn_reachable, _port_open

    pytest.importorskip("psycopg")

    async def _drop(reader: Any, writer: Any) -> None:
        writer.close()

    server = await asyncio.start_server(_drop, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    dsn = f"postgresql://u@127.0.0.1:{port}/db"
    try:
        assert await _port_open(dsn, timeout=2.0) is True  # port-only: fooled
        assert await _dsn_reachable(dsn, timeout=2.0) is False  # DB-level: correct
    finally:
        server.close()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(server.wait_closed(), 2.0)


async def test_autostart_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    # With the variable unset we must never execute anything.
    from continuum.mcp.server import _autostart

    monkeypatch.delenv("CONTINUUM_MCP_AUTOSTART", raising=False)
    assert await _autostart("postgresql://u@127.0.0.1:5599/none", timeout=1.0) is False


async def test_autostart_runs_command_then_waits_for_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from continuum.mcp import server as srv

    # Orchestration test: the command must actually RUN, and the DB must then be
    # polled until it reports ready. The readiness probe is stubbed because a
    # real one now requires a live PostgreSQL handshake (a bare socket no longer
    # counts as "up" — that was the wrong-database false positive).
    sentinel = tmp_path / "ran.txt"
    monkeypatch.setenv("CONTINUUM_MCP_AUTOSTART", f"touch {sentinel}")

    calls = {"n": 0}

    async def _fake_probe(dsn: str, timeout: float = 2.0) -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # down on the pre-check, up once the command ran

    monkeypatch.setattr(srv, "_dsn_reachable", _fake_probe)
    assert await srv._autostart("postgresql://u@127.0.0.1:5599/db", timeout=10.0) is True
    assert sentinel.exists()  # the configured command really executed


async def test_autostart_gives_up_when_backend_never_comes_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from continuum.mcp import server as srv

    monkeypatch.setenv("CONTINUUM_MCP_AUTOSTART", "true")

    async def _never(dsn: str, timeout: float = 2.0) -> bool:
        return False

    monkeypatch.setattr(srv, "_dsn_reachable", _never)
    # Bounded, and reports failure rather than hanging the session forever.
    assert await srv._autostart("postgresql://u@127.0.0.1:5599/db", timeout=2.0) is False


async def test_unreachable_backend_errors_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dead DSN must produce a reportable error, not kill the server: the pool
    # opens optimistically, so without the pre-flight probe the request's task
    # group tears down and the client gets no reply at all.
    pytest.importorskip("mcp.server.fastmcp")
    from mcp.server.fastmcp.exceptions import ToolError

    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://u@127.0.0.1:5599/none")
    monkeypatch.delenv("CONTINUUM_MCP_AUTOSTART", raising=False)
    server = build_server(memory=_FakeMemory())  # type: ignore[arg-type]
    # FastMCP wraps it as ToolError — which is exactly how the client receives
    # isError=True plus the message, rather than losing the connection.
    with pytest.raises(ToolError, match="unreachable"):
        await server.call_tool("remember", {"text": "x"})
    # …and the server is still usable afterwards.
    assert len(await server.list_tools()) == 4


def test_supersession_is_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum.mcp.server import _supersession_kwargs

    monkeypatch.delenv("CONTINUUM_MCP_SUPERSESSION", raising=False)
    assert _supersession_kwargs() == {}


def test_supersession_refuses_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enabling the decider with no LLM is worse than leaving it off: the
    # ambiguous band returns NOOP, which would silently discard the new fact.
    from continuum.mcp import server as srv

    monkeypatch.setenv("CONTINUUM_MCP_SUPERSESSION", "1")
    monkeypatch.setattr("continuum.promotion.openrouter.resolve_openrouter_key", lambda *a, **k: "")
    assert srv._supersession_kwargs() == {}


def test_supersession_wires_model_and_fn_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum.mcp import server as srv

    monkeypatch.setenv("CONTINUUM_MCP_SUPERSESSION", "yes")
    monkeypatch.setenv("CONTINUUM_MCP_SUPERSESSION_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setattr(
        "continuum.promotion.openrouter.resolve_openrouter_key", lambda *a, **k: "k-123"
    )
    kw = srv._supersession_kwargs()
    assert kw["supersession_model"] == "openai/gpt-4o-mini"
    assert callable(kw["supersession_completion_fn"])


def test_build_server_applies_host_port() -> None:
    pytest.importorskip("mcp.server.fastmcp")  # optional [mcp] extra
    server = build_server(memory=_FakeMemory(), host="0.0.0.0", port=9123)  # type: ignore[arg-type]
    assert server.settings.host == "0.0.0.0"
    assert server.settings.port == 9123


# ── server assembly ───────────────────────────────────────────────────────────


async def test_build_server_registers_four_tools() -> None:
    pytest.importorskip("mcp.server.fastmcp")  # optional [mcp] extra
    server = build_server(memory=_FakeMemory())  # type: ignore[arg-type]
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {"recall", "remember", "current", "timeline"}


def test_build_server_accepts_injected_memory() -> None:
    pytest.importorskip("mcp.server.fastmcp")  # optional [mcp] extra
    m = _FakeMemory()
    server = build_server(memory=m)  # type: ignore[arg-type]
    assert server is not None  # constructs without a DB / default memory
