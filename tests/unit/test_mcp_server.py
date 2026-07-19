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

    async def add(self, text: str, *, occurred_at: Any = None) -> None:
        self.added.append((text, occurred_at))

    async def recall(self, query: str, *, k: int = 8) -> list[Any]:
        return list(self._recall[:k])

    async def current(self, subject: str, attribute: str) -> str | None:
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


def test_default_memory_with_dsn_falls_back_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    # A DSN is set but no live DB — the server must still build an importable
    # in-memory Memory rather than raising at import/construction time.
    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://nobody@localhost:1/none")
    mem = _default_memory()
    assert mem.session is not None


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
