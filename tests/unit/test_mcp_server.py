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

from continuum.mcp.server import _current, _recall, _remember, _timeline, build_server

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


# ── server assembly ───────────────────────────────────────────────────────────


async def test_build_server_registers_four_tools() -> None:
    server = build_server(memory=_FakeMemory())  # type: ignore[arg-type]
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {"recall", "remember", "current", "timeline"}


def test_build_server_accepts_injected_memory() -> None:
    m = _FakeMemory()
    server = build_server(memory=m)  # type: ignore[arg-type]
    assert server is not None  # constructs without a DB / default memory
