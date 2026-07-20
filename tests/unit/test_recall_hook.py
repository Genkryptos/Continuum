"""
tests/unit/test_recall_hook.py
==============================
The UserPromptSubmit recall hook. The load-bearing property is that it NEVER
breaks the user's prompt — bad input, no DB, or an exception must all yield a
clean no-op, not an error.
"""

from __future__ import annotations

import io
import json

import pytest

from continuum.mcp import recall_hook

pytestmark = pytest.mark.unit


# ── context assembly ──────────────────────────────────────────────────────────


def test_build_context_lists_memories() -> None:
    ctx = recall_hook.build_context("where do we deploy?", ["staging first", "then prod"])
    assert "staging first" in ctx and "then prod" in ctx
    assert ctx.startswith(recall_hook._HEADER)


def test_build_context_dedups_preserving_order() -> None:
    # recall returns LTM+STM copies of the same fact — must not inject it twice.
    ctx = recall_hook.build_context("q", ["A fact.", "A fact.", "B fact."])
    assert ctx.count("A fact.") == 1
    assert ctx.index("A fact.") < ctx.index("B fact.")


def test_build_context_empty_when_nothing_relevant() -> None:
    assert recall_hook.build_context("q", []) == ""
    assert recall_hook.build_context("q", ["", "   "]) == ""
    assert recall_hook.build_context("", ["a fact"]) == ""  # no prompt → nothing


# ── stdin parsing ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [
        ('{"prompt": "hello"}', "hello"),
        ('{"user_prompt": "aliased"}', "aliased"),
        ("not json at all", ""),
        ("[1, 2, 3]", ""),  # not an object
        ("{}", ""),  # no prompt key
        ('{"prompt": 123}', ""),  # wrong type
    ],
)
def test_read_prompt(stdin: str, expected: str) -> None:
    assert recall_hook._read_prompt(stdin) == expected


# ── main(): never fails loudly ────────────────────────────────────────────────


def test_main_no_dsn_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTINUUM_DB_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt": "anything"}'))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert recall_hook.main() == 0
    assert out.getvalue() == ""  # nothing injected, prompt proceeds untouched


def test_main_empty_prompt_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt": "   "}'))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert recall_hook.main() == 0
    assert out.getvalue() == ""


def test_main_recall_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A DB error mid-recall must not surface — it would break every prompt.
    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://x@127.0.0.1:5599/none")

    async def _boom(_prompt: str) -> list[str]:
        raise RuntimeError("db exploded")

    monkeypatch.setattr(recall_hook, "_recall", _boom)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt": "hello"}'))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert recall_hook.main() == 0
    assert out.getvalue() == ""


def test_main_emits_hook_output_on_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_DB_DSN", "postgresql://x@127.0.0.1:5599/none")

    async def _hit(_prompt: str) -> list[str]:
        return ["we deploy to staging first"]

    monkeypatch.setattr(recall_hook, "_recall", _hit)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt": "where do we deploy?"}'))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert recall_hook.main() == 0
    payload = json.loads(out.getvalue())
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "staging first" in payload["hookSpecificOutput"]["additionalContext"]
