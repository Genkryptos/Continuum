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


# ── automatic capture (opt-in) ────────────────────────────────────────────────


class _SpyMemory:
    """Records what capture would write."""

    def __init__(self) -> None:
        self.added: list[str] = []

    async def add(self, text: str, **_kw: object) -> None:
        self.added.append(text)


async def test_capture_is_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    # A memory that starts writing on its own without being switched on is the
    # failure this whole feature has to avoid.
    monkeypatch.delenv("CONTINUUM_CAPTURE", raising=False)
    assert recall_hook._capture_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON"])
def test_capture_opt_in_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CONTINUUM_CAPTURE", value)
    assert recall_hook._capture_enabled() is True


async def test_capture_stores_only_the_durable_sentence() -> None:
    mem = _SpyMemory()
    turn = "I ran the tests. I live in Boston. Can you fix the bug?"
    stored = await recall_hook._capture(mem, turn)
    assert stored == ["I live in Boston."]
    assert mem.added == ["I live in Boston."]


async def test_capture_never_stores_a_secret() -> None:
    mem = _SpyMemory()
    await recall_hook._capture(mem, "My API key is sk-proj-abc123def456ghi789jkl.")
    assert mem.added == []


async def test_capture_is_capped_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pasted wall of first-person text must not become 40 memories in one turn.
    monkeypatch.setenv("CONTINUUM_CAPTURE_MAX", "2")
    mem = _SpyMemory()
    turn = "I live in Boston. I use Neovim daily. I own a corgi named Pixel. I speak Hindi."
    assert len(await recall_hook._capture(mem, turn)) == 2


async def test_a_bad_cap_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_CAPTURE_MAX", "not-a-number")
    mem = _SpyMemory()
    turn = "I live in Boston. I use Neovim daily. I own a corgi named Pixel. I speak Hindi."
    assert len(await recall_hook._capture(mem, turn)) == 3


def test_dry_run_writes_nothing_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt": "I live in Boston."}'))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert recall_hook.main(["--dry-run"]) == 0
    assert "I live in Boston." in out.getvalue()


def test_dry_run_on_a_turn_with_nothing_to_keep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("please fix the failing test"))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert recall_hook.main(["--dry-run"]) == 0
    assert "nothing durable" in out.getvalue()
