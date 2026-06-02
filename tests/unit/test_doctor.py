"""
tests/unit/test_doctor.py
=========================
Hermetic tests for ``continuum.doctor`` — the ``python -m continuum.doctor``
setup check. No real DB, network, model download, or session: every check
that would touch infra is exercised through its pure logic or with fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum.doctor import (
    FAIL,
    OK,
    SKIP,
    WARN,
    Check,
    Report,
    _ping_provider,
    _redact_dsn,
    check_database,
    check_python,
    detect_providers,
    load_dotenv,
    render,
)

pytestmark = pytest.mark.unit


# ── load_dotenv ────────────────────────────────────────────────────────────


def test_load_dotenv_parses_pairs_quotes_export_and_comments(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "OPENROUTER_API_KEY=sk-or-real-key",
                'OPENAI_API_KEY="quoted-value"',
                "export GROQ_API_KEY=gsk_exported",
                "CONTINUUM_DB_DSN=postgresql://u:p@localhost/db",
                "MALFORMED_NO_EQUALS",
            ]
        )
    )
    env = load_dotenv(tmp_path)
    assert env["OPENROUTER_API_KEY"] == "sk-or-real-key"
    assert env["OPENAI_API_KEY"] == "quoted-value"
    assert env["GROQ_API_KEY"] == "gsk_exported"
    assert env["CONTINUUM_DB_DSN"] == "postgresql://u:p@localhost/db"
    assert "MALFORMED_NO_EQUALS" not in env


def test_load_dotenv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path) == {}


# ── detect_providers ───────────────────────────────────────────────────────


def test_detect_providers_finds_real_keys() -> None:
    found = detect_providers({"OPENROUTER_API_KEY": "sk-or-abc", "OPENAI_API_KEY": "sk-real"})
    labels = {label for _, label, _ in found}
    assert labels == {"OpenRouter", "OpenAI"}


def test_detect_providers_ignores_placeholders_and_blanks() -> None:
    env = {
        "OPENROUTER_API_KEY": "sk-or-...",  # the .env.example placeholder
        "OPENAI_API_KEY": "",
        "GROQ_API_KEY": "your-key-here",
        "ANTHROPIC_API_KEY": "sk-ant-real",
    }
    found = detect_providers(env)
    assert [label for _, label, _ in found] == ["Anthropic"]


# ── _redact_dsn ────────────────────────────────────────────────────────────


def test_redact_dsn_hides_password() -> None:
    assert _redact_dsn("postgresql://user:secret@localhost:5432/db") == (
        "postgresql://user:***@localhost:5432/db"
    )


def test_redact_dsn_without_password_is_unchanged() -> None:
    # No credentials in the authority → nothing to redact.
    assert _redact_dsn("postgresql://localhost:5432/db") == "postgresql://localhost:5432/db"


# ── _ping_provider ─────────────────────────────────────────────────────────


def test_ping_provider_without_endpoint_is_skipped() -> None:
    c = _ping_provider("ANTHROPIC_API_KEY", "Anthropic", None, "sk-ant-x")
    assert c.status == SKIP
    assert "key present" in c.detail


# ── check_python / check_database (pure branches) ──────────────────────────


def test_check_python_ok_on_current_interpreter() -> None:
    r = Report()
    check_python(r)
    assert r.checks[0].status == OK  # tests run on >= 3.11


def test_check_database_skips_when_no_dsn() -> None:
    r = Report()

    class _Cfg:
        database = type("D", (), {"dsn": None})()

    check_database(r, _Cfg())
    assert r.checks[-1].status == SKIP


# ── Report + render ────────────────────────────────────────────────────────


def test_report_failed_flag() -> None:
    r = Report()
    r.add("a", OK)
    assert r.failed is False
    r.add("b", FAIL)
    assert r.failed is True


def test_render_summarises_fail_and_warn() -> None:
    r = Report()
    r.add("ok-check", OK, "fine")
    r.add("warn-check", WARN, "iffy", hint="do this")
    out = render(r, color=False)
    assert "RESULT: OK (with 1 warning(s)" in out

    r.add("bad-check", FAIL, "broken", hint="fix this")
    out2 = render(r, color=False)
    assert "RESULT: FAIL" in out2
    assert "↳ fix this" in out2  # hints shown for fail/warn


def test_check_dataclass_defaults() -> None:
    c = Check("x", OK)
    assert c.detail == "" and c.hint == ""
