"""
continuum.doctor
================
A one-shot "does my setup actually work?" check. Run it after copying
``.env.example`` → ``.env`` to confirm the pieces you configured are reachable
*before* you write any code::

    python -m continuum.doctor            # fast: config + keys + DB + in-mem smoke
    python -m continuum.doctor --ping     # also call your LLM provider (tiny/free)
    python -m continuum.doctor --full     # also load the embedder + embed a string
    python -m continuum.doctor --json     # machine-readable report

Nothing here is required for the offline path — the demo, benchmarks, and unit
tests run with zero config. So a *missing* provider key or *unreachable*
Postgres is reported as a WARNING (the in-memory path still works), while a
*broken* config or a session that won't start is a hard FAILURE.

Exit code is ``0`` when there are no failures, ``1`` otherwise — so it doubles
as a CI / pre-flight gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── result model ─────────────────────────────────────────────────────────────

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_GLYPH = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "–"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
_RESET = "\033[0m"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    hint: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": not self.failed,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "hint": c.hint}
                for c in self.checks
            ],
        }


# ── provider detection ───────────────────────────────────────────────────────

# (env var, human label, optional GET endpoint that validates the key on --ping).
# Endpoints are read-only model/key lookups — they don't spend tokens.
_PROVIDERS: list[tuple[str, str, str | None]] = [
    ("OPENROUTER_API_KEY", "OpenRouter", "https://openrouter.ai/api/v1/key"),
    ("OPENAI_API_KEY", "OpenAI", "https://api.openai.com/v1/models"),
    ("GROQ_API_KEY", "Groq", "https://api.groq.com/openai/v1/models"),
    ("GEMINI_API_KEY", "Gemini", "https://generativelanguage.googleapis.com/v1beta/models"),
    ("ANTHROPIC_API_KEY", "Anthropic", None),  # no free GET; presence-only
    ("NVIDIA_API_KEY", "NVIDIA", None),
]


def load_dotenv(root: Path) -> dict[str, str]:
    """
    Parse a ``.env`` file into a dict — stdlib only, no python-dotenv dependency.

    Handles ``KEY=value``, ``export KEY=value``, surrounding quotes, ``#``
    comments, and blank lines. The provider keys (OPENROUTER_API_KEY, …) live
    here but aren't ``CONTINUUM_*`` settings, so pydantic-settings doesn't
    surface them — the doctor reads them directly so "I put my key in .env"
    is actually verified.
    """
    path = root / ".env"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def detect_providers(env: dict[str, str] | None = None) -> list[tuple[str, str, str | None]]:
    """Return the subset of known providers whose key is present + non-placeholder."""
    e = env if env is not None else dict(os.environ)
    found = []
    for var, label, endpoint in _PROVIDERS:
        val = (e.get(var) or "").strip()
        if val and not val.endswith("...") and val.lower() not in {"changeme", "your-key-here"}:
            found.append((var, label, endpoint))
    return found


def _ping_provider(var: str, label: str, endpoint: str | None, key: str) -> Check:
    """Validate a key with a cheap authenticated GET. No token spend."""
    name = f"provider:{label} (ping)"
    if endpoint is None:
        return Check(name, SKIP, "no read-only check for this provider; key present")
    url = endpoint
    headers = {"Authorization": f"Bearer {key}"}
    if label == "Gemini":  # Gemini takes the key as a query param, not a header.
        url = f"{endpoint}?key={key}"
        headers = {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                return Check(name, OK, f"{label} key is valid (HTTP {resp.status})")
            return Check(name, WARN, f"unexpected HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Check(name, FAIL, f"{label} rejected the key (HTTP {exc.code})",
                         hint=f"check {var} in your .env")
        return Check(name, WARN, f"{label} returned HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Check(name, WARN, f"could not reach {label}: {exc}")


# ── individual checks ─────────────────────────────────────────────────────────


def check_python(report: Report) -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        report.add("python", OK, f"{v.major}.{v.minor}.{v.micro}")
    else:
        report.add("python", FAIL, f"{v.major}.{v.minor}.{v.micro}",
                   hint="Continuum requires Python ≥ 3.11")


def check_env_file(report: Report, root: Path) -> None:
    if (root / ".env").exists():
        report.add(".env file", OK, "found")
    else:
        report.add(".env file", WARN, "not found — using defaults / shell env",
                   hint="cp .env.example .env  then fill in what you need")


def check_config(report: Report) -> Any:
    """Load ContinuumConfig; return it (or None on failure)."""
    try:
        from continuum.core.config import ContinuumConfig

        cfg = ContinuumConfig.load()
        report.add("config loads", OK, "ContinuumConfig validated")
        return cfg
    except Exception as exc:  # pydantic.ValidationError or import error
        report.add("config loads", FAIL, str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
                   hint="a CONTINUUM_* value failed validation — see the message above")
        return None


def check_providers(report: Report, env: dict[str, str], *, ping: bool) -> None:
    found = detect_providers(env)
    if not found:
        report.add("LLM provider key", WARN, "none set — offline path only",
                   hint="set one of OPENROUTER_API_KEY / OPENAI_API_KEY / … for LLM features")
        return
    labels = ", ".join(label for _, label, _ in found)
    report.add("LLM provider key", OK, f"{len(found)} found: {labels}")
    if ping:
        for var, label, endpoint in found:
            report.checks.append(_ping_provider(var, label, endpoint, env[var].strip()))


def check_embeddings(report: Report, cfg: Any, *, full: bool) -> None:
    model = getattr(getattr(cfg, "embedding", None), "model_name", "?")
    device = getattr(getattr(cfg, "embedding", None), "device", "?")
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        report.add("embeddings", WARN, f"model={model} — sentence-transformers not importable",
                   hint="pip install sentence-transformers  (only needed for real embeddings)")
        return
    if not full:
        report.add("embeddings", OK, f"model={model} device={device} (load deferred; --full to embed)")
        return
    # --full: actually load the model + embed one string (downloads on first use).
    try:
        from continuum.embeddings.service import EmbeddingService

        svc = EmbeddingService(cfg.embedding)
        vec = asyncio.run(svc.embed(["continuum doctor smoke"]))[0]
        report.add("embeddings", OK, f"model={model} device={device} dim={len(vec)}")
    except Exception as exc:
        report.add("embeddings", FAIL, f"embed failed: {str(exc).splitlines()[0]}")


def check_database(report: Report, cfg: Any) -> None:
    """Probe the configured DSN. Unreachable = WARN (in-memory still works)."""
    dsn = getattr(getattr(cfg, "database", None), "dsn", None)
    if not dsn:
        report.add("database", SKIP, "no DSN configured — in-memory mode")
        return
    safe = _redact_dsn(dsn)
    try:
        import psycopg
    except Exception:
        report.add("database", WARN, "psycopg not installed — in-memory mode",
                   hint="pip install 'psycopg[binary,pool]'  for the Postgres path")
        return
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            ext = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            # to_regclass returns NULL when the table doesn't exist (no error).
            schema = conn.execute("SELECT to_regclass('public.memory_nodes')").fetchone()
            has_schema = bool(schema and schema[0])
            if not (ext and ext[0]):
                report.add("database", WARN, f"connected ({safe}); pgvector NOT installed",
                           hint="run `make db-migrate` (migration 001 runs CREATE EXTENSION vector)")
            elif not has_schema:
                report.add("database", WARN,
                           f"connected ({safe}); pgvector {ext[0]} but LTM schema NOT applied",
                           hint="run `make db-migrate` to create memory_nodes + the rest")
            else:
                report.add("database", OK,
                           f"connected ({safe}); pgvector {ext[0]}; LTM schema present")
    except Exception as exc:
        report.add("database", WARN, f"cannot reach {safe}: {str(exc).splitlines()[0]}",
                   hint="run `make db-up` (or skip — the in-memory path needs no DB)")


def _redact_dsn(dsn: str) -> str:
    """Hide the password in a DSN for display: postgres://user:***@host/db."""
    try:
        head, tail = dsn.split("://", 1)
        if "@" in tail and ":" in tail.split("@", 1)[0]:
            creds, rest = tail.split("@", 1)
            user = creds.split(":", 1)[0]
            return f"{head}://{user}:***@{rest}"
        return dsn
    except Exception:
        return "<dsn>"


def check_session_smoke(report: Report, cfg: Any) -> None:
    """Run one in-memory turn — proves the library works end-to-end with this config."""
    try:
        from continuum.core.session import ContinuumSession

        async def _run() -> str:
            async with ContinuumSession(cfg) as s:
                return await s.process_turn("continuum doctor: ping")

        reply = asyncio.run(_run())
        report.add("session smoke", OK, f"in-memory turn OK (reply {len(reply)} chars)")
    except Exception as exc:
        report.add("session smoke", FAIL, f"process_turn raised: {str(exc).splitlines()[0]}")


# ── orchestration + rendering ─────────────────────────────────────────────────


def run_checks(*, ping: bool, full: bool, root: Path | None = None) -> Report:
    root = root or Path.cwd()
    # Merge .env (provider keys live there) with the shell env; the shell wins.
    env = {**load_dotenv(root), **os.environ}
    report = Report()
    check_python(report)
    check_env_file(report, root)
    cfg = check_config(report)
    check_providers(report, env, ping=ping)
    if cfg is not None:
        check_embeddings(report, cfg, full=full)
        check_database(report, cfg)
        check_session_smoke(report, cfg)
    return report


def render(report: Report, *, color: bool) -> str:
    width = max((len(c.name) for c in report.checks), default=10)
    lines = ["", "Continuum setup check", "=" * 40]
    for c in report.checks:
        glyph = _GLYPH[c.status]
        if color:
            glyph = f"{_COLOR[c.status]}{glyph}{_RESET}"
        line = f"  {glyph}  {c.name:<{width}}  {c.detail}"
        lines.append(line)
        if c.hint and c.status in (WARN, FAIL):
            lines.append(f"       {'':<{width}}  ↳ {c.hint}")
    lines.append("=" * 40)
    n_fail = sum(1 for c in report.checks if c.status == FAIL)
    n_warn = sum(1 for c in report.checks if c.status == WARN)
    if report.failed:
        lines.append(f"RESULT: FAIL — {n_fail} failure(s), {n_warn} warning(s)")
    elif n_warn:
        lines.append(f"RESULT: OK (with {n_warn} warning(s) — offline path works)")
    else:
        lines.append("RESULT: OK — everything configured is reachable")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m continuum.doctor",
        description="Verify your Continuum .env / environment is set up correctly.",
    )
    p.add_argument("--ping", action="store_true",
                   help="validate each detected LLM provider key with a read-only API call")
    p.add_argument("--full", action="store_true",
                   help="also load the embedder and embed a string (downloads the model)")
    p.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    args = p.parse_args(argv)

    report = run_checks(ping=args.ping, full=args.full)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        color = (not args.no_color) and sys.stdout.isatty()
        print(render(report, color=color))
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_checks", "detect_providers", "Report", "Check"]
