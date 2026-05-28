"""
continuum/extraction/small_llm.py
=================================
Shared small-LLM helper backed by a local Ollama endpoint.

One class — :class:`SmallLLM` — exposing three task-specific methods that
wrap a tiny instruct model (default ``qwen2.5:1.5b-instruct``). Each call
goes through an on-disk sqlite cache keyed on ``(method, model,
cache_key or sha256(normalized inputs))`` so repeated extraction passes
don't re-hit the model.

Prompts are kept tight (no role-play, no system preamble) on the
assumption that the underlying model is a small instruct model — verbose
framing hurts more than it helps here. Malformed model output never
raises and is **not cached**: it falls back to a safe default and the
next call gets another chance.

Network: ``requests.post`` to ``{url}/api/generate`` with
``stream=False``. The endpoint and model are env-configurable via
``CONTINUUM_SMALL_LLM_URL`` and ``CONTINUUM_SMALL_LLM_MODEL``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

import requests

log = logging.getLogger(__name__)

Intent = Literal["lookup", "compose", "temporal", "preference"]

DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
DEFAULT_URL = "http://localhost:11434"
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "continuum" / "small_llm.db"
_INTENTS: tuple[Intent, ...] = ("lookup", "compose", "temporal", "preference")
_TIMEOUT_S = 30.0


class SmallLLM:
    """Thin wrapper around an Ollama generate endpoint with sqlite caching."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self.model = model or os.environ.get("CONTINUUM_SMALL_LLM_MODEL", DEFAULT_MODEL)
        self.url = (
            url
            or os.environ.get("CONTINUUM_SMALL_LLM_URL", DEFAULT_URL)
        ).rstrip("/")
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self._init_cache()

    # ── public API ──────────────────────────────────────────────────────────

    def span_select(
        self, question: str, passage: str, cache_key: str | None = None
    ) -> str:
        prompt = (
            f"Question: {question}\n"
            f"Passage: {passage}\n"
            "Return the shortest verbatim span from the passage that answers "
            "the question. Output the span only, no commentary."
        )
        key = self._key("span_select", cache_key, question, passage)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        raw = self._generate(prompt)
        if raw is None:
            return ""
        span = raw.strip().strip('"').strip("'")
        self._cache_put(key, span)
        return span

    def verify_claim(
        self, claim: str, evidence: str, cache_key: str | None = None
    ) -> tuple[bool, float]:
        prompt = (
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            'Does the evidence support the claim? Reply with JSON only: '
            '{"supported": true|false, "confidence": 0.0-1.0}.'
        )
        key = self._key("verify_claim", cache_key, claim, evidence)
        cached = self._cache_get(key)
        if cached is not None:
            parsed = _parse_verify(cached)
            if parsed is not None:
                return parsed
        raw = self._generate(prompt)
        if raw is None:
            return (False, 0.0)
        parsed = _parse_verify(raw)
        if parsed is None:
            return (False, 0.0)
        self._cache_put(key, json.dumps({"supported": parsed[0], "confidence": parsed[1]}))
        return parsed

    def classify_intent(
        self, question: str, cache_key: str | None = None
    ) -> Intent:
        prompt = (
            f"Question: {question}\n"
            "Classify the question intent as exactly one of: "
            "lookup, compose, temporal, preference.\n"
            "Output the single label only, lowercase, no punctuation."
        )
        key = self._key("classify_intent", cache_key, question)
        cached = self._cache_get(key)
        if cached in _INTENTS:
            return cached  # type: ignore[return-value]
        raw = self._generate(prompt)
        label = _parse_intent(raw)
        if label is None:
            return "lookup"
        self._cache_put(key, label)
        return label

    # ── HTTP ────────────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str | None:
        """POST to Ollama; return the ``response`` field, or ``None`` on any failure."""
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=_TIMEOUT_S,
            )
        except requests.RequestException:
            log.exception("SmallLLM request failed")
            return None
        if resp.status_code != 200:
            log.warning("SmallLLM non-200 status: %s", resp.status_code)
            return None
        try:
            body = resp.json()
        except ValueError:
            log.warning("SmallLLM returned non-JSON body")
            return None
        text = body.get("response") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
        return text

    # ── cache ───────────────────────────────────────────────────────────────

    def _init_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL,"
                "  created_at REAL NOT NULL"
                ")"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.cache_path), isolation_level=None)

    def _key(self, method: str, cache_key: str | None, *args: str) -> str:
        if cache_key is None:
            h = hashlib.sha256()
            for a in args:
                h.update(a.strip().encode("utf-8"))
                h.update(b"\x00")
            cache_key = h.hexdigest()
        return f"{method}|{self.model}|{cache_key}"

    def _cache_get(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def _cache_put(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) "
                "VALUES (?, ?, ?)",
                (key, value, time.time()),
            )


# ── response parsers ────────────────────────────────────────────────────────


def _parse_verify(raw: str) -> tuple[bool, float] | None:
    """Pull ``{supported, confidence}`` from a model response. Tolerates fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data: Any = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or "supported" not in data:
        return None
    supported = bool(data["supported"])
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return supported, max(0.0, min(1.0, conf))


def _parse_intent(raw: str | None) -> Intent | None:
    if not raw:
        return None
    token = raw.strip().lower().strip(".,!?\"'`").split()[0] if raw.strip() else ""
    if token in _INTENTS:
        return token  # type: ignore[return-value]
    return None


__all__ = ["SmallLLM", "Intent", "DEFAULT_MODEL", "DEFAULT_URL"]
