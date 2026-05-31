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
Backend = Literal["ollama", "openai"]

DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
DEFAULT_URL = "http://localhost:11434"
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "continuum" / "small_llm.db"
_INTENTS: tuple[Intent, ...] = ("lookup", "compose", "temporal", "preference")
_TIMEOUT_S = 30.0


def _auto_detect_backend(url: str) -> Backend:
    """
    Pick the protocol shape from the URL. LM Studio (and any other
    OpenAI-compatible local server) lives under ``/v1``; Ollama lives
    at the bare host. Either can be forced via
    ``CONTINUUM_SMALL_LLM_BACKEND={ollama,openai}``.
    """
    forced = os.environ.get("CONTINUUM_SMALL_LLM_BACKEND", "").strip().lower()
    if forced in ("ollama", "openai"):
        return forced  # type: ignore[return-value]
    return "openai" if url.rstrip("/").endswith("/v1") else "ollama"


class SmallLLM:
    """
    Thin wrapper around a tiny instruct LLM with sqlite caching.

    Two backends supported:

    * ``ollama`` — Ollama's ``/api/generate``. The default; matches a
      vanilla ``ollama serve`` running at :11434.
    * ``openai`` — OpenAI-compatible ``/v1/chat/completions``. Works
      with **LM Studio** (``http://localhost:1234/v1``), vLLM,
      LocalAI, OpenAI proper, and any other server exposing the same
      shape. Auto-detected when the URL ends with ``/v1``; force
      explicitly via ``CONTINUUM_SMALL_LLM_BACKEND=openai`` (or
      ``=ollama`` to override the auto-pick).
    """

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        cache_path: Path | str | None = None,
        backend: Backend | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("CONTINUUM_SMALL_LLM_MODEL", DEFAULT_MODEL)
        self.url = (url or os.environ.get("CONTINUUM_SMALL_LLM_URL", DEFAULT_URL)).rstrip("/")
        self.backend: Backend = backend or _auto_detect_backend(self.url)
        # OpenAI proper needs the bearer; LM Studio / vLLM accept any
        # string (or no key); Ollama ignores the kwarg entirely.
        self.api_key = (
            api_key
            or os.environ.get("CONTINUUM_SMALL_LLM_API_KEY", "")
            or ("lm-studio" if self.backend == "openai" else "")
        )
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self._init_cache()
        # Flag flipped on first ConnectionError so we don't spam the eval
        # log with a stack trace on every call when the endpoint isn't up.
        self._connection_warned = False

    def _log_connection_failure(self, exc: BaseException) -> None:
        if not self._connection_warned:
            log.warning(
                "SmallLLM endpoint %s unreachable (%s). "
                "Subsequent failures suppressed; calls will return None "
                "until the endpoint comes back up.",
                self.url,
                type(exc).__name__,
            )
            self._connection_warned = True
        else:
            log.debug("SmallLLM endpoint still unreachable")

    # ── public API ──────────────────────────────────────────────────────────

    def span_select(self, question: str, passage: str, cache_key: str | None = None) -> str:
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
            "Does the evidence support the claim? Reply with JSON only: "
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

    def classify_intent(self, question: str, cache_key: str | None = None) -> Intent:
        prompt = (
            f"Question: {question}\n"
            "Classify the question intent as exactly one of: "
            "lookup, compose, temporal, preference.\n"
            "Output the single label only, lowercase, no punctuation."
        )
        key = self._key("classify_intent", cache_key, question)
        cached = self._cache_get(key)
        if cached in _INTENTS:
            return cached
        raw = self._generate(prompt)
        label = _parse_intent(raw)
        if label is None:
            return "lookup"
        self._cache_put(key, label)
        return label

    # ── HTTP ────────────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str | None:
        """Dispatch by backend; returns the model reply or ``None`` on any failure."""
        if self.backend == "openai":
            return self._generate_openai_chat(prompt)
        return self._generate_ollama(prompt)

    def _generate_ollama(self, prompt: str) -> str | None:
        """POST to Ollama's ``/api/generate``; pull ``response``."""
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
        except requests.ConnectionError as exc:
            self._log_connection_failure(exc)
            return None
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

    def _generate_openai_chat(self, prompt: str) -> str | None:
        """
        POST to ``/v1/chat/completions``. Compatible with LM Studio,
        vLLM, LocalAI, and OpenAI proper. The request matches the same
        single-user-turn / temperature-0 shape we use for Ollama.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(
                f"{self.url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "stream": False,
                },
                headers=headers,
                timeout=_TIMEOUT_S,
            )
        except requests.ConnectionError as exc:
            self._log_connection_failure(exc)
            return None
        except requests.RequestException:
            log.exception("SmallLLM request failed")
            return None
        if resp.status_code != 200:
            # LM Studio's body carries the actual cause; surface it.
            try:
                err = resp.json()
                msg = (
                    (err.get("error") or {}).get("message")
                    if isinstance(err.get("error"), dict)
                    else err.get("error") or resp.text[:200]
                )
            except Exception:
                msg = resp.text[:200]
            log.warning("SmallLLM %d from %s: %s", resp.status_code, self.url, msg)
            return None
        try:
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            log.warning("SmallLLM (openai-chat) malformed response")
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        # Strip a leading reasoning block if the model emits one
        # (Qwen3 in LM Studio sometimes wraps replies in <think>…</think>).
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()

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
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _cache_put(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
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
        return token
    return None


__all__ = ["SmallLLM", "Intent", "DEFAULT_MODEL", "DEFAULT_URL"]
