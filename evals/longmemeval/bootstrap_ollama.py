"""
evals/longmemeval/bootstrap_ollama.py
======================================
Wire a real Continuum-style pipeline against LongMemEval-S and run the
baseline. **No Postgres required** — uses in-memory STM only — and an
Ollama model serves as the answerer so the run is free and local.

Architecture (per question)
---------------------------
1. Build a fresh in-memory ``EvalSession``.
2. Ingest every haystack message into STM, stamped with the originating
   ``session_id`` so retrieval recall is measurable.
3. On ``answer_question``: rank STM items by cosine similarity against
   the question (sentence-transformers ``all-MiniLM-L6-v2``), select the
   top-K under a token budget, format into a prompt, ask Ollama.

Run modes
---------
.. code-block:: bash

    # smoke test (5 questions) — confirms wiring before paying the
    # full ~hour-long run on consumer hardware.
    python -m evals.longmemeval.bootstrap_ollama --limit 5

    # full 500-question run after smoke looks good. The script will
    # prompt before kicking off the full run unless --yes is given.
    python -m evals.longmemeval.bootstrap_ollama --full

Dependencies
------------
* ``httpx`` (already a project dep)
* ``sentence_transformers`` (lazy-loaded; first run downloads the model)
* Ollama daemon at ``http://localhost:11434`` with the chosen model pulled
* The LongMemEval dataset at
  ``evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json``
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import random
import re
import time
from collections.abc import Iterable
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    Query,
    ScoreBreakdown,
    ScoredItem,
    TokenBudget,
)
from continuum.optimizer import OptimizerChain
from continuum.optimizer.base import estimate_tokens_text
from continuum.optimizer.strategies import (
    LLMLinguaCompress,
    MtmSummarize,
    ScoreAwareBudgetPrune,
    SemanticDedupe,
    StmTrim,
)
from evals.longmemeval.adapter import ContinuumAdapter
from evals.longmemeval.baseline import EvalRow, run_baseline
from evals.longmemeval.decompose import DecompositionRetriever

log = logging.getLogger(__name__)

DEFAULT_DATASET = (
    Path(__file__).parent / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
)
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = os.environ.get(
    "NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"
)
DEFAULT_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai"
)
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_LMSTUDIO_URL = os.environ.get(
    "LMSTUDIO_URL", "http://localhost:1234/v1"
)
DEFAULT_LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-14b")
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _openai_key_from_env() -> str | None:
    """
    Resolve the OpenAI key from the environment, tolerant of the
    non-standard names that turn up in real ``.env`` files.

    Checks ``OPENAI_API_KEY`` first (the canonical name), then the
    common typo'd variants ``OPEN_AI_KEY`` / ``OPEN_API_KEY``.
    """
    for name in ("OPENAI_API_KEY", "OPEN_AI_KEY", "OPEN_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    return None


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def _parse_lme_date(raw: Any) -> dt.datetime | None:
    """
    Parse a LongMemEval date string into a datetime.

    LongMemEval timestamps look like ``2023/05/30 (Tue) 23:40``. We
    strip the ``(Day)`` token and try a couple of formats; on any
    failure return ``None`` so recency simply falls back to uniform.
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = re.sub(r"\([A-Za-z]{3}\)", "", raw).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def load_longmemeval_rows(path: Path | str) -> list[EvalRow]:
    """
    Read the LongMemEval-S JSON and flatten each row into the
    ``EvalRow`` shape expected by :func:`run_baseline`.

    Each haystack session is a list of OpenAI-style messages. We
    concatenate them in chronological order and tag every message with
    its source ``session_id`` *and* its session date — the date is what
    makes the scorer's ``recency`` component a genuine signal.
    """
    raw = json.loads(Path(path).read_text())
    rows: list[EvalRow] = []
    for r in raw:
        flat_messages: list[dict[str, Any]] = []
        sids = r["haystack_session_ids"]
        sessions = r["haystack_sessions"]
        dates = r.get("haystack_dates", [])
        for idx, (sid, session) in enumerate(
            zip(sids, sessions, strict=False)
        ):
            sdate = _parse_lme_date(dates[idx] if idx < len(dates) else None)
            for msg in session:
                flat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "session_id": sid,  # extra field; adapter reads this
                    # ISO date of the session (or "" if unparseable) —
                    # the adapter stamps it onto MemoryItem.created_at.
                    "date": sdate.isoformat() if sdate else "",
                })
        ans_sids = r["answer_session_ids"]
        if isinstance(ans_sids, str):
            ans_sids = [ans_sids]
        rows.append(EvalRow(
            question_id=r["question_id"],
            question=r["question"],
            # Coerce non-string answers (ints, floats) — some LongMemEval
            # rows have numeric ground truth.
            expected_answer=str(r["answer"]),
            messages=flat_messages,
            answer_session_ids=list(ans_sids),
        ))
    return rows


# ---------------------------------------------------------------------------
# Ollama LLM client
# ---------------------------------------------------------------------------


class OllamaLLM:
    """
    Minimal Ollama chat client. Implements the
    ``async complete(prompt, max_tokens) -> str`` contract the
    :class:`ContinuumAdapter` expects.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", "")).strip()
        except Exception as exc:
            log.exception("Ollama call failed")
            raise RuntimeError(f"ollama: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Groq LLM client (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------


class GroqLLM:
    """
    OpenAI-compatible Groq chat client. Same surface as :class:`OllamaLLM`:
    ``async complete(prompt, max_tokens) -> str``.

    The API key is read from the ``GROQ_API_KEY`` env var. **Never** pass
    the key as a literal here — every code path reads from the
    environment so secrets stay out of source/logs.

    Rate limiting
    -------------
    Groq's free tier is ~30 req/min for 70B models. This client enforces
    a configurable client-side cap (``rpm`` — default 25 to stay safely
    under the ceiling) and also handles 429 responses with exponential
    backoff that honours ``Retry-After``.

    Pricing (USD per 1 M tokens, as of late-2025):
        llama-3.3-70b-versatile     $0.59 in  / $0.79 out
        llama-3.1-8b-instant        $0.05 in  / $0.08 out
        gemma2-9b-it                $0.20 in  / $0.20 out
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_GROQ_MODEL,
        base_url: str = DEFAULT_GROQ_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        rpm: int = 25,
        max_retries: int = 8,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY env var not set. Export it before launching:\n"
                "  export GROQ_API_KEY='gsk_...'"
            )
        self._api_key = key
        self._client = httpx.AsyncClient(timeout=timeout)
        self._throttle = _AdaptiveThrottle(rpm)
        self._max_retries = max_retries

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return await _adaptive_complete(
            provider="groq",
            client=self._client,
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=headers,
            throttle=self._throttle,
            max_retries=self._max_retries,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_retry_after(resp: httpx.Response) -> float | None:
    """
    Pull a seconds-to-wait value from Groq's response headers.

    Groq uses ``retry-after`` (RFC standard, value in seconds) and
    sometimes ``x-ratelimit-reset-*`` headers. We honour either.
    """
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = resp.headers.get(key)
        if v is None:
            continue
        try:
            return float(v.rstrip("s"))
        except ValueError:
            continue
    return None


class _AdaptiveThrottle:
    """
    AIMD (additive-increase / multiplicative-decrease) client-side rate
    limiter — the same idea as TCP congestion control.

    A fixed-pace throttle is deaf to feedback: it keeps firing at the
    configured RPM even while the server is returning 429s, so it grazes
    the ceiling forever. This throttle *listens*:

    * ``on_429()``    — widen the inter-request gap (×``widen_factor``)
                        and **keep** it widened. We just learned the
                        current pace is too fast.
    * ``on_success()``— after ``recover_after`` consecutive 200s, edge
                        the gap back toward the configured pace
                        (×``narrow_factor``). The limit may have eased.

    Net effect: a run self-tunes to the provider's *real* sustainable
    rate within a few 429s, instead of retry-storming at a fixed pace.

    Thread/task-safety: ``await_slot`` serialises on an ``asyncio.Lock``.
    ``on_429`` / ``on_success`` mutate a float without the lock — a
    benign race (a momentarily stale interval self-corrects next call).
    """

    def __init__(
        self,
        rpm: int,
        *,
        max_interval: float = 30.0,
        widen_factor: float = 1.5,
        narrow_factor: float = 0.9,
        recover_after: int = 15,
    ) -> None:
        self._target = 60.0 / max(1, rpm)
        self._interval = self._target
        self._max = max(max_interval, self._target)
        self._widen = widen_factor
        self._narrow = narrow_factor
        self._recover_after = recover_after
        self._next_allowed = 0.0
        self._ok_streak = 0
        self._lock = asyncio.Lock()

    @property
    def interval(self) -> float:
        """Current inter-request gap in seconds (for logging)."""
        return self._interval

    async def await_slot(self) -> None:
        """Block until the next request is allowed under the current pace."""
        async with self._lock:
            now = time.perf_counter()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = (
                max(now, self._next_allowed) + self._interval
            )

    def on_429(self) -> None:
        """Widen the gap — current pace exceeded the server's limit."""
        self._interval = min(self._max, self._interval * self._widen)
        self._ok_streak = 0

    def on_success(self) -> None:
        """After a clean streak, edge the pace back toward target."""
        self._ok_streak += 1
        if (
            self._ok_streak >= self._recover_after
            and self._interval > self._target
        ):
            self._interval = max(self._target, self._interval * self._narrow)
            self._ok_streak = 0


async def _adaptive_complete(
    *,
    provider: str,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    throttle: _AdaptiveThrottle,
    max_retries: int,
) -> str:
    """
    One OpenAI-compatible ``/chat/completions`` call with the shared
    adaptive-throttle + retry loop.

    Resilience features (all providers benefit):
    * Paces requests through ``throttle`` (AIMD — self-tunes to the
      real sustainable rate).
    * On 429: widens the throttle, then backs off ``2**attempt`` seconds
      (honouring ``Retry-After`` if present) **with ±20 % jitter** so
      concurrent retries don't resync, capped at 128 s.
    * On success: narrows the throttle back toward target.
    * Reasoning-model ``<think>`` blocks are stripped from the answer.
    """
    attempt = 0
    while True:
        await throttle.await_slot()
        try:
            resp = await client.post(url, json=payload, headers=headers)
        except Exception as exc:
            log.exception("%s network error", provider)
            raise RuntimeError(f"{provider}: {exc!r}") from exc

        if resp.status_code == 429:
            throttle.on_429()
            if attempt < max_retries:
                base_wait = _parse_retry_after(resp) or float(2 ** attempt)
                wait = min(128.0, base_wait * random.uniform(0.8, 1.2))
                log.warning(
                    "%s 429 — adaptive pace now %.2fs/call; "
                    "backoff %.1fs (retry %d/%d)",
                    provider, throttle.interval, wait,
                    attempt + 1, max_retries,
                )
                await asyncio.sleep(wait)
                attempt += 1
                continue
            raise RuntimeError(
                f"{provider}: 429 after {max_retries} retries"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{provider}: HTTP {resp.status_code} — {resp.text[:200]}"
            )

        throttle.on_success()
        data = resp.json()
        raw = str(data["choices"][0]["message"]["content"])
        return _strip_think(raw).strip()


# ---------------------------------------------------------------------------
# NVIDIA Build API client (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------


class NvidiaLLM:
    """
    NVIDIA Build API client. Drop-in for :class:`GroqLLM` — same
    ``async complete(prompt, max_tokens) -> str`` surface.

    The free tier gives **1000 credits** per account (1 credit per
    successful inference), enough for a full LongMemEval-S run with
    plenty of headroom. Endpoint is OpenAI-compatible.

    Reads the key from ``NVIDIA_API_KEY``. Keys are prefixed ``nvapi-``.

    Models worth knowing (all free-tier accessible)::
        meta/llama-3.1-405b-instruct        # 405 B — largest free
        meta/llama-3.3-70b-instruct         # 70 B Llama 3.3
        nvidia/llama-3.1-nemotron-70b-instruct  # RLHF on top of 70B
        deepseek-ai/deepseek-r1             # reasoning-tuned
        mistralai/mistral-large             # strong general

    Rate limits on free tier are ~40 RPM on most models. The client-
    side throttle defaults match this; if you upgrade to a paid tier,
    bump ``rpm``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_NVIDIA_MODEL,
        base_url: str = DEFAULT_NVIDIA_URL,
        timeout: float = 90.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        rpm: int = 35,
        max_retries: int = 8,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError(
                "NVIDIA_API_KEY env var not set. Export it before launching:\n"
                "  export NVIDIA_API_KEY='nvapi-...'\n"
                "Generate one at https://build.nvidia.com/ → personal key."
            )
        self._api_key = key
        self._client = httpx.AsyncClient(timeout=timeout)
        self._throttle = _AdaptiveThrottle(rpm)
        self._max_retries = max_retries

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        # NVIDIA's gateway 404s the request when Content-Type / Accept
        # aren't explicit (httpx normally sets Content-Type, but the
        # gateway is picky about exact header presence + order).
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return await _adaptive_complete(
            provider="nvidia",
            client=self._client,
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=headers,
            throttle=self._throttle,
            max_retries=self._max_retries,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Google Gemini client (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


class GeminiLLM:
    """
    Google Gemini chat client via the OpenAI-compatible endpoint at
    ``generativelanguage.googleapis.com/v1beta/openai``.

    The free tier is *per-day* (1,500 requests/day, 1 M tokens/min on
    gemini-2.0-flash) — far more generous than NVIDIA's 1,000-credit
    lifetime budget for our use case.

    Reads the key from ``GEMINI_API_KEY``. Get one at
    https://aistudio.google.com/apikey (free).

    Recommended models for LongMemEval::
        gemini-2.0-flash             # fastest, free tier
        gemini-2.5-flash             # newer / smarter, same free tier
        gemini-2.5-pro               # stronger reasoning, lower rate caps
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        base_url: str = DEFAULT_GEMINI_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        rpm: int = 25,
        max_retries: int = 8,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY env var not set. Export it before launching:\n"
                "  export GEMINI_API_KEY='AIza...'\n"
                "Generate one at https://aistudio.google.com/apikey (free)."
            )
        self._api_key = key
        self._client = httpx.AsyncClient(timeout=timeout)
        self._throttle = _AdaptiveThrottle(rpm)
        self._max_retries = max_retries

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return await _adaptive_complete(
            provider="gemini",
            client=self._client,
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=headers,
            throttle=self._throttle,
            max_retries=self._max_retries,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# OpenAI client (the original OpenAI-compatible chat API)
# ---------------------------------------------------------------------------


class OpenAILLM:
    """
    OpenAI chat client. Same ``async complete(prompt, max_tokens) -> str``
    surface as the other providers, sharing the adaptive-throttle +
    retry loop via :func:`_adaptive_complete`.

    The key is resolved by :func:`_openai_key_from_env` — it accepts the
    canonical ``OPENAI_API_KEY`` and tolerates the ``OPEN_AI_KEY`` /
    ``OPEN_API_KEY`` variants that appear in hand-written ``.env`` files.

    OpenAI has **no free tier** — the account needs billing enabled. A
    full LongMemEval-S run on ``gpt-4o-mini`` (top-K mode) costs roughly
    $0.08; long-context mode roughly $5–6.

    Pricing (USD per 1 M tokens, late-2025):
        gpt-4o-mini   $0.15 in  / $0.60 out
        gpt-4o        $2.50 in  / $10.00 out
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_URL,
        timeout: float = 90.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        rpm: int = 60,
        max_retries: int = 8,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        key = api_key or _openai_key_from_env()
        if not key:
            raise RuntimeError(
                "OpenAI key not set. Export one of:\n"
                "  export OPENAI_API_KEY='sk-...'\n"
                "(OPEN_AI_KEY / OPEN_API_KEY are also accepted.)"
            )
        self._api_key = key
        self._client = httpx.AsyncClient(timeout=timeout)
        self._throttle = _AdaptiveThrottle(rpm)
        self._max_retries = max_retries

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return await _adaptive_complete(
            provider="openai",
            client=self._client,
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=headers,
            throttle=self._throttle,
            max_retries=self._max_retries,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# LM Studio client — local OpenAI-compatible server, no API key
# ---------------------------------------------------------------------------


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """
    Remove Qwen3 / reasoning-model ``<think>...</think>`` blocks.

    Qwen3 emits chain-of-thought wrapped in <think> tags when reasoning
    mode is on. LongMemEval wants the bare answer, so strip the block.
    If the closing tag is missing (truncated output), drop everything
    up to and including the last </think> we *did* see, else leave it.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    # Truncated case: an opening <think> with no close — keep only what
    # follows it isn't possible, so just drop the dangling tag.
    if "<think>" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        if idx != -1:
            cleaned = cleaned[idx + len("</think>"):]
    return cleaned.strip()


class LMStudioLLM:
    """
    Client for a local **LM Studio** server (OpenAI-compatible).

    LM Studio exposes ``/v1/chat/completions`` at
    ``http://localhost:1234`` by default and needs **no API key** — it's
    a local process. Same ``async complete(prompt, max_tokens) -> str``
    surface as the other providers.

    Local inference has no rate limits, but a 14B model on consumer
    hardware is slow (~5–25 tok/s) — the timeout defaults high (300 s)
    and there's no throttle.

    Qwen3 reasoning-mode ``<think>`` blocks are stripped automatically.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_LMSTUDIO_MODEL,
        base_url: str = DEFAULT_LMSTUDIO_URL,
        timeout: float = 300.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            raw = str(data["choices"][0]["message"]["content"])
            return _strip_think(raw)
        except Exception as exc:
            log.exception("LM Studio call failed")
            raise RuntimeError(f"lmstudio: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Sentence-transformer embedder + STM retriever
# ---------------------------------------------------------------------------


class _Embedder:
    """Thin wrapper around sentence_transformers, loaded lazily once."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _lazy(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedder %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self._lazy()
        v = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


class FlatHaystackStore:
    """
    Trivial list-backed store for the entire LongMemEval haystack.

    We deliberately *don't* use :class:`InMemorySTM` here — its default
    4 K-token cap evicts most of a 491-message haystack silently, which
    would corrupt every recall measurement. The benchmark assumes every
    haystack message remains addressable; a flat list is the honest
    representation.
    """

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    async def append(self, item: MemoryItem) -> None:
        self.items.append(item)


class CrossEncoderReranker:
    """
    Two-stage retrieval: take top-N cosine candidates and rescore each
    (query, passage) pair with a cross-encoder. Returns the top-K with
    the highest cross-encoder scores.

    Cross-encoders are far more accurate than bi-encoder cosine matching
    because the model sees both texts together and attends across them.
    Cost: ~10–30 ms per (query, passage) pair on CPU → ~300–900 ms total
    for N=30 candidates per question (rerank is sync, runs in
    ``asyncio.to_thread`` to avoid blocking).

    Default model ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is the
    canonical small reranker: ~80 MB, trained on MS-MARCO passage
    ranking. Works well for the LongMemEval pattern (find the relevant
    turn in a long conversation).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _lazy(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder %s …", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self, query: str, items: list[MemoryItem], *, top_k: int
    ) -> list[MemoryItem]:
        if not items or top_k >= len(items):
            return items[:top_k] if top_k else items
        model = self._lazy()
        pairs = [(query, it.content) for it in items]
        # Predict is sync + can be slow; offload to a worker thread.
        scores = await asyncio.to_thread(model.predict, pairs)
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [items[int(i)] for i in order]


class STMSemanticRetriever:
    """
    Adaptive retrieval over a :class:`FlatHaystackStore`.

    Two retrieval modes, plus an ``auto`` policy that picks between them:

    * **top-K**  — cosine-similarity search, return the K best items.
      The right call when the haystack is far larger than the model's
      context window.
    * **long-context** — return *every* item (chronological order). The
      right call when the whole haystack fits comfortably in context:
      retrieving top-K and discarding the rest would throw away
      information the model could have used (this is what caps recall).

    ``mode``:
      ``"topk"`` — always cosine top-K (legacy behaviour).
      ``"long"`` — always pass everything, *unless* it exceeds
                   ``max_context_tokens``, in which case it falls back
                   to top-K (it physically can't fit).
      ``"auto"`` — long-context when the haystack fits
                   ``max_context_tokens``; top-K otherwise.

    The chosen mode for a given call is stamped onto
    ``ContextBundle.debug_info["retrieval_mode"]`` so the adapter knows
    whether to skip the optimizer chain (compressing a deliberately-
    whole context back down would defeat the purpose).
    """

    #: Default composite-score weights — Prompt 38 "Iteration 1" config.
    #: relevance + importance + recency + confidence, summing to 1.0.
    DEFAULT_WEIGHTS = {
        "relevance": 0.45,
        "importance": 0.25,
        "recency": 0.20,
        "confidence": 0.10,
    }

    def __init__(
        self,
        *,
        store: FlatHaystackStore,
        embedder: _Embedder,
        top_k: int = 8,
        session_id: str = "default",
        mode: str = "topk",
        max_context_tokens: int = 100_000,
        score_weights: dict[str, float] | None = None,
        tau_hours: float = 168.0,
    ) -> None:
        if mode not in ("topk", "long", "auto"):
            raise ValueError(f"unknown retrieval mode: {mode!r}")
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.session_id = session_id
        self.mode = mode
        self.max_context_tokens = max_context_tokens
        # Composite scorer (Prompt 38). For LongMemEval the haystack
        # carries no per-message importance/confidence, so those two
        # inputs are uniform constants — the *effective* tuning knob is
        # relevance-vs-recency. The weights are still applied verbatim
        # so the config matches ContinuumConfig.scoring 1:1.
        w = dict(self.DEFAULT_WEIGHTS)
        if score_weights:
            w.update(score_weights)
        self.score_weights = w
        self.tau_hours = float(tau_hours)
        self._cached_matrix: np.ndarray | None = None
        self._cached_items: list[MemoryItem] = []
        self._cached_len = 0

    def _refresh_index(self) -> None:
        items = self.store.items
        if not items:
            self._cached_matrix = None
            self._cached_items = []
            self._cached_len = 0
            return
        # Re-embed only if the haystack grew (always true on first call
        # per question; never refires within a single retrieve()).
        if self._cached_matrix is None or len(items) != self._cached_len:
            texts = [it.content for it in items]
            self._cached_matrix = self.embedder.encode(texts)
            self._cached_items = list(items)
            self._cached_len = len(items)

    def _resolve_mode(self, total_tokens: int) -> str:
        """Decide topk vs long-context for this call."""
        if self.mode == "topk":
            return "topk"
        fits = total_tokens <= self.max_context_tokens
        if self.mode == "long":
            # 'long' wants everything, but can't exceed the window.
            return "long" if fits else "topk"
        # 'auto': long-context only when it genuinely fits.
        return "long" if fits else "topk"

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        # Rebuild the embedding index on each call — cheap relative to
        # the LLM call, and avoids invalidation bugs.
        self._refresh_index()
        if self._cached_matrix is None or not self._cached_items:
            return ContextBundle(
                items=[], messages=[], tokens_used=0, budget=budget,
                tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
                debug_info={"retrieval_mode": "empty"},
            )

        total_tokens = sum(
            estimate_tokens_text(it.content) for it in self._cached_items
        )
        resolved = self._resolve_mode(total_tokens)

        if resolved == "long":
            return self._bundle_long_context(budget, total_tokens)
        return self._bundle_topk(query, budget)

    # ── long-context branch ────────────────────────────────────────────────

    def _bundle_long_context(
        self, budget: TokenBudget, total_tokens: int
    ) -> ContextBundle:
        """Return the *entire* haystack, chronological order."""
        items = self._cached_items
        messages = [
            {
                "role": str(it.metadata.get("role", "user"))
                if it.metadata else "user",
                "content": it.content,
            }
            for it in items
        ]
        return ContextBundle(
            items=list(items),
            messages=messages,
            tokens_used=total_tokens,
            budget=budget,
            tier_breakdown={"stm": total_tokens, "mtm": 0, "ltm": 0},
            debug_info={
                "retrieval_mode": "long",
                "long_context_items": len(items),
                "long_context_tokens": total_tokens,
            },
        )

    # ── top-K branch (cosine) ──────────────────────────────────────────────

    def _recency_scores(self) -> np.ndarray:
        """
        Exponential-decay recency in [0, 1] for every cached item.

        ``recency_i = exp(-(t_max - t_i) / tau_hours)`` where ``t_max``
        is the newest item's timestamp. Items at the end of the haystack
        timeline score ≈1; older ones decay. When timestamps are absent
        or all equal, every score is 1.0 (recency contributes a uniform
        constant — inert, as intended).
        """
        items = self._cached_items
        epochs: list[float] = []
        for it in items:
            ts = getattr(it, "created_at", None)
            epochs.append(ts.timestamp() if ts is not None else 0.0)
        arr = np.asarray(epochs, dtype=np.float64)
        t_max = arr.max() if arr.size else 0.0
        if t_max == 0.0 or float(arr.max() - arr.min()) < 1.0:
            return np.ones(len(items), dtype=np.float64)
        age_hours = (t_max - arr) / 3600.0
        tau = max(1.0, self.tau_hours)
        return np.exp(-age_hours / tau)

    def _bundle_topk(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        qv = self.embedder.encode([query.text])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = (qv @ self._cached_matrix.T).flatten()  # type: ignore[union-attr]

        # ── composite score (Prompt 38) ────────────────────────────────────
        # relevance: cosine, min-max normalised to [0,1] over candidates.
        rng = float(sims.max() - sims.min())
        rel = (sims - sims.min()) / rng if rng > 1e-9 else np.full_like(sims, 0.5)
        # recency: exponential decay over the haystack timeline.
        rec = self._recency_scores()
        # importance / confidence: per-item if present, else uniform —
        # LongMemEval messages carry neither, so these are constants.
        imp = np.asarray(
            [float(getattr(it, "importance", 0.5)) for it in self._cached_items],
            dtype=np.float64,
        )
        conf = np.asarray(
            [float(getattr(it, "confidence", 1.0)) for it in self._cached_items],
            dtype=np.float64,
        )
        w = self.score_weights
        composite = (
            w["relevance"] * rel
            + w["importance"] * imp
            + w["recency"] * rec
            + w["confidence"] * conf
        )
        order = np.argsort(-composite)[: self.top_k]
        picked = [self._cached_items[i] for i in order]

        scored = [
            ScoredItem(
                item=picked[i],
                scores=ScoreBreakdown(
                    relevance=float(rel[order[i]]),
                    importance=float(imp[order[i]]),
                    recency=float(rec[order[i]]),
                    confidence=float(conf[order[i]]),
                    composite=float(composite[order[i]]),
                ),
            )
            for i in range(len(picked))
        ]
        messages = [
            {
                "role": str(it.metadata.get("role", "user"))
                if it.metadata else "user",
                "content": it.content,
            }
            for it in picked
        ]
        token_count = sum(
            estimate_tokens_text(it.content) for it in picked
        )
        return ContextBundle(
            items=[s.item for s in scored],
            messages=messages,
            tokens_used=token_count,
            budget=budget,
            tier_breakdown={"stm": token_count, "mtm": 0, "ltm": 0},
            debug_info={"retrieval_mode": "topk"},
        )


# ---------------------------------------------------------------------------
# Per-question session + adapter
# ---------------------------------------------------------------------------


class _MiniSession:
    """
    Tiny stand-in for :class:`continuum.core.session.ContinuumSession`.

    The full ContinuumSession runs a Responder for every user turn,
    which we don't want during haystack ingestion (491 LLM calls per
    question would torch the budget). This session just exposes ``stm``
    + ``retriever`` so the :class:`ContinuumAdapter` works unchanged.
    The ``stm`` slot points at a :class:`FlatHaystackStore` — an honest
    backing for the haystack that doesn't evict messages.
    """

    def __init__(
        self, *, store: FlatHaystackStore, retriever: STMSemanticRetriever
    ) -> None:
        self.stm = store  # adapter still calls session.stm.append
        self.retriever = retriever
        self.session_id = retriever.session_id


class _IngestingAdapter(ContinuumAdapter):
    """
    Adapter variant that stamps each ingested message's ``session_id``
    onto the resulting :class:`MemoryItem.metadata`, so retrieval recall
    is measurable from ``adapter.last_ctx``.
    """

    async def process_conversation(
        self, messages: Iterable[dict[str, Any]]
    ) -> None:
        store = self.session.stm
        for msg in messages:
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            item = MemoryItem(
                content=content,
                tier=MemoryTier.STM,
                metadata={
                    "role": str(msg.get("role", "user")),
                    "session_id": str(msg.get("session_id", "")),
                },
            )
            # Stamp the session date onto created_at so the scorer's
            # recency component is a real signal (Prompt 38 tuning).
            date_iso = str(msg.get("date", "") or "")
            if date_iso:
                with contextlib.suppress(ValueError):
                    item.created_at = dt.datetime.fromisoformat(date_iso)
            await store.append(item)


# ---------------------------------------------------------------------------
# Optimizer chain wiring
# ---------------------------------------------------------------------------


def build_default_chain(
    *,
    llmlingua_compressor: Any | None = None,
    use_llmlingua: bool = True,
) -> OptimizerChain:
    """
    The 5-strategy stack from the spec.

    ``use_llmlingua=False`` drops LLMLingua entirely — useful when the
    upstream package isn't installed (the strategy would otherwise log
    a "compressor unavailable" error on every row).

    ``llmlingua_compressor`` is injectable so tests / cost-conscious
    runs swap a fake; when ``None`` *and* ``use_llmlingua`` is True,
    LLMLinguaCompress lazy-loads the real model on first call
    (~500 MB download).
    """
    strategies: list[Any] = [
        StmTrim(keep_last=10),
        MtmSummarize(keep_recent=3, method="extractive"),
        SemanticDedupe(threshold=0.92, skip_if_fewer_than=10),
    ]
    if use_llmlingua:
        # Probe for the package before adding the strategy so users
        # without it get a clean log line instead of per-row errors.
        try:
            import llmlingua  # noqa: F401
            strategies.append(LLMLinguaCompress(
                ratio=0.5,
                min_input_tokens=500,
                compressor=llmlingua_compressor,
            ))
        except ImportError:
            log.warning(
                "llmlingua not installed — skipping LLMLinguaCompress. "
                "Install with: pip install llmlingua"
            )
    strategies.append(ScoreAwareBudgetPrune(preserve_mtm_count=3))
    return OptimizerChain(strategies)


def _re_tier(items: list[MemoryItem]) -> list[MemoryItem]:
    """
    Re-tag a flat retrieved item list into STM / MTM / LTM buckets by
    **relevance rank** so every strategy in the chain has work to do
    *without* clobbering the most relevant rows.

    The retriever returns items in cosine-descending order (best match
    first). The tiering reflects that semantic, not position:

        TOP    15 %  → STM   — preserved verbatim by StmTrim
        NEXT   25 %  → MTM   — summarised by MtmSummarize
        BOTTOM 60 %  → LTM   — cosine-dedup + LLMLingua + budget-prune

    Earlier versions tiered by list position (last-15%-is-STM), which
    on a cosine-ordered list meant the *worst* matches were preserved
    and the *best* were aggressively compressed. That tanked accuracy.
    """
    n = len(items)
    if n == 0:
        return items
    stm_cut = max(1, int(n * 0.15))           # top 15 % stays STM
    mtm_cut = max(stm_cut + 1, int(n * 0.40))  # next 25 % → MTM
    out: list[MemoryItem] = []
    for idx, it in enumerate(items):
        if idx < stm_cut:
            tier = MemoryTier.STM
        elif idx < mtm_cut:
            tier = MemoryTier.MTM
        else:
            tier = MemoryTier.LTM
        out.append(dataclasses_replace(it, tier=tier))
    return out


def _tier_token_breakdown(items: list[MemoryItem]) -> dict[str, int]:
    counts: dict[str, int] = {"stm": 0, "mtm": 0, "ltm": 0}
    for it in items:
        counts[it.tier.value] = counts.get(it.tier.value, 0) + estimate_tokens_text(
            it.content
        )
    return counts


def _extract_strategy_savings(debug_info: dict[str, Any]) -> dict[str, int]:
    """
    Convert each strategy's ``debug_info`` stamp into a "tokens removed"
    integer keyed by strategy name. Strategies that didn't fire return 0.
    """
    out: dict[str, int] = {}
    if "stm_trim" in debug_info:
        d = debug_info["stm_trim"]
        # rough: summarised count * ~50 tokens − produced summary tokens
        compacted = int(d.get("summarised", 0))
        summary_chars = int(d.get("summary_chars", 0))
        out["stm_trim"] = max(0, compacted * 50 - summary_chars // 4)
    if "mtm_summarize" in debug_info:
        d = debug_info["mtm_summarize"]
        out["mtm_summarize"] = max(
            0, int(d.get("input_tokens", 0)) - int(d.get("output_tokens", 0))
        )
    if "semantic_dedupe" in debug_info:
        d = debug_info["semantic_dedupe"]
        # ~average tokens per removed item — we don't have exact data, use 30
        out["semantic_dedupe"] = int(d.get("removed", 0)) * 30
    if "llmlingua" in debug_info:
        d = debug_info["llmlingua"]
        out["llmlingua"] = max(
            0, int(d.get("input_tokens", 0)) - int(d.get("output_tokens", 0))
        )
    if "score_prune" in debug_info:
        d = debug_info["score_prune"]
        out["score_prune"] = max(
            0, int(d.get("before_tokens", 0)) - int(d.get("after_tokens", 0))
        )
    return out


class _OptimizingAdapter(_IngestingAdapter):
    """
    Variant of :class:`_IngestingAdapter` that runs an
    :class:`OptimizerChain` between retrieval and prompt formatting.

    Publishes :attr:`last_optimizer_stats` after each
    :meth:`answer_question` so the baseline runner can pull
    per-strategy savings + pre/post token deltas into the per-row record.
    """

    def __init__(
        self,
        *,
        chain: OptimizerChain,
        reranker: CrossEncoderReranker | None = None,
        rerank_to: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chain = chain
        self.reranker = reranker
        self.rerank_to = rerank_to
        self.last_optimizer_stats: dict[str, Any] = {}

    async def answer_question(self, question: str) -> str:
        # ── retrieve ───────────────────────────────────────────────────────
        retriever = getattr(self.session, "retriever", None)
        ctx: ContextBundle | None = None
        t_ret = time.perf_counter()
        if retriever is not None:
            try:
                ctx = await retriever.retrieve(Query(text=question), self.budget)
            except Exception:
                log.exception("retrieve failed for %r", question[:80])
        retrieval_ms = (time.perf_counter() - t_ret) * 1000.0

        # When retrieval ran in long-context OR decompose mode the
        # bundle is already a curated context. Reranking / re-tiering /
        # compressing it would defeat the purpose — skip to the prompt.
        retrieval_mode = (
            (ctx.debug_info or {}).get("retrieval_mode", "topk")
            if ctx is not None else "topk"
        )
        long_context = retrieval_mode in ("long", "decompose")

        # ── rerank ─────────────────────────────────────────────────────────
        rerank_ms = 0.0
        if (
            not long_context
            and self.reranker is not None
            and ctx is not None and ctx.items
        ):
            t_rer = time.perf_counter()
            try:
                top = await self.reranker.rerank(
                    question, list(ctx.items), top_k=self.rerank_to,
                )
                # Preserve the rerank order (best first) — _re_tier
                # below relies on cosine/cross-encoder ranking to pick
                # which items become STM.
                ctx = dataclasses_replace(ctx, items=top)
            except Exception:
                log.exception("reranker failed; falling back to cosine order")
            rerank_ms = (time.perf_counter() - t_rer) * 1000.0

        # Re-tier so every strategy has a population to operate on. The
        # FlatHaystackStore retains insertion order; the retriever
        # currently returns oldest-first under cosine ordering above —
        # we treat the chunk *as if* the order were chronological so the
        # tail is "STM-like" recent context. (Skipped in long-context.)
        if not long_context and ctx is not None and ctx.items:
            ctx = dataclasses_replace(ctx, items=_re_tier(list(ctx.items)))

        pre_breakdown = (
            _tier_token_breakdown(ctx.items) if ctx is not None else {}
        )
        pre_total = sum(pre_breakdown.values())

        # ── optimize ───────────────────────────────────────────────────────
        post_total = pre_total
        post_breakdown = dict(pre_breakdown)
        debug_info: dict[str, Any] = {}
        opt_ms = 0.0
        if not long_context and ctx is not None and ctx.items:
            t_opt = time.perf_counter()
            try:
                ctx = await self.chain.apply(ctx, self.budget)
                debug_info = dict(ctx.debug_info or {})
            except Exception:
                log.exception("optimizer chain failed; using unoptimised ctx")
            opt_ms = (time.perf_counter() - t_opt) * 1000.0
            post_breakdown = _tier_token_breakdown(ctx.items)
            post_total = sum(post_breakdown.values())

        # ── prompt + LLM ───────────────────────────────────────────────────
        self.last_ctx = ctx
        prompt = self.format_prompt(question, ctx)
        try:
            answer = await self.llm.complete(
                prompt=prompt, max_tokens=self.answer_max_tokens,
            )
        except Exception as exc:
            log.exception("LLM completion failed for %r", question[:80])
            answer = f"[error: {exc!r}]"

        self.last_optimizer_stats = {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "optimizer_ms": opt_ms,
            "retrieved_count": len(ctx.items) if ctx is not None else 0,
            "pre_total": pre_total,
            "post_total": post_total,
            "pre_stm": pre_breakdown.get("stm", 0),
            "pre_mtm": pre_breakdown.get("mtm", 0),
            "pre_ltm": pre_breakdown.get("ltm", 0),
            "post_stm": post_breakdown.get("stm", 0),
            "post_mtm": post_breakdown.get("mtm", 0),
            "post_ltm": post_breakdown.get("ltm", 0),
            "strategy_savings": _extract_strategy_savings(debug_info),
            "retrieval_mode": retrieval_mode,
        }
        return str(answer).strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_adapter_factory(
    llm: Any,
    embedder: _Embedder,
    *,
    top_k: int = 8,
    answer_max_tokens: int = 200,
    chain: OptimizerChain | None = None,
    reranker: CrossEncoderReranker | None = None,
    rerank_to: int = 4,
    retrieval_mode: str = "topk",
    max_context_tokens: int = 100_000,
    score_weights: dict[str, float] | None = None,
    tau_hours: float = 168.0,
    decompose: bool = False,
    decompose_max_items: int = 16,
) -> Any:
    """
    Return a zero-arg factory yielding a fresh adapter per question.

    Pass ``chain`` to enable the optimizer path. Pass ``reranker`` to
    insert a cross-encoder rerank step between cosine retrieval and the
    chain: the retriever pulls ``top_k`` candidates by cosine, the
    reranker scores all of them with a cross-encoder, and the top
    ``rerank_to`` are forwarded.

    ``retrieval_mode`` (``topk`` / ``auto`` / ``long``) selects how the
    retriever picks context — see :class:`STMSemanticRetriever`. In
    ``long`` / ``auto``-resolved-to-long, the optimizer chain + reranker
    are skipped so the whole haystack reaches the model intact.

    ``score_weights`` / ``tau_hours`` configure the composite scorer
    (Prompt 38 scorer-weight tuning).

    ``decompose`` wraps the retriever in a :class:`DecompositionRetriever`
    — the query is split into atomic sub-questions, each retrieved
    separately, and the union becomes the context. This is the lever
    for multi-session / temporal questions that single-shot retrieval
    cannot answer.
    """

    def factory() -> _IngestingAdapter:
        store = FlatHaystackStore()
        base_retriever = STMSemanticRetriever(
            store=store, embedder=embedder, top_k=top_k,
            mode=retrieval_mode, max_context_tokens=max_context_tokens,
            score_weights=score_weights, tau_hours=tau_hours,
        )
        retriever: Any = base_retriever
        if decompose:
            retriever = DecompositionRetriever(
                base=base_retriever, llm=llm, max_items=decompose_max_items,
            )
        session = _MiniSession(store=store, retriever=retriever)
        if chain is None:
            return _IngestingAdapter(
                session=session,
                llm=llm,
                answer_max_tokens=answer_max_tokens,
            )
        return _OptimizingAdapter(
            chain=chain,
            reranker=reranker,
            rerank_to=rerank_to,
            session=session,
            llm=llm,
            answer_max_tokens=answer_max_tokens,
        )

    return factory


async def _verify_ollama(model: str, base_url: str) -> None:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{base_url}/api/tags")
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
    if model not in names:
        raise RuntimeError(
            f"Ollama doesn't have {model!r}. Available: {sorted(names)}. "
            f"Pull it with: ollama pull {model}"
        )


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    # Pick the model default if the user didn't override it explicitly.
    model = args.model
    if not model:
        model = {
            "groq":     DEFAULT_GROQ_MODEL,
            "nvidia":   DEFAULT_NVIDIA_MODEL,
            "gemini":   DEFAULT_GEMINI_MODEL,
            "openai":   DEFAULT_OPENAI_MODEL,
            "lmstudio": DEFAULT_LMSTUDIO_MODEL,
            "ollama":   DEFAULT_OLLAMA_MODEL,
        }.get(args.provider, DEFAULT_OLLAMA_MODEL)

    llm: Any
    if args.provider == "groq":
        log.info("provider=groq model=%s rpm=%d", model, args.rpm)
        llm = GroqLLM(model=model, rpm=args.rpm)
    elif args.provider == "nvidia":
        log.info("provider=nvidia model=%s rpm=%d", model, args.rpm)
        llm = NvidiaLLM(model=model, rpm=args.rpm)
    elif args.provider == "gemini":
        log.info("provider=gemini model=%s rpm=%d", model, args.rpm)
        llm = GeminiLLM(model=model, rpm=args.rpm)
    elif args.provider == "openai":
        log.info("provider=openai model=%s rpm=%d", model, args.rpm)
        llm = OpenAILLM(model=model, rpm=args.rpm)
    elif args.provider == "lmstudio":
        log.info("provider=lmstudio model=%s url=%s", model, args.lmstudio_url)
        llm = LMStudioLLM(model=model, base_url=args.lmstudio_url)
    else:
        log.info("provider=ollama model=%s", model)
        await _verify_ollama(model, args.ollama_url)
        llm = OllamaLLM(model=model, base_url=args.ollama_url)

    log.info("loading dataset from %s …", args.dataset)
    rows = load_longmemeval_rows(args.dataset)
    log.info("loaded %d rows", len(rows))

    embedder = _Embedder()

    chain: OptimizerChain | None = None
    if args.optimizer:
        chain = build_default_chain(use_llmlingua=not args.no_llmlingua)
        names = [type(s).__name__ for s in chain.strategies]
        log.info("optimizer chain ENABLED: %s", " → ".join(names))

    reranker: CrossEncoderReranker | None = None
    if args.rerank:
        reranker = CrossEncoderReranker(model_name=args.rerank_model)
        log.info(
            "reranker ENABLED: %s — retrieve top-%d → rerank → keep top-%d",
            args.rerank_model, args.top_k, args.rerank_to,
        )

    # When the optimizer is on, retrieve a larger pool so the chain has
    # something to compress. Without the optimizer we keep the small
    # baseline top-k for an apples-to-apples context size.
    effective_top_k = (
        args.top_k if args.top_k != 8 or not args.optimizer else 40
    )

    if args.retrieval_mode != "topk":
        log.info(
            "retrieval mode = %s (max_context_tokens=%d) — "
            "long-context bypasses rerank + optimizer chain",
            args.retrieval_mode, args.max_context_tokens,
        )

    # Parse the composite-scorer weights (Prompt 38). Format:
    # "relevance,importance,recency,confidence" — must sum to ~1.0.
    score_weights = _parse_scorer_weights(args.scorer_weights)
    log.info(
        "scorer weights = %s   tau_hours = %.0f",
        {k: round(v, 2) for k, v in score_weights.items()}, args.tau_hours,
    )
    if args.decompose:
        log.info(
            "decomposition ENABLED — questions split into sub-questions, "
            "retrieved per-part, merged (max %d items)",
            args.decompose_max_items,
        )

    factory = make_adapter_factory(
        llm=llm, embedder=embedder, top_k=effective_top_k, chain=chain,
        reranker=reranker, rerank_to=args.rerank_to,
        retrieval_mode=args.retrieval_mode,
        max_context_tokens=args.max_context_tokens,
        score_weights=score_weights,
        tau_hours=args.tau_hours,
        decompose=args.decompose,
        decompose_max_items=args.decompose_max_items,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Smoke test ──────────────────────────────────────────────────────────
    if args.limit is None or args.limit > 5:
        smoke_limit = 5
    else:
        smoke_limit = args.limit
    log.info("=== SMOKE TEST (%d questions) ===", smoke_limit)
    t0 = time.perf_counter()
    smoke = await run_baseline(
        dataset=args.dataset_name,
        adapter_factory=factory,
        dataset_loader=lambda _: rows,
        output_dir=out_dir / "smoke",
        answerer=model,
        limit=smoke_limit,
    )
    log.info("smoke completed in %.0fs", time.perf_counter() - t0)

    if not args.full:
        log.info("Smoke run only. Pass --full to run all %d questions.", len(rows))
        return 0

    if not args.yes:
        print()
        prompt = (
            f"Smoke results: accuracy={smoke.accuracy:.1%}, "
            f"avg_tokens={smoke.avg_tokens:.0f}. "
            f"Proceed with full {len(rows)}-question run? [y/N] "
        )
        if input(prompt).strip().lower() not in {"y", "yes"}:
            log.info("aborted by user")
            return 0

    log.info("=== FULL RUN (%d questions) ===", len(rows))
    t0 = time.perf_counter()
    full = await run_baseline(
        dataset=args.dataset_name,
        adapter_factory=factory,
        dataset_loader=lambda _: rows,
        output_dir=out_dir,
        answerer=model,
        limit=args.limit,
    )
    log.info("full run completed in %.0fs", time.perf_counter() - t0)
    log.info(
        "final accuracy=%.1f%% recall=%.1f%% cost=$%.2f",
        full.accuracy * 100, full.recall * 100, full.total_cost_usd,
    )
    if args.optimizer:
        # Re-save under the spec-mandated filename + dump an analysis
        # report so the user has everything in one place.
        target = out_dir / f"optimizer_iteration_{args.iteration}.json"
        target.write_text(json.dumps(full.to_dict(), indent=2, default=str))
        log.info("optimizer metrics → %s", target)
        _print_optimizer_summary(full)
    await llm.aclose()
    return 0


def _print_optimizer_summary(results: Any) -> None:
    """Spec section 5: per-strategy contribution + recommendations."""
    print()
    print("=" * 70)
    print("OPTIMIZER ITERATION 1 — analysis")
    print("=" * 70)
    m = results.to_dict()["metrics"]
    print(f"  accuracy:               {m['accuracy']:.1%}")
    print(f"  recall:                 {m['recall']:.1%}")
    print(f"  avg context (pre  opt): {m['avg_context_tokens_pre_opt']:.0f} tok")
    print(f"  avg context (post opt): {m['avg_context_tokens_post_opt']:.0f} tok")
    print(f"  context reduction:      {m['context_token_reduction_pct']:.1f}%")
    print(f"  avg optimizer time:     {m['avg_optimizer_ms']:.0f}ms")
    print(f"  latency p50/p95:        {m['latency_p50_ms']:.0f}ms / {m['latency_p95_ms']:.0f}ms")
    print()
    print("Per-strategy token savings (total across all rows):")
    savings = m.get("strategy_savings_total", {})
    total = sum(savings.values()) or 1
    for k in ("stm_trim", "mtm_summarize", "semantic_dedupe", "llmlingua", "score_prune"):
        v = savings.get(k, 0)
        pct = 100.0 * v / total if total else 0
        print(f"  {k:<18} {v:>10,} tokens   ({pct:5.1f}% of total savings)")
    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_scorer_weights(spec: str) -> dict[str, float]:
    """
    Parse a ``--scorer-weights`` string into the composite-scorer dict.

    Accepts ``relevance,importance,recency,confidence``. Values are
    renormalised to sum to 1.0 (a small drift is tolerated rather than
    rejected). Falls back to the Iteration-1 defaults on malformed
    input.
    """
    keys = ("relevance", "importance", "recency", "confidence")
    try:
        parts = [float(x) for x in spec.split(",")]
        if len(parts) != 4 or any(p < 0 for p in parts):
            raise ValueError
    except ValueError:
        log.warning(
            "bad --scorer-weights %r — using defaults 0.45,0.25,0.20,0.10",
            spec,
        )
        parts = [0.45, 0.25, 0.20, 0.10]
    total = sum(parts) or 1.0
    return {k: v / total for k, v in zip(keys, parts, strict=True)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LongMemEval-S baseline against Ollama (no optimiser).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to longmemeval_s_cleaned.json",
    )
    p.add_argument("--dataset-name", default="longmemeval-s")
    p.add_argument(
        "--provider",
        choices=("ollama", "groq", "nvidia", "gemini", "openai", "lmstudio"),
        default="ollama",
        help="LLM provider for the answerer.",
    )
    p.add_argument(
        "--model", default="",
        help=(
            "Override model name. Defaults: "
            "ollama → llama3.2:latest, "
            "groq → llama-3.3-70b-versatile, "
            "nvidia → meta/llama-3.3-70b-instruct, "
            "gemini → gemini-2.0-flash, "
            "openai → gpt-4o-mini, "
            "lmstudio → qwen/qwen3-14b."
        ),
    )
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument(
        "--lmstudio-url", default=DEFAULT_LMSTUDIO_URL,
        help="Base URL of the local LM Studio server (OpenAI-compatible).",
    )
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--retrieval-mode",
        choices=("topk", "auto", "long"),
        default="topk",
        help=(
            "How the retriever picks context. "
            "topk — always cosine top-K (default, legacy). "
            "long — pass the whole haystack when it fits "
            "--max-context-tokens, else fall back to top-K. "
            "auto — long-context when it fits, top-K otherwise. "
            "long/auto bypass the reranker + optimizer chain."
        ),
    )
    p.add_argument(
        "--scorer-weights", default="0.45,0.25,0.20,0.10",
        help=(
            "Composite-scorer weights as "
            "'relevance,importance,recency,confidence' (Prompt 38). "
            "Default 0.45,0.25,0.20,0.10. The 'retrieval noise' branch "
            "of the tuning decision tree is 0.55,0.20,0.15,0.10."
        ),
    )
    p.add_argument(
        "--tau-hours", type=float, default=168.0,
        help=(
            "Recency half-life in hours for the composite scorer. "
            "Default 168 (7 days); the temporal-failure branch uses 96."
        ),
    )
    p.add_argument(
        "--decompose", action="store_true",
        help=(
            "Enable the DecompositionRetriever — split each question "
            "into atomic sub-questions, retrieve per sub-question, merge "
            "the union. The lever for multi-session / temporal questions. "
            "Adds ~1 cheap LLM call per question."
        ),
    )
    p.add_argument(
        "--decompose-max-items", type=int, default=16,
        help="Cap on the merged decompose-retrieval context. Default 16.",
    )
    p.add_argument(
        "--max-context-tokens", type=int, default=100_000,
        help=(
            "Token ceiling for long-context mode. If the haystack "
            "exceeds this, retrieval falls back to top-K. Default "
            "100000 (fits a 128K-context model with response headroom)."
        ),
    )
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap on rows for the FULL run (smoke is always 5)",
    )
    p.add_argument(
        "--full", action="store_true",
        help="run all questions after the smoke test",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="skip the prompt before the full run",
    )
    p.add_argument(
        "--optimizer", action="store_true",
        help=(
            "Enable the 5-strategy OptimizerChain "
            "(StmTrim+MtmSummarize+SemanticDedupe+LLMLingua+ScorePrune). "
            "Results land in results/optimizer_iteration_1.json."
        ),
    )
    p.add_argument(
        "--iteration", type=int, default=1,
        help="Iteration tag for the optimizer-run JSON filename.",
    )
    p.add_argument(
        "--no-llmlingua", action="store_true",
        help=(
            "Drop LLMLinguaCompress from the chain. Use when the "
            "`llmlingua` package isn't installed, or to isolate which "
            "strategies are doing the work."
        ),
    )
    p.add_argument(
        "--rpm", type=int, default=25,
        help=(
            "Client-side RPM cap for the Groq provider. Free tier is "
            "~30; default 25 stays safely below. Ignored for Ollama."
        ),
    )
    p.add_argument(
        "--rerank", action="store_true",
        help=(
            "Insert a cross-encoder rerank step: retrieve top-K candidates "
            "by cosine (set via --top-k), rerank with a cross-encoder, "
            "keep top-N (set via --rerank-to)."
        ),
    )
    p.add_argument(
        "--rerank-to", type=int, default=4,
        help="Number of items to keep after reranking (default 4).",
    )
    p.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace cross-encoder model id.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(_parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "OllamaLLM",
    "STMSemanticRetriever",
    "load_longmemeval_rows",
    "make_adapter_factory",
    "main",
]
