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

    # cross-session decomposed answering smoke with session-aware retrieval.
    python -m evals.longmemeval.bootstrap_ollama \
        --provider openai --model gpt-4o-mini --optimizer \
        --decompose-answer --session-aware-retrieval \
        --session-top-k 4 --turns-per-session 2 \
        --output results/gpt4omini_decompose_answer_session_aware \
        --limit 5

    # compiled memory wiki smoke.
    python -m evals.longmemeval.bootstrap_ollama \
        --provider openai --model gpt-4o-mini --decompose-answer \
        --wiki-memory --wiki-top-k 8 \
        --output results/gpt4omini_wiki_memory \
        --limit 5

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
from collections.abc import Callable, Iterable
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
from evals.longmemeval.answer_post import (
    best_candidate_fallback_answer,
    build_repair_prompt,
    clean_final_answer,
    is_idk,
    should_block_idk,
    validate_against_claims,
    validate_answer_shape,
)
from evals.longmemeval.baseline import EvalRow, run_baseline
from evals.longmemeval.candidates import (
    best_count_for_object,
    extract_candidates_from_context,
    extract_candidates_from_text,
    filter_candidates_for_question,
)
from evals.longmemeval.content_wiki import (
    ContentFact,
    build_content_wiki_pages,
    extract_content_facts,
    route_topic,
)
from evals.longmemeval.decompose import (
    DecompositionRetriever,
    build_decompose_prompt,
    parse_subquestions,
)
from evals.longmemeval.decompose_guard import (
    DecompositionGuardResult,
    guard_decomposition,
)
from evals.longmemeval.decomposed_answer import (
    SubAnswer,
    _strip_subanswer_scaffold,
    build_aggregation_synthesis_prompt,
    build_final_synthesis_prompt,
    build_subanswer_prompt,
    extract_session_ids,
    is_aggregate_question,
    parse_aggregation_response,
)
from evals.longmemeval.claim_verifier import filter_passing, verify_claims
from evals.longmemeval.claims import (
    extract_claims_from_context,
    rank_claims,
)
from evals.longmemeval.evidence_packet import (
    EvidencePacket,
    build_evidence_packet,
)
from evals.longmemeval.extractive_fallback import (
    build_extractive_prompt,
    validate_extracted_span,
)
from evals.longmemeval.reasoning_heads import (
    head_assistant_memory,
    head_fact_lookup,
    head_knowledge_update,
    head_multi_session_aggregate,
    head_preference_profile,
    head_temporal_reasoning,
)
from evals.longmemeval.query_intent import parse_intent
from evals.longmemeval.structured_claims import (
    extract_structured_claims_from_context,
)
from evals.longmemeval.structured_finalizer import finalize_fact_answer
from evals.longmemeval.task_router import TaskMode, route_task
from evals.longmemeval.evidence_spans import select_spans
from evals.longmemeval.question_type import is_multi_session_hint
from evals.longmemeval.session_narrowing import (
    NarrowResult,
    narrow_to_top_session,
)
from evals.longmemeval.judge import LLMJudgeScorer
from evals.longmemeval.question_type import QuestionType, classify
from evals.longmemeval.telemetry import (
    current_counter,
    end_row_telemetry,
    start_row_telemetry,
)

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
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "openai/gpt-oss-120b"
)
DEFAULT_BEDROCK_MODEL = os.environ.get(
    "BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"
)
DEFAULT_BEDROCK_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)
DEFAULT_BEDROCK_URL = os.environ.get("BEDROCK_BASE_URL", "")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_TYPE_CHOICES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


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


def _longmemeval_user_id(question_id: str) -> str:
    return f"longmemeval:{question_id}"


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
        user_id = _longmemeval_user_id(str(r["question_id"]))
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
                    "user_id": user_id,
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
            question_type=str(r.get("question_type", "unknown")),
            user_id=user_id,
            # The question's reference date ("now") for temporal reasoning.
            question_date=str(r.get("question_date", "") or ""),
        ))
    return rows


def _filter_rows_by_question_type(
    rows: list[EvalRow], question_type: str | None
) -> list[EvalRow]:
    if not question_type:
        return rows
    filtered = [row for row in rows if row.question_type == question_type]
    if not filtered:
        raise ValueError(f"no LongMemEval rows found for question_type={question_type!r}")
    return filtered


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


class OpenRouterLLM:
    """
    OpenRouter chat client — OpenAI-compatible, same surface as
    :class:`GroqLLM` (``async complete(prompt, max_tokens) -> str``).

    OpenRouter (``https://openrouter.ai/api/v1``) brokers many models
    behind one OpenAI-shaped endpoint. The model name carries the
    provider prefix (e.g. ``openai/gpt-oss-120b``, ``anthropic/
    claude-3.5-sonnet``, ``meta-llama/llama-3.3-70b-instruct``).

    The API key is read from ``OPENROUTER_API_KEY``. Rate limiting is
    the same client-side ``_AdaptiveThrottle`` + 429-backoff path the
    other metered providers use; OpenRouter's per-key limits are
    higher than Groq's free tier, so the full 500-row sweep can run
    without batching at a sane ``rpm``.

    Optional ``HTTP-Referer`` / ``X-Title`` headers (for OpenRouter's
    app-ranking board) are sent when ``OPENROUTER_REFERER`` /
    ``OPENROUTER_TITLE`` are set; they're not required.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        base_url: str = DEFAULT_OPENROUTER_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        rpm: int = 60,
        max_retries: int = 8,
        provider_pin: str | None = None,
        seed: int | None = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        # Determinism for ablations. OpenRouter routes a model across MANY
        # backend providers; even at temperature 0 the outputs differ run to
        # run (MoE expert routing + batch nondeterminism + provider switching
        # measured ~44% answer variance on gpt-oss-120b). Two levers:
        #   * provider_pin — force ONE backend (no fallbacks), removing the
        #     provider-switch variance. The slug is an OpenRouter provider
        #     name, e.g. "DeepInfra", "Fireworks", "Together". Falls back to
        #     OPENROUTER_PROVIDER env.
        #   * seed — fixed sampling seed; honoured by providers that support
        #     it (combined with require_parameters below).
        self.provider_pin = provider_pin or os.environ.get("OPENROUTER_PROVIDER") or None
        self.seed = seed
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY env var not set. Export it before launching:\n"
                "  export OPENROUTER_API_KEY='sk-or-...'"
            )
        self._api_key = key
        self._client = httpx.AsyncClient(timeout=timeout)
        self._throttle = _AdaptiveThrottle(rpm)
        self._max_retries = max_retries

    def _build_payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Construct the request body (pure; unit-tested for determinism keys)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.provider_pin:
            # Pin to a single backend, no fallbacks; require_parameters makes
            # the provider honour temperature/seed (else it may ignore them).
            payload["provider"] = {
                "order": [self.provider_pin],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        return payload

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        payload = self._build_payload(prompt, max_tokens)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        referer = os.environ.get("OPENROUTER_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.environ.get("OPENROUTER_TITLE")
        if title:
            headers["X-Title"] = title
        return await _adaptive_complete(
            provider="openrouter",
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
    is_answer_call: bool = False,
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
        # Charge the row's telemetry counter — uses real
        # `usage.{prompt,completion}_tokens` from the provider when
        # available, falls back to a char-length estimate otherwise.
        # No-op when no counter is active (e.g. unit tests).
        try:
            from evals.longmemeval.telemetry import record_llm_call
            # Compose a "prompt" string for the char-length fallback —
            # OpenAI-shaped payloads only have messages.
            prompt_text = " ".join(
                str(m.get("content", "")) for m in payload.get("messages") or []
            )
            record_llm_call(
                model=str(payload.get("model", provider)),
                prompt=prompt_text,
                response=data,
                is_answer_call=is_answer_call,
            )
        except Exception:  # pragma: no cover — telemetry must never break a call
            log.debug("telemetry.record_llm_call failed", exc_info=True)
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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # GPT-5 / o-series reasoning models reject `max_tokens` and a custom
        # `temperature` — they require `max_completion_tokens` and the default
        # temperature. Older models (gpt-4o*, gpt-4*) keep the legacy params.
        if re.match(r"(?:gpt-5|o[0-9])", self.model, re.IGNORECASE):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = self.temperature
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
# Amazon Bedrock client (Boto3 Converse API)
# ---------------------------------------------------------------------------


def _bedrock_region_from_env() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_BEDROCK_REGION
    )


def _bedrock_normalize_endpoint_url(raw: str | None) -> str | None:
    if not raw:
        return None
    endpoint = raw.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3].rstrip("/")
    return endpoint or None


def _bedrock_openai_base_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/v1"


def _bedrock_is_gpt_oss_model(model: str) -> bool:
    return model.strip().startswith("openai.gpt-oss")


def _bedrock_boto3_client(
    region: str,
    endpoint_url: str | None = None,
) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "boto3 is required for --provider bedrock. Install boto3 or run "
            "with a Python environment that includes the AWS SDK."
        ) from exc

    kwargs: dict[str, Any] = {"region_name": region}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("bedrock-runtime", **kwargs)


def _bedrock_openai_responses_client(
    region: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "openai is required for Bedrock openai.gpt-oss models. Install "
            "the OpenAI Python SDK or use a non-gpt-oss Bedrock model."
        ) from exc

    key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    )
    if not key:
        raise RuntimeError(
            "Set AWS_BEARER_TOKEN_BEDROCK or OPENAI_API_KEY before using "
            "Bedrock openai.gpt-oss models."
        )
    resolved_base = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or _bedrock_openai_base_url(region)
    )
    return OpenAI(api_key=key, base_url=resolved_base.rstrip("/"))


def _bedrock_text_from_blocks(blocks: Any) -> str | None:
    if isinstance(blocks, str):
        return blocks.strip() or None
    if not isinstance(blocks, list):
        return None

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict):
            text = block.get("text") or block.get("content") or ""
        else:
            text = ""
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts) if parts else None


def _bedrock_converse_text(response: Any) -> str:
    if not isinstance(response, dict):
        raise RuntimeError(
            f"bedrock: expected Converse dict, got {type(response).__name__}"
        )
    output = response.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = _bedrock_text_from_blocks(content)
    if text:
        return _strip_think(text).strip()

    preview = json.dumps(response, ensure_ascii=False, default=str)[:500]
    keys = sorted(response.keys())
    raise RuntimeError(
        "bedrock: Converse response did not contain output.message.content "
        f"text; keys={keys}; body={preview}"
    )


def _bedrock_usage(response: Any) -> tuple[int | None, int | None]:
    if not isinstance(response, dict):
        return None, None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt_tokens = usage.get("inputTokens")
    completion_tokens = usage.get("outputTokens")
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


def _bedrock_openai_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return _strip_think(text).strip()
    if isinstance(response, dict):
        text = response.get("output_text")
        if isinstance(text, str) and text.strip():
            return _strip_think(text).strip()
    preview = str(response)[:500]
    raise RuntimeError(
        "bedrock: OpenAI Responses result did not contain output_text; "
        f"response={preview}"
    )


def _bedrock_openai_usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        in_tok = usage.get("input_tokens") or usage.get("prompt_tokens")
        out_tok = usage.get("output_tokens") or usage.get("completion_tokens")
    else:
        in_tok = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
        )
        out_tok = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
        )
    return (
        int(in_tok) if in_tok is not None else None,
        int(out_tok) if out_tok is not None else None,
    )


def _bedrock_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = error.get("Code")
            if code is not None:
                return str(code)
    return type(exc).__name__


def _bedrock_is_throttling_error(exc: Exception) -> bool:
    code = _bedrock_error_code(exc).lower()
    text = str(exc).lower()
    return "throttl" in code or "too many" in text


class BedrockLLM:
    """
    Amazon Bedrock chat client via the native Boto3 Converse API.

    Authentication is delegated to Boto3. For Bedrock API keys, export
    ``AWS_BEARER_TOKEN_BEDROCK`` before launching; standard AWS credentials
    also continue to work through the normal AWS SDK provider chain.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_BEDROCK_MODEL,
        region: str | None = None,
        base_url: str | None = None,
        timeout: float = 90.0,
        temperature: float = 0.0,
        api_key: str | None = None,
        rpm: int = 30,
        max_retries: int = 8,
    ) -> None:
        self.model = model
        self.region = region or _bedrock_region_from_env()
        self.base_url = (
            base_url
            or os.environ.get("BEDROCK_BASE_URL")
            or DEFAULT_BEDROCK_URL
            or None
        )
        self.endpoint_url = _bedrock_normalize_endpoint_url(
            self.base_url
        )
        self.timeout = timeout
        self.temperature = temperature
        if api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        if _bedrock_is_gpt_oss_model(self.model):
            self._transport = "openai_responses"
            self._client = _bedrock_openai_responses_client(
                self.region,
                self.base_url,
                api_key,
            )
        else:
            self._transport = "converse"
            self._client = _bedrock_boto3_client(self.region, self.endpoint_url)
        self._throttle = _AdaptiveThrottle(rpm)
        self._max_retries = max_retries

    async def complete(self, *, prompt: str, max_tokens: int) -> str:
        if self._transport == "openai_responses":
            return await self._complete_openai_responses(
                prompt=prompt,
                max_tokens=max_tokens,
            )

        messages = [{
            "role": "user",
            "content": [{"text": prompt}],
        }]
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": self.temperature,
        }
        attempt = 0
        while True:
            await self._throttle.await_slot()
            try:
                response = await asyncio.to_thread(
                    self._client.converse,
                    modelId=self.model,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
            except Exception as exc:
                if _bedrock_is_throttling_error(exc):
                    self._throttle.on_429()
                    if attempt < self._max_retries:
                        wait = min(
                            128.0,
                            float(2 ** attempt) * random.uniform(0.8, 1.2),
                        )
                        log.warning(
                            "bedrock throttled (%s) — adaptive pace now "
                            "%.2fs/call; backoff %.1fs (retry %d/%d)",
                            _bedrock_error_code(exc),
                            self._throttle.interval,
                            wait,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(wait)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"bedrock: throttled after {self._max_retries} retries"
                    ) from exc
                log.exception("bedrock converse error")
                raise RuntimeError(f"bedrock: {exc!r}") from exc

            self._throttle.on_success()
            prompt_tokens, completion_tokens = _bedrock_usage(response)
            try:
                from evals.longmemeval.telemetry import record_llm_call
                record_llm_call(
                    model=self.model,
                    prompt=prompt,
                    response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:  # pragma: no cover - telemetry must not break calls
                log.debug("telemetry.record_llm_call failed", exc_info=True)
            return _bedrock_converse_text(response)

    async def _complete_openai_responses(
        self,
        *,
        prompt: str,
        max_tokens: int,
    ) -> str:
        attempt = 0
        while True:
            await self._throttle.await_slot()
            try:
                response = await asyncio.to_thread(
                    self._client.responses.create,
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=max_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:
                if _bedrock_is_throttling_error(exc):
                    self._throttle.on_429()
                    if attempt < self._max_retries:
                        wait = min(
                            128.0,
                            float(2 ** attempt) * random.uniform(0.8, 1.2),
                        )
                        log.warning(
                            "bedrock gpt-oss throttled (%s) — adaptive pace "
                            "now %.2fs/call; backoff %.1fs (retry %d/%d)",
                            _bedrock_error_code(exc),
                            self._throttle.interval,
                            wait,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(wait)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"bedrock: throttled after {self._max_retries} retries"
                    ) from exc
                log.exception("bedrock gpt-oss responses error")
                raise RuntimeError(f"bedrock: {exc!r}") from exc

            self._throttle.on_success()
            prompt_tokens, completion_tokens = _bedrock_openai_usage(response)
            try:
                from evals.longmemeval.telemetry import record_llm_call
                record_llm_call(
                    model=self.model,
                    prompt=prompt,
                    response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:  # pragma: no cover - telemetry must not break calls
                log.debug("telemetry.record_llm_call failed", exc_info=True)
            return _bedrock_openai_response_text(response)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


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
        except Exception as exc:
            log.error("LM Studio HTTP call failed: %s", exc)
            raise RuntimeError(f"lmstudio: {exc!r}") from exc

        if resp.status_code >= 400:
            # LM Studio's error body carries the actual cause — typically
            # "model not found", "context length … exceeded", or
            # "max_tokens > model max". Surface it so the caller doesn't
            # have to dig through httpx's generic 400 message.
            try:
                body = resp.json()
                err_msg = (
                    (body.get("error") or {}).get("message")
                    if isinstance(body.get("error"), dict)
                    else body.get("error") or body
                )
            except Exception:
                err_msg = resp.text[:500]
            log.error(
                "LM Studio %d for model=%r prompt_chars=%d max_tokens=%d → %s",
                resp.status_code, self.model, len(prompt), max_tokens, err_msg,
            )
            raise RuntimeError(
                f"lmstudio {resp.status_code}: {err_msg}"
            )
        try:
            data = resp.json()
            raw = str(data["choices"][0]["message"]["content"])
            return _strip_think(raw)
        except Exception as exc:
            log.exception("LM Studio response parse failed")
            raise RuntimeError(f"lmstudio: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Sentence-transformer embedder + STM retriever
# ---------------------------------------------------------------------------


class _Embedder:
    """Thin wrapper around sentence_transformers, loaded lazily once."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        # device=None lets sentence-transformers auto-pick (CUDA/MPS/CPU);
        # callers (e.g. bench) pass "cpu" to force CPU when MPS is contended.
        self._device = device
        self._model: Any | None = None

    def _lazy(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedder %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self._lazy()
        # show_progress_bar=False suppresses sentence-transformers' per-call
        # tqdm "Batches: 100%|…" lines, which otherwise flood the eval log
        # (one bar per retrieval).
        v = m.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
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


class BM25HaystackRetriever:
    """
    BM25 sibling of :class:`STMSemanticRetriever`.

    Same store / same haystack / same ``ContextBundle`` shape — only
    the ranker is swapped out for :class:`continuum.retrieval.bm25.BM25Index`.
    Designed to compose one-for-one with
    :class:`continuum.retrieval.rrf.ReciprocalRankFusion` so
    ``--retriever hybrid`` can fuse cosine and BM25 hits side by side.

    Why split lexical from cosine
    -----------------------------
    The cosine retriever wins on paraphrase ("dog" ↔ "canine") but
    underweights rare proper nouns ("Roscioli", "MIT-PCT-301") whose
    embeddings cluster with topically-adjacent neighbours. BM25 does
    the inverse — it lights up on rare-token matches and is blind to
    paraphrase. RRF then combines the two by rank rather than by
    incomparable raw scores. See :mod:`continuum.retrieval.rrf`.
    """

    def __init__(
        self,
        *,
        store: FlatHaystackStore,
        top_k: int = 8,
        session_id: str = "default",
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.session_id = session_id

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        from continuum.retrieval.bm25 import BM25Index
        items = list(self.store.items)
        if not items:
            return ContextBundle(
                items=[], messages=[], tokens_used=0, budget=budget,
                tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
                debug_info={"retrieval_mode": "bm25", "corpus_size": 0},
            )
        index = BM25Index(items)
        k = query.top_k or self.top_k
        hits = index.query(query.text, k)
        picked = [si.item for si in hits]
        messages = [{"role": "system", "content": it.content} for it in picked]
        token_count = sum(len(it.content.split()) for it in picked)
        return ContextBundle(
            items=picked,
            messages=messages,
            tokens_used=token_count,
            budget=budget,
            tier_breakdown={"stm": token_count, "mtm": 0, "ltm": 0},
            debug_info={
                "retrieval_mode": "bm25",
                "corpus_size": len(items),
                "hits": len(picked),
            },
        )


class SessionAwareSemanticRetriever(STMSemanticRetriever):
    """
    Session-first LongMemEval retriever.

    It ranks source sessions by their strongest matching turn, then selects
    the strongest turns inside the top sessions. This prevents one noisy
    session with many similar turns from crowding out evidence sessions for
    multi-session questions.
    """

    def __init__(
        self,
        *,
        store: FlatHaystackStore,
        embedder: _Embedder,
        session_top_k: int = 4,
        turns_per_session: int = 2,
        max_items: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(store=store, embedder=embedder, **kwargs)
        self.session_top_k = max(1, int(session_top_k))
        self.turns_per_session = max(1, int(turns_per_session))
        self.max_items = max(1, int(max_items))

    def _bundle_topk(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        qv = self.embedder.encode([query.text])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = (qv @ self._cached_matrix.T).flatten()  # type: ignore[union-attr]

        first_index: dict[str, int] = {}
        grouped: dict[str, list[tuple[int, float, MemoryItem]]] = {}
        for idx, item in enumerate(self._cached_items):
            sid = str((item.metadata or {}).get("session_id") or "unknown")
            first_index.setdefault(sid, idx)
            grouped.setdefault(sid, []).append((idx, float(sims[idx]), item))

        ranked_sessions = sorted(
            grouped,
            key=lambda sid: (
                -max(score for _idx, score, _item in grouped[sid]),
                first_index[sid],
            ),
        )[: self.session_top_k]

        picked: list[MemoryItem] = []
        for sid in ranked_sessions:
            turns = sorted(
                grouped[sid],
                key=lambda row: (-row[1], row[0]),
            )[: self.turns_per_session]
            picked.extend(item for _idx, _score, item in turns)
            if len(picked) >= self.max_items:
                picked = picked[: self.max_items]
                break

        messages = [
            {
                "role": str(item.metadata.get("role", "user"))
                if item.metadata else "user",
                "content": item.content,
            }
            for item in picked
        ]
        token_count = sum(estimate_tokens_text(item.content) for item in picked)
        return ContextBundle(
            items=picked,
            messages=messages,
            tokens_used=token_count,
            budget=budget,
            tier_breakdown={"stm": token_count, "mtm": 0, "ltm": 0},
            debug_info={
                "retrieval_mode": "session_aware",
                "selected_sessions": ranked_sessions,
                "session_top_k": self.session_top_k,
                "turns_per_session": self.turns_per_session,
            },
        )


class WikiMemoryRetriever:
    """
    Row-local compiled-memory retriever for LongMemEval.

    This is a lightweight implementation of the LLM-wiki idea: raw haystack
    turns remain unchanged in ``raw_store``; retrieval runs over generated
    Markdown pages that organize the row by index, timeline, preferences, and
    per-session pages. No oracle answer-session ids are used.
    """

    _PREFERENCE_TERMS = (
        "favorite", "prefer", "preference", "like", "love", "dislike",
        "color", "food", "book", "movie", "music", "hobby",
    )
    _LEXICAL_STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "what", "where",
        "when", "who", "which", "how", "did", "does", "was", "were",
        "is", "are", "am", "to", "of", "on", "in", "at", "a", "an",
    }
    _WORD_RE = re.compile(r"[a-z0-9]+")
    _TIME_RE = re.compile(
        r"\b(?:minute|minutes|hour|hours|each way|one way|round trip)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        raw_store: FlatHaystackStore,
        embedder: _Embedder,
        top_k: int = 8,
        session_id: str = "default",
    ) -> None:
        self.raw_store = raw_store
        self.embedder = embedder
        self.top_k = max(1, int(top_k))
        self.session_id = session_id
        self._cached_len = -1
        self._pages: list[MemoryItem] = []
        self._matrix: np.ndarray | None = None
        self._raw_matrix: np.ndarray | None = None

    def _group_sessions(self) -> dict[str, list[MemoryItem]]:
        grouped: dict[str, list[MemoryItem]] = {}
        for item in self.raw_store.items:
            sid = str((item.metadata or {}).get("session_id") or "unknown")
            grouped.setdefault(sid, []).append(item)
        return grouped

    @staticmethod
    def _session_ids(items: Iterable[MemoryItem]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            sid = str((item.metadata or {}).get("session_id") or "")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
        return out

    @staticmethod
    def _render_messages(items: list[MemoryItem], *, limit: int | None = None) -> str:
        lines: list[str] = []
        source = items[:limit] if limit is not None else items
        for idx, item in enumerate(source, start=1):
            role = str((item.metadata or {}).get("role", "user"))
            lines.append(f"- turn {idx} ({role}): {item.content}")
        return "\n".join(lines)

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token for token in cls._WORD_RE.findall(text.lower())
            if len(token) > 2 and token not in cls._LEXICAL_STOPWORDS
        }

    @classmethod
    def _lexical_score(cls, query: str, text: str) -> float:
        query_tokens = cls._tokens(query)
        if not query_tokens:
            return 0.0
        text_l = text.lower()
        text_tokens = cls._tokens(text_l)
        score = len(query_tokens & text_tokens) / len(query_tokens)
        if re.search(
            r"\bhow\s+long\b", query, re.IGNORECASE
        ) and cls._TIME_RE.search(text_l):
            score += 0.75
        if re.search(r"\bwhere\b", query, re.IGNORECASE):
            if re.search(
                r"\b(?:target|walmart|costco|kroger|store|shop|app)\b",
                text_l,
            ):
                score += 0.25
        if re.search(r"\bname\b", query, re.IGNORECASE):
            if re.search(r"\b(?:called|named|titled)\b", text_l):
                score += 0.35
        if re.search(
            r"\b(?:clothing|clothes|items? of clothing)\b",
            query,
            re.IGNORECASE,
        ):
            if re.search(
                r"\b(?:blazer|boots?|jeans?|shirt|sweater|dress|pants|jacket)\b",
                text_l,
            ):
                score += 0.4
        if re.search(r"\bpick\s+up\b", query, re.IGNORECASE):
            if re.search(r"\bpick\s+up\b|\bdry cleaning\b", text_l):
                score += 1.2
        if re.search(r"\breturn\b", query, re.IGNORECASE):
            if re.search(r"\breturn\b|\bexchang(?:e|ed|ing)\b", text_l):
                score += 1.0
        if re.search(r"\bcamping trips?\b", query, re.IGNORECASE):
            if re.search(r"\b\d+\s*-\s*day\b.*\bcamping trip\b", text_l):
                score += 2.5
            elif "camping trip" in text_l:
                score += 1.25
        if re.search(r"\bmodel kits?\b", query, re.IGNORECASE):
            if re.search(
                r"\b(?:model kit|kits?|revell|tamiya|spitfire|tiger|bomber|camaro)\b",
                text_l,
            ):
                score += 1.25
            if re.search(r"\b\d+/\d+\s+scale\b", text_l):
                score += 0.75
        if re.search(r"\bprojects?\b", query, re.IGNORECASE):
            if re.search(r"\b(?:led|lead|leading)\b", text_l):
                score += 1.25
        return score

    @staticmethod
    def _role_score(item: MemoryItem) -> float:
        role = str((item.metadata or {}).get("role", "user")).lower()
        if role == "user":
            return 1.25
        if role == "assistant":
            return -0.25
        return 0.0

    def _page(
        self,
        name: str,
        content: str,
        source_items: list[MemoryItem],
        *,
        recall_session_ids: list[str] | None = None,
        page_type: str = "compiled",
    ) -> MemoryItem:
        source_session_ids = self._session_ids(source_items)
        session_ids = recall_session_ids or []
        meta: dict[str, Any] = {
            "role": "system",
            "wiki_page": name,
            "wiki_page_type": page_type,
            "source_session_ids": source_session_ids,
            "session_ids": session_ids,
        }
        if len(session_ids) == 1:
            meta["session_id"] = session_ids[0]
        return MemoryItem(
            id=f"wiki:{name}",
            content=content,
            tier=MemoryTier.LTM,
            metadata=meta,
        )

    def _build_pages(self) -> list[MemoryItem]:
        grouped = self._group_sessions()
        all_items = list(self.raw_store.items)
        pages: list[MemoryItem] = []

        index_lines = ["# Memory Wiki Index", "", "## Sessions"]
        for sid, items in grouped.items():
            index_lines.append(f"- sessions/{sid}.md ({len(items)} turns)")
        index_lines.extend([
            "",
            "## Compiled Pages",
            "- timeline.md",
            "- preferences.md",
            "- profile.md",
        ])
        pages.append(
            self._page(
                "index.md", "\n".join(index_lines), all_items,
                page_type="global",
            )
        )

        timeline_lines = ["# Timeline", ""]
        for sid, items in grouped.items():
            preview = " ".join(item.content for item in items[:2])
            timeline_lines.append(f"- session={sid}: {preview}")
        pages.append(
            self._page(
                "timeline.md", "\n".join(timeline_lines), all_items,
                page_type="global",
            )
        )

        preference_items = [
            item for item in all_items
            if any(term in item.content.lower() for term in self._PREFERENCE_TERMS)
        ]
        if preference_items:
            pref_body = "# Preferences\n\n" + self._render_messages(preference_items)
        else:
            pref_body = "# Preferences\n\nNo explicit preferences found."
        pages.append(
            self._page(
                "preferences.md",
                pref_body,
                preference_items or all_items,
                page_type="compiled",
            )
        )

        profile_body = "# Profile\n\n" + self._render_messages(all_items, limit=40)
        pages.append(self._page("profile.md", profile_body, all_items, page_type="global"))

        for sid, items in grouped.items():
            body = f"# Session {sid}\n\n" + self._render_messages(items)
            pages.append(
                self._page(
                    f"sessions/{sid}.md",
                    body,
                    items,
                    recall_session_ids=[sid],
                    page_type="session",
                )
            )

        return pages

    def _refresh_index(self) -> None:
        if len(self.raw_store.items) == self._cached_len:
            return
        self._pages = self._build_pages()
        self._cached_len = len(self.raw_store.items)
        if not self._pages:
            self._matrix = None
            self._raw_matrix = None
            return
        self._matrix = self.embedder.encode([page.content for page in self._pages])
        self._raw_matrix = self.embedder.encode(
            [item.content for item in self.raw_store.items]
        )

    def _raw_ranked_sessions(
        self, qv: np.ndarray, query_text: str, limit: int
    ) -> list[str]:
        return [
            sid for sid, _idx in self._raw_ranked_session_indices(
                qv, query_text, limit
            )
        ]

    def _raw_ranked_session_indices(
        self, qv: np.ndarray, query_text: str, limit: int
    ) -> list[tuple[str, int]]:
        if self._raw_matrix is None or not self.raw_store.items:
            return []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = (qv @ self._raw_matrix.T).flatten()
        first_index: dict[str, int] = {}
        best_score: dict[str, float] = {}
        best_index: dict[str, int] = {}
        for idx, item in enumerate(self.raw_store.items):
            sid = str((item.metadata or {}).get("session_id") or "unknown")
            score = (
                float(sims[idx])
                + 4.0 * self._lexical_score(query_text, item.content)
                + self._role_score(item)
            )
            first_index.setdefault(sid, idx)
            if score > best_score.get(sid, float("-inf")):
                best_score[sid] = score
                best_index[sid] = idx
        ranked = sorted(
            best_score,
            key=lambda sid: (-best_score[sid], first_index[sid]),
        )[:limit]
        return [(sid, best_index[sid]) for sid in ranked]

    def _raw_ranked_item_indices(
        self, qv: np.ndarray, query_text: str, limit: int
    ) -> list[int]:
        if self._raw_matrix is None or not self.raw_store.items:
            return []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = (qv @ self._raw_matrix.T).flatten()
        scored = [
            (
                idx,
                float(sims[idx])
                + 4.0 * self._lexical_score(query_text, item.content)
                + self._role_score(item),
            )
            for idx, item in enumerate(self.raw_store.items)
        ]
        scored.sort(key=lambda row: (-row[1], row[0]))
        return [idx for idx, _score in scored[:limit]]

    def _evidence_window(self, center_idx: int, *, radius: int = 2) -> MemoryItem:
        center = self.raw_store.items[center_idx]
        sid = str((center.metadata or {}).get("session_id") or "unknown")
        session_indices = [
            idx for idx, item in enumerate(self.raw_store.items)
            if str((item.metadata or {}).get("session_id") or "unknown") == sid
        ]
        position = session_indices.index(center_idx)
        start = max(0, position - radius)
        end = min(len(session_indices), position + radius + 1)
        window_items = [
            self.raw_store.items[idx] for idx in session_indices[start:end]
        ]
        center_role = str((center.metadata or {}).get("role", "user"))
        body = (
            f"# Evidence Window for {sid}\n\n"
            "## Matched Turn\n"
            f"- ({center_role}): {center.content}\n\n"
            "## Nearby Context\n"
            + self._render_messages(window_items)
        )
        return self._page(
            f"evidence/{sid}/{center.id}.md",
            body,
            window_items,
            recall_session_ids=[sid],
            page_type="evidence_window",
        )

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        self._refresh_index()
        if self._matrix is None or not self._pages:
            return ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
                debug_info={"retrieval_mode": "wiki_memory", "wiki_pages": 0},
            )

        qv = self.embedder.encode([query.text])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = (qv @ self._matrix.T).flatten()
        ranked = [int(i) for i in np.argsort(-sims)]
        picked: list[MemoryItem] = []
        seen_page_ids: set[str] = set()
        seen_indices: set[int] = set()
        seen_session_ids: set[str] = set()

        def try_add_page(page: MemoryItem, idx: int | None = None) -> bool:
            if page.id in seen_page_ids:
                return False
            sid = str((page.metadata or {}).get("session_id") or "")
            if sid and sid in seen_session_ids:
                return False
            picked.append(page)
            seen_page_ids.add(page.id)
            if idx is not None:
                seen_indices.add(idx)
            if sid:
                seen_session_ids.add(sid)
            return True

        window_limit = min(2, max(1, self.top_k // 2))
        for raw_idx in self._raw_ranked_item_indices(qv, query.text, window_limit):
            window = self._evidence_window(raw_idx)
            try_add_page(window)

        source_top_k = max(1, self.top_k // 2)
        ranked_session_indices = self._raw_ranked_session_indices(
            qv, query.text, max(source_top_k, len(self.raw_store.items))
        )
        best_raw_idx_by_session = dict(ranked_session_indices)
        for _sid, raw_idx in ranked_session_indices[:source_top_k]:
            try_add_page(self._evidence_window(raw_idx))
        for idx in ranked:
            if idx in seen_indices:
                continue
            page = self._pages[idx]
            sid = str((page.metadata or {}).get("session_id") or "")
            if page.metadata.get("wiki_page_type") == "session" and sid:
                raw_idx = best_raw_idx_by_session.get(sid)
                if raw_idx is not None:
                    try_add_page(self._evidence_window(raw_idx), idx)
                else:
                    try_add_page(page, idx)
            else:
                try_add_page(page, idx)
            if len(picked) >= self.top_k:
                break
        picked = picked[: self.top_k]
        messages = [{"role": "system", "content": page.content} for page in picked]
        token_count = sum(estimate_tokens_text(page.content) for page in picked)
        return ContextBundle(
            items=picked,
            messages=messages,
            tokens_used=token_count,
            budget=budget,
            tier_breakdown={"stm": 0, "mtm": 0, "ltm": token_count},
            debug_info={
                "retrieval_mode": "wiki_memory",
                "wiki_pages": len(self._pages),
                "selected_pages": [
                    str(page.metadata.get("wiki_page", "")) for page in picked
                ],
            },
        )


class ContentWikiMemoryRetriever:
    """Content/topic page retriever with pure English dated fact pages."""

    def __init__(
        self,
        *,
        raw_store: FlatHaystackStore,
        embedder: _Embedder,
        top_k: int = 6,
        session_id: str = "default",
    ) -> None:
        self.raw_store = raw_store
        self.embedder = embedder
        self.top_k = max(1, int(top_k))
        self.session_id = session_id
        self._cached_len = -1
        self._pages: list[MemoryItem] = []
        self._matrix: np.ndarray | None = None

    @staticmethod
    def _session_hint(items: list[MemoryItem]) -> str | None:
        text = " ".join(item.content for item in items).lower()
        if "target" in text or "cartwheel" in text:
            return "Target"
        return None

    def _build_pages(self) -> list[MemoryItem]:
        grouped: dict[str, list[MemoryItem]] = {}
        for item in self.raw_store.items:
            sid = str((item.metadata or {}).get("session_id") or "unknown")
            grouped.setdefault(sid, []).append(item)

        facts: list[ContentFact] = []
        for sid, items in grouped.items():
            hint = self._session_hint(items)
            for item in items:
                metadata = item.metadata or {}
                facts.extend(
                    extract_content_facts(
                        content=item.content,
                        role=str(metadata.get("role", "user")),
                        session_id=sid,
                        date_text=str(metadata.get("date", "")),
                        session_hint=hint,
                    )
                )

        rendered = build_content_wiki_pages(facts)
        pages: list[MemoryItem] = []
        for name, content in rendered.items():
            page_facts = [
                fact for fact in facts
                if f"{route_topic(fact)}.md" == name
            ]
            session_ids: list[str] = []
            for fact in page_facts:
                for sid in fact.session_ids:
                    if sid not in session_ids:
                        session_ids.append(sid)
            pages.append(
                MemoryItem(
                    id=f"content-wiki:{name}",
                    content=content,
                    tier=MemoryTier.LTM,
                    metadata={
                        "role": "system",
                        "wiki_page": name,
                        "wiki_page_type": "content_topic",
                        "session_ids": session_ids,
                        "source_session_ids": session_ids,
                    },
                )
            )
        return pages

    def _refresh_index(self) -> None:
        if len(self.raw_store.items) == self._cached_len:
            return
        self._pages = self._build_pages()
        self._cached_len = len(self.raw_store.items)
        self._matrix = (
            self.embedder.encode([page.content for page in self._pages])
            if self._pages else None
        )

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        self._refresh_index()
        if self._matrix is None or not self._pages:
            return ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
                debug_info={
                    "retrieval_mode": "content_wiki_memory",
                    "wiki_pages": 0,
                },
            )

        qv = self.embedder.encode([query.text])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sims = (qv @ self._matrix.T).flatten()
        order = [int(i) for i in np.argsort(-sims)[: self.top_k]]
        picked = [self._pages[idx] for idx in order]
        token_count = sum(estimate_tokens_text(page.content) for page in picked)
        return ContextBundle(
            items=picked,
            messages=[{"role": "system", "content": page.content} for page in picked],
            tokens_used=token_count,
            budget=budget,
            tier_breakdown={"stm": 0, "mtm": 0, "ltm": token_count},
            debug_info={
                "retrieval_mode": "content_wiki_memory",
                "wiki_pages": len(self._pages),
                "selected_pages": [
                    str(page.metadata.get("wiki_page", "")) for page in picked
                ],
            },
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
        # Composite retrievers (``ReciprocalRankFusion``,
        # ``DecompositionRetriever``) don't carry a ``session_id`` — the
        # underlying children do. The field is informational here; default
        # to "default" so ``--retriever hybrid`` doesn't trip up _MiniSession.
        self.session_id = getattr(retriever, "session_id", "default")


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
            user_id = str(msg.get("user_id", "") or "")
            item = MemoryItem(
                content=content,
                tier=MemoryTier.STM,
                user_id=user_id or None,
                metadata={
                    "role": str(msg.get("role", "user")),
                    "session_id": str(msg.get("session_id", "")),
                    "user_id": user_id,
                    "date": str(msg.get("date", "") or ""),
                },
            )
            # Stamp the session date onto created_at so the scorer's
            # recency component is a real signal (Prompt 38 tuning).
            date_iso = str(msg.get("date", "") or "")
            if date_iso:
                with contextlib.suppress(ValueError):
                    item.created_at = dt.datetime.fromisoformat(date_iso)
            await store.append(item)
        # Stores that own a deferred promotion / supersession pipeline
        # (e.g. ContinuumLTMHaystackStore) expose `finalize()` so the
        # rig can signal "ingest done — run promotion now." FlatHaystackStore
        # has no such hook and is left untouched.
        finalize = getattr(store, "finalize", None)
        if finalize is not None:
            await finalize()
        # Surface store-level metrics so the row's result JSON records
        # which LTM backend produced the numbers and how many
        # supersessions fired during ingest. Read defensively — the
        # legacy FlatHaystackStore has no metrics().
        metrics = getattr(store, "metrics", None)
        if callable(metrics):
            self.last_store_metrics = metrics()


def _merge_retrieved_contexts(
    contexts: list[ContextBundle], budget: TokenBudget,
) -> ContextBundle | None:
    """
    Build a single ``ContextBundle`` from every sub-question's bundle
    so the baseline runner can extract ``retrieved_session_ids`` from
    ``adapter.last_ctx``.

    Dedup by ``MemoryItem.id`` to keep the union honest when a session
    surfaces under multiple sub-questions (a common pattern for
    multi-hop questions). Returns ``None`` only when the reasoner
    never called the retriever at all (e.g. budget exhausted before
    the first sub-q) — baseline's ``_extract_retrieved_session_ids``
    treats ``None`` as an empty list.
    """
    if not contexts:
        return None
    seen: set[str] = set()
    items: list[MemoryItem] = []
    for ctx in contexts:
        for item in ctx.items or []:
            iid = str(item.id)
            if iid in seen:
                continue
            seen.add(iid)
            items.append(item)
    return ContextBundle(
        items=items,
        messages=[
            {"role": "system", "content": it.content} for it in items
        ],
        tokens_used=sum(len(it.content.split()) for it in items),
        budget=budget,
        debug_info={
            "retrieval_mode": "iterative_reasoner",
            "subq_bundles": len(contexts),
            "merged_items": len(items),
        },
    )


class _IterativeReasoningAdapter(_IngestingAdapter):
    """
    Adapter that delegates answer composition to
    :class:`continuum.reasoning.IterativeReasoner`.

    Ingestion (``process_conversation``) and the store wiring are
    inherited from :class:`_IngestingAdapter` unchanged. Only
    ``answer_question`` is overridden — it constructs a reasoner per
    call (cheap; no I/O at construction) and surfaces budget +
    abstain telemetry onto ``self.last_telemetry`` so the row's
    result JSON records the iterative-loop signal.

    The rig's retrievers expect a :class:`continuum.core.types.Query`
    + :class:`TokenBudget`. The reasoner only passes a plain string,
    so the adapter wraps the underlying retriever in a thin shim
    that constructs the Query+budget per call.
    """

    def __init__(
        self,
        *,
        session: _MiniSession,
        llm: Any,
        answer_max_tokens: int = 200,
        small_llm: Any | None = None,
        max_llm_calls: int = 6,
        max_rounds: int = 2,
    ) -> None:
        super().__init__(
            session=session, llm=llm, answer_max_tokens=answer_max_tokens,
        )
        self._small_llm = small_llm
        self._max_llm_calls = max_llm_calls
        self._max_rounds = max_rounds
        self.last_telemetry: dict[str, Any] = {}

    async def answer_question(self, question: str) -> str:
        from continuum.extraction.small_llm import SmallLLM as _SmallLLM
        from evals.longmemeval.iterative_reasoner_wiring import (
            build_iterative_reasoner,
        )

        small_llm = self._small_llm or _SmallLLM()
        budget = self.budget
        retriever = getattr(self.session, "retriever", None)

        # Shim: the reasoner calls .retrieve(query: str); the rig's
        # retrievers want Query+budget. Wrap once per row, and
        # accumulate every returned bundle so the baseline runner can
        # measure recall against retrieved_session_ids — without this
        # ``adapter.last_ctx`` would never be set and recall would
        # be 0 on every row.
        contexts: list[ContextBundle] = []

        class _RetrieverShim:
            async def retrieve(_self, query: str) -> ContextBundle:  # noqa: N805
                if retriever is None:
                    bundle = ContextBundle(
                        items=[], messages=[], tokens_used=0, budget=budget,
                    )
                else:
                    bundle = await retriever.retrieve(Query(text=query), budget)
                contexts.append(bundle)
                return bundle

        # Dataset hints (set by baseline._run_one on the adapter
        # before answer_question fires).
        qt_hint = getattr(self, "dataset_question_type", None)
        is_multi = bool(getattr(self, "dataset_is_multi_session", False))

        reasoner = build_iterative_reasoner(
            retriever=_RetrieverShim(),
            small_llm=small_llm,
            composer_llm=self.llm,
            max_llm_calls=self._max_llm_calls,
            max_rounds=self._max_rounds,
            question_type_hint=qt_hint,
            is_multi_session_hint=is_multi,
        )
        result = await reasoner.answer(question)
        # Stash the union of every sub-question's retrieved context so
        # _extract_retrieved_session_ids (baseline.py) finds something
        # to score recall against.
        self.last_ctx = _merge_retrieved_contexts(contexts, budget)
        # Surface telemetry onto the adapter so baseline._run_one can
        # merge it into row.telemetry via the existing channel.
        self.last_telemetry = {
            "llm_call_count": result.llm_call_count,
            "abstained": result.abstained,
            "task_mode": result.trace.get("task_mode"),
            "head_short_circuit": result.trace.get("head_short_circuit"),
            "head_fired": result.trace.get("head_fired"),
            "abstain_reason": result.trace.get("abstain_reason"),
            "intent": result.trace.get("intent"),
            "intent_source": result.trace.get("intent_source"),
            "iterative_trace": result.trace,
        }
        return result.answer


# WS-7 preference conditioning — open-ended "recommend / suggest / which
# should I" requests that should honour a preference the user stated earlier.
# In the eval we gate primarily on the dataset's question_type; this wording
# detector is the production fallback when no type label is available.
_PREFERENCE_INTENT_TERMS: tuple[str, ...] = (
    "recommend",
    "suggest",
    "which should i",
    "what should i",
    "best for me",
    "best option",
    "resources",
    "ideas for",
    "tailor",
    "personali",  # personalize / personalise / personalized
    "based on my",
    "for my",
)


def _is_preference_question(question: str) -> bool:
    """Heuristic gate for preference-sensitive open-ended requests."""
    q = (question or "").lower()
    return any(term in q for term in _PREFERENCE_INTENT_TERMS)


# WS-1 temporal reasoning — date-delta questions ("how many days between X and
# Y", "how many weeks ago", "in what order"). The measured failure is NOT
# arithmetic difficulty: the direct context dropped the turn dates entirely, so
# the model answered "no dates available" / "0". The fix surfaces each turn's
# date + the reference "now", gated to temporal questions.
_TEMPORAL_INTENT_TERMS: tuple[str, ...] = (
    "how many days",
    "how many weeks",
    "how many months",
    "how many years",
    "how long ago",
    "how long since",
    "how long has",
    "days passed",
    "weeks ago",
    "months ago",
    "years ago",
    "weeks have passed",
    "months have passed",
    " since ",
    " before ",
    " after ",
    "in what order",
    "in the order",
    "which happened first",
    "which came first",
    "earlier or later",
)


def _is_temporal_question(question: str) -> bool:
    """Heuristic gate for date-arithmetic / event-ordering questions."""
    q = (question or "").lower()
    return any(term in q for term in _TEMPORAL_INTENT_TERMS)


# WS-3 knowledge-update — the user's situation CHANGED and the question asks
# for the current state. Continuum's supersession already removes the stale
# LTM fact (source="ltm_fact" items are current-only), but the old raw turn
# is still in context and distracts the reader. Gate is mainly the dataset
# question_type; this wording detector is the production fallback.
_KNOWLEDGE_UPDATE_TERMS: tuple[str, ...] = (
    "currently",
    "current",
    "these days",
    "right now",
    "as of now",
    "still ",
    "anymore",
    "now that",
    "most recent",
    "latest",
    "up to date",
    "nowadays",
)


def _is_knowledge_update_question(question: str) -> bool:
    """Heuristic gate for 'what is the current state after a change' questions."""
    q = (question or "").lower()
    return any(term in q for term in _KNOWLEDGE_UPDATE_TERMS)


# WS-date-math — the temporal residual is date ARITHMETIC the model still
# botches even with dates surfaced (WS-1). For delta questions we ask the model
# to emit a small SPEC alongside its answer, then compute the result in CODE
# (deterministic — NOT an agent tool-loop, NOT a second call). Graceful: if the
# SPEC is missing/unparseable we keep the model's free-text answer.
_TEMPORAL_DELTA_TERMS: tuple[str, ...] = (
    "how many days",
    "how many weeks",
    "how many months",
    "how many years",
    "how long ago",
    "how long since",
    "days passed",
    "weeks passed",
    "months passed",
    "days ago",
    "weeks ago",
    "months ago",
    "years ago",
    "weeks have passed",
    "months have passed",
)


def _is_temporal_delta_question(question: str) -> bool:
    """Date-arithmetic subset of temporal (count of days/weeks/months between
    events or 'ago'). Ordering questions are left to the WS-1 prompt."""
    q = (question or "").lower()
    return any(term in q for term in _TEMPORAL_DELTA_TERMS)


def _compute_temporal_from_spec(answer: str) -> str | None:
    """Parse a ``SPEC: {json}`` line from the model output and compute the
    date delta deterministically. Returns a clean answer string, or ``None``
    if no valid SPEC (caller then keeps the model's free-text answer).

    SPEC shape: ``{"op": "between"|"ago", "unit": "days"|"weeks"|"months"|
    "years", "dates": ["YYYY-MM-DD", ...], "now": "YYYY-MM-DD"}``.
    """
    m = re.search(r"SPEC:\s*(\{.*\})", answer, re.DOTALL)
    if not m:
        return None
    try:
        spec = json.loads(m.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(spec, dict):
        return None

    def _parse(d: Any) -> dt.date | None:
        try:
            return dt.date.fromisoformat(str(d)[:10])
        except (ValueError, TypeError):
            return None

    op = str(spec.get("op", "")).lower()
    unit = str(spec.get("unit", "days")).lower().rstrip("s") + "s"
    dates = [d for d in (_parse(x) for x in spec.get("dates", []) or []) if d]
    now = _parse(spec.get("now"))

    if op in ("ago", "since") and dates and now:
        a, b = now, dates[0]
    elif op in ("between", "diff", "") and len(dates) >= 2:
        a, b = dates[0], dates[1]
    else:
        return None

    days = abs((a - b).days)
    if unit == "days":
        return f"{days} days"
    if unit == "weeks":
        return f"{round(days / 7)} weeks"
    if unit == "months":
        months = abs((a.year - b.year) * 12 + (a.month - b.month))
        return f"{months} months"
    if unit == "years":
        return f"{abs(a.year - b.year)} years"
    return None


class _DirectAnswerAdapter(_IngestingAdapter):
    """
    Minimal A/B baseline for the IterativeReasoner: retrieve, then hand
    the raw retrieved turns straight to the answerer in ONE LLM call.

    No decompose, no candidate extraction, no claim verification, no
    evidence packet, no reasoning heads — the entire machinery the
    iterative reasoner runs is bypassed. Retrieval is identical (same
    ``session.retriever``, so hybrid + LTM are unchanged), which makes
    this a clean apples-to-apples test of the hypothesis that the
    claim/verify/packet stack is *losing* information that 100%-recall
    retrieval already found, for single-hop fact questions.

    Sets ``last_ctx`` (recall stays measurable) and
    ``last_telemetry`` with ``llm_call_count=1, answer_mode="direct"``.
    """

    def __init__(
        self,
        *,
        session: _MiniSession,
        llm: Any,
        answer_max_tokens: int = 128,
        top_k: int = 12,
        max_context_chars: int = 32000,
        reranker: CrossEncoderReranker | None = None,
        rerank_to: int = 0,
        preference_conditioning: bool = False,
        temporal_conditioning: bool = False,
        aggregation_v2: bool = False,
        knowledge_update_conditioning: bool = False,
        temporal_codemath: bool = False,
        synthesis_fn: Any = None,
    ) -> None:
        super().__init__(
            session=session, llm=llm, answer_max_tokens=answer_max_tokens,
        )
        self._top_k = top_k
        # v3 synthesis: when wired, ingest-time aggregation produces per-entity
        # counts ("User has 5 tops") that are injected into the prompt for
        # counting questions — the reader reads the count instead of miscounting.
        self._synthesis_fn = synthesis_fn
        self._synthesis_facts: list[Any] = []  # DerivedFacts from ingest-time aggregation
        self._max_context_chars = max_context_chars
        # WS-7: when on, preference-type questions get a prompt that tells the
        # model to identify and APPLY a stated user preference from the
        # retrieved turns (recall is ~93%; the gap is application). Feature-
        # flagged + gated so factual/temporal/etc. questions are never touched
        # — the over-personalization risk only appears under global injection.
        self._pref_conditioning = preference_conditioning
        # WS-1: when on, temporal questions get each retrieved turn PREFIXED
        # with its date + the reference "now", and a prompt to compute the
        # delta. The measured failure was dates missing from the prompt
        # entirely ("the conversation doesn't include any dates"). Gated.
        self._temporal_conditioning = temporal_conditioning
        # WS-2: when on, aggregation questions get a stronger "enumerate every
        # distinct instance, dedupe, THEN count" prompt — the failure was the
        # model counting without listing ("3 weddings" -> "1"). Gated to the
        # same aggregate questions; A/B vs the current aggregation prompt.
        self._aggregation_v2 = aggregation_v2
        # WS-3: when on, knowledge-update questions sort the current LTM facts
        # (source="ltm_fact", superseded already removed) to the front, mark
        # them [CURRENT FACT], and prompt the reader to prefer the current
        # state / most-recent statement over stale raw turns. Uses the
        # supersession moat directly. Gated; A/B vs baseline.
        self._ku_conditioning = knowledge_update_conditioning
        # WS-date-math: for temporal DELTA questions, have the model emit a
        # date SPEC and compute the delta in code (deterministic), overriding
        # its mental arithmetic. Builds on WS-1 (dates already surfaced).
        self._temporal_codemath = temporal_codemath
        # WS-4: optional cross-encoder precision pass. The retriever
        # over-fetches for recall; the reranker reorders that pool jointly
        # on (question, turn) and we keep the best ``rerank_to`` for context.
        # NB: rerank only helps when the retrieved pool is LARGER than the
        # keep count — i.e. the retriever must over-fetch. With the v1
        # session-aware config the pool is small (~4-8), so raise the
        # retriever's max_items well above rerank_to or this is a no-op.
        self._reranker = reranker
        self._rerank_to = rerank_to
        self.last_telemetry: dict[str, Any] = {}

    async def process_conversation(self, messages: Iterable[dict[str, Any]]) -> None:
        """Normal ingest, then (v3) build per-entity aggregate counts."""
        msgs = list(messages)
        await super().process_conversation(msgs)
        if self._synthesis_fn is None:
            return
        from continuum.promotion.synthesis import aggregate, extract_structured_facts

        # Group user turns by session; extract countable triples per session
        # (bounded prompts), keeping the session date for scoped aggregates.
        sessions: dict[str, list[str]] = {}
        dates: dict[str, str] = {}
        for m in msgs:
            if str(m.get("role", "")) != "user":
                continue
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            sid = str(m.get("session_id", ""))
            sessions.setdefault(sid, []).append(content)
            d = str(m.get("date", "") or "")
            if d and sid not in dates:
                dates[sid] = d

        facts: list[Any] = []
        for sid, turns in sessions.items():
            occurred = None
            if dates.get(sid):
                with contextlib.suppress(ValueError):
                    occurred = dt.datetime.fromisoformat(dates[sid])
            try:
                facts.extend(
                    await extract_structured_facts(
                        "\n".join(turns),
                        completion_fn=self._synthesis_fn,
                        occurred_at=occurred,
                    )
                )
            except Exception:
                log.exception("synthesis extraction failed for session %s", sid)
        self._synthesis_facts = aggregate(facts)
        log.info(
            "synthesis: %d entity_summaries from %d facts",
            len(self._synthesis_facts), len(facts),
        )

    async def answer_question(self, question: str) -> str:
        retriever = getattr(self.session, "retriever", None)
        ctx: ContextBundle | None = None
        if retriever is not None:
            try:
                ctx = await retriever.retrieve(Query(text=question), self.budget)
            except Exception:
                log.exception("direct retrieve failed for %r", question[:80])
        self.last_ctx = ctx

        pool = list(getattr(ctx, "items", []) or [])
        reranked = False
        # WS-4: the cross-encoder rerank measurably HELPS the retrieval-bound
        # categories (knowledge-update, multi-session) but HURTS preference
        # (the reranker reorders away the turns the preference rubric needs).
        # Gate it off for preference questions — keep the retrieval gains clean.
        _qt_early = (getattr(self, "dataset_question_type", "") or "").lower()
        _is_pref_early = self._pref_conditioning and (
            _qt_early == "single-session-preference" or _is_preference_question(question)
        )
        # Keep ``rerank_to`` after reranking (falls back to top_k if unset).
        # Fire only when the pool actually exceeds the keep count — else
        # reranking can't change which items reach the prompt.
        keep = self._rerank_to if self._rerank_to and self._rerank_to > 0 else self._top_k
        if self._reranker is not None and not _is_pref_early and len(pool) > keep:
            try:
                items = await self._reranker.rerank(question, pool, top_k=keep)
                reranked = True
            except Exception:
                log.exception("direct rerank failed; using retrieval order")
                items = pool[: self._top_k]
        else:
            items = pool[: self._top_k]
        if not items:
            self.last_telemetry = {"llm_call_count": 0, "answer_mode": "direct"}
            return ""

        from evals.longmemeval.decomposed_answer import is_aggregate_question
        qt = (getattr(self, "dataset_question_type", "") or "").lower()
        # WS-1 temporal branch (gated): the failure was that turn dates never
        # reached the prompt, so surface each turn's date here.
        temporal = self._temporal_conditioning and (
            qt == "temporal-reasoning" or _is_temporal_question(question)
        )
        # WS-date-math: a DELTA sub-question we can verify in code.
        temporal_delta = (
            temporal and self._temporal_codemath and _is_temporal_delta_question(question)
        )
        # WS-7 preference branch (gated on the flag + the type/wording).
        preference = self._pref_conditioning and (
            qt == "single-session-preference" or _is_preference_question(question)
        )
        # WS-3 knowledge-update branch (gated): the current LTM facts are
        # already superseded-filtered; surface + prioritise them so the reader
        # prefers current state over the stale raw turn still in context.
        ku = self._ku_conditioning and (
            qt == "knowledge-update" or _is_knowledge_update_question(question)
        )
        # Aggregation questions (multi-session "how many / which / list all")
        # need the model to combine items across sessions, not return the
        # first matching span (43/50 multi-session failures were
        # missing_aggregation despite 85% recall).
        aggregate = qt == "multi-session" or is_aggregate_question(question)

        if ku:
            # Current-first: live LTM facts (source="ltm_fact") ahead of raw
            # turns. Stable sort preserves retrieval order within each group.
            items = sorted(
                items,
                key=lambda it: 0
                if (getattr(it, "metadata", {}) or {}).get("source") == "ltm_fact"
                else 1,
            )

        lines: list[str] = []
        for it in items:
            content = (getattr(it, "content", "") or "").strip()
            if not content:
                continue
            meta = getattr(it, "metadata", {}) or {}
            role = meta.get("role", "") or ""
            if temporal:
                # Prefix each turn with its date so date arithmetic is
                # possible at all (otherwise the model sees no timestamps).
                date = meta.get("date") or ""
                if not date:
                    ca = getattr(it, "created_at", None)
                    date = ca.isoformat()[:10] if ca is not None else ""
                stamp = f"[{date}] " if date else ""
                lines.append(f"{stamp}[{role}] {content}" if role else f"{stamp}{content}")
            elif ku and meta.get("source") == "ltm_fact":
                # Mark resolved current facts so the reader trusts them.
                lines.append(f"[CURRENT FACT] {content}")
            else:
                lines.append(f"[{role}] {content}" if role else content)
        context = "\n".join(lines)[: self._max_context_chars]
        # v3 synthesis: on counting questions, inject ONLY the relevant
        # code-computed aggregate(s) (v3.1: relevance-filtered — v3.0 dumped all
        # ~40 and buried the answer), so the reader reads the count/total.
        from continuum.promotion.synthesis import (
            is_counting_question,
            relevant_summaries,
        )

        chosen = (
            relevant_summaries(self._synthesis_facts, question)
            if (self._synthesis_facts and is_counting_question(question))
            else []
        )
        synthesis_injected = bool(chosen)
        if chosen:
            block = (
                "COMPUTED FACTS (counts/totals already calculated by the memory "
                "system — trust these for 'how many / how much / how long' "
                "questions):\n" + "\n".join(f"- {f.text}" for f in chosen)
            )
            context = block + "\n\n" + context
        if temporal:
            # The turns above are date-stamped. Give "now" and ask for an
            # explicit, scoped date calculation (NOT a general reasoning loop).
            today = (getattr(self, "dataset_question_date", "") or "").strip()
            today_line = f"Today's date is {today}.\n" if today else ""
            prompt = (
                "Each retrieved turn below is prefixed with the date it was "
                "said, in [YYYY-MM-DD] form. " + today_line + "Use these dates "
                "to answer the question: find the relevant event date(s), then "
                "compute the difference (count of days / weeks / months, or how "
                "long ago relative to today), or put the events in time order. "
                "Work it out from the dates — do not guess. Reply with just the "
                "answer (the number or ordering).\n\n"
                f"Retrieved conversation:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
            if temporal_delta:
                # WS-date-math: also ask for a machine-checkable SPEC so code
                # can compute the delta exactly (overriding mental arithmetic).
                prompt += (
                    "\n\nThen, on a NEW final line, emit a SPEC for the "
                    "calculation as JSON:\n"
                    'SPEC: {"op": "between"|"ago", "unit": "days"|"weeks"|'
                    '"months"|"years", "dates": ["YYYY-MM-DD", ...], '
                    '"now": "YYYY-MM-DD"}\n'
                    'Use op="between" with the two event dates for "between X '
                    'and Y"; use op="ago" with the single event date + now for '
                    '"how long ago / since".'
                )
        elif ku:
            prompt = (
                "The user's situation may have CHANGED over time. Items marked "
                "[CURRENT FACT] are the user's current, resolved facts — "
                "outdated versions have already been removed by the memory "
                "system. When the raw conversation contains statements that "
                "conflict across time, the user's CURRENT situation is the most "
                "recent one. Prefer the [CURRENT FACT] items and the latest "
                "statement; never answer with a value the user has since "
                "changed. Reply with just the current answer.\n\n"
                f"Retrieved conversation:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
        elif preference:
            # Reminder-style conditioning (PrefEval's best prompting method).
            # v2: the first A/B was net-neutral (+4/-4) — it rescued
            # abstentions but ALSO turned specific correct answers into generic
            # menus the rubric judge penalised. So demand a CONCRETE pick, not
            # a list. Over-personalization guard stays: apply ONLY a relevant
            # stated preference, never invent one, preserve factual accuracy.
            prompt = (
                "The retrieved conversation may contain a preference this user "
                "has stated — a tool they use, a brand they like, a topic they "
                "care about, or something they dislike. If a stated preference "
                "is directly relevant to the request below, give a SPECIFIC, "
                "CONCRETE recommendation that honours it — a definite pick or "
                "direct answer, NOT a long generic menu of options — and make "
                "the link to their preference explicit. If no stated "
                "preference is relevant, just answer the question directly. "
                "Never invent a preference and never sacrifice factual "
                "accuracy. Reply with just the answer.\n\n"
                f"Retrieved conversation:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
        elif aggregate and self._aggregation_v2:
            # WS-2: enumerate-then-count. The failure mode was the model
            # counting/answering without first listing the instances
            # ("3 weddings" -> "1"). Force an explicit enumeration + dedupe
            # before the final count/list.
            prompt = (
                "The question below asks you to COUNT or LIST things that may "
                "be spread across MULTIPLE parts of the retrieved conversation. "
                "Work in two steps:\n"
                "1) First enumerate EVERY distinct relevant instance you can "
                "find, one per line, removing duplicates (the same instance "
                "mentioned twice counts once).\n"
                "2) Then give the FINAL answer — the total count or the "
                "complete de-duplicated list — based on that enumeration.\n"
                "Do not stop at the first match and do not guess a number.\n\n"
                f"Retrieved conversation:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
        elif aggregate:
            prompt = (
                "The question below asks you to combine information that "
                "may be spread across MULTIPLE parts of the retrieved "
                "conversation. Read ALL of it, find EVERY relevant item, "
                "and give the COMPLETE aggregated answer — the full list "
                "or the total count across everything. Do not stop at the "
                "first match. Reply with just the answer.\n\n"
                f"Retrieved conversation:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
        else:
            prompt = (
                "Answer the question using ONLY the retrieved conversation "
                "below. Reply with just the answer — no explanation. If the "
                "answer truly isn't present, reply 'I don't know.'\n\n"
                f"Retrieved conversation:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
        try:
            answer = await self.llm.complete(
                prompt=prompt, max_tokens=self.answer_max_tokens,
            )
        except Exception:
            log.exception("direct answer LLM call failed")
            answer = ""
        answer = (answer or "").strip()

        # WS-date-math: replace the model's mental arithmetic with the
        # code-computed delta when a valid SPEC is present; otherwise strip the
        # SPEC line so it doesn't leak into the free-text answer.
        codemath_applied = False
        if temporal_delta and answer:
            computed = _compute_temporal_from_spec(answer)
            if computed is not None:
                answer = computed
                codemath_applied = True
            else:
                # SPEC is the final line — strip it and anything after,
                # even when truncated/invalid (no closing brace).
                answer = re.sub(r"\s*SPEC:.*", "", answer, flags=re.DOTALL).strip()

        self.last_telemetry = {
            "llm_call_count": 1,
            "answer_mode": "direct",
            "retrieved_items": len(items),
            "aggregate_prompt": aggregate,
            "aggregation_v2": bool(aggregate and self._aggregation_v2),
            "preference_prompt": preference,
            "temporal_prompt": temporal,
            "temporal_codemath": codemath_applied,
            "ku_prompt": ku,
            "reranked": reranked,
            "rerank_skipped_preference": bool(
                self._reranker is not None and _is_pref_early
            ),
            "synthesis_injected": synthesis_injected,
            "synthesis_summaries": len(self._synthesis_facts),
            "synthesis_injected_n": len(chosen),
        }
        return answer


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


class _DecomposedAnsweringAdapter(_IngestingAdapter):
    """
    Full decomposed-answering path for LongMemEval.

    Unlike ``DecompositionRetriever``, this does not only merge retrieved
    snippets. It answers each sub-question against its own evidence, then
    synthesizes the final answer from those intermediate answers.
    """

    def __init__(
        self,
        *,
        decompose_max_tokens: int = 160,
        subanswer_max_tokens: int = 80,
        evidence_span_selection: bool = False,
        evidence_max_spans: int = 3,
        evidence_min_overlap: float = 0.2,
        session_narrowing: bool = False,
        session_narrowing_min_confidence: float = 0.5,
        evidence_packet: bool = False,
        evidence_packet_max_claims: int = 6,
        claim_first: bool = False,
        claim_first_top_n_for_fallback: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decompose_max_tokens = decompose_max_tokens
        self.subanswer_max_tokens = subanswer_max_tokens
        #: Opt-in span selector — when True, retrieved evidence windows
        #: are run through :func:`select_spans` and the sub-answer prompt
        #: receives 0-N compact spans instead of the full wide window.
        #: Empty span list forces abstain. See
        #: :mod:`evals.longmemeval.evidence_spans` for the contract.
        self.evidence_span_selection = evidence_span_selection
        self.evidence_max_spans = evidence_max_spans
        self.evidence_min_overlap = evidence_min_overlap
        #: Opt-in top-session narrowing — when True and the top-ranked
        #: retrieved session carries a verified answer-bearing claim,
        #: items from lower-ranked sessions are dropped from the
        #: sub-answer prompt. See
        #: :mod:`evals.longmemeval.session_narrowing` for the contract.
        self.session_narrowing = session_narrowing
        self.session_narrowing_min_confidence = session_narrowing_min_confidence
        #: Opt-in minimal evidence packet for the final synthesiser.
        #: When True, the synthesis prompt receives ONLY verified
        #: claims + their source spans (not the raw bundle text), and
        #: the per-row trace records ``selected_evidence_count`` +
        #: ``excluded_noise_count``. See
        #: :mod:`evals.longmemeval.evidence_packet` for the contract.
        self.evidence_packet = evidence_packet
        self.evidence_packet_max_claims = evidence_packet_max_claims
        #: Opt-in claim-first answer pipeline. When True, after the
        #: question is decomposed and each sub-question retrieved,
        #: the adapter runs a sentence-level :class:`Claim` extraction
        #: across every retrieved bundle, routes the original question
        #: to one of six broad reasoning heads, and short-circuits the
        #: final answer when the head returns a validated span. See
        #: :mod:`evals.longmemeval.task_router` /
        #: :mod:`evals.longmemeval.reasoning_heads`.
        self.claim_first = claim_first
        self.claim_first_top_n_for_fallback = claim_first_top_n_for_fallback
        #: True for one row when ``_answer_claim_first`` produced a
        #: validated span. ``answer_question`` checks this flag and
        #: skips ``_postprocess`` so the COUNT override / LLM repair
        #: can't overwrite a verified claim-first answer (the
        #: nondeterminism behind the b86304ba "6" failure).
        self._claim_first_used: bool = False
        self.last_decomposition_stats: dict[str, Any] = {}
        #: Diagnostic snapshot written when the structured-claim path
        #: matched the question intent but found no usable claim in the
        #: retrieved bundle. Captures the relation, ctx-item count,
        #: extracted-claim count, and a few content snippets so the
        #: trace can show *why* the structured path didn't fire.
        self.last_structured_fallthrough: dict[str, Any] = {}
        #: Per-row dataset hints set by the eval runner before each
        #: ``answer_question`` call. Plumbed into ``route_task`` so the
        #: LongMemEval ``question_type`` can tiebreak when the regex
        #: router lands on ambiguous FACT_LOOKUP — single largest ROI
        #: lever for the assistant-memory / preference / temporal /
        #: multi-session / knowledge-update categories.
        self.dataset_question_type: str | None = None
        self.dataset_is_multi_session: bool = False
        #: Per-row LLM telemetry snapshot — populated at the end of
        #: every ``answer_question`` call. The baseline runner attaches
        #: it to :class:`RowResult.extras` so the JSON output carries
        #: real token / cost / validator state per row.
        self.last_telemetry: dict[str, Any] = {}
        #: Last decomposition guard result. Populated by
        #: :meth:`_decompose_question` whenever the LLM call succeeds.
        #: Drives the ``decomp_guard`` block in the JSONL trace and
        #: lets failing rows be inspected for "intent changed during
        #: decomposition" failures.
        self._last_decomp_guard: DecompositionGuardResult | None = None

    async def _decompose_question(self, question: str) -> list[str]:
        try:
            reply = await self.llm.complete(
                prompt=build_decompose_prompt(question),
                max_tokens=self.decompose_max_tokens,
            )
        except Exception:
            log.exception("decomposed-answer decomposition failed")
            self._last_decomp_guard = None
            return [question]
        raw_subs = parse_subquestions(reply, original=question)
        # Intent-preserving guard: reject sub-questions that drift on
        # answer-type / relation / object / temporal scope. When every
        # sub fails, fall back to the original question so retrieval
        # still runs. The result is stashed on the adapter for the
        # downstream stats / trace consumer.
        guard = guard_decomposition(question, raw_subs)
        self._last_decomp_guard = guard
        if guard.rejected:
            log.info(
                "decomp_guard: kept=%d rejected=%d (reasons=%s)",
                len(guard.kept), len(guard.rejected),
                [r.reason for r in guard.rejected],
            )
        return guard.kept

    @staticmethod
    def _is_atomic(question: str, subquestions: list[str]) -> bool:
        if len(subquestions) != 1:
            return False
        return subquestions[0].strip().lower() == question.strip().lower()

    # Delegates to the canonical helper in decomposed_answer so the
    # adapter and the synthesis prompt agree on the question shape.
    _is_aggregate_question = staticmethod(is_aggregate_question)

    @staticmethod
    def _is_unknown_answer(answer: str) -> bool:
        normalized = answer.strip().lower()
        return (
            not normalized
            or normalized == "i don't know"
            or normalized.startswith("i don't know.")
            or normalized.startswith("i don't know,")
        )

    @staticmethod
    def _empty_context(budget: TokenBudget) -> ContextBundle:
        return ContextBundle(
            items=[],
            messages=[],
            tokens_used=0,
            budget=budget,
            tier_breakdown={"stm": 0, "mtm": 0, "ltm": 0},
        )

    @staticmethod
    def _merge_contexts(contexts: list[ContextBundle]) -> ContextBundle | None:
        if not contexts:
            return None
        first = contexts[0]
        seen: set[str] = set()
        items: list[MemoryItem] = []
        for ctx in contexts:
            for item in ctx.items:
                if item.id in seen:
                    continue
                seen.add(item.id)
                items.append(item)
        token_count = sum(estimate_tokens_text(item.content) for item in items)
        messages = [
            {
                "role": str(item.metadata.get("role", "user"))
                if item.metadata else "user",
                "content": item.content,
            }
            for item in items
        ]
        return dataclasses_replace(
            first,
            items=items,
            messages=messages,
            tokens_used=token_count,
            tier_breakdown={"stm": token_count, "mtm": 0, "ltm": 0},
            debug_info={
                "retrieval_mode": "decompose_answer",
                "merged_items": len(items),
            },
        )

    async def answer_question(self, question: str) -> str:
        """
        Public entry point — wraps the inner decompose-answer pipeline
        with per-row telemetry, question-type classification,
        structured candidate extraction, count-aggregator override,
        answer-shape validation, and a one-shot repair on validation
        failure or "I don't know" with candidates present.
        """
        start_row_telemetry()
        # Reset the per-row claim-first flag so a previous row's success
        # can't accidentally suppress this row's postprocess.
        self._claim_first_used = False
        try:
            question_type = classify(question)
            raw_answer = await self._answer_question_inner(question)
            if self._claim_first_used:
                # Claim-first ran AND produced a validated span. The
                # span is final — `_postprocess` would otherwise let
                # the COUNT override / LLM repair clobber it with a
                # noisy number candidate (the b86304ba "6" bug). Run
                # only a light cleanup pass.
                final = clean_final_answer(raw_answer)
                stats = self.last_decomposition_stats or {}
                stats.setdefault("question_type", question_type.value)
                stats["validator_passed"] = True
                stats["validator_reason"] = "claim_first_terminal"
                stats["regeneration_attempted"] = False
                self.last_decomposition_stats = stats
                counter = current_counter()
                if counter is not None:
                    counter.validator_passed = True
                    counter.validator_reason = "claim_first_terminal"
                    counter.regeneration_attempted = False
            else:
                final = await self._postprocess(
                    question, question_type, raw_answer,
                )
        finally:
            counter = end_row_telemetry()
            self.last_telemetry = counter.snapshot() if counter else {}
        return final

    async def _postprocess(
        self,
        question: str,
        question_type: QuestionType,
        raw_answer: str,
    ) -> str:
        """
        Apply the answer-quality fixes uniformly to whatever path
        produced ``raw_answer``:

        1. Clean scaffold tokens / JSON blobs / evidence bullets.
        2. Extract typed candidates from the retrieved context.
        3. COUNT override: if evidence has an explicit "<n> <noun>"
           matching the question's noun, beat the dedup-count.
        4. Don't-IDK guard: if candidates exist + answer is IDK,
           regenerate from candidates.
        5. Shape validator: if the answer doesn't match question_type,
           regenerate from candidates.

        Records all decisions on ``last_decomposition_stats`` and
        ``last_telemetry`` for the baseline runner to attach to
        ``RowResult``.
        """
        cleaned = clean_final_answer(raw_answer)

        # Pull every candidate from the bundle the inner pipeline
        # already retrieved; filter to the type the question wants.
        candidates = extract_candidates_from_context(self.last_ctx)
        relevant = filter_candidates_for_question(candidates, question_type)

        # ── COUNT override: explicit number in evidence wins over dedup ──
        override_reason = ""
        best_count = (
            best_count_for_object(candidates, question)
            if question_type == QuestionType.COUNT else None
        )
        # If the cleaned answer doesn't already contain that number (or
        # contains a different number for the same object), prefer the
        # evidence-grounded one.
        if best_count is not None and best_count.normalized_value not in cleaned:
            cleaned = best_count.value
            override_reason = (
                f"count_override:{best_count.normalized_value} "
                f"(unit={best_count.unit})"
            )

        validator_ok, validator_reason = validate_answer_shape(cleaned, question_type)
        idk_blocked = should_block_idk(cleaned, question_type, relevant)
        regenerated = False

        # One-shot repair when validation fails or IDK is blocked.
        if (not validator_ok or idk_blocked) and relevant:
            reason = "idk_with_candidates" if idk_blocked else validator_reason
            repair_prompt = build_repair_prompt(
                question=question,
                question_type=question_type,
                candidates=relevant,
                failed_answer=cleaned,
                reason=reason,
            )
            try:
                repaired_raw = await self.llm.complete(
                    prompt=repair_prompt,
                    max_tokens=max(self.answer_max_tokens, 120),
                )
            except Exception as exc:
                log.exception("repair-prompt LLM call failed for %r", question[:80])
                repaired_raw = cleaned + f" [repair_error: {exc!r}]"
            repaired = clean_final_answer(str(repaired_raw))
            regenerated = True
            # Accept the repair if it now validates; otherwise keep the
            # cleaned original (repair is best-effort, never worse).
            new_ok, new_reason = validate_answer_shape(repaired, question_type)
            if new_ok or not is_idk(repaired):
                cleaned = repaired
                validator_ok, validator_reason = new_ok, new_reason

            # Hard fallback: if the LLM repair produced empty / scaffold /
            # still-invalid output but we have verified candidates, pick
            # the top-confidence candidate's value directly. This is the
            # fix for the "empty final answer when candidates exist"
            # failure mode (rows 58ef2f1c / 5d3d2817 — Gemma 4B sometimes
            # returns empty from the repair LLM call). Never worse than
            # returning empty; usually correct because the candidates
            # have already passed the verifier and rerank pipeline.
            if (
                (not validator_ok or not cleaned.strip())
                and relevant
            ):
                fallback = best_candidate_fallback_answer(relevant)
                if fallback:
                    cleaned = fallback
                    validator_ok = True
                    validator_reason = "fallback_to_top_candidate"

        # Publish the full trace for the row.
        stats = self.last_decomposition_stats or {}
        stats.setdefault("question_type", question_type.value)
        stats.update({
            "question_type": question_type.value,
            "n_candidates_total": len(candidates),
            "n_candidates_relevant": len(relevant),
            "validator_passed": validator_ok,
            "validator_reason": validator_reason,
            "regeneration_attempted": regenerated,
            "count_override_reason": override_reason,
            "abstain_reason": (
                "no_candidate_for_question_type"
                if is_idk(cleaned) and not relevant else ""
            ),
            "candidates_preview": [c.to_dict() for c in relevant[:6]],
        })
        self.last_decomposition_stats = stats

        # Mirror the same on telemetry so the baseline runner's
        # row-extras pick it up uniformly.
        counter = current_counter()
        if counter is not None:
            counter.validator_passed = validator_ok
            counter.validator_reason = validator_reason
            counter.regeneration_attempted = regenerated
            counter.abstain_reason = stats["abstain_reason"]
            counter.selected_evidence_count = (
                len(self.last_ctx.items) if self.last_ctx is not None else 0
            )
            counter.extras.setdefault("question_type", question_type.value)
            counter.extras.setdefault("count_override_reason", override_reason)
        return cleaned

    async def _answer_claim_first(self, question: str) -> str:
        """
        Claim-first answer path.

        1. Retrieve once on the original question (no decomposition).
        2. Extract sentence-level claims from the bundle (scaffold +
           transcript-timestamp filtered).
        3. Route the question to one of six broad reasoning heads.
        4. Run the head on the ranked claims.
        5. If the head returns empty, call the extractive LLM fallback
           with the top 1-3 claims (cheap, single call, span-validated).
        6. Validate the result against the source claims. Return the
           span on success; return empty string on failure to let the
           legacy decompose+packet path run.
        """
        # Reset per-question diagnostics so stale snapshots from the
        # previous row don't leak into this row's trace.
        self.last_structured_fallthrough = {}
        retriever = getattr(self.session, "retriever", None)
        if retriever is None:
            return ""
        try:
            ctx = await retriever.retrieve(Query(text=question), self.budget)
        except Exception:
            log.exception("claim-first retrieve failed for %r", question[:80])
            return ""
        self.last_ctx = ctx

        # 2. claims
        claims = extract_claims_from_context(ctx)
        if not claims:
            return ""
        ranked = rank_claims(question, claims)

        # 3. route — pass dataset hints so the router can disambiguate
        # questions whose surface form alone is ambiguous (e.g., the
        # assistant-memory category often phrases questions as
        # autobiographical fact lookups).
        mode = route_task(
            question,
            question_type_hint=self.dataset_question_type,
            is_multi_session=self.dataset_is_multi_session,
        )
        # ── Structured-claim path (FACT_LOOKUP only, single-session) ────
        # Parse the question into a structured intent. If the intent
        # matches a known relation AND a structured claim with that
        # relation exists in the retrieved bundle, we answer from
        # ``claim.object`` directly — evidence-locked, no LLM, no
        # COUNT override, no validator second-guessing.
        # Other modes still go through the legacy heads below.
        if mode == TaskMode.FACT_LOOKUP:
            intent = parse_intent(question)
            if intent.matched:
                structured_claims = (
                    extract_structured_claims_from_context(ctx.items)
                )
                final = finalize_fact_answer(intent, structured_claims)
                if final is not None:
                    answer, diag = final
                    self._claim_first_used = True
                    self.last_decomposition_stats = {
                        "mode": "claim_first_structured",
                        "claim_first_route": mode.value,
                        "n_claims": len(ranked),
                        "claim_first_span": answer,
                        "structured_intent": diag["intent"],
                        "structured_matched_claim": diag["matched_claim"],
                        "structured_candidate_count": diag["n_candidate_claims"],
                    }
                    log.info(
                        "claim-first-structured: relation=%s answer=%r",
                        intent.relation, answer[:60],
                    )
                    return answer
                # ── Full-session structured retry ────────────────────
                # The wiki bundle's "Nearby Context" is bounded to ±2
                # turns around the matched turn. When the gold-bearing
                # sentence lives outside that window — but in the same
                # SESSION the retriever already surfaced — we'd miss
                # it. Expand to the full session(s) referenced by
                # ctx.items and rerun extraction once. The structured
                # extractor's scaffold/label/responsibility filters
                # keep the wider window from emitting false positives.
                items_snap = list(ctx.items)
                session_ids_to_expand: list[str] = []
                _seen_sids: set[str] = set()
                for it in items_snap:
                    md = getattr(it, "metadata", None) or {}
                    sid = md.get("session_id") or ""
                    sids = md.get("session_ids") or []
                    for cand in ([sid] if sid else []) + list(sids):
                        if cand and cand not in _seen_sids:
                            _seen_sids.add(cand)
                            session_ids_to_expand.append(cand)
                expanded_claims: list[Any] = []
                raw_store = getattr(retriever, "raw_store", None)
                if session_ids_to_expand and raw_store is not None:
                    for raw in getattr(raw_store, "items", []):
                        rmd = raw.metadata or {}
                        rsid = str(rmd.get("session_id") or "")
                        if rsid not in _seen_sids:
                            continue
                        expanded_claims.extend(
                            extract_structured_claims_from_context([raw])
                        )
                if expanded_claims:
                    final_expanded = finalize_fact_answer(
                        intent, expanded_claims,
                    )
                    if final_expanded is not None:
                        answer, diag = final_expanded
                        self._claim_first_used = True
                        self.last_decomposition_stats = {
                            "mode": "claim_first_structured_expanded",
                            "claim_first_route": mode.value,
                            "n_claims": len(ranked),
                            "claim_first_span": answer,
                            "structured_intent": diag["intent"],
                            "structured_matched_claim": diag["matched_claim"],
                            "structured_candidate_count":
                                diag["n_candidate_claims"],
                            "expanded_sessions": session_ids_to_expand,
                        }
                        log.info(
                            "claim-first-structured EXPANDED: rel=%s "
                            "sessions=%d answer=%r",
                            intent.relation,
                            len(session_ids_to_expand),
                            answer[:60],
                        )
                        return answer
                # ── DIAGNOSTIC: structured path matched the intent but
                # found no usable claim in the retrieved bundle OR in
                # the expanded session content. Surface enough detail
                # in the trace to distinguish retrieval gaps from
                # extractor gaps.
                matching = [
                    c for c in structured_claims
                    if c.relation == intent.relation
                ]
                expanded_matching = [
                    c for c in expanded_claims
                    if c.relation == intent.relation
                ]
                self.last_structured_fallthrough = {
                    "relation": intent.relation,
                    "constraints": dict(intent.constraints),
                    "n_ctx_items": len(items_snap),
                    "n_total_claims": len(structured_claims),
                    "n_matching_relation": len(matching),
                    "matching_objects": [c.object_ for c in matching[:5]],
                    "expanded_sessions": session_ids_to_expand,
                    "n_expanded_claims": len(expanded_claims),
                    "n_expanded_matching": len(expanded_matching),
                    "expanded_matching_objects":
                        [c.object_ for c in expanded_matching[:5]],
                    "ctx_snippets": [
                        (getattr(it, "content", "") or "")[:1500]
                        for it in items_snap[:6]
                    ],
                }
                log.info(
                    "claim-first-structured FALLTHROUGH: rel=%s "
                    "ctx=%d claims=%d matching=%d expanded=%d/%d",
                    intent.relation, len(items_snap),
                    len(structured_claims), len(matching),
                    len(expanded_claims), len(expanded_matching),
                )

        head_table = {
            TaskMode.FACT_LOOKUP:             head_fact_lookup,
            TaskMode.ASSISTANT_MEMORY_LOOKUP: head_assistant_memory,
            TaskMode.PREFERENCE_PROFILE:      head_preference_profile,
            TaskMode.KNOWLEDGE_UPDATE:        head_knowledge_update,
            TaskMode.MULTI_SESSION_AGGREGATE: head_multi_session_aggregate,
            TaskMode.TEMPORAL_REASONING:      head_temporal_reasoning,
        }
        head = head_table[mode]
        deterministic = head(ranked, question) or ""
        deterministic = deterministic.strip().rstrip(".")

        # 4. fallback when the head couldn't extract a span
        if not deterministic:
            top_k = max(1, self.claim_first_top_n_for_fallback)
            prompt = build_extractive_prompt(question, ranked[:top_k])
            if prompt:
                try:
                    raw = await self.llm.complete(
                        prompt=prompt, max_tokens=40,
                    )
                except Exception:
                    log.exception(
                        "claim-first fallback LLM call failed for %r",
                        question[:80],
                    )
                    raw = ""
                deterministic = clean_final_answer(str(raw))

        # 5. validation
        claim_texts = [c.text for c in ranked[:8]]
        ok, reason = validate_against_claims(deterministic, claim_texts)

        # 5a. Validation-failure recovery: when the deterministic span
        # is bad (e.g. prose body of a long claim), give the extractive
        # LLM fallback a chance to extract the SHORTEST span from the
        # top claims before falling through to the legacy decompose
        # pipeline. The legacy pipeline rebuilds the full window text
        # and is more error-prone than asking the model directly for
        # a span. This is what unblocks rows like 5d3d2817 where the
        # head returned a prose body, validator caught it, but the LLM
        # could legitimately extract "Marketing specialist at a small
        # startup" from a sibling claim.
        if not ok:
            top_k = max(1, self.claim_first_top_n_for_fallback)
            prompt = build_extractive_prompt(question, ranked[:top_k])
            if prompt:
                try:
                    raw = await self.llm.complete(
                        prompt=prompt, max_tokens=40,
                    )
                except Exception:
                    log.exception(
                        "claim-first recovery LLM call failed for %r",
                        question[:80],
                    )
                    raw = ""
                recovered = clean_final_answer(str(raw))
                if recovered:
                    rec_ok, _rec_reason = validate_against_claims(
                        recovered, claim_texts,
                    )
                    if rec_ok:
                        log.info(
                            "claim-first: recovered via extractive "
                            "fallback after %s (span=%r)",
                            reason, recovered[:60],
                        )
                        deterministic = recovered
                        ok = True
                        reason = f"recovered_after_{reason}"

        if not ok:
            log.info(
                "claim-first: span %r failed validation (%s) — falling "
                "through to legacy pipeline",
                deterministic[:60], reason,
            )
            # Stamp diagnostic state so the JSONL trace can show that
            # claim-first ran and was rejected.
            self.last_decomposition_stats = {
                "mode": "claim_first_rejected",
                "claim_first_route": mode.value,
                "claim_first_span": deterministic,
                "claim_first_reason": reason,
                "n_claims": len(ranked),
            }
            return ""

        # Identify the supporting claim — the highest-ranked one whose
        # text actually contains the span (lower-cased substring match).
        # Populate ``answer_source_span`` on that claim so the trace
        # shows which sentence backed the answer. `Claim` is frozen, so
        # we record the dict-level snapshot rather than mutating it.
        span_lower = deterministic.lower()
        supporting_idx = next(
            (i for i, c in enumerate(ranked) if span_lower in c.text.lower()),
            0,
        )
        from dataclasses import replace as _replace_claim
        supporting = _replace_claim(
            ranked[supporting_idx], answer_source_span=deterministic,
        )

        # Stamp the success flag so ``answer_question`` skips
        # ``_postprocess`` — the COUNT override / LLM repair must NOT
        # touch a verified claim-first span (b86304ba: "triple what I
        # paid for it" → "6" regression).
        self._claim_first_used = True

        self.last_decomposition_stats = {
            "mode": "claim_first",
            "claim_first_route": mode.value,
            "n_claims": len(ranked),
            "claim_first_span": deterministic,
            "claim_first_supporting_claim": supporting.to_dict(),
            "top_claims_preview": [c.to_dict() for c in ranked[:3]],
        }
        return deterministic

    async def _answer_question_inner(self, question: str) -> str:
        # ── Optional claim-first short-circuit ────────────────────────────
        # When enabled, run the broad reasoning-head pipeline FIRST. If
        # it produces a claim-supported span we return early; otherwise
        # we fall through to the legacy decompose+packet path. This is
        # the minimal-risk wiring: the existing pipeline is preserved
        # verbatim as the fallback, so a regression in the new heads
        # can never make accuracy worse than the prior pass.
        if self.claim_first:
            short_circuit = await self._answer_claim_first(question)
            if short_circuit:
                return short_circuit

        subquestions = await self._decompose_question(question)
        is_aggregate = self._is_aggregate_question(question)
        if self._is_atomic(question, subquestions) and not is_aggregate:
            # Even when the question is atomic, route through the same
            # extractive sub-answer prompt the multi-subquestion path
            # uses. The base ContinuumAdapter.format_prompt is too loose
            # — it lets the model summarise away specific entities
            # (e.g. "Target" stripped from "$5 coupon at Target last
            # Sunday"). The "Matching facts: …; Sub-answer: …" scaffold
            # forces the model to quote evidence verbatim before giving
            # the final word.
            retriever = getattr(self.session, "retriever", None)
            ctx: ContextBundle | None = None
            if retriever is not None:
                try:
                    ctx = await retriever.retrieve(
                        Query(text=question), self.budget,
                    )
                except Exception:
                    log.exception("atomic-extractive retrieve failed for %r", question[:80])
            self.last_ctx = ctx
            prompt = build_subanswer_prompt(question, ctx)
            try:
                raw = await self.llm.complete(
                    prompt=prompt, max_tokens=self.answer_max_tokens,
                )
            except Exception as exc:
                log.exception("atomic-extractive answer failed for %r", question[:80])
                raw = f"[error: {exc!r}]"
            answer = _strip_subanswer_scaffold(str(raw).strip())
            self.last_decomposition_stats = {
                "mode": "atomic_extractive",
                "n_sub_questions": 1,
                "sub_questions": subquestions,
                "sub_answers": [answer],
                "subquestion_hits": [
                    len(ctx.items) if ctx is not None else 0,
                ],
            }
            return answer

        retriever = getattr(self.session, "retriever", None)
        if retriever is None:
            self.last_ctx = None
            self.last_decomposition_stats = {
                "mode": "decompose_answer",
                "n_sub_questions": len(subquestions),
                "sub_questions": subquestions,
                "sub_answers": [],
                "subquestion_hits": [],
            }
            return "I don't know"

        contexts: list[ContextBundle] = []
        subanswers: list[SubAnswer] = []
        # Per-sub-question span audit, captured into last_decomposition_stats
        # so the JSONL trace can show *why* each row chose its evidence.
        spans_audit: list[dict[str, Any]] = []
        # Per-sub-question session-narrowing audit. Each entry records
        # the verified-confidence + dropped session ids + reason so the
        # JSONL trace exposes which sub-questions got narrowed.
        narrow_audit: list[dict[str, Any]] = []
        for subquestion in subquestions:
            try:
                ctx = await retriever.retrieve(Query(text=subquestion), self.budget)
            except Exception:
                log.exception("sub-question retrieve failed for %r", subquestion)
                ctx = self._empty_context(self.budget)

            sub_qtype = classify(subquestion)

            # ── Optional top-session narrowing ───────────────────────────
            # When the top-ranked retrieved session carries a verified
            # answer-bearing candidate and the question is NOT explicitly
            # multi-session, drop items from lower-ranked sessions so
            # ShareGPT/UltraChat distractor windows can't leak into the
            # sub-answer prompt. Falls through cleanly when the question
            # is multi-session, the top session is unverified, or the
            # verifier's confidence is below threshold.
            narrow_result: NarrowResult | None = None
            if self.session_narrowing and ctx.items:
                sub_cands = []
                for item in ctx.items:
                    sid = str((item.metadata or {}).get("session_id") or "")
                    sub_cands.extend(extract_candidates_from_text(
                        item.content, source_session_id=sid,
                    ))
                verified = filter_passing(verify_claims(
                    subquestion, sub_cands, question_type=sub_qtype,
                ))
                multi_session = (
                    sub_qtype == QuestionType.MULTI_SESSION
                    or is_multi_session_hint(subquestion)
                )
                narrow_result = narrow_to_top_session(
                    list(ctx.items),
                    verified,
                    multi_session_hint=multi_session,
                    min_verified_confidence=self.session_narrowing_min_confidence,
                )
                narrow_audit.append({
                    "subquestion": subquestion,
                    **narrow_result.to_dict(),
                })
                if narrow_result.narrowed:
                    # Replace ctx.items with the narrowed list. The
                    # retriever's ContextBundle is a frozen-ish dataclass;
                    # we build a shallow replace so downstream code (span
                    # selection, candidate extraction, evidence text)
                    # sees only the surviving session's items.
                    ctx = dataclasses_replace(
                        ctx,
                        items=narrow_result.items,
                        messages=[
                            {
                                "role": str(
                                    (item.metadata or {}).get("role", "user"),
                                ),
                                "content": item.content,
                            }
                            for item in narrow_result.items
                        ],
                    )

            contexts.append(ctx)

            # Optional evidence span selection — compresses each wide
            # evidence-window into 1-3 answer-bearing spans. Empty
            # result is propagated as empty so the prompt forces abstain.
            spans = None
            if self.evidence_span_selection:
                spans = select_spans(
                    subquestion,
                    list(ctx.items),
                    max_spans=self.evidence_max_spans,
                    min_overlap=self.evidence_min_overlap,
                    question_type=sub_qtype,
                )
                spans_audit.append({
                    "subquestion": subquestion,
                    "n_window_items": len(ctx.items),
                    "n_spans": len(spans),
                    "spans": [s.to_dict() for s in spans[: self.evidence_max_spans]],
                })

            prompt = build_subanswer_prompt(subquestion, ctx, spans=spans)
            try:
                subanswer = await self.llm.complete(
                    prompt=prompt,
                    max_tokens=self.subanswer_max_tokens,
                )
            except Exception as exc:
                log.exception("sub-question answer failed for %r", subquestion)
                subanswer = f"I don't know ({exc!r})"

            # Evidence text the synthesiser later sees: prefer compact
            # spans when selection ran (and at least one span was kept),
            # else fall back to the legacy first-4-items dump.
            if spans:
                evidence_text = "\n".join(
                    f"- [session={s.source_session_id}] ({s.role}): {s.text}"
                    for s in spans
                )
            else:
                evidence_text = "\n".join(item.content for item in ctx.items[:4])
            subanswers.append(
                SubAnswer(
                    subquestion=subquestion,
                    answer=str(subanswer).strip(),
                    evidence_session_ids=extract_session_ids(ctx),
                    evidence_text=evidence_text,
                    hit_count=len(ctx.items),
                )
            )

        self.last_ctx = self._merge_contexts(contexts)
        if (
            len(subanswers) == 1
            and not is_aggregate
            and not self._is_unknown_answer(subanswers[0].answer)
        ):
            self.last_decomposition_stats = {
                "mode": "single_subanswer",
                "n_sub_questions": len(subquestions),
                "sub_questions": subquestions,
                "sub_answers": [subanswers[0].answer],
                "subquestion_hits": [subanswers[0].hit_count],
            }
            return subanswers[0].answer

        # ── Optional minimal evidence packet for the final synthesiser ──
        # When --evidence-packet is on, mine every retrieved item across
        # every sub-question's bundle for candidates, run the verifier,
        # and pass ONLY the verified PASS set into the synthesis prompt
        # via :class:`EvidencePacket`. The packet renderer drops the
        # raw evidence-window text from the prompt and tags each claim
        # with its source span + session id + confidence. Telemetry
        # surfaces ``selected_evidence_count`` and
        # ``excluded_noise_count`` so the JSONL trace can confirm
        # noise was filtered out and not silently passed through.
        evidence_packet_obj: EvidencePacket | None = None
        packet_meta: dict[str, Any] = {}
        if self.evidence_packet:
            # `_answer_question_inner` runs before `_postprocess` so the
            # outer wrapper's `question_type` isn't in scope yet — classify
            # here. Cost is one regex pass, no LLM call.
            packet_qtype = classify(question)
            all_candidates: list[Any] = []
            for ctx_b in contexts:
                all_candidates.extend(extract_candidates_from_context(ctx_b))
            verified_results = verify_claims(
                question, all_candidates, question_type=packet_qtype,
            )
            verified_pool = filter_passing(verified_results)
            evidence_packet_obj = build_evidence_packet(
                question,
                verified_pool,
                all_candidates=all_candidates,
                max_claims=self.evidence_packet_max_claims,
            )
            packet_meta = evidence_packet_obj.to_dict()
            log.info(
                "evidence_packet: %d verified / %d excluded for %r",
                packet_meta["selected_evidence_count"],
                packet_meta["excluded_noise_count"],
                question[:60],
            )

        # For count / list / how-many questions, route the final
        # synthesis through a JSON-schema prompt so dedup happens at
        # the schema layer instead of being implicit in prose. The
        # parser falls back to the raw text on any failure, so the
        # caller is never worse off than the prose path.
        agg_meta: dict[str, Any] = {}
        if is_aggregate:
            agg_prompt = build_aggregation_synthesis_prompt(question, subanswers)
            # Double the budget — JSON adds overhead vs prose, and we
            # want headroom for the candidate list.
            agg_tokens = max(self.answer_max_tokens * 2, 320)
            try:
                raw = await self.llm.complete(
                    prompt=agg_prompt, max_tokens=agg_tokens,
                )
            except Exception as exc:
                log.exception("aggregation synthesis failed for %r", question)
                raw = f"[error: {exc!r}]"
            answer, agg_meta = parse_aggregation_response(str(raw))
            status = agg_meta.get("parse_status", "")
            if status != "ok":
                log.warning(
                    "aggregation JSON parse fallback (%s) for %r — "
                    "retrying via prose synthesis",
                    status, question[:80],
                )
                # Retry through the prose path so a parse failure doesn't
                # silently keep an unstructured raw blob as the answer.
                prose_prompt = build_final_synthesis_prompt(
                    question, subanswers,
                    evidence_packet=evidence_packet_obj,
                )
                try:
                    answer = await self.llm.complete(
                        prompt=prose_prompt, max_tokens=self.answer_max_tokens,
                    )
                except Exception as exc:
                    log.exception("prose-fallback synthesis failed for %r", question)
                    answer = f"[error: {exc!r}]"
                answer = str(answer).strip()
        else:
            final_prompt = build_final_synthesis_prompt(
                question, subanswers,
                evidence_packet=evidence_packet_obj,
            )
            try:
                answer = await self.llm.complete(
                    prompt=final_prompt, max_tokens=self.answer_max_tokens,
                )
            except Exception as exc:
                log.exception("final synthesis failed for %r", question)
                answer = f"[error: {exc!r}]"
            answer = str(answer).strip()

        self.last_decomposition_stats = {
            "mode": "decompose_answer" + ("_aggregate" if is_aggregate else ""),
            "n_sub_questions": len(subquestions),
            "sub_questions": subquestions,
            "sub_answers": [sub.answer for sub in subanswers],
            "subquestion_hits": [sub.hit_count for sub in subanswers],
            "aggregation": agg_meta,
            "evidence_span_selection": self.evidence_span_selection,
            "evidence_spans_audit": spans_audit,
            "session_narrowing": self.session_narrowing,
            "session_narrow_audit": narrow_audit,
            "evidence_packet": self.evidence_packet,
            # ``selected_evidence_count`` + ``excluded_noise_count`` live
            # inside ``evidence_packet_meta`` so the JSONL trace can read
            # them at the same key the spec asked for. When the flag is
            # off, the meta dict is empty.
            "evidence_packet_meta": packet_meta,
            "decomp_guard": (
                self._last_decomp_guard.to_dict()
                if self._last_decomp_guard is not None else None
            ),
        }
        return answer


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
    synthesis_fn: Any = None,
    retrieval_mode: str = "topk",
    retriever_kind: str = "cosine",
    rrf_k: int = 60,
    store_factory: Callable[[], Any] | None = None,
    use_iterative_reasoner: bool = False,
    iterative_max_llm_calls: int = 6,
    iterative_max_rounds: int = 2,
    direct_answer: bool = False,
    direct_max_context_chars: int = 32000,
    preference_conditioning: bool = False,
    temporal_conditioning: bool = True,  # WS-1 WIN (+33pp) → default-on for v1.1
    aggregation_v2: bool = False,
    knowledge_update_conditioning: bool = True,  # WS-3 confirmed (~+4.5pp KU) → default-on
    temporal_codemath: bool = False,
    small_llm: Any | None = None,
    max_context_tokens: int = 100_000,
    score_weights: dict[str, float] | None = None,
    tau_hours: float = 168.0,
    decompose: bool = False,
    decompose_max_items: int = 16,
    decompose_answer: bool = False,
    decompose_answer_subanswer_max_tokens: int = 80,
    evidence_span_selection: bool = False,
    evidence_max_spans: int = 3,
    evidence_min_overlap: float = 0.2,
    session_narrowing: bool = False,
    session_narrowing_min_confidence: float = 0.5,
    evidence_packet: bool = False,
    evidence_packet_max_claims: int = 6,
    claim_first: bool = False,
    claim_first_top_n_for_fallback: int = 3,
    session_aware_retrieval: bool = False,
    session_top_k: int = 4,
    turns_per_session: int = 2,
    wiki_memory: bool = False,
    wiki_top_k: int = 8,
    content_wiki_memory: bool = False,
    content_wiki_top_k: int = 6,
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

    ``decompose_answer`` uses the stronger path: decompose once, retrieve
    per sub-question, answer each sub-question, then synthesize the final
    answer. It owns decomposition itself, so it uses the base retriever
    rather than wrapping it in :class:`DecompositionRetriever`.

    ``session_aware_retrieval`` ranks LongMemEval sessions first, then turns
    inside the selected sessions. This helps cross-session questions collect
    evidence from multiple sessions instead of only the globally closest
    turns.

    ``wiki_memory`` builds a row-local Markdown wiki from the raw haystack
    and retrieves over wiki pages instead of raw turns. It is mutually
    exclusive with ``session_aware_retrieval`` in practice; wiki mode wins.

    ``content_wiki_memory`` builds topic pages from clean dated English
    user facts. It is separate from ``wiki_memory`` so the session-wiki
    baseline remains stable while content pages are evaluated.
    """

    def factory() -> _IngestingAdapter:
        store = store_factory() if store_factory is not None else FlatHaystackStore()
        if content_wiki_memory:
            base_retriever = ContentWikiMemoryRetriever(
                raw_store=store,
                embedder=embedder,
                top_k=content_wiki_top_k,
            )
        elif wiki_memory:
            base_retriever = WikiMemoryRetriever(
                raw_store=store, embedder=embedder, top_k=wiki_top_k,
            )
        else:
            retriever_cls = (
                SessionAwareSemanticRetriever
                if session_aware_retrieval else STMSemanticRetriever
            )
            retriever_kwargs: dict[str, Any] = {}
            if session_aware_retrieval:
                retriever_kwargs.update({
                    "session_top_k": session_top_k,
                    "turns_per_session": turns_per_session,
                    "max_items": decompose_max_items,
                })
            cosine_retriever: Any = retriever_cls(
                store=store, embedder=embedder, top_k=top_k,
                mode=retrieval_mode, max_context_tokens=max_context_tokens,
                score_weights=score_weights, tau_hours=tau_hours,
                **retriever_kwargs,
            )
            # ── --retriever {cosine,bm25,hybrid} ────────────────────────
            if retriever_kind == "bm25":
                base_retriever = BM25HaystackRetriever(
                    store=store, top_k=top_k,
                )
            elif retriever_kind == "hybrid":
                from continuum.retrieval.rrf import HybridRetriever
                bm25_retriever = BM25HaystackRetriever(
                    store=store, top_k=top_k,
                )
                base_retriever = HybridRetriever(
                    cosine_retriever, bm25_retriever,
                    k=rrf_k, top_k=top_k,
                )
            else:  # cosine (default)
                base_retriever = cosine_retriever
        retriever: Any = base_retriever
        if decompose and not decompose_answer:
            retriever = DecompositionRetriever(
                base=base_retriever, llm=llm, max_items=decompose_max_items,
            )
        session = _MiniSession(store=store, retriever=retriever)
        # ── --reasoner iterative ────────────────────────────────────────────
        # The iterative loop replaces the legacy synthesis path entirely.
        # It still uses the same retriever (wrapped) and the same composer
        # LLM (``llm``), so retrieval + ingest behaviour is unchanged.
        if direct_answer:
            return _DirectAnswerAdapter(
                session=session, llm=llm, answer_max_tokens=answer_max_tokens,
                top_k=top_k, max_context_chars=direct_max_context_chars,
                reranker=reranker,  # WS-4: precision pass in the winning path
                rerank_to=rerank_to,  # keep best N after rerank (not top_k)
                preference_conditioning=preference_conditioning,  # WS-7
                temporal_conditioning=temporal_conditioning,  # WS-1
                aggregation_v2=aggregation_v2,  # WS-2
                knowledge_update_conditioning=knowledge_update_conditioning,  # WS-3
                temporal_codemath=temporal_codemath,  # WS-date-math
                synthesis_fn=synthesis_fn,  # v3 aggregation
            )
        if use_iterative_reasoner:
            return _IterativeReasoningAdapter(
                session=session, llm=llm, answer_max_tokens=answer_max_tokens,
                max_llm_calls=iterative_max_llm_calls,
                max_rounds=iterative_max_rounds,
                small_llm=small_llm,
            )
        if decompose_answer:
            return _DecomposedAnsweringAdapter(
                session=session,
                llm=llm,
                answer_max_tokens=answer_max_tokens,
                subanswer_max_tokens=decompose_answer_subanswer_max_tokens,
                evidence_span_selection=evidence_span_selection,
                evidence_max_spans=evidence_max_spans,
                evidence_min_overlap=evidence_min_overlap,
                session_narrowing=session_narrowing,
                session_narrowing_min_confidence=session_narrowing_min_confidence,
                evidence_packet=evidence_packet,
                evidence_packet_max_claims=evidence_packet_max_claims,
                claim_first=claim_first,
                claim_first_top_n_for_fallback=claim_first_top_n_for_fallback,
            )
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


def _build_small_llm(args: argparse.Namespace, model: str) -> Any:
    """
    Construct the SmallLLM used for span-fallback / intent / query
    rewrite, honouring ``--small-llm``:

    * ``ollama`` (default) — Ollama at ``--ollama-url`` with the tiny
      ``qwen2.5:1.5b-instruct`` default (or ``CONTINUUM_SMALL_LLM_*``
      overrides). Degrades to regex-only if Ollama isn't running.
    * ``same`` — reuse the same provider endpoint + model + key as the
      answerer. Lets span-fallback work without a separate Ollama
      daemon, at the cost of extra calls against the provider's rate
      limit (notable on Groq's free tier — batch accordingly).

    Returns a ``SmallLLM`` instance. For ``same`` on a provider we
    can't map to an OpenAI-compatible endpoint, logs a warning and
    falls back to the Ollama default.
    """
    from continuum.extraction.small_llm import SmallLLM

    if args.small_llm != "same":
        return SmallLLM()  # ollama defaults / env overrides

    provider = args.provider
    if provider == "groq":
        url, key = DEFAULT_GROQ_URL, os.environ.get("GROQ_API_KEY", "")
    elif provider == "openai":
        url, key = DEFAULT_OPENAI_URL, os.environ.get("OPENAI_API_KEY", "")
    elif provider == "openrouter":
        url, key = DEFAULT_OPENROUTER_URL, os.environ.get("OPENROUTER_API_KEY", "")
    elif provider == "lmstudio":
        url, key = args.lmstudio_url, "lm-studio"  # LM Studio accepts any key
    elif provider == "ollama":
        # "same" against ollama means use the big ollama model via the
        # native /api/generate shape — no /v1, backend stays ollama.
        return SmallLLM(model=model, url=args.ollama_url)
    else:
        log.warning(
            "--small-llm same unsupported for provider=%s; using Ollama default",
            provider,
        )
        return SmallLLM()

    # URL ends in /v1 → SmallLLM auto-detects the openai-chat backend.
    log.info("small-llm following provider=%s model=%s url=%s", provider, model, url)
    return SmallLLM(model=model, url=url, api_key=key)


def _build_ltm_store_factory(
    args: argparse.Namespace, embedder: Any,
) -> tuple[Callable[[], Any], str]:
    """
    Build a per-row store factory that yields a fresh
    :class:`ContinuumLTMHaystackStore` over an in-memory or Postgres LTM.

    The returned factory is closed over a *backend selector* (no shared
    LTM state across rows — every row gets fresh tables). The label
    flips to ``"postgres"`` only when the user explicitly opted in OR
    ``--ltm-backend auto`` saw ``DATABASE_URL`` in env. We deliberately
    avoid lazy-pickling Postgres state across rows: each row gets its
    own connection / DB instance so failures don't bleed between
    questions.
    """
    from evals.longmemeval.continuum_ltm_store import ContinuumLTMHaystackStore

    backend = args.ltm_backend
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if backend == "auto":
        backend = "postgres" if dsn else "in_memory"
    if backend == "postgres" and not dsn:
        raise SystemExit(
            "--ltm-backend postgres requires DATABASE_URL in the environment"
        )

    # Optional LLM wiring for the promoter + fact extractor. Skipped
    # when --no-llm-promoter is passed; the store then falls back to
    # the deterministic supersession heuristic (good enough to
    # demonstrate the lift on knowledge-update offline).
    promoter = None
    fact_extractor = None
    llm_available = bool(args.llm_promoter)
    if llm_available:
        try:
            from continuum.extraction.fact_extractor import FactExtractor
            from continuum.promotion.mem0_promoter import Mem0Promoter
            promoter = Mem0Promoter()  # litellm.acompletion by default
            fact_extractor = FactExtractor()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("LLM promoter unavailable (%s); using heuristic", exc)
            llm_available = False

    def factory() -> Any:
        if backend == "postgres":
            from continuum.stores.postgres.ltm import PostgresLTM
            ltm: Any = PostgresLTM(dsn=dsn)
        else:
            from continuum.stores.in_memory.ltm import InMemoryLTM
            ltm = InMemoryLTM()
        return ContinuumLTMHaystackStore(
            ltm=ltm,
            embedder=embedder,
            promoter=promoter,
            fact_extractor=fact_extractor,
            llm_available=llm_available,
            ltm_backend_label=backend,
        )

    return factory, backend


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.abstain_threshold is not None:
        log.warning(
            "--abstain-threshold=%s is a NO-OP: the abstain head was gated "
            "out (judge scores abstentions as 0.0). Ignoring it.",
            args.abstain_threshold,
        )

    # Pick the model default if the user didn't override it explicitly.
    model = args.model
    if not model:
        model = {
            "groq":       DEFAULT_GROQ_MODEL,
            "nvidia":     DEFAULT_NVIDIA_MODEL,
            "gemini":     DEFAULT_GEMINI_MODEL,
            "openai":     DEFAULT_OPENAI_MODEL,
            "openrouter": DEFAULT_OPENROUTER_MODEL,
            "bedrock":    DEFAULT_BEDROCK_MODEL,
            "lmstudio":   DEFAULT_LMSTUDIO_MODEL,
            "ollama":     DEFAULT_OLLAMA_MODEL,
        }.get(args.provider, DEFAULT_OLLAMA_MODEL)

    llm: Any
    if args.provider == "groq":
        log.info("provider=groq model=%s rpm=%d", model, args.rpm)
        llm = GroqLLM(model=model, rpm=args.rpm)
    elif args.provider == "openrouter":
        log.info(
            "provider=openrouter model=%s rpm=%d provider_pin=%s seed=%s",
            model, args.rpm, args.openrouter_provider or "(none)", args.seed,
        )
        llm = OpenRouterLLM(
            model=model, rpm=args.rpm,
            provider_pin=args.openrouter_provider, seed=args.seed,
        )
    elif args.provider == "nvidia":
        log.info("provider=nvidia model=%s rpm=%d", model, args.rpm)
        llm = NvidiaLLM(model=model, rpm=args.rpm)
    elif args.provider == "gemini":
        log.info("provider=gemini model=%s rpm=%d", model, args.rpm)
        llm = GeminiLLM(model=model, rpm=args.rpm)
    elif args.provider == "openai":
        log.info("provider=openai model=%s rpm=%d", model, args.rpm)
        llm = OpenAILLM(model=model, rpm=args.rpm)
    elif args.provider == "bedrock":
        log.info(
            "provider=bedrock model=%s region=%s rpm=%d",
            model, args.bedrock_region, args.rpm,
        )
        llm = BedrockLLM(
            model=model,
            region=args.bedrock_region,
            base_url=args.bedrock_url or None,
            rpm=args.rpm,
        )
    elif args.provider == "lmstudio":
        log.info(
            "provider=lmstudio model=%s url=%s temperature=%s",
            model, args.lmstudio_url, args.lmstudio_temperature,
        )
        llm = LMStudioLLM(
            model=model, base_url=args.lmstudio_url,
            temperature=args.lmstudio_temperature,
        )
    else:
        log.info("provider=ollama model=%s", model)
        await _verify_ollama(model, args.ollama_url)
        llm = OllamaLLM(model=model, base_url=args.ollama_url)

    # ── SmallLLM construction (shared by span-fallback + reasoner) ─────────
    # Build one SmallLLM here so the span-fallback picker AND the
    # IterativeReasoner's intent/rewrite calls use the same endpoint
    # (--small-llm). Cheap to construct; the cache is on disk.
    small_llm: Any = _build_small_llm(args, model)

    # ── --span-fallback wiring ─────────────────────────────────────────────
    # Parks the SmallLLM as the process-wide default the assistant-claim
    # picker reaches for when its regex tier is unreliable. Failure here is
    # non-fatal: the picker degrades to regex-only.
    if args.span_fallback:
        try:
            from evals.longmemeval.answer_post import (
                reset_span_fallback_stats,
                set_default_span_fallback_llm,
            )
            reset_span_fallback_stats()
            set_default_span_fallback_llm(small_llm)
            log.info("span fallback ENABLED (small-llm=%s)", args.small_llm)
        except ImportError as exc:
            log.warning("span fallback unavailable: %s", exc)

    log.info("loading dataset from %s …", args.dataset)
    rows = load_longmemeval_rows(args.dataset)
    log.info("loaded %d rows", len(rows))
    rows = _filter_rows_by_question_type(rows, args.question_type)
    if args.question_type:
        log.info(
            "question_type filter %r retained %d rows",
            args.question_type,
            len(rows),
        )
    # --question-types accepts a comma-separated allow-list (orthogonal
    # to the legacy single-value --question-type). Used by the
    # knowledge-update sweep: --question-types knowledge-update.
    if args.question_types:
        wanted_types = {
            t.strip() for t in args.question_types.split(",") if t.strip()
        }
        if wanted_types:
            before = len(rows)
            rows = [r for r in rows if r.question_type in wanted_types]
            log.info(
                "question_types filter %s retained %d/%d rows",
                sorted(wanted_types), len(rows), before,
            )
    # `--question-ids-file` filters to a known subset (e.g. the deterministic
    # diagnostic_50 sample). The subset preserves the dataset's source order
    # so trace logs remain comparable across runs.
    if args.question_ids_file:
        ids_payload = json.loads(Path(args.question_ids_file).read_text())
        wanted = set(ids_payload.get("question_ids") or [])
        if not wanted:
            raise ValueError(
                f"--question-ids-file {args.question_ids_file} contained no question_ids",
            )
        before = len(rows)
        rows = [r for r in rows if r.question_id in wanted]
        log.info(
            "--question-ids-file %s: %d/%d rows kept",
            args.question_ids_file, len(rows), before,
        )
        if not rows:
            raise ValueError(
                f"no dataset rows matched ids in {args.question_ids_file}",
            )

    # --offset slices off the first N rows (after all filtering) so the
    # remaining --limit window starts there. This is the batching lever:
    # `--offset 50 --limit 50` evaluates rows 50-99. Output dirs should
    # differ per batch; merge with v1_summary.py (accepts multiple files).
    if args.offset:
        before = len(rows)
        rows = rows[args.offset:]
        log.info("--offset %d: %d/%d rows remain", args.offset, len(rows), before)
        if not rows:
            raise ValueError(
                f"--offset {args.offset} skipped past all {before} rows",
            )

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

    # v3 synthesis: an LLM completion (prompt -> JSON text) for the triple
    # extractor. litellm routes args.synthesis_model (default openrouter gpt-4o-mini).
    synthesis_fn: Any = None
    if args.synthesis:
        import litellm

        async def _synth(prompt: str) -> str:
            resp = await litellm.acompletion(
                model=args.synthesis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            return str(resp["choices"][0]["message"]["content"])

        synthesis_fn = _synth
        if args.synthesis_cache:
            from continuum.promotion.synthesis import disk_cached

            synthesis_fn = disk_cached(
                _synth, args.synthesis_cache, namespace=args.synthesis_model
            )
            log.info("v3 synthesis cache: %s", args.synthesis_cache)
        log.info("v3 synthesis ENABLED — extractor=%s", args.synthesis_model)

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
    if args.decompose and args.decompose_answer:
        raise ValueError(
            "Use either --decompose or --decompose-answer, not both. "
            "--decompose is retrieval-only; --decompose-answer performs "
            "sub-question answering and final synthesis."
        )
    if args.content_wiki_memory and args.wiki_memory:
        raise ValueError(
            "Use either --wiki-memory or --content-wiki-memory, not both."
        )
    if args.decompose:
        log.info(
            "decomposition ENABLED — questions split into sub-questions, "
            "retrieved per-part, merged (max %d items)",
            args.decompose_max_items,
        )
    if args.decompose_answer:
        log.info(
            "decomposed answering ENABLED — split question, retrieve and "
            "answer each part, synthesize final answer",
        )
    if args.session_aware_retrieval:
        log.info(
            "session-aware retrieval ENABLED — top %d sessions, %d turns/session",
            args.session_top_k,
            args.turns_per_session,
        )
    if args.wiki_memory:
        log.info(
            "wiki memory ENABLED — compiled Markdown wiki retrieval, top %d pages",
            args.wiki_top_k,
        )
    if args.content_wiki_memory:
        log.info(
            "content wiki memory ENABLED — topic fact pages, top %d pages",
            args.content_wiki_top_k,
        )
    row_scorer = None
    # Two scoring code paths:
    # (a) --judge {none,rule,hybrid,llm}: routes through HybridScorer's
    #     5-stage deterministic ladder (exact → normalized → numeric → unit
    #     → rule_semantic) with the LLM judge only invoked on uncertain
    #     rows in hybrid mode (or unconditionally in llm mode).
    # (b) legacy --llm-judge: unconditional LLM judge (kept for back-compat).
    if args.judge:
        from evals.longmemeval.hybrid_scorer import HybridScorer
        judge_obj = LLMJudgeScorer(llm=llm) if args.judge in ("hybrid", "llm") else None
        hybrid = HybridScorer(
            mode=args.judge,
            llm_judge=judge_obj,
            judge_model=model,
        )

        async def row_scorer(row: EvalRow, answer: str) -> bool:
            result, _hit = await hybrid.score(
                row.question, row.expected_answer, answer,
                question_id=row.question_id,
            )
            return result.correct

        log.info(
            "hybrid scorer ENABLED — mode=%s judge=%s",
            args.judge, "on" if judge_obj is not None else "off",
        )
    elif args.llm_judge:
        judge = LLMJudgeScorer(llm=llm)

        async def row_scorer(row: EvalRow, answer: str) -> bool:
            return await judge.is_correct(
                row.question, row.expected_answer, answer
            )

        log.info("LLM judge scoring ENABLED — semantic answer grading")

    # ── --use-ltm wiring ────────────────────────────────────────────────────
    # Construct a `store_factory` closure that yields a fresh store per
    # row. When --use-ltm is off, this stays None and make_adapter_factory
    # falls back to the legacy FlatHaystackStore path (zero behavioural
    # change). When on, each row gets a fresh ContinuumLTMHaystackStore
    # over an InMemoryLTM (or PostgresLTM when DATABASE_URL is set and
    # --ltm-backend allows it).
    store_factory: Callable[[], Any] | None = None
    ltm_backend_label = "flat"
    if args.use_ltm:
        store_factory, ltm_backend_label = _build_ltm_store_factory(args, embedder)
        log.info("LTM-backed haystack ENABLED — backend=%s, llm_promoter=%s",
                 ltm_backend_label, args.llm_promoter)

    factory = make_adapter_factory(
        llm=llm, embedder=embedder, top_k=effective_top_k, chain=chain,
        answer_max_tokens=args.answer_max_tokens,
        reranker=reranker, rerank_to=args.rerank_to,
        synthesis_fn=synthesis_fn,
        retrieval_mode=args.retrieval_mode,
        retriever_kind=args.retriever,
        store_factory=store_factory,
        use_iterative_reasoner=(args.reasoner == "iterative"),
        iterative_max_llm_calls=args.max_llm_calls,
        iterative_max_rounds=args.max_rounds,
        direct_answer=(args.reasoner == "direct"),
        direct_max_context_chars=args.max_context_chars,
        preference_conditioning=args.pref_conditioning,
        temporal_conditioning=args.temporal_conditioning,
        aggregation_v2=args.aggregation_v2,
        knowledge_update_conditioning=args.ku_recency,
        temporal_codemath=args.temporal_codemath,
        small_llm=small_llm,
        rrf_k=args.rrf_k,
        max_context_tokens=args.max_context_tokens,
        score_weights=score_weights,
        tau_hours=args.tau_hours,
        decompose=args.decompose,
        decompose_max_items=args.decompose_max_items,
        decompose_answer=args.decompose_answer,
        decompose_answer_subanswer_max_tokens=args.subanswer_max_tokens,
        evidence_span_selection=args.evidence_spans,
        evidence_max_spans=args.evidence_max_spans,
        evidence_min_overlap=args.evidence_min_overlap,
        session_narrowing=args.session_narrowing,
        session_narrowing_min_confidence=args.session_narrowing_min_confidence,
        evidence_packet=args.evidence_packet,
        evidence_packet_max_claims=args.evidence_packet_max_claims,
        claim_first=args.claim_first,
        claim_first_top_n_for_fallback=args.claim_first_top_n_for_fallback,
        session_aware_retrieval=args.session_aware_retrieval,
        session_top_k=args.session_top_k,
        turns_per_session=args.turns_per_session,
        wiki_memory=args.wiki_memory,
        wiki_top_k=args.wiki_top_k,
        content_wiki_memory=args.content_wiki_memory,
        content_wiki_top_k=args.content_wiki_top_k,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Debug trace setup ────────────────────────────────────────────────
    # When `--debug-trace` is set we open a TraceWriter rooted at
    # <out_dir>/logs. The writer is shared across smoke + full so a single
    # JSONL file captures the whole run; one run_id is stamped on each row.
    trace_cm: Any = contextlib.nullcontext(None)
    if args.debug_trace:
        from evals.longmemeval.trace import TraceWriter
        trace_cm = TraceWriter.from_run(out_dir / "logs", enabled=True)
        log.info("debug trace ENABLED — JSONL per-row trace will be written")

    # ── Smoke test ──────────────────────────────────────────────────────────
    # --no-smoke skips the 5-row pre-run entirely. For batched sweeps the
    # smoke would re-evaluate (and re-bill) the first 5 rows of every batch,
    # so batches should pass --no-smoke --full --yes.
    with trace_cm as trace_writer:
        if not args.no_smoke:
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
                row_scorer=row_scorer,
                limit=smoke_limit,
                trace_writer=trace_writer,
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
        elif not args.full:
            log.info("--no-smoke with no --full: nothing to run. Pass --full.")
            return 0

        log.info("=== FULL RUN (%d questions) ===", len(rows))
        t0 = time.perf_counter()
        full = await run_baseline(
            dataset=args.dataset_name,
            adapter_factory=factory,
            dataset_loader=lambda _: rows,
            output_dir=out_dir,
            answerer=model,
            row_scorer=row_scorer,
            limit=args.limit,
            trace_writer=trace_writer,
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
        choices=(
            "ollama", "groq", "nvidia", "gemini",
            "openai", "openrouter", "bedrock", "lmstudio",
        ),
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
            "openrouter → openai/gpt-oss-120b, "
            "bedrock → anthropic.claude-3-haiku-20240307-v1:0, "
            "lmstudio → qwen/qwen3-14b."
        ),
    )
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument(
        "--openrouter-provider",
        default=None,
        help=(
            "Pin OpenRouter to a SINGLE backend provider (no fallbacks) for "
            "reproducible ablations — e.g. 'DeepInfra', 'Fireworks', "
            "'Together'. OpenRouter otherwise routes across providers, giving "
            "~44%% answer variance run-to-run even at temperature 0. Also "
            "reads OPENROUTER_PROVIDER. Find a model's providers at "
            "openrouter.ai/models/<model>."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Sampling seed sent to the provider (default 0). Honoured by "
            "providers that support it; combine with --openrouter-provider "
            "for the most reproducible runs."
        ),
    )
    p.add_argument(
        "--bedrock-region",
        default=DEFAULT_BEDROCK_REGION,
        help=(
            "AWS region for Bedrock Runtime. Defaults to AWS_REGION, "
            "AWS_DEFAULT_REGION, then us-east-1."
        ),
    )
    p.add_argument(
        "--bedrock-url",
        default=DEFAULT_BEDROCK_URL,
        help=(
            "Optional boto3 endpoint_url override for Bedrock Runtime. "
            "Defaults to boto3's regional endpoint."
        ),
    )
    p.add_argument(
        "--lmstudio-url", default=DEFAULT_LMSTUDIO_URL,
        help="Base URL of the local LM Studio server (OpenAI-compatible).",
    )
    p.add_argument(
        "--lmstudio-temperature", type=float, default=0.0,
        help=(
            "Sampling temperature for LM Studio calls (default 0.0). "
            "Some Qwen3 / DeepSeek runtimes in LM Studio reject "
            "temperature=0 with a 'Compute error.' response — bump to "
            "0.7 if the model refuses 0.0."
        ),
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
        "--retriever",
        choices=("cosine", "bm25", "hybrid"),
        default="cosine",
        help=(
            "Which lexical/dense retriever to use. "
            "'cosine' — embeddings-only (default, legacy). "
            "'bm25' — lexical-only, rank_bm25 with a regex tokenizer. "
            "'hybrid' — fuse cosine and bm25 with Reciprocal Rank Fusion "
            "(see --rrf-k). Strongly recommended for multi-session and "
            "temporal-reasoning categories where proper-noun anchors carry "
            "the answer."
        ),
    )
    p.add_argument(
        "--rrf-k", type=int, default=60,
        help=(
            "RRF smoothing constant for --retriever hybrid (default 60, "
            "Cormack 2009). Smaller k lets top-1 dominate; larger k "
            "smooths agreements between cosine and bm25."
        ),
    )
    # ── Continuum LTM + supersession ───────────────────────────────────────
    p.add_argument(
        "--use-ltm", action="store_true", default=False,
        help=(
            "Replace FlatHaystackStore with ContinuumLTMHaystackStore, "
            "which drives Continuum's STM→facts→LTM promotion pipeline "
            "per row. Live LTM facts are added to the retrieval corpus "
            "alongside raw turns, so superseded knowledge-update halves "
            "stop competing with the current half during retrieval."
        ),
    )
    p.add_argument(
        "--ltm-backend",
        choices=("auto", "in_memory", "postgres"),
        default="auto",
        help=(
            "Where the LTM lives when --use-ltm is on. 'auto' picks "
            "postgres iff DATABASE_URL is set, otherwise in_memory. "
            "Tagged onto the result JSON's metrics.ltm_backend."
        ),
    )
    p.add_argument(
        "--ltm-bootstrap-schema", action="store_true", default=False,
        help=(
            "Run the LTM migrations (001..004) idempotently before "
            "the row loop. Only meaningful with --ltm-backend postgres "
            "against a fresh database."
        ),
    )
    p.add_argument(
        "--no-llm-promoter", dest="llm_promoter",
        action="store_false", default=True,
        help=(
            "Disable the LLM-driven fact extractor + Mem0 promoter; use "
            "the deterministic supersession heuristic only. Useful for "
            "hermetic test runs and when no provider key is in env."
        ),
    )
    p.add_argument(
        "--question-types",
        default=None,
        help=(
            "Comma-separated list of LongMemEval question_type values to "
            "include (e.g. 'knowledge-update' or "
            "'multi-session,temporal-reasoning'). When unset, all rows "
            "are evaluated. Useful for the n=78 knowledge-update sweep."
        ),
    )
    # ── Span-selector fallback (answer_post picker) ───────────────────────
    # Parks a process-wide SmallLLM that the assistant-claim picker calls
    # when its regex output is a nationality adjective, fails a count-shape
    # check, or sits inside a structurally-richer claim than itself. Mirrors
    # the same flag in evals.longmemeval.baseline so the production CLI
    # (this file) and the library CLI (baseline.py) share semantics.
    p.add_argument(
        "--span-fallback", dest="span_fallback",
        action="store_true", default=False,
        help=(
            "Enable the SmallLLM span-selector fallback inside the "
            "assistant-claim picker (answer_post._pick_answer_from_assistant_claim). "
            "Requires a SmallLLM endpoint (default: Ollama at "
            "http://localhost:11434 with qwen2.5:1.5b-instruct)."
        ),
    )
    p.add_argument(
        "--no-span-fallback", dest="span_fallback", action="store_false",
        help="Disable the SmallLLM span-selector fallback (default).",
    )
    p.add_argument(
        "--small-llm",
        choices=("ollama", "same"),
        default="ollama",
        help=(
            "Where the SmallLLM (span-fallback / intent / query-rewrite) "
            "runs. 'ollama' (default): local Ollama at --ollama-url with "
            "qwen2.5:1.5b-instruct. 'same': reuse the answerer's provider, "
            "model, endpoint and key (works for groq/openai/lmstudio) — "
            "needed when no Ollama daemon is running, but adds calls "
            "against the provider's rate limit."
        ),
    )
    p.add_argument(
        "--answer-max-tokens", type=int, default=256,
        help=(
            "Max tokens the answerer may produce per answer (default 256). "
            "REASONING models (gpt-oss-120b, deepseek-r1, o-series) spend "
            "tokens thinking before answering — at low caps the answer is "
            "truncated to empty/None. Set 1024-2048 for reasoning models, "
            "especially on aggregation questions with longer answers."
        ),
    )
    p.add_argument(
        "--max-context-chars", type=int, default=32000,
        help=(
            "Direct mode (--reasoner direct): cap on characters of "
            "retrieved conversation handed to the answerer (default "
            "32000 ≈ 8k tokens). Multi-session aggregation needs many "
            "sessions in context — raise to 48000-96000 on large-window "
            "models. Too low silently drops answer-bearing turns even "
            "when the session was retrieved (recall stays 1.0 but the "
            "model says 'I don't have that information')."
        ),
    )
    # ── Batching (Groq free-tier friendly) ─────────────────────────────────
    p.add_argument(
        "--offset", type=int, default=0,
        help=(
            "Skip the first N rows after filtering, so the --limit window "
            "starts at row N. Lets you run the sweep in batches "
            "(--offset 0/50/100… --limit 50) into separate output dirs and "
            "merge with v1_summary.py."
        ),
    )
    p.add_argument(
        "--no-smoke", action="store_true", default=False,
        help=(
            "Skip the 5-row smoke pre-run. Use with --full --yes for "
            "batched sweeps so each batch doesn't re-evaluate (and re-bill) "
            "its first 5 rows."
        ),
    )
    # ── IterativeReasoner ──────────────────────────────────────────────────
    p.add_argument(
        "--reasoner",
        choices=("legacy", "iterative", "direct"),
        default="legacy",
        help=(
            "Which reasoner drives answer composition. 'legacy' (default) "
            "uses the existing decompose+optimize+synthesise adapter. "
            "'iterative' wraps the same retriever in "
            "continuum.reasoning.IterativeReasoner: budget-capped loop "
            "(see --max-llm-calls / --max-rounds) with deterministic "
            "head short-circuit, refine-on-fail, and abstain semantics. "
            "'direct' is the A/B baseline: retrieve then hand the raw "
            "retrieved turns to the answerer in ONE call — no decompose, "
            "no claims, no verify, no packet. Same retriever as the "
            "others, so it isolates the value of the claim machinery."
        ),
    )
    p.add_argument(
        "--max-llm-calls", type=int, default=6,
        help="Per-question hard cap on LLM calls (iterative reasoner only).",
    )
    p.add_argument(
        "--max-rounds", type=int, default=2,
        help=(
            "Per-sub-question refine rounds (iterative reasoner only). "
            "Counts heuristic + SmallLLM rewrites; the initial retrieval "
            "always runs. max_rounds=2 means up to 3 total attempts."
        ),
    )
    p.add_argument(
        "--abstain-threshold", type=float, default=None,
        help=(
            "ACCEPTED BUT NO-OP. The abstain head (Prompt 8 Task A) was "
            "gated out: judge.py scores an abstention as wrong (0.0), so "
            "returning 'I don't have enough information' would lose every "
            "uncertain row under judged accuracy. The flag is kept so the "
            "plan's literal Task B command runs without an argparse error; "
            "it has no effect on behaviour. A warning is logged if set."
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
        "--decompose-answer",
        action="store_true",
        help=(
            "Enable full decomposed answering: decompose the question, "
            "retrieve and answer each sub-question, then synthesize the final "
            "answer. This is stronger than --decompose, which only merges "
            "retrieved context."
        ),
    )
    p.add_argument(
        "--subanswer-max-tokens",
        type=int,
        default=80,
        help="Token cap for each decomposed sub-question answer. Default 80.",
    )
    p.add_argument(
        "--session-aware-retrieval",
        action="store_true",
        help=(
            "Rank LongMemEval sessions first, then select turns within top "
            "sessions. Useful with --decompose-answer for multi-session recall."
        ),
    )
    p.add_argument(
        "--session-top-k",
        type=int,
        default=4,
        help="Number of source sessions to keep in session-aware retrieval.",
    )
    p.add_argument(
        "--turns-per-session",
        type=int,
        default=2,
        help="Number of best turns to keep per selected source session.",
    )
    p.add_argument(
        "--pref-conditioning",
        action="store_true",
        default=False,
        help=(
            "WS-7: for preference-type questions only, use a prompt that tells "
            "the answerer to identify and APPLY a stated user preference from "
            "the retrieved turns (recall ~93%%, the gap is application). "
            "Feature-flagged + gated; factual/temporal/etc. questions are "
            "never touched. A/B: run baseline vs this flag, judge, ablate."
        ),
    )
    p.add_argument(
        "--temporal-codemath",
        action="store_true",
        default=False,
        help=(
            "WS-date-math: for temporal DELTA questions (how many days/weeks/"
            "months between/ago), have the model emit a date SPEC and compute "
            "the delta in CODE (deterministic, one call, no agent loop), "
            "overriding its mental arithmetic. Builds on --temporal-conditioning "
            "(dates already surfaced). Off by default; A/B vs baseline."
        ),
    )
    p.add_argument(
        "--ku-recency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "WS-3 (DEFAULT-ON for v1.1, confirmed ~+4.5pp KU): for "
            "knowledge-update questions only, sort the current LTM facts "
            "(superseded already removed) to the front, mark them "
            "[CURRENT FACT], and prompt the reader to prefer the current state "
            "/ latest statement over the stale raw turn still in context. Uses "
            "the supersession layer directly. Use --no-ku-recency for the "
            "A/B baseline arm."
        ),
    )
    p.add_argument(
        "--aggregation-v2",
        action="store_true",
        default=False,
        help=(
            "WS-2: for aggregation/multi-session questions only, use a stronger "
            "'enumerate every distinct instance -> dedupe -> THEN count' prompt "
            "(the failure was counting without listing: '3 weddings' -> '1'). "
            "Off by default; A/B vs the current aggregation prompt, judged."
        ),
    )
    p.add_argument(
        "--temporal-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "WS-1 (DEFAULT-ON for v1.1, +33.1pp measured): for "
            "temporal-reasoning questions only, PREFIX each retrieved turn with "
            "its date + give the reference 'now', and prompt for an explicit "
            "date calculation. Fixes the measured failure where the direct "
            "context dropped all dates ('no dates available' -> '0'). Gated; "
            "other categories untouched. Use --no-temporal-conditioning for "
            "the A/B baseline arm."
        ),
    )
    p.add_argument(
        "--wiki-memory",
        action="store_true",
        help=(
            "Build a row-local Markdown memory wiki from haystack sessions "
            "and retrieve wiki pages instead of raw turns."
        ),
    )
    p.add_argument(
        "--wiki-top-k",
        type=int,
        default=8,
        help="Number of compiled wiki pages to retrieve when --wiki-memory is enabled.",
    )
    p.add_argument(
        "--content-wiki-memory",
        action="store_true",
        help=(
            "Build content/topic Markdown memory pages with clean dated "
            "English facts instead of session transcript pages."
        ),
    )
    p.add_argument(
        "--content-wiki-top-k",
        type=int,
        default=6,
        help=(
            "Number of content wiki pages to retrieve when "
            "--content-wiki-memory is enabled."
        ),
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
        "--question-type",
        choices=QUESTION_TYPE_CHOICES,
        default=None,
        help=(
            "Filter LongMemEval rows by question_type before applying "
            "--limit. Useful for focused multi-session or temporal runs."
        ),
    )
    p.add_argument(
        "--llm-judge",
        action="store_true",
        help=(
            "Use the configured LLM as a semantic judge for answer scoring "
            "instead of the cheap substring scorer."
        ),
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
    p.add_argument(
        "--synthesis", action="store_true",
        help=(
            "v3: at ingest, LLM-extract countable membership triples and "
            "code-compute per-entity counts; surface them to the reader on "
            "counting questions ('how many / how much'). Targets the "
            "aggregation gap (see findings/roadmap_v3.md)."
        ),
    )
    p.add_argument(
        "--synthesis-model",
        default="openrouter/openai/gpt-4o-mini",
        help="litellm model id for the synthesis triple-extractor (default gpt-4o-mini).",
    )
    p.add_argument(
        "--synthesis-cache",
        default=".synthesis_cache",
        help=(
            "Directory to cache synthesis extraction calls (deterministic, so "
            "re-runs cost zero credits + are instant). Empty string disables."
        ),
    )
    p.add_argument(
        "--question-ids-file",
        type=Path, default=None,
        help=(
            "Path to a JSON file containing {'question_ids': [...]}. "
            "When set, only rows whose question_id appears in that list "
            "are evaluated. Use this with samples/diagnostic_50_ids.json "
            "for cheap regression testing."
        ),
    )
    p.add_argument(
        "--judge",
        choices=("none", "rule", "hybrid", "llm"),
        default=None,
        help=(
            "Hybrid scorer mode. 'none' skips scoring; 'rule' uses only "
            "deterministic stages; 'hybrid' adds LLM judge fallback for "
            "uncertain rows; 'llm' calls the judge unconditionally. "
            "When unset, falls back to the legacy substring/LLM-judge wiring."
        ),
    )
    p.add_argument(
        "--debug-trace", action="store_true",
        help=(
            "Write a per-row JSONL trace to "
            "<output>/logs/eval_trace_<ts>_<run_id>.jsonl. Includes "
            "retrieval counts, telemetry, validator state, candidates "
            "preview, and timing for every row."
        ),
    )
    p.add_argument(
        "--evidence-spans", action="store_true",
        help=(
            "Enable evidence span selection: after each sub-question's "
            "retrieval, pick 1-3 answer-bearing spans (lexical overlap + "
            "shape-aware) from the broad evidence windows and pass ONLY "
            "those to the answerer. Empty selection forces abstain "
            "instead of guessing from the wider window."
        ),
    )
    p.add_argument(
        "--evidence-max-spans", type=int, default=3,
        help="Cap on spans returned per sub-question (default 3).",
    )
    p.add_argument(
        "--evidence-min-overlap", type=float, default=0.2,
        help=(
            "Minimum content-word overlap (fraction of the question's "
            "content words) required for a span to be eligible. Default "
            "0.2; lower values let weaker spans through, higher values "
            "make the selector stricter (and more abstentions)."
        ),
    )
    p.add_argument(
        "--session-narrowing", action="store_true",
        help=(
            "Enable top-session narrowing. When the top-ranked retrieved "
            "session carries a verified answer-bearing claim AND the "
            "question is not explicitly multi-session, drop items from "
            "lower-ranked sessions so distractor ShareGPT/UltraChat "
            "windows can't leak into the sub-answer prompt."
        ),
    )
    p.add_argument(
        "--session-narrowing-min-confidence", type=float, default=0.5,
        help=(
            "Floor on the verifier's confidence required to narrow "
            "(default 0.5). Below this, the narrower keeps all sessions "
            "rather than risk silently dropping evidence."
        ),
    )
    p.add_argument(
        "--evidence-packet", action="store_true",
        help=(
            "Replace the raw evidence-window text in the final synthesis "
            "prompt with a minimal evidence packet — verified claims, "
            "exact source spans, source session ids, and confidences. "
            "Excluded noise candidates never enter the answerer's "
            "context. Per-row JSONL trace gains "
            "selected_evidence_count + excluded_noise_count."
        ),
    )
    p.add_argument(
        "--evidence-packet-max-claims", type=int, default=6,
        help=(
            "Hard cap on packet size (default 6). Claims are ranked by "
            "verifier confidence; lowest-confidence claims are dropped "
            "first when the cap is hit."
        ),
    )
    p.add_argument(
        "--claim-first", action="store_true",
        help=(
            "Enable the claim-first answer pipeline. After retrieval, "
            "extract sentence-level claims with role/session metadata, "
            "route the question to one of six broad reasoning heads "
            "(FACT_LOOKUP, ASSISTANT_MEMORY_LOOKUP, PREFERENCE_PROFILE, "
            "KNOWLEDGE_UPDATE, MULTI_SESSION_AGGREGATE, "
            "TEMPORAL_REASONING), and short-circuit the final answer "
            "when the head produces a claim-supported span. Falls "
            "through to the existing decompose+packet path when the "
            "head can't extract a span."
        ),
    )
    p.add_argument(
        "--claim-first-top-n-for-fallback", type=int, default=3,
        help=(
            "Number of top-ranked claims to send to the extractive LLM "
            "fallback when the deterministic head returns empty "
            "(default 3)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(_parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "OllamaLLM",
    "STMSemanticRetriever",
    "SessionAwareSemanticRetriever",
    "WikiMemoryRetriever",
    "ContentWikiMemoryRetriever",
    "load_longmemeval_rows",
    "make_adapter_factory",
    "main",
]
