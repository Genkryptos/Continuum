"""
evals/longmemeval/openrouter_completion.py
==========================================
OpenRouter-backed ``completion_fn`` for the ingest-side LLM calls, replacing
litellm in the ``--use-ltm`` path.

Why
---
litellm cost this project a full debugging session for no benefit:

* **Three independent routing configs.** The answerer used ``--provider`` /
  ``--model``; :class:`FactExtractor` used ``CONTINUUM_FACT_MODEL``;
  :class:`ContinuumPromoter` used ``CONTINUUM_PROMOTER_LLM_MODEL``. Each defaulted to
  a bare ``gpt-4o-mini`` that resolves to *OpenAI*, not the OpenRouter account
  actually in use, so each failed separately and had to be found separately.
* **Three different silent failure modes.** The extractor raised and was
  swallowed into "no facts"; the store degraded to regex mining; the promoter
  returned ``NOOP`` — indistinguishable from a legitimate "no contradiction".
  Zero supersessions fired and the run still reported a clean accuracy number.
* **Log noise** that buried the real errors, and a bare-``MODEL`` env collision
  that silently overrode the prefixed setting.

OpenRouter is OpenAI-compatible, so the stock ``openai`` SDK returns objects
that :meth:`FactExtractor._content` and :meth:`ContinuumPromoter._parse_tool_calls`
already parse (both read ``resp.choices[0].message``). No adapter, no library
surgery — just the injection point both classes already expose.

Design
------
**The model is forced here, not read from per-component config.** That is the
whole point: one flag decides what the ingest side runs, so the three-way
routing split cannot reappear. The ``model=`` kwarg the callers pass is
deliberately ignored.

Retry and timeout stay with the callers (``FactExtractor`` wraps calls in
tenacity with a per-attempt timeout), so this layer is a thin passthrough.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Small, cheap, reliable at structured output (JSON mode + tool calls). Kept as
#: the default because it is what every prior ``--use-ltm`` run effectively used,
#: so switching transport does not also change the measurement.
DEFAULT_INGEST_MODEL = "openai/gpt-4o-mini"

#: Passed through unchanged — these are the kwargs the two call sites send.
#: ``model`` is intentionally absent: it is overridden.
_PASSTHROUGH = (
    "messages",
    "temperature",
    "max_tokens",
    "response_format",
    "tools",
    "tool_choice",
)


class MissingOpenRouterKey(RuntimeError):
    """Raised at construction rather than on the first call, so the run fails
    before spending time on ingest it cannot complete."""


def make_completion_fn(
    *,
    model: str = DEFAULT_INGEST_MODEL,
    api_key: str | None = None,
    base_url: str = OPENROUTER_BASE_URL,
    timeout: float = 120.0,
    rpm: int = 25,
    max_retries: int = 5,
) -> Any:
    """
    Build an async ``completion_fn`` for ``FactExtractor`` / ``ContinuumPromoter``.

    Both call it with keyword arguments and expect an OpenAI-shaped response.
    Returns the SDK object directly — the callers do their own parsing.

    Rate limiting
    -------------
    Calls are admitted through the same AIMD throttle the answerer uses
    (``_AdaptiveThrottle``): a client-side RPM cap that *widens* on 429 and
    edges back after a clean streak. **This is what makes concurrent ingest
    safe** — the throttle paces admission, so running N extractions in flight
    does not raise the request rate above ``rpm``; it only stops the pipe from
    going idle while responses are in flight.

    On 429 the throttle is widened and the call retried with the server's
    ``Retry-After`` when present, exponential backoff otherwise.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise MissingOpenRouterKey(
            "OPENROUTER_API_KEY is not set — the ingest path (fact extraction + "
            "promoter decisions) cannot run.\n"
            "Set it, or `set -a; source .env; set +a` if it lives there."
        )

    try:
        from openai import AsyncOpenAI, RateLimitError
    except ImportError as exc:  # pragma: no cover - environment
        raise ImportError(
            "the openai SDK is required for the OpenRouter ingest path.\n"
            "Install it with:  pip install openai"
        ) from exc

    # Lazy import: bootstrap_ollama is what calls this, so it is fully loaded by
    # the time we get here — importing at module scope would be a cycle.
    from evals.longmemeval.bootstrap_ollama import _AdaptiveThrottle

    throttle = _AdaptiveThrottle(rpm)
    # max_retries=0 — we handle 429 ourselves so the throttle *learns* from it.
    # The SDK's internal retry would hide the signal and keep the pace too high.
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)

    async def completion_fn(**kwargs: Any) -> Any:
        call: dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in _PASSTHROUGH and v is not None
        }
        # Force the model. Ignoring the caller's value is what collapses the
        # three-config routing split down to one knob.
        call["model"] = model

        last: Exception | None = None
        for attempt in range(max_retries + 1):
            await throttle.await_slot()
            try:
                resp = await client.chat.completions.create(**call)
            except RateLimitError as exc:
                last = exc
                throttle.on_429()
                wait = _retry_after(exc) or float(2**attempt)
                log.warning(
                    "ingest 429 (attempt %d/%d) — throttle now %.2fs/req, "
                    "sleeping %.1fs",
                    attempt + 1, max_retries + 1, throttle.interval, wait,
                )
                await asyncio.sleep(wait)
                continue
            throttle.on_success()
            return resp
        raise RuntimeError(
            f"ingest call failed after {max_retries + 1} attempts against rate "
            f"limits (throttle at {throttle.interval:.2f}s/req). Lower --ingest-rpm "
            f"or --ingest-concurrency."
        ) from last

    completion_fn.model = model  # type: ignore[attr-defined]  # for logging/metrics
    completion_fn.throttle = throttle  # type: ignore[attr-defined]
    return completion_fn


def _retry_after(exc: Any) -> float | None:
    """Honour the server's ``Retry-After`` when it sends one."""
    try:
        raw = exc.response.headers.get("retry-after")
    except AttributeError:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
