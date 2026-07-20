"""
continuum.promotion.openrouter
==============================
An OpenRouter-backed ``completion_fn`` for the Mem0 supersession decider.

The decider adjudicates whether a new fact ADDs, UPDATEs or DELETEs an existing
one. It only reaches an LLM for the ambiguous similarity band — contradiction
pairs like *"pricing is 9 dollars"* vs *"switched pricing to 12 dollars"* sit at
cosine ≈0.6-0.8, too close to call deterministically and too far apart to treat
as duplicates.

OpenRouter's chat API is OpenAI-shaped, so returning the decoded JSON is exactly
what :class:`~continuum.promotion.mem0_promoter.Mem0Promoter` expects to read
``choices[0].message.tool_calls`` from.

Lives here rather than in ``continuum.chat`` so the MCP server can use it
without importing the interactive CLI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["OPENROUTER_URL", "build_openrouter_completion_fn", "resolve_openrouter_key"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def resolve_openrouter_key(root: Path | None = None) -> str:
    """The OpenRouter key from the environment, falling back to a local ``.env``.

    Returns ``""`` when absent so callers can decide whether that is fatal —
    the decider is opt-in, and enabling it without a key would make every write
    return NOOP and silently drop the fact.
    """
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    try:
        from continuum.doctor import load_dotenv

        return (load_dotenv(root or Path.cwd()).get("OPENROUTER_API_KEY") or "").strip()
    except Exception:
        return ""


def build_openrouter_completion_fn(api_key: str, *, timeout: float = 90.0) -> Any:
    """A litellm-shaped ``async completion_fn`` bound to *api_key*."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Genkryptos/Continuum",
        "X-Title": "Continuum",
    }

    async def complete(
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **_: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    return complete
