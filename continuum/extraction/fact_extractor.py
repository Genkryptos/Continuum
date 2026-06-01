"""
continuum/extraction/fact_extractor.py
=======================================
``FactExtractor`` — distil an MTM :class:`SummaryBlock` into a list of
**atomic facts** (one proposition each) using an LLM with structured JSON
output.

Atomicity
---------
One fact = one subject-verb-object proposition, self-contained.

* Bad : "Alice works at Acme and Bob works at Globex" → must become 2 facts
* Good: "Alice works at Acme Corp"

The prompt instructs the model to split conjunctions; the ``max_fact_len``
quality filter is the safety net (a 500-char "fact" is almost never atomic,
so it is rejected).

Quality filters (configurable via :class:`FactExtractionConfig`)
---------------------------------------------------------------
* ``confidence >= min_confidence`` (default 0.6)
* ``min_fact_len <= len(text) <= max_fact_len`` (default 10..500 chars)

Robustness mirrors :class:`LLMEntityExtractor`: litellm is lazy-imported,
``completion_fn`` is injectable for tests, transient/rate-limit errors are
retried (tenacity), each attempt is bounded by ``config.timeout``, and *any*
failure degrades to ``[]`` — fact extraction never raises into the Promoter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from continuum.core.config import FactExtractionConfig
from continuum.core.types import SummaryBlock
from continuum.extraction.entity_extractor import Entity

log = logging.getLogger(__name__)

#: ``completion_fn(**kwargs) -> awaitable[response]`` — litellm-shaped.
CompletionFn = Callable[..., Awaitable[Any]]

_SYSTEM_PROMPT = """\
You extract ATOMIC FACTS from a summary block. Return ONLY a JSON object:

{
  "facts": [
    {"text": "Alice works at Acme Corp",
     "confidence": 0.0-1.0,
     "entities": ["Alice", "Acme Corp"],
     "type": "employment"}
  ]
}

Rules:
- Each fact is ONE self-contained proposition (subject-verb-object).
- SPLIT conjunctions. "Alice works at Acme and Bob works at Globex" becomes
  TWO facts: "Alice works at Acme" and "Bob works at Globex".
- A fact must stand alone without the surrounding text (resolve pronouns
  using the provided entities as context).
- ``text``: a single clear sentence. ``confidence``: your certainty [0, 1].
- ``entities``: the surface strings of entities the fact mentions.
- ``type``: a short lowercase category (optional, e.g. "employment").
- Output JSON only. No prose, no markdown fences.
"""


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _is_transient(exc: BaseException) -> bool:
    """Retry rate-limit/connection/5xx; never retry timeout or bad output."""
    if isinstance(exc, TimeoutError | ValueError):
        return False
    return True


@dataclass
class Fact:
    """
    One atomic proposition mined from a :class:`SummaryBlock`.

    ``entities_mentioned`` are the surface strings the fact references;
    ``source_block_id`` ties the fact back to its originating MTM block for
    provenance. ``category`` is the optional LLM-supplied ``type`` tag.
    """

    text: str
    confidence: float
    entities_mentioned: list[str]
    source_block_id: uuid.UUID
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FactExtractor:
    """
    LLM atomic-fact extractor over MTM summary blocks.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.FactExtractionConfig`.
    completion_fn:
        Async litellm-style completion. Injected in tests; the default
        lazily wraps ``litellm.acompletion``.
    """

    def __init__(
        self,
        config: FactExtractionConfig | None = None,
        *,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.config = config or FactExtractionConfig()
        self._completion_fn = completion_fn
        self._max_attempts = 3
        self._backoff_initial = 0.3
        self._backoff_max = 8.0

    # ── public API ──────────────────────────────────────────────────────────

    async def extract_facts(
        self,
        block: SummaryBlock,
        entities: list[Entity],
    ) -> list[Fact]:
        """
        Extract quality-filtered atomic :class:`Fact`s from *block*.

        *entities* (e.g. from GLiNER / the LLM extractor) are passed to the
        model as disambiguation context. Returns ``[]`` on empty input or
        any failure — never raises into the caller (Promoter) path.
        """
        if not block.text or not block.text.strip():
            return []

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(block, entities)},
        ]
        try:
            content = await self._complete(messages)
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("LLM did not return a JSON object")
        except Exception:
            log.exception(
                "fact extraction failed for block %s — returning no facts",
                block.id,
            )
            return []

        return self._parse_and_filter(data.get("facts", []), block)

    # ── LLM call (retry + per-attempt timeout) ──────────────────────────────

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=self._backoff_initial, max=self._backoff_max),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        async for attempt in self._retrying():
            with attempt:
                resp = await asyncio.wait_for(self._call(messages), timeout=self.config.timeout)
                return self._content(resp)
        raise RuntimeError("unreachable retry exit")  # pragma: no cover

    async def _call(self, messages: list[dict[str, str]]) -> Any:
        if self._completion_fn is not None:
            return await self._completion_fn(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - via completion_fn
            raise ImportError(
                "litellm is required for FactExtractor.\nInstall it with:  pip install litellm"
            ) from exc
        return await litellm.acompletion(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )

    @staticmethod
    def _content(resp: Any) -> str:
        try:
            return str(resp.choices[0].message.content)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        try:
            return str(resp["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unrecognised completion response: {resp!r}") from exc

    # ── prompt ──────────────────────────────────────────────────────────────

    @staticmethod
    def _user_prompt(block: SummaryBlock, entities: Sequence[Entity]) -> str:
        ents = sorted({e.text.strip() for e in entities if e.text.strip()})
        return (
            f"Summary block:\n{block.text}\n\n"
            f"Known entities (context for disambiguation):\n"
            f"{json.dumps(ents, ensure_ascii=False)}\n\n"
            f"Extract every atomic fact."
        )

    # ── parse + quality filters ─────────────────────────────────────────────

    def _parse_and_filter(self, rows: Any, block: SummaryBlock) -> list[Fact]:
        if not isinstance(rows, list):
            return []
        cfg = self.config
        out: list[Fact] = []
        seen: set[str] = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            text = str(r.get("text", "")).strip()
            if not (cfg.min_fact_len <= len(text) <= cfg.max_fact_len):
                continue  # too short → trivial; too long → not atomic
            conf = _clamp(r.get("confidence", 0.0))
            if conf < cfg.min_confidence:
                continue
            dedup = text.lower()
            if dedup in seen:
                continue
            seen.add(dedup)

            raw_ents = r.get("entities", [])
            mentioned = (
                [str(x).strip() for x in raw_ents if str(x).strip()]
                if isinstance(raw_ents, list)
                else []
            )
            category = r.get("type")
            out.append(
                Fact(
                    text=text,
                    confidence=conf,
                    entities_mentioned=mentioned,
                    source_block_id=block.id,
                    category=str(category) if category else None,
                )
            )
        return out


__all__ = ["FactExtractor", "Fact"]
