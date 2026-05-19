"""
continuum/extraction/llm_extractor.py
======================================
``LLMEntityExtractor`` — the **enhancement layer** over the deterministic
GLiNER baseline (:mod:`continuum.extraction.entity_extractor`).

It takes the GLiNER entities and asks a small LLM (via litellm, structured
JSON output) to *augment* them:

* add entities GLiNER missed,
* add entity **types** GLiNER's label set doesn't cover,
* extract **relations** (GLiNER is span-only).

Contract: it **merges, never replaces**. On *any* failure — LLM error after
retries, timeout, or unparseable output — it returns the GLiNER entities it
was given (plus no relations), so this layer can only ever *improve* on the
baseline, never regress below it.

Cost / robustness
-----------------
* Smallest capable model by default (``gpt-4o-mini``); ``temperature=0``;
  ``max_tokens`` capped.
* The system prompt is **static** (entity/relation taxonomy + few-shots) so
  OpenAI/Anthropic prompt-caching can amortise it across calls.
* litellm gives multi-provider support; rate-limit/transient errors are
  retried with exponential backoff (tenacity); each attempt is bounded by
  ``config.timeout`` seconds.

litellm is imported lazily; unit tests inject ``completion_fn`` and need
neither litellm nor network.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from continuum.core.config import LLMExtractionConfig
from continuum.extraction.entity_extractor import ENTITY_TYPES, Entity, Relation

log = logging.getLogger(__name__)

#: Relation predicates the prompt steers toward (open set — the model may
#: emit others; we keep whatever it returns).
RELATION_TYPES: tuple[str, ...] = (
    "EMPLOYED_BY",
    "USES",
    "CREATED",
    "PART_OF",
    "LOCATED_IN",
    "RELATED_TO",
)

#: ``completion_fn(**kwargs) -> awaitable[response]`` — litellm-shaped.
CompletionFn = Callable[..., Awaitable[Any]]

_SYSTEM_PROMPT = f"""\
You are a precise information-extraction engine. Extract named entities and
relations from the user's text and return ONLY a JSON object with this exact
shape:

{{
  "entities": [{{"text": "...", "type": "...", "confidence": 0.0-1.0}}],
  "relations": [{{"subject": "...", "predicate": "...",
                  "object": "...", "confidence": 0.0-1.0}}]
}}

Entity types (prefer these; you MAY introduce others if clearly warranted):
{", ".join(ENTITY_TYPES)}.

Relation predicates (prefer these SCREAMING_SNAKE_CASE forms; others allowed):
{", ".join(RELATION_TYPES)}.

Rules:
- The user message lists entities ALREADY FOUND by a fast NER model. Do NOT
  repeat them unless you are adding information. Focus on what was MISSED:
  entities that model could not see, types it does not support, and ALL
  relations (it cannot extract relations).
- ``text`` must be the exact surface string as it appears in the source.
- confidence is your calibrated certainty in [0, 1].
- Output JSON only. No prose, no markdown fences.

Example
-------
Text: "Ada Lovelace wrote the first algorithm for Babbage's Analytical Engine."
Already found: [{{"text": "Ada Lovelace", "type": "PERSON"}}]
Output:
{{"entities": [{{"text": "Analytical Engine", "type": "PRODUCT",
                 "confidence": 0.9}},
               {{"text": "Babbage", "type": "PERSON", "confidence": 0.85}}],
 "relations": [{{"subject": "Ada Lovelace", "predicate": "CREATED",
                 "object": "the first algorithm", "confidence": 0.88}}]}}
"""


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _is_transient(exc: BaseException) -> bool:
    """
    Retry policy: rate-limit / connection / 5xx errors are transient.

    Timeouts (``asyncio.wait_for``) and bad output (``ValueError`` /
    ``JSONDecodeError``) are **not** retried — they fail fast into the
    graceful-degradation path so we never block or loop on a bad response.
    """
    if isinstance(exc, TimeoutError | ValueError):
        return False
    return True


class LLMEntityExtractor:
    """
    LLM augmentation of GLiNER entities + relation extraction.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.LLMExtractionConfig`.
    completion_fn:
        Async ``litellm``-style completion callable. Injected in tests; the
        default lazily wraps ``litellm.acompletion``.
    """

    def __init__(
        self,
        config: LLMExtractionConfig | None = None,
        *,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.config = config or LLMExtractionConfig()
        self._completion_fn = completion_fn
        # Retry policy (tests set the backoff to 0 for speed).
        self._max_attempts = 3
        self._backoff_initial = 0.3
        self._backoff_max = 8.0

    # ── public API ──────────────────────────────────────────────────────────

    async def extract(
        self,
        text: str,
        gliner_entities: list[Entity],
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Return ``(merged_entities, relations)``.

        ``merged_entities`` = the GLiNER entities (spans preserved) plus any
        the LLM adds. On failure the GLiNER entities are returned unchanged
        with no relations — this layer never regresses the baseline.
        """
        if not text or not text.strip():
            return list(gliner_entities), []

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(text, gliner_entities)},
        ]
        try:
            content = await self._complete(messages)
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("LLM did not return a JSON object")
        except Exception:
            # Retry exhaustion (tenacity reraise=True), per-attempt timeout,
            # bad JSON, or any provider error → never regress the baseline.
            log.exception(
                "LLM extraction failed — falling back to GLiNER entities only"
            )
            return list(gliner_entities), []

        llm_entities = self._parse_entities(data.get("entities", []))
        relations = self._parse_relations(data.get("relations", []))
        merged = self._merge_entities(text, gliner_entities, llm_entities)
        return merged, relations

    # ── LLM call (retry + per-attempt timeout) ──────────────────────────────

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=self._backoff_initial, max=self._backoff_max
            ),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        async for attempt in self._retrying():
            with attempt:
                resp = await asyncio.wait_for(
                    self._call(messages), timeout=self.config.timeout
                )
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
                "litellm is required for LLMEntityExtractor.\n"
                "Install it with:  pip install litellm"
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
        """Pull the message text out of a litellm-shaped response."""
        try:
            return str(resp.choices[0].message.content)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        try:  # dict-shaped fallback
            return str(resp["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unrecognised completion response: {resp!r}") from exc

    # ── prompt ──────────────────────────────────────────────────────────────

    @staticmethod
    def _user_prompt(text: str, gliner_entities: Sequence[Entity]) -> str:
        already = [{"text": e.text, "type": e.type} for e in gliner_entities]
        return (
            f"Text:\n{text}\n\n"
            f"Already found (do not repeat; add what is missing):\n"
            f"{json.dumps(already, ensure_ascii=False)}"
        )

    # ── parsing / merge ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_entities(rows: Any) -> list[Entity]:
        out: list[Entity] = []
        if not isinstance(rows, list):
            return out
        for r in rows:
            if not isinstance(r, dict):
                continue
            txt = str(r.get("text", "")).strip()
            if not txt:
                continue
            out.append(
                Entity(
                    text=txt,
                    type=str(r.get("type", "CONCEPT")).upper(),
                    start=-1,  # spans are GLiNER's job; located in _merge
                    end=-1,
                    confidence=_clamp(r.get("confidence", 0.5)),
                )
            )
        return out

    @staticmethod
    def _parse_relations(rows: Any) -> list[Relation]:
        out: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()
        if not isinstance(rows, list):
            return out
        for r in rows:
            if not isinstance(r, dict):
                continue
            s = str(r.get("subject", "")).strip()
            p = str(r.get("predicate", "")).strip().upper()
            o = str(r.get("object", "")).strip()
            if not (s and p and o):
                continue
            key = (s.lower(), p, o.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Relation(
                    subject=s,
                    predicate=p,
                    object=o,
                    confidence=_clamp(r.get("confidence", 0.5)),
                )
            )
        return out

    @staticmethod
    def _merge_entities(
        text: str,
        gliner: Sequence[Entity],
        llm: Sequence[Entity],
    ) -> list[Entity]:
        """
        GLiNER is authoritative for spans; LLM only *adds*.

        Dedup key = ``(text.lower(), type.upper())``. A duplicate keeps the
        GLiNER row (real offsets) but bumps confidence to the max of the
        two. LLM-only entities get a best-effort span via case-insensitive
        substring search (``(-1, -1)`` if not located).
        """
        merged: list[Entity] = list(gliner)
        index: dict[tuple[str, str], Entity] = {
            (e.text.strip().lower(), e.type.upper()): e for e in merged
        }
        lowered = text.lower()
        for e in llm:
            key = (e.text.strip().lower(), e.type.upper())
            existing = index.get(key)
            if existing is not None:
                existing.confidence = max(existing.confidence, e.confidence)
                continue
            pos = lowered.find(e.text.strip().lower())
            located = Entity(
                text=e.text,
                type=e.type,
                start=pos if pos >= 0 else -1,
                end=pos + len(e.text) if pos >= 0 else -1,
                confidence=e.confidence,
            )
            merged.append(located)
            index[key] = located
        return merged


__all__ = ["LLMEntityExtractor", "RELATION_TYPES"]
