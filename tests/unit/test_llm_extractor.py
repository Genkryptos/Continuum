"""
tests/unit/test_llm_extractor.py
================================
Unit tests for ``continuum.extraction.llm_extractor.LLMEntityExtractor``.

A fake litellm-shaped ``completion_fn`` is injected — no litellm, no network.
Backoff is zeroed so retry tests run instantly.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.config import LLMExtractionConfig
from continuum.extraction import Entity, Relation
from continuum.extraction.llm_extractor import LLMEntityExtractor

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(content: str) -> SimpleNamespace:
    """A minimal litellm-shaped response object."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _payload(entities: list[dict] | None = None, relations: list[dict] | None = None) -> str:
    return json.dumps({"entities": entities or [], "relations": relations or []})


class FakeLLM:
    """
    Configurable fake completion.

    behaviours: list popped per call; each is either a str (returned as
    content) or an Exception instance (raised).
    """

    def __init__(self, behaviours: list[Any], *, sleep: float = 0.0) -> None:
        self._behaviours = list(behaviours)
        self._sleep = sleep
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls += 1
        self.kwargs.append(kw)
        if self._sleep:
            await asyncio.sleep(self._sleep)
        b = self._behaviours.pop(0) if self._behaviours else self._behaviours
        if isinstance(b, BaseException):
            raise b
        return _resp(b)


def _ex(fake: Any, **cfg: Any) -> LLMEntityExtractor:
    ex = LLMEntityExtractor(LLMExtractionConfig(**cfg), completion_fn=fake)
    ex._backoff_initial = 0.0
    ex._backoff_max = 0.0
    return ex


GLINER = [Entity(text="Ada Lovelace", type="PERSON", start=0, end=12, confidence=0.95)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    c = LLMExtractionConfig()
    assert c.model == "gpt-4o-mini"
    assert c.temperature == 0.0
    assert c.max_tokens == 1000
    assert c.timeout == 30


# ---------------------------------------------------------------------------
# Enhancement: adds what GLiNER missed
# ---------------------------------------------------------------------------


class TestEnhancement:
    async def test_adds_missed_entities_and_types(self) -> None:
        content = _payload(
            entities=[
                {"text": "Analytical Engine", "type": "PRODUCT", "confidence": 0.9},
                {"text": "Babbage", "type": "PERSON", "confidence": 0.8},
            ],
            relations=[
                {
                    "subject": "Ada Lovelace",
                    "predicate": "CREATED",
                    "object": "the first algorithm",
                    "confidence": 0.88,
                },
            ],
        )
        ex = _ex(FakeLLM([content]))
        text = "Ada Lovelace wrote the first algorithm for the Analytical Engine."

        ents, rels = await ex.extract(text, GLINER)

        texts = {e.text for e in ents}
        assert "Ada Lovelace" in texts  # original kept
        assert "Analytical Engine" in texts  # missed entity added
        assert "Babbage" in texts  # added (not in source → span -1)
        # LLM-only entity got a best-effort span via substring search.
        ae = next(e for e in ents if e.text == "Analytical Engine")
        assert ae.start == text.index("Analytical Engine")
        assert ae.end == ae.start + len("Analytical Engine")
        bab = next(e for e in ents if e.text == "Babbage")
        assert (bab.start, bab.end) == (-1, -1)  # not present verbatim
        assert [(r.subject, r.predicate, r.object) for r in rels] == [
            ("Ada Lovelace", "CREATED", "the first algorithm")
        ]
        assert all(isinstance(r, Relation) for r in rels)

    async def test_merge_dedups_and_preserves_gliner_span(self) -> None:
        # LLM re-emits the GLiNER entity with higher confidence.
        content = _payload(
            entities=[
                {"text": "Ada Lovelace", "type": "PERSON", "confidence": 0.99},
            ]
        )
        ex = _ex(FakeLLM([content]))
        ents, _ = await ex.extract("Ada Lovelace", GLINER)

        ada = [e for e in ents if e.text == "Ada Lovelace"]
        assert len(ada) == 1  # not duplicated
        assert (ada[0].start, ada[0].end) == (0, 12)  # GLiNER span kept
        assert ada[0].confidence == 0.99  # bumped to the max

    async def test_relation_clamp_and_dedup(self) -> None:
        content = _payload(
            relations=[
                {
                    "subject": "X",
                    "predicate": "uses",
                    "object": "Y",
                    "confidence": 5,
                },  # → clamped to 1.0
                {
                    "subject": "X",
                    "predicate": "USES",
                    "object": "Y",
                    "confidence": 0.7,
                },  # dup (case-normalised)
                {
                    "subject": "A",
                    "predicate": "PART_OF",
                    "object": "B",
                    "confidence": -1,
                },  # → clamped to 0.0
            ]
        )
        ex = _ex(FakeLLM([content]))
        _, rels = await ex.extract("text", [])
        triples = {(r.predicate, r.confidence) for r in rels}
        assert ("USES", 1.0) in triples
        assert ("PART_OF", 0.0) in triples
        assert len(rels) == 2  # dup collapsed

    async def test_static_system_prompt_sent(self) -> None:
        fake = FakeLLM([_payload()])
        await _ex(fake).extract("hello", GLINER)
        msgs = fake.kwargs[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert "JSON object" in msgs[0]["content"]
        assert "Already found" in msgs[1]["content"]
        # JSON mode + cost knobs forwarded.
        assert fake.kwargs[0]["response_format"] == {"type": "json_object"}
        assert fake.kwargs[0]["temperature"] == 0.0
        assert fake.kwargs[0]["max_tokens"] == 1000


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_empty_text_returns_gliner_unchanged_no_call(self) -> None:
        fake = FakeLLM([_payload()])
        ex = _ex(fake)
        ents, rels = await ex.extract("   ", GLINER)
        assert ents == GLINER and rels == []
        assert fake.calls == 0  # no LLM round-trip

    async def test_llm_error_falls_back_to_gliner(self) -> None:
        fake = FakeLLM([RuntimeError("provider 500")] * 3)
        ex = _ex(fake)
        ents, rels = await ex.extract("Ada Lovelace", GLINER)
        assert ents == GLINER and rels == []
        assert fake.calls == 3  # retried to exhaustion

    async def test_bad_json_falls_back(self) -> None:
        ex = _ex(FakeLLM(["this is not json {{"]))
        ents, rels = await ex.extract("Ada", GLINER)
        assert ents == GLINER and rels == []

    async def test_non_object_json_falls_back(self) -> None:
        ex = _ex(FakeLLM([json.dumps(["a", "list", "not", "object"])]))
        ents, rels = await ex.extract("Ada", GLINER)
        assert ents == GLINER and rels == []


# ---------------------------------------------------------------------------
# Retry on transient errors
# ---------------------------------------------------------------------------


class TestRetry:
    async def test_transient_then_success(self) -> None:
        good = _payload(entities=[{"text": "Git", "type": "TECH", "confidence": 0.9}])
        fake = FakeLLM(
            [
                ConnectionError("rate limited"),
                ConnectionError("rate limited"),
                good,
            ]
        )
        ex = _ex(fake)
        ents, _ = await ex.extract("Linus created Git", [])
        assert fake.calls == 3  # 2 fails + 1 success
        assert any(e.text == "Git" for e in ents)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_respects_timeout_and_does_not_retry(self) -> None:
        fake = FakeLLM([_payload()], sleep=0.30)  # slower than the timeout
        ex = _ex(fake)
        ex.config.timeout = 0.05  # tiny per-attempt cap

        t0 = time.perf_counter()
        ents, rels = await ex.extract("Ada Lovelace", GLINER)
        elapsed = time.perf_counter() - t0

        assert ents == GLINER and rels == []  # graceful
        assert elapsed < 0.25  # bailed at ~0.05s
        assert fake.calls == 1  # timeout NOT retried
