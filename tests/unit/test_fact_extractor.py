"""
tests/unit/test_fact_extractor.py
=================================
Unit tests for ``continuum.extraction.fact_extractor.FactExtractor``.

A fake litellm-shaped ``completion_fn`` is injected — no litellm, no network.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.config import FactExtractionConfig
from continuum.core.types import SummaryBlock
from continuum.extraction import Entity, Fact, FactExtractor

pytestmark = pytest.mark.unit

BLOCK_ID = uuid.UUID("00000000-0000-0000-0000-0000000000b1")
BLOCK = SummaryBlock(
    text="Alice works at Acme Corp and Bob works at Globex.",
    id=BLOCK_ID,
    session_id="s1",
)
ENTITIES = [
    Entity("Alice", "PERSON", 0, 5, 0.95),
    Entity("Acme Corp", "ORG", 15, 24, 0.9),
]


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _facts_payload(facts: list[dict]) -> str:
    return json.dumps({"facts": facts})


class FakeLLM:
    def __init__(self, behaviours: list[Any], *, sleep: float = 0.0) -> None:
        self._b = list(behaviours)
        self._sleep = sleep
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls += 1
        self.kwargs.append(kw)
        if self._sleep:
            await asyncio.sleep(self._sleep)
        b = self._b.pop(0) if self._b else self._b
        if isinstance(b, BaseException):
            raise b
        return _resp(b)


def _fx(fake: Any, **cfg: Any) -> FactExtractor:
    fx = FactExtractor(FactExtractionConfig(**cfg), completion_fn=fake)
    fx._backoff_initial = 0.0
    fx._backoff_max = 0.0
    return fx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    c = FactExtractionConfig()
    assert c.model == "gpt-4o-mini"
    assert c.min_confidence == 0.6
    assert c.min_fact_len == 10
    assert c.max_fact_len == 500


# ---------------------------------------------------------------------------
# Extraction + atomicity + provenance
# ---------------------------------------------------------------------------


class TestExtraction:
    async def test_extracts_atomic_facts(self) -> None:
        payload = _facts_payload(
            [
                {
                    "text": "Alice works at Acme Corp",
                    "confidence": 0.95,
                    "entities": ["Alice", "Acme Corp"],
                    "type": "employment",
                },
                {"text": "Bob works at Globex", "confidence": 0.9, "entities": ["Bob", "Globex"]},
            ]
        )
        fx = _fx(FakeLLM([payload]))
        facts = await fx.extract_facts(BLOCK, ENTITIES)

        assert [f.text for f in facts] == [
            "Alice works at Acme Corp",
            "Bob works at Globex",
        ]
        assert all(isinstance(f, Fact) for f in facts)
        # Provenance + entity refs captured.
        assert all(f.source_block_id == BLOCK_ID for f in facts)
        assert facts[0].entities_mentioned == ["Alice", "Acme Corp"]
        assert facts[0].category == "employment"
        assert facts[1].category is None

    async def test_prompt_demands_atomic_and_uses_entities(self) -> None:
        fake = FakeLLM([_facts_payload([])])
        await _fx(fake).extract_facts(BLOCK, ENTITIES)
        sys_msg, user_msg = (m["content"] for m in fake.kwargs[0]["messages"])
        assert "ATOMIC FACTS" in sys_msg
        assert "SPLIT conjunctions" in sys_msg
        assert "Alice" in user_msg and "Acme Corp" in user_msg  # entity context
        assert fake.kwargs[0]["response_format"] == {"type": "json_object"}

    async def test_dedup_identical_facts(self) -> None:
        payload = _facts_payload(
            [
                {"text": "Alice works at Acme Corp", "confidence": 0.9},
                {"text": "alice works at acme corp", "confidence": 0.8},  # dup
            ]
        )
        facts = await _fx(FakeLLM([payload])).extract_facts(BLOCK, [])
        assert len(facts) == 1


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


class TestQualityFilters:
    async def test_confidence_threshold(self) -> None:
        payload = _facts_payload(
            [
                {"text": "Alice works at Acme Corp", "confidence": 0.9},  # keep
                {"text": "Bob might know Carol somehow", "confidence": 0.4},  # drop
            ]
        )
        facts = await _fx(FakeLLM([payload])).extract_facts(BLOCK, [])
        assert [f.text for f in facts] == ["Alice works at Acme Corp"]

    async def test_custom_confidence_threshold(self) -> None:
        payload = _facts_payload(
            [
                {"text": "Alice works at Acme Corp", "confidence": 0.7},
            ]
        )
        facts = await _fx(FakeLLM([payload]), min_confidence=0.9).extract_facts(BLOCK, [])
        assert facts == []  # 0.7 < 0.9

    async def test_min_length_filter(self) -> None:
        payload = _facts_payload(
            [
                {"text": "Hi.", "confidence": 0.99},  # < 10 chars
                {"text": "Alice works at Acme Corp", "confidence": 0.9},
            ]
        )
        facts = await _fx(FakeLLM([payload])).extract_facts(BLOCK, [])
        assert [f.text for f in facts] == ["Alice works at Acme Corp"]

    async def test_max_length_filter_enforces_atomicity(self) -> None:
        long_non_atomic = (
            "Alice works at Acme Corp and " * 30
        ).strip()  # > 500 chars, clearly multi-proposition
        assert len(long_non_atomic) > 500
        payload = _facts_payload(
            [
                {"text": long_non_atomic, "confidence": 0.99},
                {"text": "Bob works at Globex", "confidence": 0.9},
            ]
        )
        facts = await _fx(FakeLLM([payload])).extract_facts(BLOCK, [])
        assert [f.text for f in facts] == ["Bob works at Globex"]

    async def test_confidence_clamped(self) -> None:
        payload = _facts_payload(
            [
                {"text": "Alice works at Acme Corp", "confidence": 5},  # → 1.0
            ]
        )
        facts = await _fx(FakeLLM([payload])).extract_facts(BLOCK, [])
        assert facts[0].confidence == 1.0


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_empty_block_no_call(self) -> None:
        fake = FakeLLM([_facts_payload([])])
        facts = await _fx(fake).extract_facts(SummaryBlock(text="   ", id=BLOCK_ID), ENTITIES)
        assert facts == []
        assert fake.calls == 0

    async def test_llm_error_returns_empty(self) -> None:
        fake = FakeLLM([RuntimeError("boom")] * 3)
        facts = await _fx(fake).extract_facts(BLOCK, ENTITIES)
        assert facts == []
        assert fake.calls == 3  # retried to exhaustion

    async def test_bad_json_returns_empty(self) -> None:
        facts = await _fx(FakeLLM(["not json {{"])).extract_facts(BLOCK, [])
        assert facts == []

    async def test_non_object_json_returns_empty(self) -> None:
        facts = await _fx(FakeLLM([json.dumps(["a", "list"])])).extract_facts(BLOCK, [])
        assert facts == []

    async def test_transient_then_success(self) -> None:
        good = _facts_payload([{"text": "Alice works at Acme Corp", "confidence": 0.9}])
        fake = FakeLLM([ConnectionError("rate limit"), ConnectionError("rate limit"), good])
        facts = await _fx(fake).extract_facts(BLOCK, [])
        assert fake.calls == 3
        assert len(facts) == 1


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_respects_timeout(self) -> None:
        fake = FakeLLM([_facts_payload([])], sleep=0.30)
        fx = _fx(fake)
        fx.config.timeout = 0.05

        t0 = time.perf_counter()
        facts = await fx.extract_facts(BLOCK, ENTITIES)
        elapsed = time.perf_counter() - t0

        assert facts == []
        assert elapsed < 0.25
        assert fake.calls == 1  # timeout not retried
