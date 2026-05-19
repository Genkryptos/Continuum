"""
tests/unit/test_entity_extractor.py
====================================
Unit tests for ``continuum.extraction.entity_extractor`` — a fake GLiNER is
injected via ``model_factory`` so neither ``gliner`` nor ``torch`` is needed.
"""
from __future__ import annotations

from typing import Any

import pytest

from continuum.core.config import ExtractionConfig
from continuum.extraction import ENTITY_TYPES, Entity, EntityExtractor, Relation

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGLiNER:
    """Returns canned spans; records how it was called."""

    def __init__(self, spans: list[dict[str, Any]]) -> None:
        self.spans = spans
        self.calls: list[tuple[str, tuple[str, ...], float]] = []

    def predict_entities(
        self, text: str, labels: list[str], *, threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        self.calls.append((text, tuple(labels), threshold))
        return self.spans


def _span(text: str, label: str, start: int, end: int, score: float) -> dict[str, Any]:
    return {"text": text, "label": label, "start": start, "end": end, "score": score}


def _extractor(
    spans: list[dict[str, Any]] | None = None,
    *,
    config: ExtractionConfig | None = None,
    factory: Any | None = None,
) -> tuple[EntityExtractor, FakeGLiNER | None]:
    fake = FakeGLiNER(spans or []) if factory is None else None
    fac = factory if factory is not None else (lambda _m, _d: fake)
    ex = EntityExtractor(config=config, model_factory=fac)
    return ex, fake


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        c = ExtractionConfig()
        assert c.gliner_model == "urchade/gliner_multi-v2.1"
        assert c.confidence_threshold == 0.7
        assert c.device == "auto"

    def test_entity_types_constant(self) -> None:
        assert ENTITY_TYPES == (
            "PERSON", "ORG", "LOCATION", "DATE",
            "PRODUCT", "TECH", "CONCEPT",
        )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    async def test_extracts_entities(self) -> None:
        spans = [
            _span("Alice", "person", 0, 5, 0.95),
            _span("Acme", "org", 15, 19, 0.88),
        ]
        ex, fake = _extractor(spans)
        ents, _ = await ex.extract("Alice works at Acme")

        assert [(e.text, e.type, e.start, e.end) for e in ents] == [
            ("Alice", "PERSON", 0, 5),
            ("Acme", "ORG", 15, 19),
        ]
        assert all(isinstance(e, Entity) for e in ents)
        # GLiNER queried with lowercase labels + the config threshold.
        assert fake is not None
        _text, labels, thr = fake.calls[0]
        assert labels == tuple(t.lower() for t in ENTITY_TYPES)
        assert thr == 0.7

    async def test_unknown_label_uppercased_passthrough(self) -> None:
        ex, _ = _extractor([_span("Friday", "weird", 0, 6, 0.9)])
        ents, _ = await ex.extract("Friday")
        assert ents[0].type == "WEIRD"

    async def test_empty_text_short_circuits(self) -> None:
        ex, fake = _extractor([_span("x", "person", 0, 1, 0.9)])
        assert await ex.extract("   ") == ([], [])
        assert fake is not None and fake.calls == []  # model never called


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------


class TestConfidenceFiltering:
    async def test_below_threshold_dropped(self) -> None:
        spans = [
            _span("Bob", "person", 0, 3, 0.92),     # keep
            _span("maybe", "concept", 4, 9, 0.40),  # drop (< 0.7)
        ]
        ex, _ = _extractor(spans)
        ents, _ = await ex.extract("Bob maybe")
        assert [e.text for e in ents] == ["Bob"]

    async def test_custom_threshold(self) -> None:
        spans = [_span("X", "tech", 0, 1, 0.80)]
        ex, _ = _extractor(
            spans, config=ExtractionConfig(confidence_threshold=0.9)
        )
        ents, _ = await ex.extract("X")
        assert ents == []                      # 0.80 < 0.90

        ex2, _ = _extractor(
            spans, config=ExtractionConfig(confidence_threshold=0.5)
        )
        ents2, _ = await ex2.extract("X")
        assert len(ents2) == 1                 # 0.80 ≥ 0.50


# ---------------------------------------------------------------------------
# Relation extraction (heuristics, model-free)
# ---------------------------------------------------------------------------


class TestRelationExtraction:
    def _rels(self, text: str) -> list[tuple[str, str, str]]:
        ex, _ = _extractor([])
        return [
            (r.subject, r.predicate, r.object) for r in ex.extract_relations(text)
        ]

    def test_employed_by_works_at(self) -> None:
        assert ("Alice", "EMPLOYED_BY", "Acme") in self._rels(
            "Alice works at Acme."
        )

    def test_employed_by_passive(self) -> None:
        assert ("Bob", "EMPLOYED_BY", "Globex") in self._rels(
            "Bob is employed by Globex."
        )

    def test_uses(self) -> None:
        assert ("Carol", "USES", "Python") in self._rels("Carol uses Python.")

    def test_created(self) -> None:
        rels = self._rels("Linus created Linux and developed Git.")
        assert ("Linus", "CREATED", "Linux") in rels

    def test_relation_dataclass_and_confidence(self) -> None:
        ex, _ = _extractor([])
        (r,) = ex.extract_relations("Dave works for IBM")
        assert isinstance(r, Relation)
        assert (r.subject, r.predicate, r.object) == ("Dave", "EMPLOYED_BY", "IBM")
        assert 0.0 < r.confidence <= 1.0

    def test_dedup(self) -> None:
        rels = self._rels("Alice works at Acme. Alice works at Acme.")
        assert rels.count(("Alice", "EMPLOYED_BY", "Acme")) == 1

    def test_no_false_positive(self) -> None:
        assert self._rels("The weather is nice today.") == []

    async def test_relations_returned_by_extract(self) -> None:
        # Model finds no entities, but heuristic relations still surface.
        ex, _ = _extractor([])
        ents, rels = await ex.extract("Eve works at Initech.")
        assert ents == []
        assert ("Eve", "EMPLOYED_BY", "Initech") in [
            (r.subject, r.predicate, r.object) for r in rels
        ]


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_model_build_failure_returns_empty(self) -> None:
        def boom(_m: str, _d: str) -> Any:
            raise RuntimeError("model download failed")

        ex = EntityExtractor(model_factory=boom)
        assert await ex.extract("Alice works at Acme") == ([], [])

    async def test_predict_failure_returns_empty(self) -> None:
        class Broken:
            def predict_entities(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("CUDA exploded")

        ex = EntityExtractor(model_factory=lambda _m, _d: Broken())
        assert await ex.extract("Bob uses Python") == ([], [])

    async def test_relation_failure_keeps_entities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ex, _ = _extractor([_span("Z", "tech", 0, 1, 0.99)])

        def bad(_text: str) -> list[Relation]:
            raise ValueError("regex blew up")

        monkeypatch.setattr(ex, "extract_relations", bad)
        ents, rels = await ex.extract("Z")
        assert [e.text for e in ents] == ["Z"]   # entities survive
        assert rels == []                         # relations degraded


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestDeviceResolution:
    def test_cpu_forced(self) -> None:
        ex = EntityExtractor(
            config=ExtractionConfig(device="cpu"), model_factory=lambda m, d: None
        )
        assert ex._resolve_device() == "cpu"

    def test_auto_without_cuda_is_cpu(self) -> None:
        # No GPU in CI/dev → auto resolves to cpu (never raises).
        ex = EntityExtractor(
            config=ExtractionConfig(device="auto"),
            model_factory=lambda m, d: None,
        )
        assert ex._resolve_device() in ("cpu", "cuda")

    async def test_device_passed_to_factory(self) -> None:
        seen: dict[str, str] = {}

        def fac(model_name: str, device: str) -> Any:
            seen["model"] = model_name
            seen["device"] = device
            return FakeGLiNER([])

        ex = EntityExtractor(
            config=ExtractionConfig(device="cpu"), model_factory=fac
        )
        await ex.extract("hello world")
        assert seen["model"] == "urchade/gliner_multi-v2.1"
        assert seen["device"] == "cpu"
