"""
continuum/extraction/entity_extractor.py
=========================================
``EntityExtractor`` — the **deterministic Phase-0 baseline** for turning raw
text into ``(entities, relations)``.

* **Entities** — zero-shot spans from GLiNER (``urchade/gliner_multi-v2.1``
  by default), filtered by a configurable confidence threshold.
* **Relations** — simple, transparent regex heuristics (no model): the
  baseline an LLM-based extractor must beat before it replaces this.

Design (mirrors ``EmbeddingService``)
-------------------------------------
* GLiNER / torch are **lazy-loaded** on first ``extract`` and cached for the
  process; importing this module is cheap and tests inject a fake model via
  ``model_factory`` (no ``gliner``/``torch`` needed).
* Device: ``auto`` → cuda if available else cpu (``cuda``/``cpu`` force it).
* **Graceful degradation**: any model/inference failure is logged and
  yields ``([], [])`` — extraction never raises into the caller's path.

Install the real backend with::

    pip install gliner
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from continuum.core.config import ExtractionConfig

log = logging.getLogger(__name__)

#: Canonical entity labels (spec). GLiNER is queried with the lowercased
#: forms; the reverse map restores the canonical casing.
ENTITY_TYPES: tuple[str, ...] = (
    "PERSON",
    "ORG",
    "LOCATION",
    "DATE",
    "PRODUCT",
    "TECH",
    "CONCEPT",
)

#: Fixed confidence for heuristic (non-model) relations.
_RELATION_CONFIDENCE = 0.6

# Heuristic relation patterns. Subject/object are short noun phrases built
# from "content" tokens — a negative lookahead bars relation verbs,
# prepositions and conjunctions from the span, and excluding '.' from the
# token class stops a phrase crossing a sentence boundary. This keeps the
# baseline deterministic and clause-local (it is intentionally simple — the
# bar an LLM extractor must clear, not the final word).
_STOP = (
    r"(?:works?|working|use|uses|using|created|built|develops?|developed|"
    r"authored|made|employed|is|are|was|were|by|at|for|and|or|the|a|an)"
)
_TOK = rf"(?!{_STOP}\b)[A-Za-z0-9][\w&'’-]*"
_NP = rf"{_TOK}(?:\s+{_TOK}){{0,3}}"
_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            rf"\b(?P<s>{_NP})\s+(?:works?|working)\s+(?:at|for)\s+(?P<o>{_NP})",
            re.IGNORECASE,
        ),
        "EMPLOYED_BY",
    ),
    (
        re.compile(
            rf"\b(?P<s>{_NP})\s+(?:is|are|was|were)\s+employed\s+by\s+"
            rf"(?P<o>{_NP})",
            re.IGNORECASE,
        ),
        "EMPLOYED_BY",
    ),
    (
        re.compile(
            rf"\b(?P<s>{_NP})\s+(?:uses?|using|use)\s+(?P<o>{_NP})",
            re.IGNORECASE,
        ),
        "USES",
    ),
    (
        re.compile(
            rf"\b(?P<s>{_NP})\s+(?:created|built|developed|authored|made)\s+"
            rf"(?P<o>{_NP})",
            re.IGNORECASE,
        ),
        "CREATED",
    ),
)


@dataclass
class Entity:
    """A typed text span: ``[start, end)`` are character offsets into the
    source text; ``type`` is one of :data:`ENTITY_TYPES`."""

    text: str
    type: str
    start: int
    end: int
    confidence: float


@dataclass
class Relation:
    """A ``(subject, predicate, object)`` triple with a confidence score."""

    subject: str
    predicate: str
    object: str
    confidence: float


class GLiNERModel(Protocol):
    """Structural type for the bits of GLiNER we use."""

    def predict_entities(
        self, text: str, labels: list[str], *, threshold: float = ...
    ) -> list[dict[str, Any]]: ...


#: ``model_factory(model_name, device) -> GLiNERModel``. Injected in tests;
#: the default lazily builds a real GLiNER model.
ModelFactory = Callable[[str, str], GLiNERModel]


class EntityExtractor:
    """
    GLiNER entity extraction + heuristic relation extraction.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.ExtractionConfig`. Defaults to one
        built from env / defaults.
    model_factory:
        ``(model_name, device) -> GLiNERModel``. Inject a fake in tests;
        the default lazily imports and builds GLiNER.
    entity_types:
        Override the canonical label set (defaults to :data:`ENTITY_TYPES`).
    """

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
        entity_types: Sequence[str] | None = None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self._factory = model_factory
        self._types: tuple[str, ...] = tuple(entity_types or ENTITY_TYPES)
        # GLiNER is queried with lowercase labels; map back to canonical.
        self._labels = [t.lower() for t in self._types]
        self._label_to_canonical = {t.lower(): t for t in self._types}

        self._model: GLiNERModel | None = None
        self._device: str | None = None
        self._lock = asyncio.Lock()

    # ── model loading (lazy, cached) ────────────────────────────────────────

    def _resolve_device(self) -> str:
        want = self.config.device
        if want == "cpu":
            return "cpu"
        if want in ("auto", "cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                log.debug("torch unavailable — extraction on cpu")
            if want == "cuda":
                log.warning("device='cuda' requested but unavailable — using cpu")
            return "cpu"
        return "cpu"

    def _build_model(self, device: str) -> GLiNERModel:
        if self._factory is not None:
            return self._factory(self.config.gliner_model, device)
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - exercised via factory
            raise ImportError(
                "gliner is required for EntityExtractor.\n"
                "Install it with:  pip install gliner"
            ) from exc
        log.info(
            "loading GLiNER model %s on %s", self.config.gliner_model, device
        )
        model = GLiNER.from_pretrained(self.config.gliner_model)
        try:
            model = model.to(device)
        except Exception:  # pragma: no cover - cpu fallback
            log.debug("model.to(%s) failed — staying on cpu", device)
        return model  # type: ignore[no-any-return]

    async def _get_model(self) -> GLiNERModel:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._device = self._resolve_device()
                self._model = await asyncio.to_thread(
                    self._build_model, self._device
                )
        return self._model

    # ── public API ──────────────────────────────────────────────────────────

    async def extract(
        self, text: str
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Extract ``(entities, relations)`` from *text*.

        Entities come from GLiNER (filtered to
        ``config.confidence_threshold``); relations from regex heuristics.
        On any model/inference error this logs and returns ``([], [])`` so
        the caller's pipeline degrades instead of breaking.
        """
        if not text or not text.strip():
            return [], []

        entities: list[Entity] = []
        try:
            model = await self._get_model()
            raw = await asyncio.to_thread(
                model.predict_entities,
                text,
                self._labels,
                threshold=self.config.confidence_threshold,
            )
            entities = self._to_entities(raw)
        except Exception:
            log.exception("entity extraction failed — returning empty result")
            return [], []

        # Relations are pure regex and must not fail the whole call.
        try:
            relations = self.extract_relations(text)
        except Exception:
            log.exception("relation extraction failed — returning entities only")
            relations = []

        return entities, relations

    # ── entity post-processing ──────────────────────────────────────────────

    def _to_entities(self, raw: list[dict[str, Any]]) -> list[Entity]:
        """Map GLiNER spans → :class:`Entity`, re-filtering by threshold."""
        thr = self.config.confidence_threshold
        out: list[Entity] = []
        for r in raw:
            score = float(r.get("score", 0.0))
            if score < thr:
                continue
            label = str(r.get("label", "")).lower()
            canonical = self._label_to_canonical.get(label, label.upper())
            out.append(
                Entity(
                    text=str(r.get("text", "")),
                    type=canonical,
                    start=int(r.get("start", 0)),
                    end=int(r.get("end", 0)),
                    confidence=score,
                )
            )
        return out

    # ── heuristic relations ─────────────────────────────────────────────────

    def extract_relations(self, text: str) -> list[Relation]:
        """
        Find ``(subject, predicate, object)`` triples via regex heuristics.

        Patterns (case-insensitive on the verb):
        * "X works at/for Y" / "X is employed by Y" → ``EMPLOYED_BY``
        * "X uses Y"                                → ``USES``
        * "X created/built/developed Y"            → ``CREATED``

        Deterministic and model-free — this is the baseline an LLM extractor
        must beat. Duplicate triples are de-duplicated.
        """
        seen: set[tuple[str, str, str]] = set()
        out: list[Relation] = []
        for pattern, predicate in _RELATION_PATTERNS:
            for m in pattern.finditer(text):
                subj = m.group("s").strip(" .,;:'\"")
                obj = m.group("o").strip(" .,;:'\"")
                if not subj or not obj:
                    continue
                key = (subj.lower(), predicate, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Relation(
                        subject=subj,
                        predicate=predicate,
                        object=obj,
                        confidence=_RELATION_CONFIDENCE,
                    )
                )
        return out


__all__ = ["EntityExtractor", "Entity", "Relation", "ENTITY_TYPES"]
