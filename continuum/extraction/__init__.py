"""
continuum.extraction
=====================
Deterministic entity / relation extraction.

Exports
-------
EntityExtractor    — GLiNER-backed zero-shot entity extractor + heuristic
                     relation extractor (the deterministic Phase-0 baseline).
LLMEntityExtractor       — LLM enhancement layer: merges with GLiNER, adds
                            missed entities/types and extracts relations.
FactExtractor            — distil an MTM SummaryBlock into atomic facts.
MemoryCandidateExtractor — bridge SummaryBlock → MemoryCandidate list for
                            the Policy Engine (regex + reuses FactExtractor).
Fact                     — (text, confidence, entities_mentioned, block_id).
Entity                   — (text, type, start, end, confidence) span.
Relation           — (subject, predicate, object, confidence) triple.
ENTITY_TYPES       — the default canonical label set.
RELATION_TYPES     — predicates the LLM prompt steers toward.

``gliner`` / ``torch`` / ``litellm`` are imported lazily, so importing this
package is cheap and unit tests inject fakes without those packages.
"""

from __future__ import annotations

from continuum.extraction.candidate_extractor import MemoryCandidateExtractor
from continuum.extraction.entity_extractor import (
    ENTITY_TYPES,
    Entity,
    EntityExtractor,
    Relation,
)
from continuum.extraction.fact_extractor import Fact, FactExtractor
from continuum.extraction.llm_extractor import RELATION_TYPES, LLMEntityExtractor

__all__ = [
    "EntityExtractor",
    "LLMEntityExtractor",
    "FactExtractor",
    "MemoryCandidateExtractor",
    "Entity",
    "Relation",
    "Fact",
    "ENTITY_TYPES",
    "RELATION_TYPES",
]
