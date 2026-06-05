"""
tests/unit/test_mem0_retraction.py
==================================
Mem0 decider — retraction handling in ``_short_circuit``.

A retraction fact ("I was never in Bengaluru") must deterministically RETIRE
its contradicted neighbor (op=DELETE → ``ltm.invalidate``), not be stored as a
new fact and not chronologically supersede. Gated on retrieval similarity so a
stray negation can't delete an unrelated memory. All decisions stay audited.
No LLM is involved on this path — these are pure short-circuit assertions.
"""

from __future__ import annotations

import uuid

import pytest

from continuum.core.config import PromoterConfig
from continuum.core.types import MemoryItem, MemoryTier, ScoreBreakdown, ScoredItem
from continuum.extraction.fact_extractor import Fact
from continuum.promotion.mem0_promoter import Mem0Promoter

pytestmark = pytest.mark.unit

BLOCK = uuid.UUID("00000000-0000-0000-0000-0000000000b1")
N_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c1")


def _fact(text: str) -> Fact:
    return Fact(text=text, confidence=0.9, entities_mentioned=[], source_block_id=BLOCK)


def _si(text: str, cos: float, nid: uuid.UUID = N_ID) -> ScoredItem:
    return ScoredItem(
        item=MemoryItem(id=str(nid), content=text, tier=MemoryTier.LTM),
        scores=ScoreBreakdown(
            relevance=cos, importance=0.0, recency=0.0, confidence=1.0, composite=cos
        ),
    )


def _promoter(**cfg: object) -> Mem0Promoter:
    # No completion_fn needed: every assertion here is a short-circuit path.
    return Mem0Promoter(PromoterConfig(**cfg))


def test_retraction_deletes_contradicted_neighbor() -> None:
    p = _promoter(add_threshold=0.5, noop_threshold=0.97)
    # The stored memory says the user is in Bengaluru; the retraction negates it.
    neighbor = _si("user is located in Bengaluru", cos=0.62)
    dec = p._short_circuit(_fact("I was never in Bengaluru"), [neighbor])
    assert dec is not None
    assert dec.op == "DELETE"
    assert dec.target_id == N_ID  # retires the contradicted memory
    assert dec.short_circuited is True
    assert dec.metadata.get("retraction") is True


def test_retraction_without_matching_memory_is_noop_not_add() -> None:
    p = _promoter(add_threshold=0.5)
    # Weakly-related neighbor (below add_threshold) → nothing to retire, and we
    # must NOT store the bare negation as a new fact.
    weak = _si("user likes hiking", cos=0.20)
    dec = p._short_circuit(_fact("I was never in Bengaluru"), [weak])
    assert dec is not None
    assert dec.op == "NOOP"
    assert dec.target_id is None
    assert dec.metadata.get("retraction") is True


def test_retraction_with_no_neighbors_is_noop() -> None:
    p = _promoter()
    dec = p._short_circuit(_fact("Actually I never said that"), [])
    assert dec is not None
    assert dec.op == "NOOP"
    assert dec.metadata.get("retraction") is True


def test_non_retraction_change_is_unaffected_add() -> None:
    # "I moved to Boston" is a normal new fact with no similar neighbor → ADD,
    # exactly as before (retraction branch must not interfere).
    p = _promoter(add_threshold=0.5)
    dec = p._short_circuit(_fact("I moved to Boston"), [_si("user likes hiking", 0.10)])
    assert dec is not None
    assert dec.op == "ADD"
    assert dec.metadata.get("retraction") is None


def test_non_retraction_duplicate_still_noops() -> None:
    # A near-duplicate non-retraction still short-circuits to NOOP (unchanged).
    p = _promoter(add_threshold=0.5, noop_threshold=0.9)
    dec = p._short_circuit(_fact("user lives in Boston"), [_si("user lives in Boston", 0.99)])
    assert dec is not None
    assert dec.op == "NOOP"
    assert dec.metadata.get("retraction") is None


def test_standing_negative_preference_is_not_a_retraction() -> None:
    # "I never eat meat" is a preference, not a retraction: even with a related
    # neighbor it must NOT be forced to DELETE — it flows to the normal path.
    p = _promoter(add_threshold=0.5, noop_threshold=0.97)
    dec = p._short_circuit(_fact("I never eat meat"), [_si("user eats meat", 0.70)])
    # cosine 0.70 is between add/noop thresholds → not a short-circuit → LLM path.
    assert dec is None
