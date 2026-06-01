"""
tests/unit/test_scorer.py
=========================
Unit tests for ``continuum.scoring.scorer.Scorer`` — the composite
memory-scoring formula. Pure / deterministic (fixed ``now``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from continuum.core.config import ScoringConfig, ScoringWeights
from continuum.core.protocols import ScorerProtocol
from continuum.core.types import MemoryItem, MemoryTier, Query
from continuum.scoring import Scorer

pytestmark = pytest.mark.unit

NOW = datetime(2025, 1, 10, 12, 0, 0, tzinfo=UTC)


def _item(
    *,
    embedding: list[float] | None = None,
    importance: float = 0.5,
    confidence: float = 1.0,
    tier: MemoryTier = MemoryTier.MTM,
    created_at: datetime | None = None,
    last_access: datetime | None = None,
    access_count: int = 0,
) -> MemoryItem:
    return MemoryItem(
        content="x",
        tier=tier,
        importance=importance,
        confidence=confidence,
        created_at=created_at or NOW,
        last_access=last_access,
        access_count=access_count,
        embedding=embedding,
    )


def _q(embedding: list[float] | None = None) -> Query:
    return Query(text="q", embedding=embedding)


def _cfg(**w: float) -> ScoringConfig:
    if w:
        return ScoringConfig(weights=ScoringWeights(**w))
    return ScoringConfig()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_satisfies_scorer_protocol() -> None:
    s = Scorer()
    assert isinstance(s, ScorerProtocol)
    assert callable(s.score) and callable(s.breakdown)


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


class TestRelevance:
    def test_identical_embeddings_relevance_one(self) -> None:
        s = Scorer()
        b = s.breakdown(_item(embedding=[1.0, 2.0, 3.0]), _q([1.0, 2.0, 3.0]), NOW)
        assert b.relevance == pytest.approx(1.0)

    def test_orthogonal_embeddings_relevance_zero(self) -> None:
        s = Scorer()
        b = s.breakdown(_item(embedding=[1.0, 0.0]), _q([0.0, 1.0]), NOW)
        assert b.relevance == pytest.approx(0.0)

    def test_opposite_embeddings_clamped_to_zero(self) -> None:
        s = Scorer()
        b = s.breakdown(_item(embedding=[1.0, 0.0]), _q([-1.0, 0.0]), NOW)
        assert b.relevance == 0.0  # cosine -1 clamped to [0,1]

    def test_missing_embedding_relevance_zero(self) -> None:
        s = Scorer()
        assert s.breakdown(_item(embedding=None), _q([1.0]), NOW).relevance == 0.0
        assert s.breakdown(_item(embedding=[1.0]), _q(None), NOW).relevance == 0.0

    def test_relevance_only_weighting(self) -> None:
        # weight 100% on relevance → composite == relevance.
        s = Scorer(_cfg(rel=1.0, imp=0.0, rec=0.0, conf=0.0))
        b = s.breakdown(_item(embedding=[1.0, 1.0]), _q([1.0, 1.0]), NOW)
        assert b.composite == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recency decay + reinforcement
# ---------------------------------------------------------------------------


class TestRecency:
    def test_decays_with_age(self) -> None:
        s = Scorer()  # tau = 168 h
        fresh = s.breakdown(_item(created_at=NOW), _q(), NOW).recency
        week = s.breakdown(_item(created_at=NOW - timedelta(hours=168)), _q(), NOW).recency
        month = s.breakdown(_item(created_at=NOW - timedelta(hours=720)), _q(), NOW).recency
        assert fresh == pytest.approx(1.0)  # exp(0)
        assert week == pytest.approx(math.exp(-1.0))  # age == tau
        assert fresh > week > month  # monotone decay

    def test_access_count_reinforces(self) -> None:
        s = Scorer()
        old = _item(created_at=NOW - timedelta(hours=504))  # 3·tau
        cold = s.breakdown(old, _q(), NOW).recency
        warm = s.breakdown(
            _item(created_at=NOW - timedelta(hours=504), access_count=5),
            _q(),
            NOW,
        ).recency
        assert warm > cold  # recall strengthens it
        # exact: base=exp(-3); strength=1+log1p(5)
        base = math.exp(-3.0)
        assert warm == pytest.approx(base * (1.0 + math.log1p(5)))

    def test_recency_clamped_to_one(self) -> None:
        s = Scorer()
        # Recent + heavily accessed → base·strength > 1 → clamps to 1.0.
        r = s.breakdown(
            _item(created_at=NOW - timedelta(hours=24), access_count=50),
            _q(),
            NOW,
        ).recency
        assert r == 1.0

    def test_last_access_overrides_created_at(self) -> None:
        s = Scorer()
        it = _item(
            created_at=NOW - timedelta(hours=720),  # old…
            last_access=NOW,  # …but just re-read
        )
        assert s.breakdown(it, _q(), NOW).recency == pytest.approx(1.0)

    def test_now_defaults_to_utcnow(self) -> None:
        # No 'now' passed → uses current time; a just-created item is ~fresh.
        s = Scorer()
        r = s.breakdown(_item(created_at=datetime.now(UTC)), _q()).recency
        assert r == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Configurable weights + validation
# ---------------------------------------------------------------------------


class TestWeights:
    def test_weights_are_used(self) -> None:
        item = _item(
            embedding=[1.0, 1.0],
            importance=0.2,
            confidence=0.4,
            created_at=NOW,
        )
        q = _q([1.0, 1.0])  # relevance = 1.0; recency = 1.0 (fresh)
        s = Scorer(_cfg(rel=0.45, imp=0.25, rec=0.20, conf=0.10))
        b = s.breakdown(item, q, NOW)
        expected = 0.45 * 1.0 + 0.25 * 0.2 + 0.20 * 1.0 + 0.10 * 0.4
        assert b.composite == pytest.approx(expected)

    def test_importance_only_weighting(self) -> None:
        s = Scorer(_cfg(rel=0.0, imp=1.0, rec=0.0, conf=0.0))
        b = s.breakdown(_item(importance=0.73), _q(), NOW)
        assert b.composite == pytest.approx(0.73)

    @pytest.mark.parametrize(
        "w",
        [
            {"rel": 0.5, "imp": 0.5, "rec": 0.5, "conf": 0.5},  # sums 2.0
            {"rel": 0.4, "imp": 0.25, "rec": 0.20, "conf": 0.10},  # 0.95
        ],
    )
    def test_weights_must_sum_to_one(self, w: dict[str, float]) -> None:
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            ScoringWeights(**w)

    def test_valid_weights_accepted(self) -> None:
        ScoringWeights(rel=0.7, imp=0.1, rec=0.1, conf=0.1)  # no raise

    def test_importance_confidence_clamped(self) -> None:
        s = Scorer(_cfg(rel=0.0, imp=0.5, rec=0.0, conf=0.5))
        b = s.breakdown(_item(importance=2.0, confidence=-1.0), _q(), NOW)
        assert b.importance == 1.0 and b.confidence == 0.0


# ---------------------------------------------------------------------------
# Layer boost
# ---------------------------------------------------------------------------


class TestLayerBoost:
    def test_boost_applied_per_tier(self) -> None:
        # composite == 1.0 (100% relevance, identical embeddings).
        s = Scorer(_cfg(rel=1.0, imp=0.0, rec=0.0, conf=0.0))
        q = _q([1.0, 1.0])

        stm = s.score(_item(embedding=[1.0, 1.0], tier=MemoryTier.STM), q, NOW)
        mtm = s.score(_item(embedding=[1.0, 1.0], tier=MemoryTier.MTM), q, NOW)
        ltm = s.score(_item(embedding=[1.0, 1.0], tier=MemoryTier.LTM), q, NOW)

        assert stm == pytest.approx(1.05)
        assert mtm == pytest.approx(1.00)
        assert ltm == pytest.approx(1.10)  # LTM may exceed 1.0 by design
        assert ltm > stm > mtm

    def test_breakdown_is_unboosted(self) -> None:
        s = Scorer(_cfg(rel=1.0, imp=0.0, rec=0.0, conf=0.0))
        item = _item(embedding=[1.0], tier=MemoryTier.LTM)
        q = _q([1.0])
        assert s.breakdown(item, q, NOW).composite == pytest.approx(1.0)
        assert s.score(item, q, NOW) == pytest.approx(1.10)  # boost on top

    def test_unknown_tier_boost_defaults_to_one(self) -> None:
        cfg = ScoringConfig(layer_boost={})  # no keys → fallback 1.0
        s = Scorer(cfg)
        item = _item(embedding=[1.0], tier=MemoryTier.LTM)
        assert s.score(item, _q([1.0]), NOW) == pytest.approx(
            s.breakdown(item, _q([1.0]), NOW).composite
        )

    def test_custom_layer_boost(self) -> None:
        cfg = ScoringConfig(layer_boost={"MTM": 2.0})
        s = Scorer(cfg)
        item = _item(embedding=[1.0], tier=MemoryTier.MTM)
        comp = s.breakdown(item, _q([1.0]), NOW).composite
        assert s.score(item, _q([1.0]), NOW) == pytest.approx(comp * 2.0)


# ---------------------------------------------------------------------------
# End-to-end formula
# ---------------------------------------------------------------------------


def test_full_formula_matches_spec() -> None:
    s = Scorer()  # defaults: .45/.25/.20/.10, tau=168, LTM boost 1.1
    item = _item(
        embedding=[1.0, 0.0, 0.0],
        importance=0.8,
        confidence=0.9,
        tier=MemoryTier.LTM,
        created_at=NOW - timedelta(hours=168),  # age == tau
        access_count=0,
    )
    q = _q([1.0, 0.0, 0.0])  # relevance = 1.0

    relevance, importance = 1.0, 0.8
    recency = math.exp(-1.0)  # base, strength 1 (no accesses)
    confidence = 0.9
    composite = 0.45 * relevance + 0.25 * importance + 0.20 * recency + 0.10 * confidence
    assert s.breakdown(item, q, NOW).composite == pytest.approx(composite)
    assert s.score(item, q, NOW) == pytest.approx(composite * 1.1)
