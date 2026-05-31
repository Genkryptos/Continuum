"""
tests/unit/test_semantic_dedupe.py
==================================
Unit tests for
:class:`continuum.optimizer.strategies.semantic_dedupe.SemanticDedupe`.

Covers
------
* Protocol satisfaction + Pydantic config validation
* No-op when fewer than ``skip_if_fewer_than`` LTM items
* No-op when no items carry embeddings
* Near-duplicates removed at threshold
* Highest-scored duplicate kept (default scorer)
* Custom ``score_key`` honoured
* Transitive duplicates (A≈B≈C) collapse to a single survivor
* Threshold is configurable — higher threshold keeps more
* Dissimilar items untouched (no false positives)
* Items without embeddings pass through
* :func:`cosine_similarity_matrix` matches a numpy reference
* Bundle plumbing: messages / tier_breakdown / debug_info updated
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    TokenBudget,
)
from continuum.optimizer.protocol import Optimizer
from continuum.optimizer.strategies.semantic_dedupe import (
    SemanticDedupe,
    SemanticDedupeConfig,
    cosine_similarity_matrix,
)

_BASE = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8_000,
        stm_reserved=500,
        mtm_reserved=2_000,
        ltm_reserved=2_000,
        response_reserved=500,
    )


def _ltm(
    content: str,
    embedding: list[float] | None,
    *,
    importance: float = 0.5,
    confidence: float = 1.0,
    offset_seconds: int = 0,
) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        content=content,
        tier=MemoryTier.LTM,
        embedding=embedding,
        importance=importance,
        confidence=confidence,
        created_at=_BASE + timedelta(seconds=offset_seconds),
    )


def _bundle(items: list[MemoryItem]) -> ContextBundle:
    return ContextBundle(
        items=items,
        messages=[{"role": "system", "content": i.content} for i in items],
        tokens_used=sum(len(i.content.split()) for i in items),
        budget=_budget(),
    )


def _emb(*vals: float, dim: int = 8) -> list[float]:
    """Build a length-*dim* embedding padded with zeros."""
    out = list(vals) + [0.0] * (dim - len(vals))
    return out[:dim]


# ---------------------------------------------------------------------------
# Protocol + config
# ---------------------------------------------------------------------------


def test_protocol_satisfaction() -> None:
    assert isinstance(SemanticDedupe(), Optimizer)


def test_config_defaults() -> None:
    cfg = SemanticDedupeConfig()
    assert cfg.threshold == 0.92
    assert cfg.skip_if_fewer_than == 10


def test_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValidationError):
        SemanticDedupeConfig(threshold=1.5)


def test_config_rejects_skip_below_two() -> None:
    with pytest.raises(ValidationError):
        SemanticDedupeConfig(skip_if_fewer_than=1)


# ---------------------------------------------------------------------------
# cosine_similarity_matrix — sanity vs numpy reference
# ---------------------------------------------------------------------------


def test_cosine_matrix_matches_reference() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(size=(20, 64)).astype(np.float32)
    out = cosine_similarity_matrix(m)
    # Reference: row-normalise + matmul (safe div in case of zero rows).
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalised = m / safe
    ref = normalised @ normalised.T
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-6)
    # Diagonal must be ~1.
    assert np.allclose(np.diag(out), 1.0, atol=1e-5)


def test_cosine_matrix_handles_zero_row() -> None:
    m = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    out = cosine_similarity_matrix(m)
    # Zero row → zeros throughout (no NaNs).
    assert not np.isnan(out).any()
    assert out[1, 0] == 0.0
    assert out[1, 2] == 0.0


# ---------------------------------------------------------------------------
# No-op branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_below_threshold_count() -> None:
    items = [_ltm(f"fact {i}", _emb(float(i))) for i in range(5)]
    out = await SemanticDedupe(skip_if_fewer_than=10).apply(_bundle(items), _budget())
    assert out is _bundle(items) or len(out.items) == len(items)


@pytest.mark.asyncio
async def test_no_op_when_no_embeddings() -> None:
    items = [_ltm(f"fact {i}", None) for i in range(20)]
    out = await SemanticDedupe(skip_if_fewer_than=2).apply(_bundle(items), _budget())
    # No embeddings → no dedup candidates → no-op (returns original).
    assert len(out.items) == len(items)


@pytest.mark.asyncio
async def test_no_op_when_all_dissimilar() -> None:
    # Orthogonal one-hot embeddings — pairwise cosine = 0.
    items = [_ltm(f"distinct fact {i}", _emb(*([0.0] * i + [1.0]))) for i in range(12)]
    out = await SemanticDedupe(
        threshold=0.92,
        skip_if_fewer_than=2,
    ).apply(_bundle(items), _budget())
    assert len(out.items) == len(items)


# ---------------------------------------------------------------------------
# Dedup behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_near_duplicates_removed() -> None:
    """
    Two near-identical vectors → one removed; the rest survive.
    """
    base = _emb(1.0, 0.0, 0.0, 0.0)
    very_close = _emb(0.99, 0.14, 0.0, 0.0)  # cosine ≈ 0.99
    others = [_emb(*([0.0] * (i + 4) + [1.0])) for i in range(8)]

    a = _ltm("Acme launched analytics", base, importance=0.4)
    b = _ltm("Acme launched analytics product", very_close, importance=0.9)
    rest = [_ltm(f"unrelated {i}", e) for i, e in enumerate(others)]
    ctx = _bundle([a, b, *rest])

    out = await SemanticDedupe(
        threshold=0.92,
        skip_if_fewer_than=2,
    ).apply(ctx, _budget())

    # The lower-scored duplicate (a, importance=0.4) is removed; b kept.
    surviving_ids = {it.id for it in out.items}
    assert b.id in surviving_ids
    assert a.id not in surviving_ids
    # All unrelated items are still present.
    for it in rest:
        assert it.id in surviving_ids
    # debug_info stamped.
    assert out.debug_info["semantic_dedupe"]["removed"] == 1


@pytest.mark.asyncio
async def test_highest_scored_wins() -> None:
    near = _emb(1.0, 0.0, 0.0)
    near2 = _emb(0.98, 0.2, 0.0)
    low = _ltm("low score", near, importance=0.2, confidence=0.5)
    high = _ltm("high score", near2, importance=0.9, confidence=0.9)
    filler = [_ltm(f"filler {i}", _emb(*([0.0] * (i + 3) + [1.0]))) for i in range(8)]
    out = await SemanticDedupe(skip_if_fewer_than=2).apply(_bundle([low, high, *filler]), _budget())
    survivors = {it.id for it in out.items}
    assert high.id in survivors
    assert low.id not in survivors


@pytest.mark.asyncio
async def test_custom_score_key_used() -> None:
    near = _emb(1.0, 0.0)
    near2 = _emb(0.99, 0.14)
    # By default importance=0.5 ties → tie-break by created_at.
    a = _ltm("first", near, importance=0.5, offset_seconds=0)
    b = _ltm("second", near2, importance=0.5, offset_seconds=10)
    filler = [_ltm(f"f{i}", _emb(*([0.0] * (i + 3) + [1.0]))) for i in range(8)]
    # Score by content length: "second" > "first".
    out = await SemanticDedupe(
        skip_if_fewer_than=2,
        score_key=lambda it: float(len(it.content)),
    ).apply(_bundle([a, b, *filler]), _budget())
    survivors = {it.id for it in out.items}
    assert b.id in survivors
    assert a.id not in survivors


@pytest.mark.asyncio
async def test_transitive_duplicates_collapse_to_one() -> None:
    """
    Three vectors all > threshold pairwise → only the best-scored
    survives. (Asserts the (i, j) pairwise loop catches all three pairs.)
    """
    a = _ltm("paraphrase A", _emb(1.0, 0.05, 0.05), importance=0.3)
    b = _ltm("paraphrase B", _emb(0.99, 0.1, 0.05), importance=0.5)
    c = _ltm("paraphrase C", _emb(0.98, 0.15, 0.05), importance=0.9)
    filler = [_ltm(f"f{i}", _emb(*([0.0] * (i + 3) + [1.0]))) for i in range(8)]
    out = await SemanticDedupe(
        threshold=0.92,
        skip_if_fewer_than=2,
    ).apply(_bundle([a, b, c, *filler]), _budget())
    survivors = {it.id for it in out.items}
    assert c.id in survivors
    assert a.id not in survivors
    assert b.id not in survivors


# ---------------------------------------------------------------------------
# Threshold tunability + no false positives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_higher_threshold_keeps_more() -> None:
    # Cosine ≈ 0.95 between the first two.
    a = _ltm("a", _emb(1.0, 0.0, 0.0), importance=0.3)
    b = _ltm("b", _emb(0.95, 0.31, 0.0), importance=0.9)
    filler = [_ltm(f"f{i}", _emb(*([0.0] * (i + 3) + [1.0]))) for i in range(8)]
    items = [a, b, *filler]

    permissive = await SemanticDedupe(
        threshold=0.90,
        skip_if_fewer_than=2,
    ).apply(_bundle(items), _budget())
    strict = await SemanticDedupe(
        threshold=0.99,
        skip_if_fewer_than=2,
    ).apply(_bundle(items), _budget())

    assert len(permissive.items) < len(items)  # dedupe fired
    assert len(strict.items) == len(items)  # nothing crossed threshold


@pytest.mark.asyncio
async def test_dissimilar_items_never_removed() -> None:
    items = [
        _ltm("Acme launched analytics", _emb(1.0, 0.0, 0.0)),
        _ltm("Weather is warm", _emb(0.0, 1.0, 0.0)),
        _ltm("Birds are nesting", _emb(0.0, 0.0, 1.0)),
    ] + [_ltm(f"distinct {i}", _emb(*([0.0] * (i + 4) + [1.0]))) for i in range(8)]
    out = await SemanticDedupe(
        threshold=0.92,
        skip_if_fewer_than=2,
    ).apply(_bundle(items), _budget())
    assert len(out.items) == len(items)


# ---------------------------------------------------------------------------
# Bundle plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_and_breakdown_rebuilt() -> None:
    base = _emb(1.0, 0.0)
    dup = _emb(0.99, 0.14)
    a = _ltm("removed me", base, importance=0.1)
    b = _ltm("kept me", dup, importance=0.9)
    filler = [_ltm(f"f{i}", _emb(*([0.0] * (i + 3) + [1.0]))) for i in range(8)]
    ctx = _bundle([a, b, *filler])

    out = await SemanticDedupe(skip_if_fewer_than=2).apply(ctx, _budget())

    # Messages dropped the row paired with the removed item.
    assert all(m["content"] != "removed me" for m in out.messages)
    # Tier breakdown still sums to tokens_used.
    assert sum(out.tier_breakdown.values()) == out.tokens_used
    assert out.tier_breakdown["ltm"] > 0
