"""
tests/unit/evals/test_scorer_weights.py
=======================================
Unit tests for the Prompt-38 composite scorer added to
``evals.longmemeval.bootstrap_ollama.STMSemanticRetriever``.

Covers ``_parse_scorer_weights``, ``_parse_lme_date``, the recency
decay, and that the weighted composite actually reorders top-K results.
Runs offline with a deterministic fake embedder.
"""
from __future__ import annotations

import datetime as dt
import uuid

import numpy as np
import pytest

from continuum.core.types import MemoryItem, MemoryTier, Query, TokenBudget
from evals.longmemeval.bootstrap_ollama import (
    FlatHaystackStore,
    STMSemanticRetriever,
    _parse_lme_date,
    _parse_scorer_weights,
)


class _FixedEmbedder:
    """
    Returns caller-supplied vectors keyed by exact text. Any text not in
    the table gets a deterministic hashed 3-dim vector — enough for
    ``_refresh_index`` to populate the cache when the test only cares
    about recency, not relevance.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t in self.table:
                rows.append(self.table[t])
            else:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                rows.append(rng.normal(size=3).tolist())
        return np.asarray(rows, dtype=np.float32)


def _budget() -> TokenBudget:
    return TokenBudget(
        total=8000, stm_reserved=500, mtm_reserved=500,
        ltm_reserved=2000, response_reserved=500,
    )


# ---------------------------------------------------------------------------
# _parse_scorer_weights
# ---------------------------------------------------------------------------


def test_parse_weights_normalises_to_one() -> None:
    w = _parse_scorer_weights("0.45,0.25,0.20,0.10")
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["relevance"] == pytest.approx(0.45)


def test_parse_weights_renormalises_off_sum() -> None:
    # Sums to 2.0 → each halved.
    w = _parse_scorer_weights("0.9,0.5,0.4,0.2")
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["relevance"] == pytest.approx(0.45)


def test_parse_weights_bad_input_falls_back() -> None:
    w = _parse_scorer_weights("garbage")
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["relevance"] == pytest.approx(0.45)   # defaults

    w2 = _parse_scorer_weights("0.5,0.5")          # wrong count
    assert w2["relevance"] == pytest.approx(0.45)


def test_parse_weights_retrieval_noise_branch() -> None:
    # The Prompt-38 "retrieval noise" decision-tree branch.
    w = _parse_scorer_weights("0.55,0.20,0.15,0.10")
    assert w["relevance"] == pytest.approx(0.55)
    assert w["recency"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# _parse_lme_date
# ---------------------------------------------------------------------------


def test_parse_date_with_day_token() -> None:
    d = _parse_lme_date("2023/05/30 (Tue) 23:40")
    assert d == dt.datetime(2023, 5, 30, 23, 40)


def test_parse_date_variants() -> None:
    assert _parse_lme_date("2023/05/30") == dt.datetime(2023, 5, 30)
    assert _parse_lme_date("2023-05-30 09:15") == dt.datetime(2023, 5, 30, 9, 15)


def test_parse_date_bad_input_returns_none() -> None:
    assert _parse_lme_date("not a date") is None
    assert _parse_lme_date("") is None
    assert _parse_lme_date(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recency_uniform_when_no_dates() -> None:
    store = FlatHaystackStore()
    for i in range(4):
        await store.append(MemoryItem(
            id=str(uuid.uuid4()), content=f"m{i}", tier=MemoryTier.STM,
        ))
    r = STMSemanticRetriever(
        store=store,
        embedder=_FixedEmbedder({}),
    )
    r._refresh_index()
    rec = r._recency_scores()
    # No meaningful timestamp spread → all 1.0.
    assert np.allclose(rec, 1.0)


@pytest.mark.asyncio
async def test_recency_decays_with_age() -> None:
    base = dt.datetime(2025, 1, 1, 12, 0)
    store = FlatHaystackStore()
    # Three items: newest, 7 days old, 30 days old.
    ages_days = [0, 7, 30]
    for i, days in enumerate(ages_days):
        it = MemoryItem(id=str(uuid.uuid4()), content=f"m{i}", tier=MemoryTier.STM)
        it.created_at = base - dt.timedelta(days=days)
        await store.append(it)
    r = STMSemanticRetriever(
        store=store, embedder=_FixedEmbedder({}), tau_hours=168.0,  # 7-day
    )
    r._refresh_index()
    rec = r._recency_scores()
    # Newest ≈ 1.0; 7-day-old ≈ e^-1 ≈ 0.368; 30-day-old much smaller.
    assert rec[0] == pytest.approx(1.0, abs=1e-6)
    assert rec[1] == pytest.approx(np.e ** -1, abs=0.02)
    assert rec[2] < rec[1] < rec[0]


# ---------------------------------------------------------------------------
# Composite reordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recency_weight_reorders_topk() -> None:
    """
    Two items with *identical* cosine relevance but different ages.
    With recency weight > 0, the newer one must rank first.
    """
    base = dt.datetime(2025, 1, 1, 12, 0)
    store = FlatHaystackStore()
    old = MemoryItem(id="old", content="OLD", tier=MemoryTier.STM)
    old.created_at = base - dt.timedelta(days=60)
    new = MemoryItem(id="new", content="NEW", tier=MemoryTier.STM)
    new.created_at = base
    await store.append(old)
    await store.append(new)

    # Both items embed to the SAME vector → identical cosine to the query.
    same = [1.0, 0.0, 0.0]
    embedder = _FixedEmbedder({"OLD": same, "NEW": same, "q": same})

    # Pure relevance (recency weight 0) — order is arbitrary/stable.
    r_norec = STMSemanticRetriever(
        store=store, embedder=embedder, top_k=1,
        score_weights={"relevance": 1.0, "importance": 0.0,
                       "recency": 0.0, "confidence": 0.0},
    )
    ctx0 = await r_norec.retrieve(Query(text="q"), _budget())
    # With recency weighting, the NEW item must win the single slot.
    r_rec = STMSemanticRetriever(
        store=store, embedder=embedder, top_k=1, tau_hours=168.0,
        score_weights={"relevance": 0.5, "importance": 0.0,
                       "recency": 0.5, "confidence": 0.0},
    )
    ctx1 = await r_rec.retrieve(Query(text="q"), _budget())
    assert ctx1.items[0].id == "new", (
        "recency weight should promote the newer item to the top slot"
    )
    # ctx0 just needs to have returned something (order undefined at tie).
    assert len(ctx0.items) == 1
