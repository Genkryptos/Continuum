"""
tests/unit/test_reranker.py
===========================
Unit tests for ``continuum.retrieval.reranker.Reranker``. A fake
CrossEncoder is injected via ``model_factory`` — no sentence-transformers /
torch / network.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import pytest

from continuum.core.config import RerankerConfig
from continuum.core.protocols import RerankerProtocol
from continuum.core.types import MemoryItem, MemoryTier, ScoreBreakdown, ScoredItem
from continuum.retrieval import Reranker

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _si(text: str, composite: float = 0.5) -> ScoredItem:
    return ScoredItem(
        item=MemoryItem(content=text, tier=MemoryTier.LTM),
        scores=ScoreBreakdown(
            relevance=composite, importance=0.3, recency=0.4,
            confidence=0.6, composite=composite,
        ),
    )


class FakeCE:
    """Returns a configured logit per document text."""

    def __init__(self, logits: dict[str, float], *, sleep: float = 0.0) -> None:
        self._logits = logits
        self._sleep = sleep
        self.calls: list[tuple[int, int]] = []  # (n_pairs, batch_size)

    def predict(
        self, sentences: list[list[str]], *, batch_size: int = 32, **_: Any
    ) -> list[float]:
        if self._sleep:
            time.sleep(self._sleep)            # blocking — must run in a thread
        self.calls.append((len(sentences), batch_size))
        return [self._logits.get(doc, 0.0) for _q, doc in sentences]


def _rr(
    logits: dict[str, float] | None = None,
    *,
    factory: Any | None = None,
    **cfg: Any,
) -> tuple[Reranker, FakeCE | None]:
    fake = FakeCE(logits or {}) if factory is None else None
    fac = factory if factory is not None else (lambda _m, _d: fake)
    return Reranker(RerankerConfig(**cfg), model_factory=fac), fake


# ---------------------------------------------------------------------------
# Protocol / config
# ---------------------------------------------------------------------------


def test_satisfies_reranker_protocol() -> None:
    rr, _ = _rr()
    assert isinstance(rr, RerankerProtocol)


def test_config_defaults() -> None:
    c = RerankerConfig()
    assert c.model_name == "BAAI/bge-reranker-v2-m3"
    assert c.batch_size == 32
    assert c.rerank_top_n == 50
    assert c.skip_if_fewer_than == 10


# ---------------------------------------------------------------------------
# Order improvement
# ---------------------------------------------------------------------------


class TestReorder:
    async def test_relevant_item_moves_up(self) -> None:
        # Recall order is poor: the truly relevant doc is last.
        items = [_si(f"junk-{i}") for i in range(10)] + [_si("the answer")]
        logits = {"the answer": 8.0}            # everything else → 0.0
        rr, _ = _rr(logits, skip_if_fewer_than=2)

        out = await rr.rerank("q", items)

        assert out[0].item.content == "the answer"
        assert out[0].score > out[1].score      # composite now = sigmoid(logit)
        assert out[0].score == pytest.approx(1.0 / (1.0 + math.exp(-8.0)))
        # Provenance recorded on the item.
        meta = out[0].item.metadata["rerank"]
        assert meta["cross_encoder_logit"] == 8.0
        assert meta["prev_composite"] == 0.5

    async def test_same_items_returned_just_reordered(self) -> None:
        items = [_si("a"), _si("b"), _si("c")] + [_si(f"x{i}") for i in range(8)]
        rr, _ = _rr({"c": 5.0, "a": 1.0})
        out = await rr.rerank("q", items)
        assert {i.item.content for i in out} == {i.item.content for i in items}
        assert len(out) == len(items)
        assert out[0].item.content == "c"       # highest logit first

    async def test_stable_for_ties(self) -> None:
        items = [_si("p"), _si("q"), _si("r")] + [_si(f"z{i}") for i in range(8)]
        rr, _ = _rr({})                          # all logits 0.0 → all tie
        out = await rr.rerank("query", items)
        assert [i.item.content for i in out[:3]] == ["p", "q", "r"]  # stable


# ---------------------------------------------------------------------------
# Batching + top-N
# ---------------------------------------------------------------------------


class TestBatchingAndTopN:
    async def test_batch_size_forwarded(self) -> None:
        items = [_si(f"d{i}") for i in range(12)]
        rr, fake = _rr({}, batch_size=8)
        await rr.rerank("q", items)
        assert fake is not None
        assert fake.calls == [(12, 8)]           # all pairs, configured batch

    async def test_only_top_n_reranked_tail_preserved(self) -> None:
        items = [_si(f"item-{i}") for i in range(20)]
        # Give the LAST in-window item the top logit; tail (>=5) untouched.
        rr, fake = _rr({"item-4": 9.0}, rerank_top_n=5, skip_if_fewer_than=2)
        out = await rr.rerank("q", items)

        assert fake is not None
        assert fake.calls[0][0] == 5             # only 5 pairs scored
        assert out[0].item.content == "item-4"   # reranked into the lead
        # The tail (items 5..19) keeps its original recall order.
        assert [i.item.content for i in out[5:]] == [
            f"item-{i}" for i in range(5, 20)
        ]


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


class TestSkip:
    async def test_skips_when_too_few(self) -> None:
        called = {"built": False}

        def factory(_m: str, _d: str) -> Any:
            called["built"] = True
            return FakeCE({})

        rr = Reranker(
            RerankerConfig(skip_if_fewer_than=10), model_factory=factory
        )
        items = [_si(f"d{i}") for i in range(9)]   # 9 < 10 → skip
        out = await rr.rerank("q", items)

        assert out == items                        # identical order, untouched
        assert called["built"] is False            # model never loaded

    async def test_runs_at_threshold(self) -> None:
        rr, fake = _rr({"d0": 3.0}, skip_if_fewer_than=10)
        items = [_si(f"d{i}") for i in range(10)]   # exactly 10 → run
        out = await rr.rerank("q", items)
        assert fake is not None and fake.calls      # model was used
        assert out[0].item.content == "d0"


# ---------------------------------------------------------------------------
# Async non-blocking
# ---------------------------------------------------------------------------


class TestAsync:
    async def test_predict_runs_off_event_loop(self) -> None:
        # Blocking 0.2s predict must not stall a concurrent coroutine.
        items = [_si(f"d{i}") for i in range(10)]
        fake = FakeCE({"d0": 5.0}, sleep=0.20)
        rr = Reranker(RerankerConfig(skip_if_fewer_than=2),
                      model_factory=lambda _m, _d: fake)

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.01)
                ticks += 1

        out, _ = await asyncio.gather(rr.rerank("q", items), ticker())
        assert out[0].item.content == "d0"
        assert ticks >= 10            # loop kept running during predict()


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGraceful:
    async def test_model_build_failure_returns_recall_order(self) -> None:
        def boom(_m: str, _d: str) -> Any:
            raise RuntimeError("download failed")

        rr = Reranker(RerankerConfig(skip_if_fewer_than=2),
                      model_factory=boom)
        items = [_si(f"d{i}") for i in range(10)]
        out = await rr.rerank("q", items)
        assert out == items                        # unchanged, no raise

    async def test_predict_failure_returns_recall_order(self) -> None:
        class Broken:
            def predict(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("CUDA OOM")

        rr = Reranker(RerankerConfig(skip_if_fewer_than=2),
                      model_factory=lambda _m, _d: Broken())
        items = [_si(f"d{i}") for i in range(10)]
        assert await rr.rerank("q", items) == items

    async def test_length_mismatch_is_graceful(self) -> None:
        class WrongLen:
            def predict(self, sentences: list[list[str]],
                        **_k: Any) -> list[float]:
                return [1.0]                       # too few scores

        rr = Reranker(RerankerConfig(skip_if_fewer_than=2),
                      model_factory=lambda _m, _d: WrongLen())
        items = [_si(f"d{i}") for i in range(10)]
        assert await rr.rerank("q", items) == items
