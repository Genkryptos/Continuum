"""
tests/unit/test_embeddings.py
=============================
Unit tests for ``EmbeddingService``.

No ML stack and no Redis server are needed: a deterministic ``FakeEncoder``
and a dict-backed ``FakeRedis`` are injected through the constructor's
dependency-injection hooks.  Sections mirror the spec:

* Embedding generation   — shape, determinism, FP16 output
* Batching behavior      — chunking into ``batch_size`` forward passes
* Cache hits/misses      — L1, L2, in-call dedupe, cache-disabled
* halfvec conversion     — exact FP16 round-trip semantics
* Error handling         — Redis-down skip, CUDA-OOM → CPU fallback
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from continuum.core.config import EmbeddingConfig
from continuum.embeddings.service import EmbeddingService

pytestmark = pytest.mark.unit

DIM = 1024


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []  # one entry per encode() call

    @staticmethod
    def _seed(text: str) -> int:
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")

    def encode(self, sentences: list[str], *, batch_size: int = 32, **_: Any) -> np.ndarray:
        self.calls.append(list(sentences))
        rows = [np.random.default_rng(self._seed(s)).standard_normal(self.dim) for s in sentences]
        return np.asarray(rows, dtype=np.float32)


class FakeRedis:
    """Async dict-backed Redis with optional failure injection."""

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.fail = fail
        self.closed = False

    async def mget(self, keys: list[str]) -> list[str | None]:
        if self.fail:
            raise ConnectionError("redis down")
        return [self.store.get(k) for k in keys]

    async def mset(self, mapping: dict[str, str]) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.store.update(mapping)

    async def aclose(self) -> None:
        self.closed = True


def _cfg(**overrides: Any) -> EmbeddingConfig:
    base: dict[str, Any] = {
        "batch_size": 32,
        "cache_size": 1000,
        "cache_enabled": True,
        "redis_url": None,
        "device": "cpu",
        "dim": DIM,
    }
    base.update(overrides)
    return EmbeddingConfig(**base)


def _svc(
    cfg: EmbeddingConfig | None = None,
    encoder: FakeEncoder | None = None,
    redis_client: Any | None = None,
) -> tuple[EmbeddingService, FakeEncoder]:
    enc = encoder or FakeEncoder()
    svc = EmbeddingService(
        cfg or _cfg(),
        encoder_factory=lambda _device: enc,
        redis_client=redis_client,
    )
    return svc, enc


# ---------------------------------------------------------------------------
# 1. Embedding generation
# ---------------------------------------------------------------------------


class TestEmbeddingGeneration:
    async def test_returns_one_vector_per_text(self) -> None:
        svc, _ = _svc()
        out = await svc.embed(["hello", "world", "foo"])
        assert len(out) == 3
        assert all(len(v) == DIM for v in out)

    async def test_empty_input_returns_empty(self) -> None:
        svc, enc = _svc()
        assert await svc.embed([]) == []
        assert enc.calls == []

    async def test_output_is_deterministic(self) -> None:
        svc, _ = _svc()
        a = await svc.embed(["repeatable"])
        svc2, _ = _svc()
        b = await svc2.embed(["repeatable"])
        assert a == b

    async def test_output_is_fp16_rounded(self) -> None:
        svc, enc = _svc()
        out = await svc.embed(["abc"])
        raw = enc.encode(["abc"])  # same deterministic seed
        expected = np.asarray(raw, np.float32).astype(np.float16).astype(np.float32).tolist()
        assert out == expected

    async def test_dim_mismatch_raises(self) -> None:
        svc, _ = _svc(_cfg(dim=512), encoder=FakeEncoder(dim=DIM))
        with pytest.raises(ValueError, match="expected \\(\\*, 512\\)"):
            await svc.embed(["bad dim"])


# ---------------------------------------------------------------------------
# 2. Batching behavior
# ---------------------------------------------------------------------------


class TestBatching:
    async def test_splits_into_batch_size_chunks(self) -> None:
        svc, enc = _svc(_cfg(batch_size=2))
        await svc.embed(["t1", "t2", "t3", "t4", "t5"])
        # 5 texts / batch_size 2 → batches of [2, 2, 1]
        assert [len(c) for c in enc.calls] == [2, 2, 1]
        assert svc.stats["batches"] == 3
        assert svc.stats["encode_calls"] == 3

    async def test_single_batch_when_under_limit(self) -> None:
        svc, enc = _svc(_cfg(batch_size=32))
        await svc.embed(["a", "b", "c"])
        assert len(enc.calls) == 1
        assert enc.calls[0] == ["a", "b", "c"]

    async def test_in_call_dedupe(self) -> None:
        svc, enc = _svc(_cfg(batch_size=32))
        out = await svc.embed(["x", "x", "y", "x"])
        # "x" embedded once, "y" once → 2 unique texts in a single batch
        assert sorted(enc.calls[0]) == ["x", "y"]
        assert out[0] == out[1] == out[3]
        assert out[2] != out[0]


# ---------------------------------------------------------------------------
# 3. Cache hits / misses
# ---------------------------------------------------------------------------


class TestCaching:
    async def test_l1_hit_skips_encoder(self) -> None:
        svc, enc = _svc()
        await svc.embed(["cached"])
        assert svc.stats["misses"] == 1
        await svc.embed(["cached"])  # served from L1
        assert len(enc.calls) == 1  # encoder NOT called again
        assert svc.stats["hits_l1"] == 1

    async def test_partial_cache_only_embeds_misses(self) -> None:
        svc, enc = _svc()
        await svc.embed(["known"])
        enc.calls.clear()
        await svc.embed(["known", "new"])  # only "new" is a miss
        assert enc.calls == [["new"]]

    async def test_cache_disabled_always_encodes(self) -> None:
        svc, enc = _svc(_cfg(cache_enabled=False))
        await svc.embed(["z"])
        await svc.embed(["z"])
        assert len(enc.calls) == 2  # no caching

    async def test_l2_redis_hit_populates_l1(self) -> None:
        redis = FakeRedis()
        cfg = _cfg(redis_url="redis://x")
        # First service warms Redis.
        svc1, _enc1 = _svc(cfg, redis_client=redis)
        await svc1.embed(["shared"])
        assert len(redis.store) == 1

        # Fresh service (empty L1) shares the same Redis → L2 hit.
        svc2, enc2 = _svc(cfg, redis_client=redis)
        await svc2.embed(["shared"])
        assert enc2.calls == []  # never touched the model
        assert svc2.stats["hits_l2"] == 1

    async def test_key_is_sha256_of_model_and_text(self) -> None:
        svc, _ = _svc()
        expected = hashlib.sha256(b"BAAI/bge-m3" + b"\x00" + b"hi").hexdigest()
        assert svc._key("hi") == expected


# ---------------------------------------------------------------------------
# 4. halfvec conversion
# ---------------------------------------------------------------------------


class TestHalfvec:
    def test_fp16_round_trip_values(self) -> None:
        src = [[0.1, 0.2, 0.3], [1.0, -2.5, 3.14159]]
        got = EmbeddingService.to_halfvec(src)
        expected = np.asarray(src, np.float32).astype(np.float16).astype(np.float32).tolist()
        assert got == expected

    def test_precision_is_actually_reduced(self) -> None:
        # A value needing >FP16 mantissa precision must change.
        precise = 1.0 + 1e-4
        (rounded,) = EmbeddingService.to_halfvec([[precise]])[0:1]
        assert rounded[0] != precise
        assert rounded[0] == float(np.float16(precise))

    def test_shape_preserved(self) -> None:
        out = EmbeddingService.to_halfvec(np.zeros((4, DIM), np.float32))
        assert len(out) == 4
        assert len(out[0]) == DIM


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_redis_failure_is_skipped_gracefully(self) -> None:
        redis = FakeRedis(fail=True)
        svc, enc = _svc(_cfg(redis_url="redis://x"), redis_client=redis)
        out = await svc.embed(["still works"])  # must NOT raise
        assert len(out) == 1
        assert svc._redis_ok is False  # L2 latched off
        # Subsequent call doesn't retry Redis, still fine.
        await svc.embed(["again"])
        assert len(enc.calls) == 2

    async def test_transient_error_is_retried(self) -> None:
        class FlakyEncoder(FakeEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            def encode(
                self, sentences: list[str], *, batch_size: int = 32, **kw: Any
            ) -> np.ndarray:
                self.attempts += 1
                if self.attempts < 3:
                    raise ConnectionError("transient blip")
                return super().encode(sentences, batch_size=batch_size, **kw)

        flaky = FlakyEncoder()
        svc, _ = _svc(encoder=flaky)
        out = await svc.embed(["retry me"])
        assert flaky.attempts == 3
        assert len(out) == 1

    async def test_cuda_oom_falls_back_to_cpu(self) -> None:
        built: list[str] = []

        def factory(device: str) -> Any:
            """First build OOMs on encode; the CPU rebuild succeeds."""
            built.append(device)
            is_first = len(built) == 1
            good = FakeEncoder()

            class _Enc:
                def encode(
                    self, sentences: list[str], *, batch_size: int = 32, **kw: Any
                ) -> np.ndarray:
                    if is_first:
                        raise RuntimeError("CUDA out of memory")
                    return good.encode(sentences, batch_size=batch_size, **kw)

            return _Enc()

        svc = EmbeddingService(_cfg(), encoder_factory=factory)
        out = await svc.embed(["big batch"])

        assert len(out) == 1
        assert len(built) == 2  # original + CPU rebuild
        assert built[1] == "cpu"  # reload forced cpu
        assert svc._device == "cpu"

    async def test_aclose_closes_redis(self) -> None:
        redis = FakeRedis()
        svc, _ = _svc(_cfg(redis_url="redis://x"), redis_client=redis)
        await svc.embed(["warm"])
        async with svc:
            pass
        assert redis.closed is True
