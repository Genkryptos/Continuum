"""
continuum/embeddings/service.py
===============================
``EmbeddingService`` — async, batched, cached text embedding for Continuum.

What it does
------------
* **Async API** — ``await service.embed([...])`` returns one vector per text.
* **Batching** — misses are grouped into ``batch_size`` chunks (default 32)
  before hitting the model, so a 1 000-text call is a handful of forward
  passes, not 1 000.
* **Two-level cache** — an in-process ``cachetools.LRUCache`` (L1) plus an
  optional shared Redis cache (L2).  Key = ``sha256(model_name + text)``.
* **BGE-M3** — defaults to ``BAAI/bge-m3`` (1024-dim, multilingual).  The
  output dimensionality is validated against ``config.dim``.
* **Storage-efficient FP16** — vectors are down-cast to ``float16`` (the
  ``halfvec`` representation pgvector stores) before they leave the service.

Operational robustness
----------------------
* The heavy stack (``sentence-transformers`` / ``torch``) is **lazy-loaded**
  on first real embed, so importing this module is cheap and tests can
  inject a fake encoder without those packages installed.
* **CUDA OOM → CPU**: an out-of-memory error transparently reloads the model
  on CPU and retries the batch once.
* **Transient failures** are retried with exponential backoff (tenacity).
* **Redis is best-effort**: any Redis error disables L2 for the rest of the
  process and falls back to L1 — embedding never fails because the cache is
  down.

Dependency injection (testing)
------------------------------
    svc = EmbeddingService(
        config,
        encoder_factory=lambda device: FakeEncoder(),  # no torch needed
        redis_client=FakeRedis(),                       # no redis needed
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from cachetools import LRUCache
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from continuum.core.config import EmbeddingConfig

log = logging.getLogger(__name__)

#: Redis key namespace so embedding keys never collide with other app data.
_REDIS_PREFIX = "continuum:emb:"


class Encoder(Protocol):
    """Structural type for anything sentence-transformers-shaped."""

    def encode(
        self, sentences: list[str], *, batch_size: int = ..., **kwargs: Any
    ) -> Any:  # numpy array / list of lists
        ...


#: ``encoder_factory(device) -> Encoder``.  Injected in tests; the default
#: builds a real ``SentenceTransformer`` lazily.
EncoderFactory = Callable[[str], Encoder]


def _is_oom(exc: BaseException) -> bool:
    """Heuristically classify *exc* as a CUDA out-of-memory error."""
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _is_transient(exc: BaseException) -> bool:
    """
    Retry policy predicate.

    OOM is handled by the CPU-fallback path (not retried on the same
    device); ``ValueError`` is a programming/contract error.  Everything
    else (network blips loading weights, transient CUDA faults) is retried.
    """
    if _is_oom(exc):
        return False
    if isinstance(exc, ValueError):
        return False
    return True


class EmbeddingService:
    """
    Async batched embedding with LRU + optional Redis caching.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.EmbeddingConfig` instance.
    encoder_factory:
        ``(device) -> Encoder``.  Defaults to a lazy ``SentenceTransformer``
        loader.  Inject a fake here in unit tests.
    redis_client:
        Pre-built async Redis client (``.get``/``.set`` coroutines).  When
        omitted and ``config.redis_url`` is set, a ``redis.asyncio`` client
        is created lazily on first use.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        encoder_factory: EncoderFactory | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self.config = config
        self._encoder_factory = encoder_factory
        self._encoder: Encoder | None = None
        self._device: str | None = None
        self._model_lock = asyncio.Lock()

        # L1 — in-process LRU. Values are FP16-rounded list[float].
        self._lru: LRUCache[str, list[float]] = LRUCache(maxsize=max(1, config.cache_size))

        # L2 — optional Redis. ``_redis_ok`` latches False on first failure.
        self._redis = redis_client
        self._redis_ok = bool(redis_client) or bool(config.redis_url)

        # Lightweight observability for tests / dashboards.
        self.stats: dict[str, int] = {
            "hits_l1": 0,
            "hits_l2": 0,
            "misses": 0,
            "encode_calls": 0,
            "batches": 0,
        }

    # ── Cache key ───────────────────────────────────────────────────────────

    def _key(self, text: str) -> str:
        """``sha256(model_name + "\\0" + text)`` — stable across processes."""
        h = hashlib.sha256()
        h.update(self.config.model_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    # ── halfvec (FP16) conversion ───────────────────────────────────────────

    @staticmethod
    def to_halfvec(
        vectors: npt.NDArray[Any] | Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """
        Down-cast float32 vectors to ``float16`` (pgvector ``halfvec``).

        Returns native Python floats whose values are exactly representable
        in FP16, halving storage versus full ``vector`` columns.
        """
        arr = np.asarray(vectors, dtype=np.float32).astype(np.float16)
        # 2-D float32 array → list[list[float]]; numpy types tolist() as a
        # dimensionality-agnostic union, so narrow it explicitly.
        return cast(
            "list[list[float]]", arr.astype(np.float32).tolist()
        )  # JSON/DB-friendly py floats

    # ── Public API ──────────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Return one FP16 embedding per input text, in input order.

        Identical texts within a single call are embedded only once.  Cache
        lookups (L1 then L2) precede any model work; only the misses are
        batched and sent to the model.

        Passing a bare ``str`` raises. A string is iterable, so it would
        otherwise be embedded *per character* and ``embed(text)[0]`` would
        quietly return the vector for its first letter — a plausible-looking
        1024-dim unit vector that ranks nothing correctly. Silent nonsense is
        worse than a TypeError.
        """
        if isinstance(texts, str):
            raise TypeError(
                "embed() takes a list of strings, not a single string — "
                f"use embed([{texts[:32]!r}]) (a str would be embedded per character)."
            )
        if not texts:
            return []

        keys = [self._key(t) for t in texts]
        results: dict[str, list[float]] = {}

        # ── L1: in-process LRU ───────────────────────────────────────────────
        if self.config.cache_enabled:
            for k in keys:
                if k in self._lru:
                    results[k] = self._lru[k]
                    self.stats["hits_l1"] += 1

        # ── L2: Redis (only for keys still missing) ──────────────────────────
        missing = [k for k in dict.fromkeys(keys) if k not in results]
        if missing and self.config.cache_enabled and self._redis_ok:
            for k, vec in zip(missing, await self._redis_mget(missing), strict=True):
                if vec is not None:
                    results[k] = vec
                    self._lru[k] = vec
                    self.stats["hits_l2"] += 1
            missing = [k for k in missing if k not in results]

        # ── Model: embed the still-missing unique texts ──────────────────────
        if missing:
            # Map each missing key to one representative source text.
            key_to_text: dict[str, str] = {}
            for k, t in zip(keys, texts, strict=True):
                if k in missing and k not in key_to_text:
                    key_to_text[k] = t

            order = list(key_to_text)
            to_embed = [key_to_text[k] for k in order]
            self.stats["misses"] += len(order)

            vectors = await self._embed_uncached(to_embed)

            for k, vec in zip(order, vectors, strict=True):
                results[k] = vec
                if self.config.cache_enabled:
                    self._lru[k] = vec
            if self.config.cache_enabled and self._redis_ok:
                await self._redis_mset({k: results[k] for k in order})

        return [results[k] for k in keys]

    # ── Model loading (lazy) ────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Resolve ``config.device`` ('auto'/'cuda'/'cpu') to a concrete device."""
        want = self.config.device
        if want == "cpu":
            return "cpu"
        if want in ("auto", "cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                log.debug("torch unavailable — using cpu")
            if want == "cuda":
                log.warning("device='cuda' requested but unavailable — using cpu")
            return "cpu"
        return "cpu"

    def _build_encoder(self, device: str) -> Encoder:
        """Construct an encoder for *device* (injected factory or real model)."""
        if self._encoder_factory is not None:
            return self._encoder_factory(device)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via factory
            raise ImportError(
                "sentence-transformers is required for EmbeddingService.\n"
                "Install it with:  pip install 'continuum-memory[embed]'"
            ) from exc

        log.info("loading embedding model %s on %s", self.config.model_name, device)
        # SentenceTransformer is untyped (mypy sees Any); pin to Encoder so
        # the declared return type is honoured under --strict no-any-return.
        encoder: Encoder = SentenceTransformer(self.config.model_name, device=device)
        return encoder

    async def _get_encoder(self) -> Encoder:
        """Lazily build (once) and return the encoder."""
        if self._encoder is not None:
            return self._encoder
        async with self._model_lock:
            if self._encoder is None:
                self._device = self._resolve_device()
                self._encoder = await asyncio.to_thread(self._build_encoder, self._device)
        return self._encoder

    async def _reload_on_cpu(self) -> Encoder:
        """Force-rebuild the encoder on CPU (CUDA-OOM recovery path)."""
        async with self._model_lock:
            log.warning("CUDA OOM — reloading embedding model on cpu")
            self._device = "cpu"
            self._encoder = await asyncio.to_thread(self._build_encoder, "cpu")
        return self._encoder

    # ── Encoding ────────────────────────────────────────────────────────────

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Batch *texts*, run the model with retry + OOM fallback, return FP16."""
        out: list[list[float]] = []
        bs = max(1, self.config.batch_size)
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            self.stats["batches"] += 1
            raw = await self._encode_batch_resilient(batch)
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.config.dim:
                raise ValueError(
                    f"embedding model returned shape {arr.shape}; "
                    f"expected (*, {self.config.dim}) for {self.config.model_name}"
                )
            out.extend(self.to_halfvec(arr))
        return out

    async def _encode_batch_resilient(self, batch: list[str]) -> Any:
        """
        Encode one batch with exponential-backoff retry on transient errors
        and a one-shot CPU fallback on CUDA OOM.
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=0.2, max=5.0),
                retry=retry_if_exception(_is_transient),
                reraise=True,
            ):
                with attempt:
                    return await self._encode_once(batch)
        except Exception as exc:
            if _is_oom(exc):
                await self._reload_on_cpu()
                return await self._encode_once(batch)
            raise
        raise RuntimeError("unreachable encode retry exit")  # pragma: no cover

    async def _encode_once(self, batch: list[str]) -> Any:
        """Single blocking ``encoder.encode`` call, off the event loop."""
        encoder = await self._get_encoder()
        self.stats["encode_calls"] += 1
        return await asyncio.to_thread(
            lambda: encoder.encode(batch, batch_size=self.config.batch_size)
        )

    # ── Redis (best-effort L2) ──────────────────────────────────────────────

    async def _ensure_redis(self) -> Any | None:
        """Lazily create the async Redis client; latch off on failure."""
        if self._redis is not None or not self._redis_ok:
            return self._redis
        if not self.config.redis_url:
            self._redis_ok = False
            return None
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.config.redis_url, decode_responses=True)
        except Exception:
            log.warning("redis client init failed — L2 cache disabled")
            self._redis_ok = False
            self._redis = None
        return self._redis

    async def _redis_mget(self, keys: list[str]) -> list[list[float] | None]:
        """Best-effort multi-get; any failure disables L2 and returns misses."""
        client = await self._ensure_redis()
        if client is None:
            return [None] * len(keys)
        try:
            raw = await client.mget([_REDIS_PREFIX + k for k in keys])
        except Exception:
            log.warning("redis mget failed — disabling L2 cache", exc_info=True)
            self._redis_ok = False
            return [None] * len(keys)
        out: list[list[float] | None] = []
        for item in raw:
            try:
                out.append(json.loads(item) if item else None)
            except (TypeError, ValueError):
                out.append(None)
        return out

    async def _redis_mset(self, mapping: dict[str, list[float]]) -> None:
        """Best-effort multi-set; failure disables L2 without raising."""
        client = await self._ensure_redis()
        if client is None:
            return
        try:
            payload = {_REDIS_PREFIX + k: json.dumps(v) for k, v in mapping.items()}
            await client.mset(payload)
        except Exception:
            log.warning("redis mset failed — disabling L2 cache", exc_info=True)
            self._redis_ok = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Release the Redis connection if one was created."""
        if self._redis is not None:
            closer = getattr(self._redis, "aclose", None) or getattr(self._redis, "close", None)
            if closer is not None:
                try:
                    res = closer()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    log.debug("error closing redis client", exc_info=True)

    async def __aenter__(self) -> EmbeddingService:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()


__all__ = ["EmbeddingService", "Encoder", "EncoderFactory"]
