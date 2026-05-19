"""
continuum/retrieval/reranker.py
===============================
``Reranker`` — a BGE cross-encoder precision pass over a recalled set
(``RerankerProtocol``).

Stage one of retrieval (bi-encoder vectors + lexical) optimises *recall* and
over-fetches. This reranks the top of that list with
``BAAI/bge-reranker-v2-m3``, which scores each ``(query, item.content)``
pair *jointly* — far more precise than cosine on independently-embedded
vectors — then re-sorts.

Process
-------
1. Take the first ``rerank_top_n`` items (the tail keeps its recall order).
2. Build ``(query, content)`` pairs, score them with the cross-encoder
   (``batch_size`` per forward pass).
3. Map raw logits → [0, 1] via sigmoid; write that into each item's
   ``ScoreBreakdown`` (``relevance`` + ``composite``) so ``ScoredItem.score``
   reflects the rerank, and stash provenance in ``item.metadata['rerank']``.
4. Sort the head desc by rerank score; return head + untouched tail.

Optimisation / robustness
-------------------------
* **Skip** entirely when ``len(items) < skip_if_fewer_than`` — the model
  latency is not worth it for a tiny set (and the model is never loaded).
* The blocking ``predict`` runs in a worker thread (``asyncio.to_thread``)
  so reranking never blocks the event loop.
* Lazy, cached model load; ``auto`` device → cuda if available else cpu.
* **Graceful**: any model/inference failure logs and returns the input
  order unchanged — reranking can only improve, never break, retrieval.

``sentence-transformers``/``torch`` are imported lazily; ``model_factory``
is injectable so unit tests need neither.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from continuum.core.config import RerankerConfig

log = logging.getLogger(__name__)


class CrossEncoderModel(Protocol):
    """Structural type for the bit of sentence-transformers we use."""

    def predict(
        self, sentences: list[list[str]], *, batch_size: int = ..., **kw: Any
    ) -> Any:  # np.ndarray | list[float]
        ...


#: ``model_factory(model_name, device) -> CrossEncoderModel``. Injected in
#: tests; the default lazily builds a real ``CrossEncoder``.
ModelFactory = Callable[[str, str], CrossEncoderModel]


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _to_floats(raw: Any) -> list[float]:
    """Coerce a CrossEncoder.predict result (np array / list / scalar)."""
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(raw)]


class Reranker:
    """
    Cross-encoder reranker. Stateless given its config; one instance is safe
    to share across queries.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.RerankerConfig`.
    model_factory:
        ``(model_name, device) -> CrossEncoderModel``. Inject a fake in
        tests; the default lazily imports/builds a real ``CrossEncoder``.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or RerankerConfig()
        self._factory = model_factory
        self._model: CrossEncoderModel | None = None
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
                log.debug("torch unavailable — reranking on cpu")
            if want == "cuda":
                log.warning("device='cuda' requested but unavailable — cpu")
            return "cpu"
        return "cpu"

    def _build_model(self, device: str) -> CrossEncoderModel:
        if self._factory is not None:
            return self._factory(self.config.model_name, device)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - via factory in tests
            raise ImportError(
                "sentence-transformers is required for Reranker.\n"
                "Install it with:  pip install 'continuum-memory[embed]'"
            ) from exc
        log.info(
            "loading cross-encoder %s on %s", self.config.model_name, device
        )
        # sentence-transformers ships no stubs → CrossEncoder is Any.
        return cast(
            "CrossEncoderModel", CrossEncoder(self.config.model_name, device=device)
        )

    async def _get_model(self) -> CrossEncoderModel:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._device = self._resolve_device()
                self._model = await asyncio.to_thread(
                    self._build_model, self._device
                )
        return self._model

    # ── RerankerProtocol ────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        items: Sequence[Any],
    ) -> list[Any]:
        """Re-order *items* best-first by cross-encoder score (see module doc)."""
        from continuum.core.types import ScoreBreakdown, ScoredItem

        ordered: list[ScoredItem] = list(items)
        if len(ordered) < self.config.skip_if_fewer_than:
            return ordered  # too few → skip (model never loaded)

        head = ordered[: self.config.rerank_top_n]
        tail = ordered[self.config.rerank_top_n :]
        if not head:
            return ordered

        pairs = [[query, it.item.content] for it in head]
        try:
            model = await self._get_model()
            raw = await asyncio.to_thread(
                lambda: model.predict(
                    pairs, batch_size=self.config.batch_size
                )
            )
            logits = _to_floats(raw)
            if len(logits) != len(head):
                raise ValueError(
                    f"cross-encoder returned {len(logits)} scores for "
                    f"{len(head)} pairs"
                )
        except Exception:
            log.exception("rerank failed — returning recall order unchanged")
            return ordered

        scored: list[tuple[float, ScoredItem]] = []
        for it, logit in zip(head, logits, strict=True):
            rr = _sigmoid(logit)
            it.item.metadata["rerank"] = {
                "cross_encoder_logit": logit,
                "cross_encoder_score": rr,
                "prev_composite": it.scores.composite,
            }
            scored.append(
                (
                    rr,
                    ScoredItem(
                        item=it.item,
                        scores=ScoreBreakdown(
                            relevance=rr,
                            importance=it.scores.importance,
                            recency=it.scores.recency,
                            confidence=it.scores.confidence,
                            composite=rr,
                        ),
                    ),
                )
            )

        # Stable sort: ties keep their recall order.
        scored.sort(key=lambda p: p[0], reverse=True)
        return [si for _, si in scored] + tail


__all__ = ["Reranker"]
