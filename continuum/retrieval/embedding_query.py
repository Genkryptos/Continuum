"""
continuum.retrieval.embedding_query
===================================
``EmbeddingQueryRetriever`` — wraps any retriever and embeds ``query.text`` once
(when the query arrives without an embedding) so the LTM **dense** channel fires.

Without it, :class:`~continuum.retrieval.Retriever` sees ``query.embedding is
None`` and silently runs sparse + recency only — which is why a session with no
retriever (or a bare one) returns recency-ordered, not relevance-ranked, results.
This is the piece that makes semantic recall work end-to-end; it is shared by the
chat REPL and :meth:`continuum.Memory.from_postgres`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from continuum.core.protocols import RetrieverProtocol
    from continuum.core.types import ContextBundle, Query, TokenBudget

__all__ = ["EmbeddingQueryRetriever"]


class _Embedder(Protocol):
    """Minimal structural type for the local embedding model."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingQueryRetriever:
    """Embed ``query.text`` (once) before delegating, so dense recall fires.

    Degrades gracefully: if the embedder is absent or raises, the query passes
    through unchanged and the inner retriever does what it can on its own
    (sparse + recency).
    """

    def __init__(self, inner: RetrieverProtocol, embedder: _Embedder | None) -> None:
        self._inner = inner
        self._embedder = embedder

    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle:
        if self._embedder is not None and query.embedding is None and query.text:
            try:
                vec = (await self._embedder.embed([query.text]))[0]
                query = replace(query, embedding=vec)
            except Exception:
                pass  # degrade to sparse-only retrieval
        return await self._inner.retrieve(query, budget)
