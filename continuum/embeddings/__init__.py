"""
continuum.embeddings
====================
Embedding generation with batching, caching, and storage-efficient FP16.

Exports
-------
EmbeddingService — async, batched, LRU(+optional Redis)-cached BGE-M3
                   embedder that emits ``halfvec``-ready FP16 vectors.

The heavy ML stack (``sentence-transformers`` / ``torch``) and the optional
``redis`` client are imported lazily, so importing this package is cheap and
unit tests can inject fakes without those packages installed.
"""

from __future__ import annotations

from continuum.embeddings.service import EmbeddingService

__all__ = ["EmbeddingService"]
