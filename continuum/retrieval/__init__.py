"""
continuum.retrieval
===================
Retrieval-time refinement.

Exports
-------
Retriever — the full ``RetrieverProtocol`` pipeline (LTM hybrid → graph →
            score → rerank → STM/MTM → ContextBundle).
Reranker  — ``RerankerProtocol`` cross-encoder (BGE-reranker-v2-m3) second
            pass that re-sorts a recalled candidate set for precision.

``sentence-transformers`` / ``torch`` are imported lazily, so importing this
package is cheap and unit tests inject a fake model without those packages.
"""
from __future__ import annotations

from continuum.retrieval.reranker import Reranker
from continuum.retrieval.retriever import Retriever

__all__ = ["Retriever", "Reranker"]
