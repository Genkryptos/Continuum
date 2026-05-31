"""
continuum/retrieval/bm25.py
===========================
BM25 retrieval — lexical sibling to the cosine-similarity path used in
:mod:`continuum.retrieval.retriever`. The cosine retriever wins on
paraphrases; BM25 wins on rare-token / proper-noun queries where the
embedding model spreads probability mass across topically-related but
non-matching content. The pair is fused later in
:mod:`continuum.retrieval.rrf`.

The module exposes two layers:

* :class:`BM25Index` — pure algorithm. Takes a list of
  :class:`MemoryItem` and answers ``query(text, k)`` with a list of
  :class:`ScoredItem`. No I/O, no async.
* :class:`BM25Retriever` — :class:`RetrieverProtocol` implementation
  on top of the index. Takes a ``corpus_provider`` callable that
  returns the items to score (sync or async). Returns a
  :class:`ContextBundle` so the result composes one-for-one with the
  cosine retriever in :class:`ReciprocalRankFusion`.

Tokenization (v1)
-----------------
Lowercase + a single ``\\w+`` regex split. No stemming, no stop-word
filter, no case folding beyond ``.lower()``. The simplicity is
deliberate — we want the BM25 path to react to *exactly* the user's
surface terms, leaving paraphrase-recovery to the embedding side of
the fusion. Stemming + stop-word filters can be added later if smoke
data shows token-recall is hurting precision.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from rank_bm25 import BM25Okapi

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    Query,
    ScoreBreakdown,
    ScoredItem,
    TokenBudget,
)

log = logging.getLogger(__name__)

#: Word-splitter — matches runs of word characters (``[A-Za-z0-9_]+``).
#: ``re.UNICODE`` is implicit on Python 3 so accented letters survive,
#: which matters for LongMemEval rows with non-ASCII names.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase + regex word-split. Empty strings → ``[]``."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Pure index
# ---------------------------------------------------------------------------


class BM25Index:
    """
    Pure BM25 over an in-memory corpus.

    Build once per ``retrieve()`` call (the corpus is small for
    LongMemEval row sizes; the full re-index avoids stale state
    between rows). Pass an empty list for an empty corpus — queries
    against an empty index always return ``[]``.

    Score normalisation
    -------------------
    Raw BM25 scores are unbounded and corpus-dependent — comparing
    them across queries is meaningless. The index normalises every
    query's score vector into ``[0, 1]`` via min-max so the value
    that lands in :class:`ScoreBreakdown.relevance` is comparable
    across runs and queries. RRF uses ranks, not scores, so this is
    safe — it only affects callers reading the breakdown directly.
    """

    def __init__(self, items: list[MemoryItem]):
        self._items: list[MemoryItem] = list(items)
        tokenized = [tokenize(it.content) for it in self._items]
        # rank_bm25 raises on an empty corpus — guard so empty-corpus
        # callers get a fast empty-result path instead of an exception.
        self._bm25: BM25Okapi | None = BM25Okapi(tokenized) if tokenized else None

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None or not self._items

    def query(self, query_text: str, k: int) -> list[ScoredItem]:
        """Top-``k`` items by BM25 against ``query_text``."""
        if self.is_empty or k <= 0:
            return []
        q_tokens = tokenize(query_text)
        if not q_tokens:
            return []
        assert self._bm25 is not None  # narrowed by is_empty above
        raw_scores = self._bm25.get_scores(q_tokens)
        # Sort indices by raw score desc. Stable sort so ties keep
        # corpus order — deterministic across runs.
        order = sorted(
            range(len(self._items)),
            key=lambda i: (-float(raw_scores[i]), i),
        )
        top = order[:k]
        norm = _minmax(raw_scores[i] for i in top)
        out: list[ScoredItem] = []
        for i, rel in zip(top, norm, strict=True):
            sb = ScoreBreakdown.compute(
                relevance=rel,
                importance=float(self._items[i].importance),
                recency=0.0,  # BM25 has no recency notion; leave for scorer.
                confidence=float(self._items[i].confidence),
            )
            out.append(ScoredItem(item=self._items[i], scores=sb))
        return out


def _minmax(values: Any) -> list[float]:
    nums = [float(v) for v in values]
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        # All ties (often all zeros for misses) — flat 0.0 row.
        return [0.0 for _ in nums]
    span = hi - lo
    return [(v - lo) / span for v in nums]


# ---------------------------------------------------------------------------
# Protocol-conforming retriever
# ---------------------------------------------------------------------------


CorpusProvider = Callable[[Query], list[MemoryItem] | Awaitable[list[MemoryItem]]]


class BM25Retriever:
    """
    :class:`RetrieverProtocol` implementation backed by :class:`BM25Index`.

    Parameters
    ----------
    corpus_provider:
        Callable invoked once per :meth:`retrieve` call. Receives the
        live :class:`Query` and returns the items to index. Sync or
        async — both are accepted. Letting the provider see the query
        means callers can scope the corpus (e.g. by ``session_id``)
        without rebuilding the retriever.
    top_k:
        How many items to surface. Defaults to ``query.top_k`` at
        call time when not overridden.
    name:
        Identifier copied into ``ContextBundle.debug_info["retriever"]``
        so RRF can attribute scores back to a specific child.

    The retriever is best-effort: provider exceptions are swallowed
    and surfaced as an empty bundle with the error captured in
    ``debug_info``. This matches the contract of the canonical
    :class:`continuum.retrieval.Retriever`.
    """

    def __init__(
        self,
        *,
        corpus_provider: CorpusProvider,
        top_k: int | None = None,
        name: str = "bm25",
    ) -> None:
        self._provider = corpus_provider
        self._top_k = top_k
        self.name = name

    async def retrieve(self, query: Query, budget: TokenBudget) -> ContextBundle:
        debug: dict[str, Any] = {"retriever": self.name}
        try:
            items = await _maybe_await(self._provider(query))
        except Exception as exc:
            log.exception("%s corpus_provider failed", self.name)
            debug["error"] = repr(exc)
            return ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                debug_info=debug,
            )
        if not items:
            debug["corpus_size"] = 0
            return ContextBundle(
                items=[],
                messages=[],
                tokens_used=0,
                budget=budget,
                debug_info=debug,
            )
        index = BM25Index(items)
        k = self._top_k if self._top_k is not None else query.top_k
        hits = index.query(query.text, k)
        debug["corpus_size"] = len(items)
        debug["hits"] = len(hits)
        bundle_items = [si.item for si in hits]
        messages = [{"role": "system", "content": it.content} for it in bundle_items]
        return ContextBundle(
            items=bundle_items,
            messages=messages,
            tokens_used=sum(len(it.content.split()) for it in bundle_items),
            budget=budget,
            debug_info=debug,
        )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


__all__ = ["BM25Index", "BM25Retriever", "tokenize"]
