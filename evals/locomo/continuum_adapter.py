"""
evals/locomo/continuum_adapter.py
=================================
Continuum's answerer for LOCOMO — v1's *actual* winning architecture:
**direct retrieval** (hybrid cosine+BM25, session-aware) over a
FlatHaystackStore, one answerer call per question.

This deliberately does NOT use the IterativeReasoner. This session's
LongMemEval evals showed the reasoner net-negative vs direct mode
(it lost on both single-hop and multi-hop); benchmarking Mem0 against
a component we know underperforms would be dishonest. The continuum
side here mirrors exactly the config that scored 60.8% judged on
LongMemEval-S.

Everything is reused from :mod:`evals.longmemeval.bootstrap_ollama` —
the retrievers and ``_DirectAnswerAdapter`` operate on
:class:`MemoryItem`, not on any LongMemEval-specific shape, so they
drop straight onto LOCOMO conversations.
"""

from __future__ import annotations

import time
from typing import Any

from continuum.core.types import MemoryItem, MemoryTier
from continuum.retrieval.rrf import HybridRetriever
from evals.locomo.loader import LocomoConversation
from evals.longmemeval.bootstrap_ollama import (
    BM25HaystackRetriever,
    FlatHaystackStore,
    SessionAwareSemanticRetriever,
    STMSemanticRetriever,
    _DirectAnswerAdapter,
    _Embedder,
    _MiniSession,
)

# LOCOMO categories that benefit from the aggregation prompt (combine
# across sessions). multi-hop (1) + temporal (2) need cross-session
# synthesis; we flag them so _DirectAnswerAdapter's aggregation branch
# fires (it keys on dataset_question_type == "multi-session").
_AGGREGATE_CATEGORIES = {1, 2}


class ContinuumLocomoAnswerer:
    """
    Builds the direct-mode stack over one LOCOMO conversation and answers
    its questions. One instance per sample (fresh store per conversation).
    """

    def __init__(
        self,
        conversation: LocomoConversation,
        *,
        llm: Any,
        embedder: _Embedder,
        session_aware: bool = True,
        session_top_k: int = 12,
        turns_per_session: int = 6,
        top_k: int = 80,
        rrf_k: int = 60,
        answer_max_tokens: int = 2048,
        max_context_chars: int = 64000,
    ) -> None:
        self._conversation = conversation
        self._llm = llm
        self._embedder = embedder
        self._top_k = top_k
        self._answer_max_tokens = answer_max_tokens
        self._max_context_chars = max_context_chars
        self._session_aware = session_aware
        self._session_top_k = session_top_k
        self._turns_per_session = turns_per_session
        self._rrf_k = rrf_k
        self._adapter: _DirectAnswerAdapter | None = None

    async def _ensure_built(self) -> _DirectAnswerAdapter:
        if self._adapter is not None:
            return self._adapter
        store = FlatHaystackStore()
        # Ingest every turn as a MemoryItem tagged with session/speaker/
        # dia_id so the retriever can rank sessions and the runner can
        # score recall against LOCOMO's evidence dia_ids.
        for turn in self._conversation.turns:
            await store.append(
                MemoryItem(
                    content=f"{turn.speaker}: {turn.text}",
                    tier=MemoryTier.STM,
                    metadata={
                        "role": "user",
                        "session_id": turn.session_id,
                        "speaker": turn.speaker,
                        "dia_id": turn.dia_id,
                        "date": turn.session_date,
                    },
                )
            )

        # Cosine side: session-aware (cross-session coverage) or plain.
        if self._session_aware:
            cosine: Any = SessionAwareSemanticRetriever(
                store=store,
                embedder=self._embedder,
                top_k=self._top_k,
                session_top_k=self._session_top_k,
                turns_per_session=self._turns_per_session,
                max_items=self._top_k,
            )
        else:
            cosine = STMSemanticRetriever(
                store=store,
                embedder=self._embedder,
                top_k=self._top_k,
            )
        bm25 = BM25HaystackRetriever(store=store, top_k=self._top_k)
        retriever = HybridRetriever(cosine, bm25, k=self._rrf_k, top_k=self._top_k)

        session = _MiniSession(store=store, retriever=retriever)
        self._adapter = _DirectAnswerAdapter(
            session=session,
            llm=self._llm,
            answer_max_tokens=self._answer_max_tokens,
            top_k=self._top_k,
            max_context_chars=self._max_context_chars,
        )
        return self._adapter

    async def answer(
        self,
        question: str,
        *,
        category: int | None = None,
    ) -> dict[str, Any]:
        """
        Answer one question. Returns ``{answer, retrieved_dia_ids,
        latency_ms, llm_calls}``. ``retrieved_dia_ids`` lets the runner
        score recall against LOCOMO's evidence list.
        """
        adapter = await self._ensure_built()
        # Flag aggregation categories so the direct adapter combines
        # across sessions instead of returning a single span.
        adapter.dataset_question_type = (
            "multi-session" if category in _AGGREGATE_CATEGORIES else "single"
        )
        t0 = time.perf_counter()
        answer = await adapter.answer_question(question)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        dia_ids: list[str] = []
        ctx = getattr(adapter, "last_ctx", None)
        for it in getattr(ctx, "items", []) or []:
            did = (getattr(it, "metadata", {}) or {}).get("dia_id")
            if did:
                dia_ids.append(str(did))
        telem = getattr(adapter, "last_telemetry", {}) or {}
        return {
            "answer": answer,
            "retrieved_dia_ids": dia_ids,
            "latency_ms": latency_ms,
            "llm_calls": int(telem.get("llm_call_count", 1)),
        }


__all__ = ["ContinuumLocomoAnswerer"]
