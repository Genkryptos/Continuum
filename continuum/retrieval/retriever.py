"""
continuum/retrieval/retriever.py
================================
``Retriever`` — the full context-assembly pipeline (``RetrieverProtocol``).

    retrieve(query, budget) ->
        1. (optional) HyDE query rewrite              [off by default]
        2. LTM hybrid search (dense ⊕ sparse ⊕ RRF)   k = config.k1
        3. Graph expansion: 1-hop neighbours of the top entity hits
        4. Scoring: Scorer over the whole candidate pool, sort desc
        5. Reranking: cross-encoder second pass, take config.ltm_top_k
        6. STM (recent turns verbatim) + MTM (recent summaries ≤ budget.mtm)
        7. Assemble a ContextBundle (items + chat messages + token breakdown)

Design
------
Every collaborator is **injected and optional** (duck-typed). A missing or
failing component degrades that stage to a no-op — ``retrieve`` never raises
into the agent's hot path; partial context beats an exception. The default
``scorer`` is the dependency-free :class:`continuum.scoring.Scorer`, so
scoring always runs even with nothing else wired.

``ContextBundle`` is the project's real shape (flat ``items`` +
prompt-ready ``messages`` + ``tier_breakdown`` token map). The spec's
"stm/mtm/ltm" framing maps onto ``tier_breakdown`` and the assembled order
(LTM facts → MTM summaries → STM recent). Token accounting uses an
injectable counter (whitespace split by default; pass tiktoken in prod).
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from continuum.core.config import PolicyEngineConfig, RetrieverConfig
from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    Query,
    ScoreBreakdown,
    ScoredItem,
    TokenBudget,
)
from continuum.scoring import Scorer

log = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]
HydeFn = Callable[[Query], Awaitable[str]]
ClockFn = Callable[[], datetime]

_ZERO = ScoreBreakdown(
    relevance=0.0, importance=0.0, recency=0.0, confidence=0.0, composite=0.0
)


class Retriever:
    """
    End-to-end retrieval pipeline.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.RetrieverConfig`.
    ltm / stm / mtm:
        Tier stores (duck-typed). Each optional — a missing tier is skipped.
        * ``ltm.search_hybrid(query, k)`` + ``ltm.neighbors(id, hops=1)``
        * ``stm.get_recent(session_id, n)``
        * ``mtm.recent(token_budget, session_id=)`` → ``SummaryBlock``s
    scorer:
        ``score(item, query, now)`` + ``breakdown(...)``. Defaults to
        :class:`continuum.scoring.Scorer`.
    reranker:
        ``rerank(query_text, items)``. Optional — skipped if ``None``.
    hyde_fn:
        ``async (query) -> str`` rewrite, used only when
        ``config.hyde_enabled``.
    token_counter / clock / session_id:
        Token estimator (default whitespace), recency clock, default
        session for STM when ``query.session_id`` is unset.
    """

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        *,
        ltm: Any | None = None,
        stm: Any | None = None,
        mtm: Any | None = None,
        scorer: Any | None = None,
        reranker: Any | None = None,
        hyde_fn: HydeFn | None = None,
        token_counter: TokenCounter | None = None,
        clock: ClockFn | None = None,
        session_id: str = "default",
        policy_engine_config: PolicyEngineConfig | None = None,
    ) -> None:
        self.config = config or RetrieverConfig()
        self._ltm = ltm
        self._stm = stm
        self._mtm = mtm
        self._scorer = scorer or Scorer()
        self._reranker = reranker
        self._hyde_fn = hyde_fn
        self._count = token_counter or (lambda s: len(s.split()))
        self._clock = clock or (lambda: datetime.now(UTC))
        self._session_id = session_id
        # Policy-engine config is optional. Filtering and boosting only
        # activate when items in the pool carry the policy metadata
        # introduced by migration 004 — old rows without those fields
        # are untouched (backward compatible).
        self._policy_config = policy_engine_config or PolicyEngineConfig()

    # ── RetrieverProtocol ───────────────────────────────────────────────────

    async def retrieve(
        self, query: Query, budget: TokenBudget
    ) -> ContextBundle:
        """Run the full pipeline; never raises (best-effort context)."""
        debug: dict[str, Any] = {}
        try:
            search_q = await self._maybe_hyde(query, debug)
            pool = await self._ltm_hybrid(search_q, debug)
            await self._graph_expand(pool, debug)
            scored = self._score_all(pool, query, debug)
            ltm_items = await self._rerank(query, scored, debug)
            stm_items = await self._stm_recent(query)
            mtm_items = await self._mtm_recent(query, budget, debug)
            return self._assemble(
                budget, ltm_items, mtm_items, stm_items, debug
            )
        except Exception as exc:  # pragma: no cover - stages already guarded
            log.exception("retrieval pipeline aborted")
            return ContextBundle(
                items=[], messages=[], tokens_used=0, budget=budget,
                tier_breakdown={}, debug_info={"error": repr(exc), **debug},
            )

    # ── 1. HyDE (optional) ──────────────────────────────────────────────────

    async def _maybe_hyde(self, query: Query, debug: dict[str, Any]) -> Query:
        if not (self.config.hyde_enabled and self._hyde_fn is not None):
            return query
        try:
            hypothetical = await self._hyde_fn(query)
            debug["hyde"] = True
            return replace(query, text=hypothetical)
        except Exception:
            log.exception("HyDE rewrite failed — using the original query")
            debug["hyde"] = "failed"
            return query

    # ── 2. LTM hybrid search ────────────────────────────────────────────────

    async def _ltm_hybrid(
        self, query: Query, debug: dict[str, Any]
    ) -> dict[str, ScoredItem]:
        pool: dict[str, ScoredItem] = {}
        if self._ltm is None:
            debug["ltm_hybrid"] = 0
            return pool
        search_query = query
        if (
            self.config.cross_session_user_recall_enabled
            and query.user_id
        ):
            filt = dict(query.metadata_filter or {})
            filt.setdefault("user_id", query.user_id)
            search_query = replace(query, metadata_filter=filt)
            debug["cross_session_ltm_filter"] = {"user_id": query.user_id}
        try:
            hits: Sequence[ScoredItem] = await self._ltm.search_hybrid(
                search_query, self.config.k1
            )
        except Exception:
            log.exception("LTM hybrid search failed")
            debug["ltm_hybrid"] = "failed"
            return pool
        for si in hits:
            pool.setdefault(str(si.item.id), si)
        debug["ltm_hybrid"] = len(pool)
        return pool

    # ── 3. Graph expansion ──────────────────────────────────────────────────

    async def _graph_expand(
        self, pool: dict[str, ScoredItem], debug: dict[str, Any]
    ) -> None:
        n = self.config.graph_expand_n
        if n <= 0 or self._ltm is None or not hasattr(self._ltm, "neighbors"):
            debug["graph_added"] = 0
            return
        seeds = [
            si
            for si in list(pool.values())[: max(n * 3, n)]
            if si.item.metadata.get("kind") == "entity"
        ][:n]
        added = 0
        for si in seeds:
            try:
                edges = await self._ltm.neighbors(si.item.id, hops=1)
            except Exception:
                log.debug("neighbors failed for %s", si.item.id, exc_info=True)
                continue
            for edge in edges:
                tgt = getattr(edge, "target", None)
                if tgt is None:
                    continue
                key = str(tgt.id)
                if key in pool:
                    continue
                tgt.metadata.setdefault("via_graph", str(si.item.id))
                pool[key] = ScoredItem(item=tgt, scores=_ZERO)
                added += 1
        debug["graph_added"] = added

    # ── 4. Scoring ──────────────────────────────────────────────────────────

    def _score_all(
        self,
        pool: dict[str, ScoredItem],
        query: Query,
        debug: dict[str, Any],
    ) -> list[ScoredItem]:
        now = self._clock()
        rescored: list[ScoredItem] = []
        filtered_out = 0
        for si in pool.values():
            if not self._should_retrieve(si.item, now):
                filtered_out += 1
                continue
            try:
                sb = self._scorer.breakdown(si.item, query, now)
            except Exception:
                log.debug("scoring failed for %s", si.item.id, exc_info=True)
                sb = si.scores
            boosted = self._apply_policy_boost(si.item, query, sb)
            rescored.append(ScoredItem(item=si.item, scores=boosted))
        rescored.sort(key=lambda s: s.score, reverse=True)
        debug["scored"] = len(rescored)
        if filtered_out:
            debug["policy_filtered"] = filtered_out
        return rescored

    # ── policy-aware filtering / boosting ───────────────────────────────────

    def _should_retrieve(self, item: MemoryItem, now: datetime) -> bool:
        """
        Policy filter applied during scoring.

        * Drop items whose ``expires_at`` is in the past.
        * Drop RESTRICTED items still pending approval (status != active).
        Items without any of the policy fields are untouched.
        """
        if not self._policy_config.enabled:
            return True
        md = item.metadata or {}
        # 1. expiry
        expires = md.get("expires_at")
        if isinstance(expires, datetime) and expires < now:
            return False
        # 2. restricted-without-approval
        sensitivity = md.get("sensitivity")
        status = md.get("status")
        if (
            sensitivity == "restricted"
            and status not in (None, "active")
            and self._policy_config.require_approval_for_restricted
        ):
            return False
        return True

    def _apply_policy_boost(
        self,
        item: MemoryItem,
        query: Query,
        scores: ScoreBreakdown,
    ) -> ScoreBreakdown:
        """
        Bump composite for high-signal candidate types.

        * Tasks marked urgent get a recency boost.
        * Procedures get boosted on "how" queries.
        * User preferences get boosted on all queries.
        Untyped items are untouched.
        """
        if not self._policy_config.enabled:
            return scores
        md = item.metadata or {}
        ctype = md.get("candidate_type")
        urgency = md.get("urgency")
        boost = 0.0

        if ctype == "user_preference":
            boost += 0.10
        if ctype == "task" and urgency in ("immediate", "soon"):
            boost += 0.15
        if ctype == "procedure" and any(
            kw in (query.text or "").lower() for kw in ("how", "workflow", "process")
        ):
            boost += 0.10

        if boost == 0.0:
            return scores
        new_composite = min(1.0, scores.composite + boost)
        return ScoreBreakdown(
            relevance=scores.relevance,
            importance=scores.importance,
            recency=scores.recency,
            confidence=scores.confidence,
            composite=new_composite,
        )

    # ── 5. Reranking ────────────────────────────────────────────────────────

    async def _rerank(
        self,
        query: Query,
        scored: list[ScoredItem],
        debug: dict[str, Any],
    ) -> list[ScoredItem]:
        ranked = scored
        if self._reranker is not None and scored:
            try:
                ranked = list(
                    await self._reranker.rerank(query.text, scored)
                )
                debug["reranked"] = True
            except Exception:
                log.exception("rerank failed — keeping scored order")
                debug["reranked"] = "failed"
        else:
            debug["reranked"] = False
        return ranked[: self.config.ltm_top_k]

    # ── 6. STM / MTM ────────────────────────────────────────────────────────

    async def _stm_recent(self, query: Query) -> list[MemoryItem]:
        if self._stm is None:
            return []
        sid = query.session_id or self._session_id
        try:
            recent = await self._stm.get_recent(sid, self.config.stm_turns)
            return list(recent)
        except Exception:
            log.exception("STM retrieval failed")
            return []

    async def _mtm_recent(
        self, query: Query, budget: TokenBudget, debug: dict[str, Any]
    ) -> list[MemoryItem]:
        if self._mtm is None:
            return []
        recall_enabled = (
            self.config.cross_session_user_recall_enabled
            and bool(query.user_id)
        )
        try:
            if recall_enabled:
                blocks = await self._mtm.recent(
                    budget.mtm_reserved, user_id=query.user_id
                )
            else:
                blocks = await self._mtm.recent(
                    budget.mtm_reserved, session_id=query.session_id
                )
        except Exception:
            log.exception("MTM retrieval failed")
            return []
        out: list[MemoryItem] = []
        for b in blocks:
            to_item = getattr(b, "to_memory_item", None)
            out.append(to_item() if callable(to_item) else b)
        debug["cross_session_mtm"] = bool(recall_enabled)
        debug["mtm_user_scoped"] = bool(recall_enabled)
        debug["mtm_count"] = len(out)
        return out

    # ── 7. Assembly ─────────────────────────────────────────────────────────

    def _assemble(
        self,
        budget: TokenBudget,
        ltm_items: list[ScoredItem],
        mtm_items: list[MemoryItem],
        stm_items: list[MemoryItem],
        debug: dict[str, Any],
    ) -> ContextBundle:
        ltm = [si.item for si in ltm_items]
        # Prompt order: durable facts → project summaries → recent turns.
        items = ltm + mtm_items + stm_items
        messages: list[dict[str, str]] = []
        for it in ltm:
            messages.append({"role": "system", "content": it.content})
        for it in mtm_items:
            messages.append({"role": "system", "content": it.content})
        for it in stm_items:
            messages.append(
                {"role": str(it.metadata.get("role", "user")),
                 "content": it.content}
            )

        t_ltm = sum(self._count(i.content) for i in ltm)
        t_mtm = sum(self._count(i.content) for i in mtm_items)
        t_stm = sum(self._count(i.content) for i in stm_items)
        breakdown = {"stm": t_stm, "mtm": t_mtm, "ltm": t_ltm}

        debug.update(
            {"ltm_final": len(ltm), "mtm": len(mtm_items),
             "stm": len(stm_items)}
        )
        return ContextBundle(
            items=items,
            messages=messages,
            tokens_used=t_ltm + t_mtm + t_stm,
            budget=budget,
            tier_breakdown=breakdown,
            debug_info=debug,
        )


__all__ = ["Retriever"]
