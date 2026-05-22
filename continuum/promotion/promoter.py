"""
continuum/promotion/promoter.py
===============================
``Promoter`` — the full MTM → LTM promotion orchestrator (implements the
runtime ``PromoterProtocol``).

Pipeline (per :meth:`promote`)
------------------------------
1. Scan unprocessed MTM blocks (``mtm.scan_unprocessed``), filtered by
   :class:`~continuum.core.types.PromotionScope` (session / since / all).
2. Extract entities (GLiNER baseline + optional LLM enhancement).
3. Extract atomic facts from each block (``fact_extractor.extract_facts``).
4. For every candidate fact:
   a. Retrieve top-K similar LTM facts (``ltm.search_hybrid``).
   b. Decide ADD / UPDATE / DELETE / NOOP (``Mem0Promoter``; this also
      writes the ``memory_promotions`` audit log).
   c. Execute: ADD→``ltm.upsert``, UPDATE→``ltm.update``,
      DELETE→``ltm.invalidate(at=now)``, NOOP→skip.
5. Mark successfully-extracted blocks processed (``mtm.mark_processed``).

Error isolation
---------------
A failure extracting a block skips *that block* (it is **left unprocessed**
so a transient failure is retried next run). A failure deciding/executing a
fact skips *that fact*. Every error is collected into the report; the run
never crashes and ``promote`` never raises.

Compatibility
-------------
``PromoterProtocol`` is ``@runtime_checkable`` (name-based) and
``ContinuumSession.checkpoint`` calls ``promote("session:<id>")``. So
:meth:`promote` accepts a :class:`PromotionScope` **or** the legacy string
scope, and :class:`PromotionRunReport` exposes ``scope`` /
``promoted_count`` / ``skipped_count`` so it is a drop-in for the existing
core ``PromotionReport`` consumers.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from continuum.core.config import PolicyEngineConfig, PromoterConfig
from continuum.core.types import (
    MemoryItem,
    MemoryTier,
    PromotionScope,
    Query,
    ScoredItem,
)
from continuum.extraction.fact_extractor import Fact
from continuum.policies.models import (
    MemoryAction,
    PolicyEvaluationContext,
)

log = logging.getLogger(__name__)

#: Rough litellm pricing (USD per 1K tokens) for the small models we default
#: to. Unknown models cost 0.0 (better to under-report than fabricate).
_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-5-haiku": (0.00080, 0.00400),
}

MetricsFn = Callable[[str, float], None]
ClockFn = Callable[[], datetime]


@dataclass
class PromotionRunReport:
    """
    Outcome of one :meth:`Promoter.promote` run.

    ``promoted_count`` / ``skipped_count`` / ``scope`` are provided so this
    is a drop-in for the core ``PromotionReport`` consumers
    (``ContinuumSession.checkpoint``, ``make_promotion_sweep``).
    """

    scope: str = "global"
    added: list[uuid.UUID] = field(default_factory=list)
    updated: list[uuid.UUID] = field(default_factory=list)
    deleted: list[uuid.UUID] = field(default_factory=list)
    noop_count: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    blocks_processed: int = 0

    @property
    def promoted_count(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)

    @property
    def skipped_count(self) -> int:
        return self.noop_count


class Promoter:
    """
    MTM → LTM promotion orchestrator.

    All collaborators are injected and duck-typed, so unit tests wire fakes
    and no DB / LLM / network is needed.

    Required
    --------
    mtm:
        ``scan_unprocessed()`` (async iterator of ``SummaryBlock``) +
        ``mark_processed(ids)``.
    ltm:
        ``search_hybrid(query, k)`` + ``upsert(item)`` + ``update(id, patch)``
        + ``invalidate(id, at)``.
    fact_extractor:
        ``extract_facts(block, entities) -> list[Fact]``.
    decider:
        :class:`~continuum.promotion.mem0_promoter.Mem0Promoter`-like —
        ``decide_operations_batch([(fact, neighbors)]) -> list[Decision]``
        (also performs the memory_promotions audit).

    Optional
    --------
    entity_extractor:
        GLiNER baseline — ``extract(text) -> (entities, relations)``.
    llm_entity_extractor:
        LLM enhancer — ``extract(text, entities) -> (entities, relations)``.
    metrics_fn:
        ``(name, value)`` sink for ``promoter.processed/added/errors``.
    clock:
        ``() -> datetime`` (injected for deterministic DELETE timestamps).
    """

    def __init__(
        self,
        config: PromoterConfig | None = None,
        *,
        mtm: Any,
        ltm: Any,
        fact_extractor: Any,
        decider: Any,
        entity_extractor: Any | None = None,
        llm_entity_extractor: Any | None = None,
        metrics_fn: MetricsFn | None = None,
        clock: ClockFn | None = None,
        cost_per_1k: dict[str, tuple[float, float]] | None = None,
        # ── Policy Engine (optional, backward-compatible) ────────────────
        policy_engine: Any | None = None,
        candidate_extractor: Any | None = None,
        storage_router: Any | None = None,
        trace_writer: Any | None = None,
        policy_engine_config: PolicyEngineConfig | None = None,
    ) -> None:
        self.config = config or PromoterConfig()
        self._mtm = mtm
        self._ltm = ltm
        self._facts = fact_extractor
        self._decider = decider
        self._entities = entity_extractor
        self._llm_entities = llm_entity_extractor
        self._metrics = metrics_fn
        self._clock = clock or (lambda: datetime.now(UTC))
        self._prices = cost_per_1k or _PRICE_PER_1K
        # Policy-engine collaborators are entirely optional. The legacy
        # Mem0 path runs unchanged unless *all* of policy_engine,
        # candidate_extractor, and storage_router are wired AND the
        # config has enabled=True.
        self._policy_engine = policy_engine
        self._candidate_extractor = candidate_extractor
        self._storage_router = storage_router
        self._trace_writer = trace_writer
        self._policy_config = policy_engine_config or PolicyEngineConfig()

    # ── policy-engine activation guard ──────────────────────────────────────

    @property
    def _policy_engine_active(self) -> bool:
        """True only when fully wired AND enabled in config."""
        return bool(
            self._policy_config.enabled
            and self._policy_engine is not None
            and self._candidate_extractor is not None
            and self._storage_router is not None
        )

    # ── protocol entry point ────────────────────────────────────────────────

    async def promote(
        self, scope: PromotionScope | str | None = None
    ) -> PromotionRunReport:
        """Run the full pipeline for *scope*; never raises."""
        t0 = time.perf_counter()
        sc = PromotionScope.parse(scope)
        report = PromotionRunReport(scope=sc.as_str())
        try:
            await self._run(sc, report)
        except Exception as exc:  # catastrophic — still return a report
            log.exception("promotion run aborted")
            report.errors.append(f"run aborted: {exc!r}")
        report.duration_seconds = time.perf_counter() - t0
        self._emit("promoter.processed", report.blocks_processed)
        self._emit("promoter.added", len(report.added))
        self._emit("promoter.errors", len(report.errors))
        log.info(
            "promotion done scope=%s blocks=%d add=%d upd=%d del=%d "
            "noop=%d err=%d %.0fms $%.5f",
            report.scope, report.blocks_processed, len(report.added),
            len(report.updated), len(report.deleted), report.noop_count,
            len(report.errors), report.duration_seconds * 1000,
            report.cost_usd,
        )
        return report

    # ── pipeline ────────────────────────────────────────────────────────────

    async def _run(
        self, sc: PromotionScope, report: PromotionRunReport
    ) -> None:
        # Branch on policy engine availability. The legacy Mem0 path is the
        # default and ships unchanged for callers that don't wire a policy
        # engine.
        if self._policy_engine_active:
            await self._run_policy_engine(sc, report)
            return
        await self._run_legacy(sc, report)

    async def _run_legacy(
        self, sc: PromotionScope, report: PromotionRunReport
    ) -> None:
        # Steps 1–3: scan + extract → (fact, neighbors) candidate pairs.
        pairs: list[tuple[Fact, list[ScoredItem]]] = []
        processed_blocks: list[uuid.UUID] = []
        scanned = 0

        async for block in self._mtm.scan_unprocessed():
            if not self._in_scope(block, sc):
                continue
            scanned += 1
            if scanned % 10 == 0:
                log.info("promotion progress: %d blocks scanned", scanned)
                self._emit("promoter.processed", scanned)

            try:
                entities = await self._extract_entities(block.text)
                facts = await self._facts.extract_facts(block, entities)
            except Exception as exc:
                # Skip the block; leave it UNPROCESSED so it is retried.
                report.errors.append(f"block {block.id}: extract: {exc!r}")
                self._emit("promoter.errors", 1)
                log.exception("extraction failed for block %s", block.id)
                continue

            # Extraction succeeded → this block is handled (mark processed
            # even if some downstream fact decisions fail).
            processed_blocks.append(block.id)
            report.blocks_processed += 1

            for fact in facts:
                if fact.confidence < self.config.confidence_threshold:
                    continue  # below promotion bar → silently drop
                try:
                    nbrs = await self._neighbors(fact)
                except Exception as exc:
                    report.errors.append(
                        f"fact {fact.text[:48]!r}: neighbors: {exc!r}"
                    )
                    self._emit("promoter.errors", 1)
                    continue
                pairs.append((fact, nbrs))

        if not pairs:
            if processed_blocks:
                await self._mark_processed(processed_blocks, report)
            return

        # Step 4b: batched ADD/UPDATE/DELETE/NOOP decisions (audited inside).
        try:
            decisions = await self._decider.decide_operations_batch(pairs)
        except Exception as exc:
            report.errors.append(f"decision batch failed: {exc!r}")
            self._emit("promoter.errors", 1)
            await self._mark_processed(processed_blocks, report)
            return

        # Step 4c: execute each decision.
        for (fact, _nbrs), decision in zip(pairs, decisions, strict=False):
            self._account(decision, report)
            try:
                await self._execute(fact, decision, report)
            except Exception as exc:
                report.errors.append(
                    f"fact {fact.text[:48]!r}: execute "
                    f"{getattr(decision, 'op', '?')}: {exc!r}"
                )
                self._emit("promoter.errors", 1)
                log.exception("execute failed for a %s decision",
                              getattr(decision, "op", "?"))

        # Step 6: mark blocks processed.
        await self._mark_processed(processed_blocks, report)

    # ── policy-engine pipeline ──────────────────────────────────────────────

    async def _run_policy_engine(
        self, sc: PromotionScope, report: PromotionRunReport
    ) -> None:
        """
        Policy-driven promotion path.

        For each MTM block: entities → candidates → engine.evaluate_many →
        storage_router.execute → trace_writer.write. Errors per block /
        per candidate are isolated, mirroring the legacy path.
        """
        # _policy_engine_active is True iff all three are non-None; help
        # mypy narrow the unions.
        assert self._policy_engine is not None
        assert self._candidate_extractor is not None
        assert self._storage_router is not None

        processed_blocks: list[uuid.UUID] = []
        traces: list[Any] = []
        scanned = 0

        async for block in self._mtm.scan_unprocessed():
            if not self._in_scope(block, sc):
                continue
            scanned += 1
            if scanned % 10 == 0:
                log.info("promotion progress: %d blocks scanned", scanned)
                self._emit("promoter.processed", scanned)

            try:
                entities = await self._extract_entities(block.text)
                candidates = await self._candidate_extractor.extract_candidates(
                    block, entities
                )
            except Exception as exc:
                report.errors.append(
                    f"block {block.id}: extract_candidates: {exc!r}"
                )
                self._emit("promoter.errors", 1)
                log.exception(
                    "candidate extraction failed for block %s", block.id
                )
                continue

            # Block-level handling is considered done once extraction
            # succeeded — mirror the legacy contract.
            processed_blocks.append(block.id)
            report.blocks_processed += 1

            if not candidates:
                continue

            ctx = PolicyEvaluationContext(
                now=self._clock(),
                session_id=self._coerce_uuid(
                    getattr(block, "session_id", None)
                ),
                existing_neighbors=[],
                config=self._policy_config.model_dump(),
            )

            try:
                evaluated = await self._policy_engine.evaluate_many(
                    candidates, ctx
                )
            except Exception as exc:
                report.errors.append(
                    f"block {block.id}: policy.evaluate_many: {exc!r}"
                )
                self._emit("promoter.errors", 1)
                log.exception("policy evaluation failed for block %s", block.id)
                continue

            for candidate, plan, trace in evaluated:
                traces.append(trace)
                self._account_policy_decision(plan, report)
                try:
                    await self._storage_router.execute(candidate, plan)
                except Exception as exc:
                    report.errors.append(
                        f"candidate {candidate.id}: storage_router: {exc!r}"
                    )
                    self._emit("promoter.errors", 1)
                    log.exception(
                        "storage_router failed for candidate %s", candidate.id
                    )

        # Persist traces best-effort (TraceWriter never raises).
        if traces and self._trace_writer is not None:
            try:
                await self._trace_writer.write(traces)
            except Exception:
                # TraceWriter promises to swallow, but be defensive.
                log.exception("trace_writer.write raised — ignoring")

        await self._mark_processed(processed_blocks, report)

    def _account_policy_decision(
        self, plan: Any, report: PromotionRunReport
    ) -> None:
        """Map a MemoryAction onto the legacy add/upd/del/noop counters."""
        action = getattr(plan, "action", None)
        cand_id = getattr(plan, "candidate_id", None)
        try:
            cand_uuid = self._as_uuid(cand_id) if cand_id is not None else None
        except Exception:
            cand_uuid = None

        if action == MemoryAction.STORE:
            if cand_uuid is not None:
                report.added.append(cand_uuid)
            self._emit("promoter.added", 1)
        elif action in (
            MemoryAction.UPDATE,
            MemoryAction.MERGE,
            MemoryAction.SUPERSEDE,
            MemoryAction.COMPACT,
        ):
            if cand_uuid is not None:
                report.updated.append(cand_uuid)
        elif action in (MemoryAction.DELETE, MemoryAction.EXPIRE):
            if cand_uuid is not None:
                report.deleted.append(cand_uuid)
        else:
            # IGNORE / ASK_USER / NOOP all count as "skipped" from the
            # promotion-report perspective.
            report.noop_count += 1

    # ── steps ───────────────────────────────────────────────────────────────

    async def _extract_entities(self, text: str) -> list[Any]:
        ents: list[Any] = []
        if self._entities is not None:
            ents, _ = await self._entities.extract(text)
        if self._llm_entities is not None:
            ents, _ = await self._llm_entities.extract(text, ents)
        return ents

    async def _neighbors(self, fact: Fact) -> list[ScoredItem]:
        q = Query(
            text=fact.text,
            top_k=self.config.max_neighbors,
            tiers=[MemoryTier.LTM],
        )
        result = await self._ltm.search_hybrid(q, self.config.max_neighbors)
        return list(result)

    async def _execute(
        self,
        fact: Fact,
        decision: Any,
        report: PromotionRunReport,
    ) -> None:
        op = getattr(decision, "op", "NOOP")
        if op == "ADD":
            new_id = await self._ltm.upsert(self._fact_to_item(fact))
            report.added.append(self._as_uuid(new_id))
            self._emit("promoter.added", 1)
        elif op == "UPDATE":
            tid = decision.target_id
            if tid is None:
                report.errors.append(
                    f"UPDATE without target_id for {fact.text[:48]!r}"
                )
                return
            merged = getattr(decision, "merged_text", None) or fact.text
            await self._ltm.update(tid, {"content": merged})
            report.updated.append(self._as_uuid(tid))
        elif op == "DELETE":
            tid = decision.target_id
            if tid is None:
                report.errors.append(
                    f"DELETE without target_id for {fact.text[:48]!r}"
                )
                return
            await self._ltm.invalidate(tid, self._clock())
            report.deleted.append(self._as_uuid(tid))
        else:  # NOOP / unknown
            report.noop_count += 1

    async def _mark_processed(
        self, block_ids: Sequence[uuid.UUID], report: PromotionRunReport
    ) -> None:
        if not block_ids:
            return
        try:
            await self._mtm.mark_processed(list(block_ids))
        except Exception as exc:
            report.errors.append(f"mark_processed failed: {exc!r}")
            self._emit("promoter.errors", 1)
            log.exception("mark_processed failed")

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _in_scope(block: Any, sc: PromotionScope) -> bool:
        if sc.session_id is not None and str(
            getattr(block, "session_id", None)
        ) != str(sc.session_id):
            return False
        if sc.since is not None:
            created = getattr(block, "created_at", None)
            if created is not None and created < sc.since:
                return False
        return True

    def _fact_to_item(self, fact: Fact) -> MemoryItem:
        return MemoryItem(
            content=fact.text,
            tier=MemoryTier.LTM,
            confidence=fact.confidence,
            importance=0.5,
            metadata={
                "kind": "fact",
                "source_block_id": str(fact.source_block_id),
                "entities": list(fact.entities_mentioned),
                "category": fact.category,
            },
        )

    def _account(self, decision: Any, report: PromotionRunReport) -> None:
        tin = int(getattr(decision, "tokens_in", 0) or 0)
        tout = int(getattr(decision, "tokens_out", 0) or 0)
        report.tokens_used += tin + tout
        model = getattr(decision, "model", None)
        if model:
            report.cost_usd += self._cost(str(model), tin, tout)

    def _cost(self, model: str, tin: int, tout: int) -> float:
        key = next((k for k in self._prices if k in model), None)
        if key is None:
            return 0.0
        in_rate, out_rate = self._prices[key]
        return (tin / 1000.0) * in_rate + (tout / 1000.0) * out_rate

    @staticmethod
    def _as_uuid(value: Any) -> uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))

    @staticmethod
    def _coerce_uuid(value: Any) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except (ValueError, AttributeError):
            return None

    def _emit(self, name: str, value: float) -> None:
        if self._metrics is not None:
            try:
                self._metrics(name, value)
            except Exception:  # metrics must never break promotion
                log.debug("metrics sink failed for %s", name, exc_info=True)
        else:
            log.debug("metric %s=%s", name, value)


__all__ = ["Promoter", "PromotionRunReport"]
