"""
continuum/promotion/triggers.py
===============================
``TriggerManager`` — decides *when* the :class:`Promoter` should run, so
callers don't have to.

Triggers
--------
a. **New entity**       — a turn mentions an entity absent from LTM.
b. **Block accumulation** — ≥ ``block_threshold`` unprocessed MTM blocks.
c. **Session checkpoint** — the user explicitly forces it
   (``session.checkpoint()`` → :meth:`force_now`).
d. **Periodic**         — a background sweep every
   ``periodic_interval_seconds`` (default 6 h), via ``BackgroundQueue``.

Never blocks the response path
------------------------------
The trigger *checks* (a, b) are cheap and awaited; the *promotion itself* is
always queued on the injected ``BackgroundQueue`` (fire-and-forget). An
in-flight guard collapses a burst of triggers into a single queued run, so a
chatty session can't stack dozens of promotion jobs. ``force_now`` (manual
checkpoint) is the one path that awaits promotion synchronously.

All collaborators are injected / duck-typed (mtm, ltm, promoter,
background), so unit tests need no DB / LLM / network.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from continuum.core.config import TriggerConfig
from continuum.core.types import MemoryTier, Query
from continuum.extraction.entity_extractor import Entity

log = logging.getLogger(__name__)


class TriggerManager:
    """
    Watch promotion conditions and queue / force promotion.

    Parameters
    ----------
    config:
        :class:`continuum.core.config.TriggerConfig`.
    mtm:
        Source of unprocessed blocks — ``scan_unprocessed()`` async iterator.
    ltm:
        Existing-fact store — ``search_hybrid(query, k)`` (entity presence).
    promoter:
        ``promote(scope=None) -> report``-shaped orchestrator.
    background:
        ``BackgroundQueue``-like: ``schedule_nowait(factory) -> bool`` and
        ``schedule_periodic(factory, interval_seconds=, name=) -> task``.
    entity_lookup:
        Optional ``async (text) -> bool`` override for "is this entity in
        LTM?" (defaults to a ``ltm.search_hybrid`` probe).
    match_threshold:
        Cosine at/above which the top LTM hit counts as the same entity
        (also matched if the text is identical, case-insensitive).
    """

    def __init__(
        self,
        config: TriggerConfig | None = None,
        *,
        mtm: Any,
        ltm: Any,
        promoter: Any,
        background: Any,
        entity_lookup: Any | None = None,
        match_threshold: float = 0.85,
    ) -> None:
        self.config = config or TriggerConfig()
        self._mtm = mtm
        self._ltm = ltm
        self._promoter = promoter
        self._bg = background
        self._entity_lookup = entity_lookup
        self._match_threshold = match_threshold
        self._inflight = False  # at most one queued/running promotion

    # ── trigger (a): new entity ─────────────────────────────────────────────

    async def check_new_entity(self, entities: Sequence[Entity]) -> bool:
        """
        ``True`` if *any* entity is not already represented in LTM.

        Disabled (always ``False``) when ``config.on_new_entity`` is off. A
        lookup error is treated as "present" (conservative — avoids noisy
        promotions; accumulation/periodic triggers still catch it).
        """
        if not self.config.on_new_entity:
            return False
        for ent in entities:
            text = ent.text.strip()
            if not text:
                continue
            try:
                present = await self._entity_in_ltm(text)
            except Exception:
                log.debug("entity lookup failed for %r — assuming present",
                          text, exc_info=True)
                present = True
            if not present:
                log.debug("new entity %r → promotion trigger", text)
                return True
        return False

    async def _entity_in_ltm(self, text: str) -> bool:
        if self._entity_lookup is not None:
            return bool(await self._entity_lookup(text))
        q = Query(text=text, top_k=1, tiers=[MemoryTier.LTM])
        results = await self._ltm.search_hybrid(q, 1)
        if not results:
            return False
        top = results[0]
        if top.item.content.strip().lower() == text.lower():
            return True
        return float(top.scores.relevance) >= self._match_threshold

    # ── trigger (b): block accumulation ─────────────────────────────────────

    async def check_block_accumulation(self) -> bool:
        """
        ``True`` once ≥ ``config.block_threshold`` unprocessed MTM blocks
        exist. Scans with early-exit so it never reads more than the
        threshold + 1 blocks.
        """
        threshold = self.config.block_threshold
        count = 0
        try:
            async for _ in self._mtm.scan_unprocessed():
                count += 1
                if count >= threshold:
                    log.debug("block accumulation %d ≥ %d → trigger",
                              count, threshold)
                    return True
        except Exception:
            log.exception("block-accumulation check failed")
            return False
        return False

    # ── trigger (d): periodic ───────────────────────────────────────────────

    async def schedule_periodic(
        self,
        promoter: Any | None = None,
        interval_seconds: int | None = None,
    ) -> Any:
        """
        Register a recurring background promotion.

        Returns the ``BackgroundQueue`` timer task so callers can cancel an
        individual schedule; :meth:`BackgroundQueue.stop` also cancels it.
        """
        target = promoter or self._promoter
        interval = interval_seconds or self.config.periodic_interval_seconds

        def _factory() -> Any:
            return self._run_promotion(target, "periodic")

        return await self._bg.schedule_periodic(
            _factory,
            interval_seconds=interval,
            name="promotion-periodic",
        )

    # ── integration: after-turn hook ────────────────────────────────────────

    async def after_turn(
        self, entities: Sequence[Entity] | None = None
    ) -> bool:
        """
        Called once per turn (off the response path). Queues a promotion if
        the new-entity or block-accumulation trigger fires. Returns whether a
        promotion was queued.
        """
        triggered = False
        reason = ""
        if entities and await self.check_new_entity(entities):
            triggered, reason = True, "new_entity"
        elif await self.check_block_accumulation():
            triggered, reason = True, "block_accumulation"
        if not triggered:
            return False
        return self.fire(reason)

    # ── queue / force ───────────────────────────────────────────────────────

    def fire(self, reason: str = "manual") -> bool:
        """
        Queue a promotion (non-blocking). De-duplicated: returns ``False`` if
        a promotion is already queued or running.
        """
        if self._inflight:
            log.debug("promotion already in flight — skipping (%s)", reason)
            return False
        self._inflight = True

        def _factory() -> Any:
            return self._run_promotion(self._promoter, reason)

        queued = self._bg.schedule_nowait(_factory)
        if not queued:
            self._inflight = False  # queue closed/full — let the next try re-arm
        return bool(queued)

    async def force_now(self, scope: Any | None = None) -> Any:
        """
        Run promotion **immediately** and await it — the manual
        ``session.checkpoint()`` path. Bypasses the queue and the in-flight
        guard (the caller explicitly wants it now).
        """
        log.info("forced promotion (checkpoint) scope=%s", scope)
        return await self._promoter.promote(scope)

    async def _run_promotion(self, promoter: Any, reason: str) -> None:
        try:
            log.info("promotion triggered by %s", reason)
            report = await promoter.promote()
            log.info(
                "triggered promotion done (%s): promoted=%s",
                reason, getattr(report, "promoted_count", "?"),
            )
        except Exception:
            log.exception("triggered promotion (%s) failed", reason)
        finally:
            self._inflight = False  # re-arm for the next trigger


__all__ = ["TriggerManager"]
