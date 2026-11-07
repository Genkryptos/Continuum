"""
evals/longmemeval/continuum_ltm_store.py
========================================
``ContinuumLTMHaystackStore`` — drop-in replacement for
:class:`FlatHaystackStore` (defined in
:mod:`evals.longmemeval.bootstrap_ollama`) that drives the real
Continuum LTM pipeline against each LongMemEval row's haystack.

Why this exists
---------------
``FlatHaystackStore`` is a dumb append-list. The standard rig pumps
every raw turn into it and lets cosine / BM25 rank the bag. For
knowledge-update questions — n=78 in LongMemEval-S — this is a
structural ceiling: the stale half of every update pair still
competes with the current half during retrieval, and the answerer
has no way to tell which is current. The rig scores ~33% (substring)
/ ~40% (judge) on this category.

This store wires in Continuum's promotion path so the LTM rows the
retriever sees are the **current** facts only:

1. ``append(item)``  buckets raw turns by ``metadata["session_id"]``
   exactly like :class:`FlatHaystackStore` — ``items`` keeps the
   raw turns for the lexical fallback path.
2. ``finalize()`` (called by the rig after the per-message loop):
   * For each session bucket, synthesise one ``SummaryBlock`` (we
     deliberately skip the LLM MTM-compression step in v1 — see plan).
   * Run :class:`FactExtractor` (or a heuristic-only path when no LLM
     is available) to mint atomic ``Fact``\\s.
   * Upsert each fact into the LTM.
   * Run :class:`ContinuumPromoter.decide_operations_batch` over the new
     facts + their nearest live LTM neighbours.
   * Execute the resulting decisions: ``DELETE`` → ``ltm.invalidate``
     (supersession); ``UPDATE`` → ``ltm.update``; ``ADD`` is already
     in (we upserted ahead of time); ``NOOP`` skipped.
   * Stream the resulting **live** LTM facts onto ``items`` tagged
     ``metadata["source"]="ltm_fact"`` so the rig's hybrid retriever
     can rank them alongside raw turns.

Composition (decided in the plan)
---------------------------------
Items-level union. The store does not replace the retriever — it
enriches ``items``. ``--retriever hybrid`` (cosine + BM25 RRF) then
ranks raw turns + live LTM facts together. The "two queries and
merge" the original spec described is realised here as how ``items``
is populated.

Heuristic fallback
------------------
When no LLM provider key is in the environment (or the caller
passes ``llm_available=False``), the LLM-driven :class:`ContinuumPromoter`
is skipped. A deterministic supersession heuristic runs in its
place: regex over "no longer / used to / now / instead of /
changed to / moved to" markers emits ``DELETE(target=most-recent
same-entity fact) + ADD(new)`` pairs. This won't generalise but it
captures most knowledge-update signal and lets the rig run
hermetically. The path taken is reported in :meth:`metrics`.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from continuum.core.types import (
    MemoryItem,
    MemoryTier,
    ScoredItem,
    SummaryBlock,
)
from continuum.extraction.fact_extractor import Fact
from continuum.promotion.continuum_promoter import Decision
from evals.longmemeval import ingest_cache as ic

log = logging.getLogger(__name__)

# Regex markers the heuristic path uses to flag "this fact contradicts
# an earlier one." Kept narrow on purpose — the pattern is meant to
# fire ONLY on knowledge-update style language; false positives are
# more harmful than false negatives because they invalidate a live
# fact the answerer needs.
_UPDATE_MARKERS = re.compile(
    r"\b("
    r"no\s+longer|used\s+to|previously|formerly|"
    r"now\s+(?:lives?|works?|owns?|uses?|prefers?|drives?|studies?)|"
    r"changed\s+(?:to|from)|switched\s+(?:to|from)|"
    r"replaced\s+with|instead\s+of|"
    r"moved\s+(?:to|from)|relocated\s+to"
    r")\b",
    re.IGNORECASE,
)


# ── canonical fact-bearing item shape ──────────────────────────────────────


@dataclass
class _PromoterStats:
    """Per-row counters surfaced via :meth:`ContinuumLTMHaystackStore.metrics`."""
    added: int = 0
    updated: int = 0
    deleted: int = 0
    noop: int = 0
    errors: list[str] = field(default_factory=list)


# Type aliases for the duck-typed dependencies the store accepts.
# Real types from continuum.* are accepted at runtime; aliases here
# keep this file independent of those import paths for testing.
LTMLike = Any  # supports upsert / update / invalidate / search_hybrid / iter_live

#: The two arms of the A5 design-space experiment (see
#: ``docs/ACCURACY_AND_BENCHMARK_PLAN.md`` §A5). Everything else in the pipeline
#: is held constant; only what happens to a *superseded* fact differs.
#:
#: * ``invalidate`` — **arm A**, the historical default. Mem0 0.1.67's model:
#:   ``DELETE`` stamps ``invalidated_at`` (transaction time) so the fact becomes
#:   unreachable, and ``UPDATE`` mutates the target's content in place,
#:   destroying the old value. History is not recoverable.
#: * ``bitemporal`` — **arm B**. Closes the superseded fact's *valid* time
#:   (``valid_to``) and leaves ``invalidated_at`` NULL, so the row survives and
#:   stays reachable by an as-of query. ``UPDATE`` becomes a supersession too:
#:   the old fact is closed, the new one already exists, and no content is
#:   overwritten.
#:
#: Retrieval sees the same *current* view under both arms (the drain in
#: :meth:`finalize` filters closed facts), so a difference in accuracy is
#: attributable to information retained, not to a changed candidate set.
_SUPERSESSION_MODES = {"invalidate", "bitemporal"}


PromoterLike = Any  # supports decide_operations_batch
FactExtractorLike = Any  # supports extract_facts
EmbedderLike = Any  # supports encode(list[str]) -> np.ndarray-like


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity; 0.0 when either vector is missing or degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def _rescore_as_cosine(
    qvec: list[float] | None, neighbours: list[ScoredItem]
) -> list[ScoredItem]:
    """
    Replace each neighbour's ``scores.relevance`` with **true cosine** against
    the candidate.

    ``ContinuumPromoter`` documents ``ScoreBreakdown.relevance`` as cosine similarity
    and short-circuits on it: ``< add_threshold`` (0.5) → ADD without ever
    calling the LLM. But ``search_hybrid`` returns **RRF** scores there —
    rank-derived, ~1/(60+rank) ≈ 0.016 — so every candidate fell below 0.5 and
    short-circuited. Measured: 912 facts → 912 ADDs, 0 NOOP/UPDATE/DELETE, and
    the LLM promoter never invoked once.

    A unit mismatch, not a tuning problem: lowering the threshold would just
    make rank-1 always "similar" regardless of content.
    """
    if not qvec:
        return neighbours
    out: list[ScoredItem] = []
    for si in neighbours:
        emb = getattr(si.item, "embedding", None)
        cos = _cosine(qvec, list(emb) if emb else [])
        out.append(replace(si, scores=replace(si.scores, relevance=cos)))
    return out


def _is_closed(row: MemoryItem, at: datetime) -> bool:
    """True when *row*'s valid time has closed on or before *at* (arm B)."""
    vr = getattr(row, "valid_range", None)
    valid_to = getattr(vr, "valid_to", None) if vr is not None else None
    return valid_to is not None and valid_to <= at


class ContinuumLTMHaystackStore:
    """
    Raw-turn haystack with a live LTM view woven in at finalize time.

    Parameters
    ----------
    ltm:
        :class:`continuum.core.protocols.LTMProtocol` implementation.
        Either :class:`continuum.stores.in_memory.ltm.InMemoryLTM` or
        :class:`continuum.stores.postgres.ltm.PostgresLTM`.
    embedder:
        Embedder with ``encode(list[str]) -> ndarray-like`` (the
        :class:`evals.longmemeval.bootstrap_ollama._Embedder` shape).
        Used to embed extracted facts before LTM upsert so the
        in-mem LTM's cosine channel has something to rank.
    promoter:
        :class:`continuum.promotion.continuum_promoter.ContinuumPromoter` (or
        any object with the same ``decide_operations_batch`` shape).
        Ignored when ``llm_available=False``.
    fact_extractor:
        :class:`continuum.extraction.fact_extractor.FactExtractor`
        (or compatible). Ignored when ``llm_available=False``; the
        heuristic path treats every assistant turn as a candidate
        "fact" with simple entity extraction.
    llm_available:
        ``True`` to run the LLM-driven extraction + promotion path.
        ``False`` to fall back to the deterministic heuristic.
    ltm_backend_label:
        Tag stamped onto ``metrics()`` so the result JSON records
        which backend produced the numbers (``"in_memory"`` /
        ``"postgres"``).
    """

    def __init__(
        self,
        *,
        ltm: LTMLike,
        embedder: EmbedderLike | None = None,
        promoter: PromoterLike | None = None,
        fact_extractor: FactExtractorLike | None = None,
        llm_available: bool = False,
        ltm_backend_label: str = "in_memory",
        supersession_mode: str = "invalidate",
        cache: ic.IngestCache | None = None,
        cache_fingerprint: str = "",
        concurrency: int = 1,
        replay_facts_only: bool = False,
    ) -> None:
        self.items: list[MemoryItem] = []  # FlatHaystackStore-compatible
        self._ltm = ltm
        self._embedder = embedder
        self._promoter = promoter
        self._fact_extractor = fact_extractor
        self._llm_available = llm_available and (
            promoter is not None and fact_extractor is not None
        )
        self._ltm_backend_label = ltm_backend_label
        if supersession_mode not in _SUPERSESSION_MODES:
            raise ValueError(
                f"supersession_mode must be one of {sorted(_SUPERSESSION_MODES)}, "
                f"got {supersession_mode!r}"
            )
        self._supersession_mode = supersession_mode

        self._session_buffers: dict[str, list[MemoryItem]] = {}
        self._stats = _PromoterStats()
        self._supersession_count = 0
        self._ltm_facts_live = 0
        self._superseded_retained = 0
        self._blocks_seen = 0
        self._llm_fallbacks = 0
        self._cache = cache
        self._cache_fingerprint = cache_fingerprint
        self._replay_facts_only = bool(replay_facts_only)
        self._cache_state = "off"
        self._concurrency = max(1, int(concurrency))
        self._pending_cache: tuple[str | None, list[tuple[str, Fact]]] | None = None
        self._promoter_path: str = "disabled"
        self._finalized = False

    # ── FlatHaystackStore-compatible surface ───────────────────────────────

    async def append(self, item: MemoryItem) -> None:
        """Append a raw turn. Also bucket it by session for finalize()."""
        self.items.append(item)
        sid = _session_id(item)
        self._session_buffers.setdefault(sid, []).append(item)

    # ── eval rig entry point ───────────────────────────────────────────────

    async def finalize(self) -> None:
        """
        Drive STM → MTM-block → facts → LTM → supersession.

        Idempotent: a second call is a no-op. The rig builds a fresh
        store per row, but defensive idempotency keeps the surface
        safe for callers that might double-fire.
        """
        if self._finalized:
            return
        self._finalized = True

        # ── ingest cache ────────────────────────────────────────────────────
        # Extraction + promoter decisions are identical across supersession
        # arms — only _execute_decisions differs — so the expensive half runs
        # once per row and both arms replay it. See ingest_cache.py.
        blocks: dict[str, Any] = {}
        for session_id, turns in self._session_buffers.items():
            blocks[session_id] = self._synthesise_block(session_id, turns)
        self._blocks_seen = blocks_seen = len(blocks)

        cache_key_ = None
        cached = None
        if self._cache is not None and blocks:
            cache_key_ = ic.cache_key(
                {sid: (b.text or "") for sid, b in blocks.items()},
                self._cache_fingerprint,
            )
            cached = self._cache.load(cache_key_)

        all_facts: list[tuple[Fact, MemoryItem | None]] = []  # (fact, raw_item)
        cached_decisions: list[dict[str, Any]] | None = None

        if cached is not None:
            self._cache_state = "hit"
            self._promoter_path = cached.get("promoter_path", self._promoter_path)
            self._llm_fallbacks = int(cached.get("llm_fallbacks", 0))
            cached_with_session: list[tuple[str, Fact]] = []
            for sid, fact in ic.payload_to_facts(cached.get("facts", []), Fact):
                turns = self._session_buffers.get(sid) or []
                all_facts.append((fact, _representative_turn(turns) if turns else None))
                cached_with_session.append((sid, fact))
            # Carry session attribution across the cache boundary. Decisions are
            # grouped per session so each candidate sees only genuinely *prior*
            # facts; without this the replay path collapses every fact in the row
            # into one batch decided against an empty store, every candidate gets
            # an empty neighbour set, and the result is all-ADD / zero
            # supersessions — indistinguishable from "no contradictions here".
            # Measured: 4 supersessions when replaying cached decisions, 0 when
            # recomputing them, on the same 78 rows.
            #
            # The key is ``None`` so :meth:`finalize`'s save path leaves the
            # cache entry alone — these facts came from it and have not changed.
            self._pending_cache = (None, cached_with_session)
            # ``replay_facts_only`` keeps the expensive half of the cache (fact
            # extraction, ~53 LLM calls per row) and throws away the cheap half
            # (the promoter's decisions), so a change to promotion logic can be
            # measured without paying to re-extract.
            #
            # It exists because ``ic.config_fingerprint`` hashes only models and
            # sampling settings — fact_model, max_tokens, temperature,
            # promoter_model, promoter_path — and *not* the promoter's decision
            # rules. Add a deterministic short-circuit and the fingerprint does
            # not move, so a warm cache would replay the old decisions and the
            # new rule would never run: the same class of silent no-op as the
            # supersessions=0 bug. Until the fingerprint covers a rules version,
            # pass this flag whenever promotion logic has changed.
            cached_decisions = (
                None if self._replay_facts_only else cached.get("decisions", [])
            )
        else:
            self._cache_state = "miss" if self._cache is not None else "off"
            facts_with_session: list[tuple[str, Fact]] = []

            # Sessions are independent — nothing about session N depends on
            # N-1 — so extract them concurrently. Sequentially this is ~53
            # round-trips per row (measured: 511s). Concurrency does NOT raise
            # the request rate: the completion fn admits every call through an
            # AIMD throttle capped at --ingest-rpm, which widens on 429. The
            # semaphore only stops the pipe going idle while responses are in
            # flight, so the throttle stays the governor.
            sem = asyncio.Semaphore(max(1, self._concurrency))

            async def _one(sid: str, blk: Any) -> tuple[str, list[Fact]]:
                async with sem:
                    return sid, await self._extract_facts(blk, self._session_buffers[sid])

            results = await asyncio.gather(
                *(_one(sid, blk) for sid, blk in blocks.items())
            )
            # Reassemble in the original session order — fact order determines
            # decision indices, which the ingest cache stores. Gather preserves
            # input order, but be explicit: a reorder here would silently
            # invalidate every cached entry's target mapping.
            for session_id, facts in results:
                turns = self._session_buffers[session_id]
                for fact in facts:
                    all_facts.append((fact, _representative_turn(turns)))
                    facts_with_session.append((session_id, fact))
            self._pending_cache = (cache_key_, facts_with_session)

        # Fail loudly on a *silent mechanism swap*. When the LLM extractor
        # errors, ``FactExtractor`` logs and swallows it, and ``_extract_facts``
        # quietly falls back to regex mining — so the run keeps going, reports a
        # clean accuracy number, and measures the heuristic promoter instead of
        # the one under test. Every block falling back means the LLM path is
        # dead, not flaky. This is the failure PREREGISTRATION.md P1c exists to
        # catch; without it an A5 result is meaningless.
        if self._llm_available and blocks_seen and self._llm_fallbacks >= blocks_seen:
            raise RuntimeError(
                f"LLM fact extraction failed on all {blocks_seen} session block(s) "
                "— silently downgraded to the heuristic regex path, so this run "
                "would measure the wrong mechanism.\n"
                "The extractor authenticates separately from --provider/--model: "
                "it uses FactExtractionConfig (default gpt-4o-mini via litellm).\n"
                "Check OPENAI_API_KEY is set (.env may define it under a different "
                "name), or route it with "
                "CONTINUUM_FACT_MODEL=openrouter/openai/gpt-4o-mini.\n"
                "To use the heuristic path deliberately, pass --no-llm-promoter."
            )

        if not all_facts:
            return

        # 1-2. Decide, THEN upsert — session by session, in order.
        #
        # This ordering is load-bearing. Upserting everything first (what this
        # store used to do) puts each candidate's own just-inserted row into its
        # neighbour set, and the promoter then sees its own identical text
        # already in memory and returns NOOP/ADD without ever comparing against
        # the older contradicting fact. Measured directly:
        #
        #     neighbours=[Berlin]              -> DELETE   (supersession fires)
        #     neighbours=[Paris(self), Berlin] -> NOOP     (suppressed)
        #
        # That is why every --use-ltm run reported supersessions=0 with ~all
        # ADDs. Production ``continuum/promotion/promoter.py`` already gets this
        # right (neighbours are fetched before any upsert); only this eval store
        # had it inverted.
        #
        # Per-session granularity is the right unit: LongMemEval haystacks are
        # chronological, and knowledge-update contradictions are
        # later-session-supersedes-earlier. Deciding a whole session's facts as
        # one batch keeps the promoter's batching (no extra LLM calls) while
        # giving each candidate only genuinely *prior* facts to compare against.
        new_ids: list[uuid.UUID] = []
        decisions: list[Any] = []
        if cached_decisions is None:
            # Group facts by session, preserving the order they were assembled.
            by_session: dict[str, list[tuple[Fact, MemoryItem | None]]] = {}
            order: list[str] = []
            src = (
                self._pending_cache[1]
                if self._pending_cache
                else [(None, f) for f, _ in all_facts]
            )
            for (sid, _fact), pair in zip(src, all_facts, strict=False):
                key = sid or "_"
                if key not in by_session:
                    by_session[key] = []
                    order.append(key)
                by_session[key].append(pair)

            for sid in order:
                batch = by_session[sid]
                # Decide against the LTM as it stands *before* this session.
                decisions.extend(await self._decide(batch))
                # Only now insert this session's facts, so the next session
                # sees them as prior context.
                for fact, _ in batch:
                    new_ids.append(await self._ltm.upsert(self._fact_to_item(fact)))
        else:
            # Replay: decisions were computed under the same ordering, so the
            # upserts can happen in one pass.
            for fact, _ in all_facts:
                new_ids.append(await self._ltm.upsert(self._fact_to_item(fact)))
            decisions = ic.payload_to_decisions(cached_decisions, new_ids, Decision)

        if cached_decisions is None:  # noqa: SIM102 — kept flat for readability
            if self._cache is not None and self._pending_cache is not None:
                key, facts_with_session = self._pending_cache
                payload_decisions = ic.decisions_to_payload(
                    decisions, {str(nid): i for i, nid in enumerate(new_ids)}
                )
                if key is not None and payload_decisions is not None:
                    self._cache.save(
                        key,
                        {
                            "facts": ic.facts_to_payload(facts_with_session),
                            "decisions": payload_decisions,
                            "promoter_path": self._promoter_path,
                            "llm_fallbacks": self._llm_fallbacks,
                        },
                    )
                elif payload_decisions is None:
                    # A decision targeted a row this ingest did not create, so
                    # it cannot be replayed faithfully. Stay uncached.
                    self._cache_state = "uncacheable"

        # 3. Execute decisions.
        await self._execute_decisions(new_ids, all_facts, decisions)

        # 4. Drain live LTM rows onto items so the rig's retriever
        # can rank them. Each row gets ``source=ltm_fact`` so the
        # picker / claim heads can tell facts from raw turns.
        # Under arm B a superseded fact is still "live" (invalidated_at is
        # NULL), so filter the *current* view here. This keeps the candidate set
        # identical to arm A — the arms differ in what is retained, not in what
        # retrieval can see. Without this the stale fact would compete with its
        # replacement and knowledge-update would get worse, not better.
        cutoff = datetime.now(UTC)
        live = 0
        retained = 0
        async for row in self._ltm.iter_live():
            if self._supersession_mode == "bitemporal" and _is_closed(row, cutoff):
                retained += 1
                continue
            self.items.append(_clone_as_haystack_item(row))
            live += 1
        self._ltm_facts_live = live
        self._superseded_retained = retained

    # ── telemetry ──────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        return {
            "ltm_backend": self._ltm_backend_label,
            "promoter_path": self._promoter_path,
            "supersession_mode": self._supersession_mode,
            "blocks_seen": self._blocks_seen,
            # "hit" means extraction + promoter decisions were replayed rather
            # than recomputed — the row cost no LLM calls.
            "ingest_cache": self._cache_state,
            "ingest_concurrency": self._concurrency,
            # >0 means the LLM extractor failed on some blocks and the regex
            # path silently took over — the run is measuring a mixed mechanism.
            "llm_fallbacks": self._llm_fallbacks,
            "supersessions": self._supersession_count,
            "ltm_facts_live": self._ltm_facts_live,
            # arm B only: facts whose valid time closed but which were kept and
            # remain answerable via as-of. Always 0 under arm A, where the
            # equivalent facts are unreachable.
            "superseded_retained": self._superseded_retained,
            "promoter_added": self._stats.added,
            "promoter_updated": self._stats.updated,
            "promoter_deleted": self._stats.deleted,
            "promoter_noop": self._stats.noop,
            "promoter_errors": list(self._stats.errors),
        }

    # ── private helpers ────────────────────────────────────────────────────

    def _synthesise_block(
        self, session_id: str, turns: Iterable[MemoryItem],
    ) -> SummaryBlock:
        """Treat the session's turn sequence as one synthetic MTM block."""
        text = "\n".join(t.content for t in turns)
        return SummaryBlock(
            text=text,
            tokens=len(text.split()),
            id=uuid.uuid4(),
            session_id=session_id or None,
            created_at=datetime.now(UTC),
            processed=False,
            metadata={"synthesised": True, "turn_count": len(list(turns))},
        )

    async def _extract_facts(
        self, block: SummaryBlock, turns: list[MemoryItem],
    ) -> list[Fact]:
        if self._llm_available and self._fact_extractor is not None:
            try:
                facts = await self._fact_extractor.extract_facts(block, [])
                if facts:
                    self._promoter_path = "llm"
                    return facts
                # Falls through to heuristic when LLM returned nothing —
                # rare but possible if the upstream call timed out.
            except Exception:
                log.exception("FactExtractor failed — falling back to heuristic")
            # Count the downgrade. ``FactExtractor`` logs and swallows its own
            # errors, so without this the swap from the LLM path to the regex
            # path is invisible in the results.
            self._llm_fallbacks += 1
        return self._heuristic_facts(block, turns)

    def _heuristic_facts(
        self, block: SummaryBlock, turns: list[MemoryItem],
    ) -> list[Fact]:
        """
        Deterministic fact mining used when no LLM is wired.

        For knowledge-update specifically the lexical signal is
        strong: a user fact is the entire turn body trimmed; an
        entity is the longest capitalised noun phrase (a coarse but
        consistent proxy). The Mem0 promoter path is bypassed
        (:meth:`_decide`) so DELETE decisions come from regex.
        """
        self._promoter_path = "heuristic"
        out: list[Fact] = []
        for t in turns:
            text = (t.content or "").strip()
            if not text:
                continue
            role = (t.metadata or {}).get("role", "user") if hasattr(t, "metadata") else "user"
            # Only user-side statements are facts we want to promote;
            # assistant responses are echoes or recommendations.
            if str(role).lower() != "user":
                continue
            entities = _coarse_entities(text)
            out.append(Fact(
                text=text,
                confidence=0.5,
                entities_mentioned=entities,
                source_block_id=block.id,
                category=None,
            ))
        return out

    def _fact_to_item(self, fact: Fact) -> MemoryItem:
        """Mirror ``Promoter._fact_to_item`` so the LTM row shape lines up."""
        embedding: list[float] | None = None
        if self._embedder is not None:
            try:
                vec = self._embedder.encode([fact.text])
                embedding = [float(x) for x in list(vec[0])]
            except Exception:
                log.exception("embedder failed for fact %r — storing without vector", fact.text[:60])
        return MemoryItem(
            content=fact.text,
            tier=MemoryTier.LTM,
            confidence=fact.confidence,
            importance=0.5,
            embedding=embedding,
            metadata={
                "kind": "fact",
                "source": "ltm_fact",
                "source_block_id": str(fact.source_block_id),
                "entities": list(fact.entities_mentioned),
                "category": fact.category,
            },
        )

    async def _decide(
        self, facts_with_origin: list[tuple[Fact, MemoryItem | None]],
    ) -> list[Any]:
        """Run promoter (LLM) or heuristic (deterministic). Decision-shaped objects."""
        if self._promoter_path == "llm" and self._promoter is not None:
            # Run ContinuumPromoter against each fact with its live LTM neighbours.
            inputs: list[tuple[Fact, list[ScoredItem]]] = []
            for fact, _ in facts_with_origin:
                neighbours: list[ScoredItem] = []
                try:
                    from continuum.core.types import Query
                    q = Query(text=fact.text)
                    qvec: list[float] | None = None
                    if self._embedder is not None:
                        vec = self._embedder.encode([fact.text])
                        qvec = [float(x) for x in list(vec[0])]
                        q.embedding = qvec
                    neighbours = await self._ltm.search_hybrid(q, 5)
                    neighbours = _rescore_as_cosine(qvec, neighbours)
                except Exception:
                    log.debug("neighbour fetch failed", exc_info=True)
                inputs.append((fact, neighbours))
            try:
                return await self._promoter.decide_operations_batch(inputs)
            except Exception:
                log.exception("ContinuumPromoter failed — falling back to heuristic")
                self._promoter_path = "heuristic"
        return self._heuristic_decisions(facts_with_origin)

    def _heuristic_decisions(
        self, facts_with_origin: list[tuple[Fact, MemoryItem | None]],
    ) -> list["_HeuristicDecision"]:
        """
        Emit DELETE(target=stale match) when an update-marker fires
        AND a previously-stored fact shares one of the candidate's
        entities. Otherwise NOOP — the upsert already counts as ADD.
        """
        decisions: list[_HeuristicDecision] = []
        # Snapshot what's currently in the LTM by walking `_rows` if
        # the in-mem variant is wired; otherwise we walk known facts
        # via `iter_live` (sync would block). For Postgres-backed
        # runs the heuristic path is mostly disabled in practice
        # (Postgres + heuristic is an unusual combination); we keep
        # the path functional by querying via the asyncio loop.
        rows = list(getattr(self._ltm, "_rows", {}).items())
        for fact, _ in facts_with_origin:
            if not _UPDATE_MARKERS.search(fact.text):
                decisions.append(_HeuristicDecision(op="NOOP", target_id=None,
                                                    rationale="no update marker"))
                continue
            entities = set(e.lower() for e in fact.entities_mentioned)
            # Find the most recently inserted live row whose
            # content lexically references one of our entities AND
            # does NOT itself carry the update marker (i.e. it's
            # the stale half of the pair).
            stale_target: uuid.UUID | None = None
            for nid, row in reversed(rows):
                # Skip the just-upserted row (same id).
                if str(nid) == fact.text:  # dummy guard; ids never equal text
                    continue
                lower = row.content.lower()
                if _UPDATE_MARKERS.search(row.content):
                    continue
                if any(ent in lower for ent in entities):
                    stale_target = nid
                    break
            if stale_target is None:
                decisions.append(_HeuristicDecision(op="NOOP", target_id=None,
                                                    rationale="no stale match"))
                continue
            decisions.append(_HeuristicDecision(
                op="DELETE", target_id=stale_target,
                rationale="heuristic supersession",
            ))
        return decisions

    async def _execute_decisions(
        self,
        new_ids: list[uuid.UUID],
        facts: list[tuple[Fact, MemoryItem | None]],
        decisions: list[Any],
    ) -> None:
        now = datetime.now(UTC)
        for new_id, (fact, _), decision in zip(new_ids, facts, decisions, strict=False):
            op = str(getattr(decision, "op", "NOOP")).upper()
            target_id = getattr(decision, "target_id", None)
            if op == "ADD":
                # We already upserted; just count it.
                self._stats.added += 1
            elif op == "UPDATE":
                if target_id is None:
                    self._stats.errors.append(f"UPDATE without target_id: {fact.text[:48]!r}")
                    continue
                try:
                    if self._supersession_mode == "bitemporal":
                        # Arm B: an UPDATE *is* a supersession. Close the old
                        # fact's valid time and keep it; the replacement was
                        # already upserted, so nothing is overwritten and the
                        # prior value stays answerable via an as-of query.
                        await self._close_valid_time(target_id, now)
                    else:
                        # Arm A: overwrite the target in place. The previous
                        # value is destroyed.
                        merged = getattr(decision, "merged_text", None) or fact.text
                        await self._ltm.update(target_id, {"content": merged})
                    self._stats.updated += 1
                    # An UPDATE retires a prior fact just as a DELETE does —
                    # the promoter's choice between the two verbs is
                    # model-dependent (gpt-4o-mini emits DELETE where
                    # llama-3.3-70b emits UPDATE for the same contradiction).
                    # Counting only DELETE made `supersessions` read 0 on a run
                    # where supersession was happening throughout, which is the
                    # exact gate PREREGISTRATION.md P1c keys on.
                    self._supersession_count += 1
                except Exception as exc:
                    self._stats.errors.append(f"update failed: {exc!r}")
            elif op == "DELETE":
                if target_id is None:
                    self._stats.errors.append(f"DELETE without target_id: {fact.text[:48]!r}")
                    continue
                try:
                    if self._supersession_mode == "bitemporal":
                        await self._close_valid_time(target_id, now)
                    else:
                        await self._ltm.invalidate(target_id, now)
                    self._stats.deleted += 1
                    self._supersession_count += 1
                except Exception as exc:
                    self._stats.errors.append(f"supersede failed: {exc!r}")
            else:  # NOOP / unknown — keep the upsert as a plain add
                self._stats.noop += 1

    async def _close_valid_time(self, target_id: Any, at: datetime) -> None:
        """
        Arm B supersession: close *target_id*'s **valid** time at *at*.

        ``invalidated_at`` is deliberately left NULL — the row stays live, so it
        remains reachable by an as-of query. Only the current-view filter in
        :meth:`finalize` hides it from retrieval.
        """
        await self._ltm.update(target_id, {"valid_to": at})


# ── small decision shape for the heuristic path ────────────────────────────


@dataclass
class _HeuristicDecision:
    op: str
    target_id: uuid.UUID | None
    rationale: str
    merged_text: str | None = None


# ── module-level helpers ───────────────────────────────────────────────────


def _session_id(item: MemoryItem) -> str:
    md = getattr(item, "metadata", None) or {}
    return str(md.get("session_id", "") or item.session_id or "")


def _representative_turn(turns: list[MemoryItem]) -> MemoryItem | None:
    return turns[0] if turns else None


def _clone_as_haystack_item(row: MemoryItem) -> MemoryItem:
    """
    Project an LTM fact row back into the rig's MemoryItem shape with
    ``metadata["role"]="assistant"`` so the assistant-claim picker
    treats it as authoritative. ``source`` survives untouched.
    """
    md = dict(getattr(row, "metadata", None) or {})
    md.setdefault("source", "ltm_fact")
    md.setdefault("role", "assistant")
    md.setdefault("kind", "fact")
    return MemoryItem(
        id=row.id,
        content=row.content,
        tier=MemoryTier.LTM,
        importance=row.importance,
        confidence=row.confidence,
        created_at=row.created_at,
        session_id=row.session_id,
        agent_id=row.agent_id,
        user_id=row.user_id,
        embedding=list(row.embedding) if row.embedding is not None else None,
        metadata=md,
    )


_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def _coarse_entities(text: str) -> list[str]:
    """Capitalised noun phrases — a coarse stand-in for the GLiNER+LLM extractor."""
    return _PROPER_NOUN_RE.findall(text)[:6]


# ── public factory wired by bootstrap_ollama ───────────────────────────────


BackendChoice = Callable[[], Awaitable[Any]]


__all__ = ["ContinuumLTMHaystackStore"]
