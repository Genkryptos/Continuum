"""
continuum/core/protocols.py
===========================
Structural Protocol interfaces for the Continuum memory framework (PEP 544).

Every class here is a ``typing.Protocol``.  Implementors do *not* inherit from
these classes — structural (duck-type) compatibility is enough for both static
type-checkers (mypy / pyright) and runtime ``isinstance`` checks via
``@runtime_checkable``.

The eight protocols map directly to the architectural layers:

    ScorerProtocol      — dimension scoring & composite formula
    STMProtocol         — session-level message buffer
    MTMProtocol         — project-context persistence & work queue
    LTMProtocol         — bi-temporal durable fact store + knowledge graph
    RetrieverProtocol   — unified fan-out across all three tiers
    PromoterProtocol    — LLM-driven MTM → LTM promotion
    OptimizerProtocol   — token-budget trimming / compression
    PersistenceProtocol — swappable raw storage backend

Scoring formula (canonical, encoded in ScoreBreakdown.compute)
--------------------------------------------------------------
    composite = 0.45·relevance + 0.25·importance + 0.20·recency + 0.10·confidence

Quick usage
-----------
    from continuum.core.protocols import RetrieverProtocol, STMProtocol
    from continuum.core.types import Query, TokenBudget

    async def chat(retriever: RetrieverProtocol, query: Query, budget: TokenBudget):
        bundle = await retriever.retrieve(query, budget)
        return bundle.messages
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from continuum.core.types import (
    BiTemporalRange,
    ContextBundle,
    MemoryItem,
    MemoryTier,
    ProcessingState,
    PromotionReport,
    Query,
    ScoreBreakdown,
    ScoredItem,
    TokenBudget,
)

# Dict[str, Any] used in STMProtocol.stats return type (stdlib only)
_StatsDict = dict[str, Any]

# ============================================================================
# ScorerProtocol
# ============================================================================


@runtime_checkable
class ScorerProtocol(Protocol):
    """
    Computes relevance / importance / recency / confidence scores for a
    MemoryItem against a Query.

    Canonical composite formula
    ---------------------------
        composite = 0.45·relevance + 0.25·importance + 0.20·recency + 0.10·confidence

    The two methods intentionally separate concerns:

    * ``score``     — hot path, called once per candidate during retrieval.
                      Returns only the scalar composite; implementations should
                      be fast (no LLM calls, minimal allocations).

    * ``breakdown`` — introspection path, called by observability tooling,
                      re-ranking UI, or tests that assert individual dimensions.
                      Returns the full ScoreBreakdown dataclass.

    Implementations may add extra dimensions (e.g. session-affinity boost) as
    long as the returned composite stays in [0, 1] and ``breakdown`` faithfully
    reflects the components used.
    """

    def score(
        self,
        item: MemoryItem,
        query: Query,
        now: datetime,
    ) -> float:
        """
        Return the composite relevance score for *item* w.r.t. *query*.

        Parameters
        ----------
        item:
            The memory item being evaluated.
        query:
            The retrieval query supplying semantic context.
        now:
            Explicit reference timestamp for recency decay.  Passing a fixed
            value keeps scoring deterministic in tests.

        Returns
        -------
        float
            A value in [0, 1].  1.0 means perfectly relevant / important /
            recent / confident.
        """
        ...

    def breakdown(
        self,
        item: MemoryItem,
        query: Query,
        now: datetime,
    ) -> ScoreBreakdown:
        """
        Return the full per-dimension breakdown for *item*.

        Callers use this when they need to log, visualise, or override
        individual dimensions (e.g. a downstream re-ranker that replaces the
        relevance component with a cross-encoder score).
        """
        ...


# ============================================================================
# STMProtocol
# ============================================================================


@runtime_checkable
class STMProtocol(Protocol):
    """
    Short-term memory — the bounded, session-local conversation buffer.

    STM holds the most recent turns of an ongoing conversation.  When the
    token cap is reached, the oldest messages are evicted and handed to the
    on_evict callback (which typically calls ``flush_to`` to persist them in
    MTM before they are discarded).

    All six methods are ``async`` so STM can be backed by a fast in-process
    deque (``InMemorySTM``), a distributed cache (Redis), or a relational
    store (``PostgresSTM``) without any caller changes.

    Typical lifecycle
    -----------------
    1. ``append``    — called after every LLM turn (user + assistant messages)
    2. ``window``    — called by the Retriever to include recent context
    3. ``get_recent``— light retrieval when a fixed message count is preferred
    4. ``flush_to``  — called at end-of-session to promote items to MTM
    5. ``clear``     — discard buffer without flushing (e.g. context reset)
    6. ``stats``     — utilisation telemetry for monitoring and compression

    Method surface
    --------------
    All methods that scope to a session accept *session_id* explicitly.
    This makes ``STMProtocol`` implementations session-agnostic — a single
    instance can serve multiple concurrent sessions (``PostgresSTM``,
    ``InMemorySTM``) or delegate to a per-session object (``ConversationSTM``
    wrapper) by validating the id against a stored value.
    """

    async def append(self, item: MemoryItem) -> None:
        """
        Append *item* to the STM buffer for its session.

        Implementations must:
        * Set ``item.tier = MemoryTier.STM`` if not already set.
        * Compute token count and store it in ``item.metadata["tokens"]`` if
          not already present.
        * Enforce the configured token / message cap, evicting oldest items
          first and invoking any on_evict callbacks before dropping.
        * Be concurrency-safe (use an asyncio.Lock if the store is not
          naturally concurrent).

        Parameters
        ----------
        item:
            A new message or system note.  ``item.session_id`` must be set;
            ``item.metadata["role"]`` should be one of ``user`` / ``assistant``
            / ``system`` / ``tool``.
        """
        ...

    async def window(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> Sequence[MemoryItem]:
        """
        Return the conversation window for *session_id*, oldest-first.

        Parameters
        ----------
        session_id:
            Identifies the conversation whose buffer is requested.
        max_tokens:
            When provided, trim the returned window to fit this token budget.
            Most-recent messages have priority; older ones are dropped from
            the front.  ``None`` returns the full untruncated buffer up to
            the implementation's own ``max_tokens - reserved_for_response``.

        Returns
        -------
        Sequence[MemoryItem]
            Ordered oldest-first, ready for serialisation into chat messages.
        """
        ...

    async def get_recent(
        self,
        session_id: str,
        n: int = 10,
    ) -> Sequence[MemoryItem]:
        """
        Return the *n* most-recent messages for *session_id*, oldest-first.

        Unlike ``window``, this method ignores token budgets — it is a
        lightweight alternative when a fixed message count is preferred over
        a token-budget window (e.g. building a short "last N turns" context).

        Parameters
        ----------
        session_id:
            The conversation to query.
        n:
            Maximum number of messages to return.  If the buffer has fewer
            than *n* messages, all of them are returned.

        Returns
        -------
        Sequence[MemoryItem]
            The *n* most-recent items, ordered oldest-first.
        """
        ...

    async def flush_to(
        self,
        session_id: str,
        target: MTMProtocol,
    ) -> int:
        """
        Evict all items for *session_id* and hand them to *target* (MTM).

        Called at end-of-session or when a compression trigger fires.
        Implementations must clear the buffer *only after* the target confirms
        receipt so no data is lost on failure.

        If the target raises for some items, implementations may choose to:
        * Abort and return the partial transfer count (recommended), OR
        * Continue and report failures via logging.

        In both cases the STM buffer must not be cleared for items that failed
        to transfer.

        Parameters
        ----------
        session_id:
            The session whose buffer should be flushed.
        target:
            The MTMProtocol implementation that will receive the evicted items.

        Returns
        -------
        int
            Number of items successfully transferred to *target*.
        """
        ...

    async def clear(self, session_id: str) -> None:
        """
        Discard all STM messages for *session_id* without flushing to MTM.

        Use this for context resets (e.g. user starts a new topic), hard
        error recovery, or test teardown.  Unlike ``flush_to``, no MTM call
        is made — items are simply dropped.

        Parameters
        ----------
        session_id:
            The conversation to clear.
        """
        ...

    async def stats(self, session_id: str) -> _StatsDict:
        """
        Return utilisation statistics for *session_id*.

        Implementations should return at minimum:

        .. code-block:: python

            {
                "session_id":      str,
                "message_count":   int,
                "total_tokens":    int,
                "max_tokens":      int,
                "utilization":     float,   # 0.0–1.0
                "budget_remaining": int,
            }

        Extra keys (e.g. ``"compression_count"``, ``"eviction_count"``) are
        allowed and will be passed through transparently by the Retriever.

        Parameters
        ----------
        session_id:
            The conversation to inspect.
        """
        ...


# ============================================================================
# MTMProtocol
# ============================================================================


@runtime_checkable
class MTMProtocol(Protocol):
    """
    Mid-term memory — project / session context persisted across turns.

    MTM stores LLM-generated summaries and high-importance episodes from the
    conversation.  Items accumulate across turns and sessions.  A background
    Promoter periodically scans UNPROCESSED items to decide which ones deserve
    promotion to LTM as durable facts.

    The ``scan_unprocessed`` / ``mark_processed`` pair implements a lightweight
    at-least-once work-queue so the Promoter can run incrementally in batches
    without re-evaluating items it has already handled.

    Typical lifecycle
    -----------------
    1. STM eviction / compression  →  ``add_summary``
    2. Retriever fan-out           →  ``recent``
    3. Promoter batch              →  ``scan_unprocessed`` → evaluate → ``mark_processed``
    """

    async def add_summary(self, summary: MemoryItem) -> str:
        """
        Persist *summary* (typically the output of STM compression) in MTM.

        Implementations must:
        * Set ``summary.tier = MemoryTier.MTM``.
        * Set ``summary.processing_state = ProcessingState.UNPROCESSED`` so the
          Promoter will evaluate this item in a future batch.
        * Generate and store an embedding if ``summary.embedding`` is ``None``.

        Parameters
        ----------
        summary:
            A MemoryItem whose ``content`` is an LLM-generated summary or an
            important episode evicted from STM.

        Returns
        -------
        str
            The assigned ``id`` of the persisted MTM item.
        """
        ...

    async def recent(
        self,
        token_budget: int,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        exclude_session_id: str | None = None,
    ) -> Sequence[MemoryItem]:
        """
        Return the most recent MTM items within *token_budget*, newest-first.

        Used by the Retriever to surface fresh project context without running
        a full vector search. Implementations may scope to the current session
        or same-user memories from other sessions.

        Parameters
        ----------
        token_budget:
            Maximum context budget to spend on returned MTM items.
        session_id:
            Optional single-session scope.
        user_id:
            Optional same-user scope for cross-session recall.
        exclude_session_id:
            Optional session to omit when querying same-user memories.
        """
        ...

    async def scan_unprocessed(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> Sequence[MemoryItem]:
        """
        Return MTM items whose ``processing_state`` is UNPROCESSED.

        The Promoter calls this to obtain its next batch of promotion
        candidates.  Items must be returned in ascending ``created_at`` order
        (oldest-first) so the Promoter evaluates them chronologically.

        Implementations should *not* lock the returned rows — the work-queue
        is idempotent because ``mark_processed`` is called after evaluation
        regardless of the promotion outcome.

        Parameters
        ----------
        agent_id:
            Narrows the scan to one agent's items; prevents cross-agent leakage
            when multiple agents share the same MTM store.
        limit:
            Batch size cap.  Keep it small (≤ 100) to bound transaction length.
        """
        ...

    async def mark_processed(self, item_ids: Sequence[str]) -> int:
        """
        Transition the given items to ``ProcessingState.PROCESSED``.

        Called by the Promoter after evaluation — regardless of whether the
        item was promoted — so that future ``scan_unprocessed`` calls skip it.

        Items that fail promotion should still be marked PROCESSED (not FAILED)
        unless a hard error occurred, to avoid infinite retry loops.

        Parameters
        ----------
        item_ids:
            The ``MemoryItem.id`` values to update.

        Returns
        -------
        int
            Number of rows actually updated (≤ ``len(item_ids)`` when some ids
            were already processed or do not exist).
        """
        ...


# ============================================================================
# LTMProtocol
# ============================================================================


@runtime_checkable
class LTMProtocol(Protocol):
    """
    Long-term memory — bi-temporal, durable, cross-session fact store with a
    knowledge graph.

    LTM items represent stable knowledge (user preferences, project decisions,
    domain facts) that must persist across many sessions.  They carry a
    ``BiTemporalRange`` that tracks *when the fact was true in the world*,
    independent of when Continuum stored it.

    ``search_hybrid`` blends dense vector similarity with keyword / metadata
    filters, letting callers constrain by date range, importance threshold, or
    custom tags without sacrificing semantic recall.

    ``neighbors`` exposes the knowledge graph: LTM items reference each other
    via ``MemoryItem.related_ids``, and this method traverses those edges up to
    *depth* hops from a seed item.

    Bi-temporal write pattern
    -------------------------
    1. ``upsert``      — create or supersede an existing fact; implementation
                         closes the old record's valid_to before opening the new one.
    2. ``update``      — partial field change; same bi-temporal bookkeeping.
    3. ``invalidate``  — close a fact's validity window without creating a successor.
    """

    async def upsert(self, item: MemoryItem) -> str:
        """
        Insert *item* or supersede an existing LTM record with the same
        logical identity.

        Implementations must:
        * Set ``item.tier = MemoryTier.LTM``.
        * Close the ``valid_to`` of any previous version before opening the new
          one, preserving full bi-temporal history.
        * Generate an embedding if ``item.embedding`` is ``None``.

        Returns
        -------
        str
            The ``id`` of the created or updated LTM item.
        """
        ...

    async def update(
        self,
        item_id: str,
        updates: dict[str, Any],
    ) -> MemoryItem:
        """
        Apply a partial update to an existing LTM item.

        Only the fields present in *updates* are changed.  Implementations must
        close the previous bi-temporal record and open a new one so that the
        full history remains queryable via point-in-time ``as_of`` queries.

        Parameters
        ----------
        item_id:
            The ``MemoryItem.id`` to update.
        updates:
            Mapping of field names to their new values.
            Example: ``{"importance": 0.9, "confidence": 0.75}``

        Returns
        -------
        MemoryItem
            The full updated item reflecting its new state.

        Raises
        ------
        KeyError
            If *item_id* does not exist.
        """
        ...

    async def invalidate(
        self,
        item_id: str,
        as_of: datetime | None = None,
    ) -> None:
        """
        Close the validity window of *item_id* as of *as_of*.

        The item is *not* deleted — it becomes historical.  Queries with a
        ``Query.as_of`` value strictly before *as_of* still return it; queries
        at or after *as_of* do not.

        Parameters
        ----------
        item_id:
            The fact to invalidate.
        as_of:
            The world-time at which the fact ceased to be true.
            Defaults to ``datetime.now(utc)`` when ``None``.

        Raises
        ------
        KeyError
            If *item_id* does not exist or is already invalidated.
        """
        ...

    async def search_hybrid(
        self,
        query: Query,
        top_k: int = 10,
    ) -> Sequence[ScoredItem]:
        """
        Retrieve LTM items using a blend of vector similarity and keyword /
        metadata filtering.

        Implementations should:
        1. Embed ``query.text`` if ``query.embedding`` is ``None`` (once).
        2. Filter by ``query.metadata_filter`` (tags, importance threshold, …).
        3. Apply bi-temporal filtering: only items valid at ``query.as_of``
           (defaults to now) are eligible.
        4. Rank survivors with the canonical formula:
               0.45·relevance + 0.25·importance + 0.20·recency + 0.10·confidence

        Parameters
        ----------
        query:
            The retrieval request, including optional bi-temporal ``as_of``.
        top_k:
            Maximum number of results.

        Returns
        -------
        Sequence[ScoredItem]
            Items sorted by descending composite score.
        """
        ...

    async def neighbors(
        self,
        item_id: str,
        depth: int = 1,
    ) -> Sequence[MemoryItem]:
        """
        Traverse the LTM knowledge graph starting from *item_id*.

        Edges are defined by ``MemoryItem.related_ids``.  At ``depth=1`` only
        direct neighbours are returned; at ``depth=2`` neighbours-of-neighbours
        are included, and so on.

        Implementations must:
        * Guard against cycles (track visited ids).
        * Apply a sensible node cap to prevent runaway traversal on dense graphs.

        Parameters
        ----------
        item_id:
            The seed node; excluded from the result set.
        depth:
            Hop limit.  Use depth ≥ 2 with care on large graphs — callers
            should treat it as a hint, not a guarantee.

        Returns
        -------
        Sequence[MemoryItem]
            All reachable items within *depth* hops, excluding the seed.

        Raises
        ------
        KeyError
            If *item_id* does not exist in LTM.
        """
        ...


# ============================================================================
# RetrieverProtocol
# ============================================================================


@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Unified retrieval across STM, MTM, and LTM.

    The Retriever is the single entry point for assembling context before an
    LLM call.  It fans out to each tier, scores candidates with a
    ScorerProtocol implementation, then calls OptimizerProtocol to pack the
    winning items into the available token budget.

    From the agent's perspective a single ``await retriever.retrieve(q, budget)``
    is all that is needed:

        bundle = await retriever.retrieve(query, budget)
        response = await llm.chat(bundle.messages)

    Internal fan-out sequence (canonical implementation)
    ----------------------------------------------------
    1. Embed ``query.text`` once (reuse across all tiers).
    2. Fan out to ``query.tiers`` concurrently.
    3. Merge and score all candidates with ScorerProtocol.score.
    4. Pass the ranked list to OptimizerProtocol.apply.
    5. Return the resulting ContextBundle.
    """

    async def retrieve(
        self,
        query: Query,
        budget: TokenBudget,
    ) -> ContextBundle:
        """
        Retrieve and assemble a token-budgeted context bundle for *query*.

        Parameters
        ----------
        query:
            What to retrieve and from which tiers.  The implementation fills
            ``query.embedding`` before fan-out so it is computed only once.
        budget:
            Token envelope that caps the returned context.

        Returns
        -------
        ContextBundle
            Assembled, budget-fitting context with prompt-ready ``messages``,
            source ``items``, and ``tier_breakdown`` telemetry.
        """
        ...


# ============================================================================
# PromoterProtocol
# ============================================================================


@runtime_checkable
class PromoterProtocol(Protocol):
    """
    MTM → LTM promotion with LLM-driven decisions.

    The Promoter runs periodically — end-of-session, nightly cron, or
    post-turn hook — and evaluates UNPROCESSED MTM items to decide whether
    any deserves promotion to LTM as a durable fact.

    The decision is expected to be LLM-driven: the Promoter assembles a
    prompt from the candidate's content, submits it to an LLM with a
    structured-output schema (promote: bool, reason: str, importance: float),
    then acts on the answer:
    * promote=True  → call LTMProtocol.upsert, then MTMProtocol.mark_processed
    * promote=False → call MTMProtocol.mark_processed only

    Scope convention
    ----------------
    "session:<session_id>"  — process one session's unprocessed MTM items
    "agent:<agent_id>"      — process all sessions for one agent
    "global"                — process all unprocessed items across all agents
    """

    async def promote(self, scope: str) -> PromotionReport:
        """
        Evaluate UNPROCESSED MTM items within *scope* for LTM promotion.

        Parameters
        ----------
        scope:
            Identifies what to scan.  See class docstring for conventions.

        Returns
        -------
        PromotionReport
            Counts, the full candidate list, and per-candidate decisions.
            ``elapsed_ms`` should be populated for observability.
        """
        ...


# ============================================================================
# OptimizerProtocol
# ============================================================================


@runtime_checkable
class OptimizerProtocol(Protocol):
    """
    Token-budget optimisation — trim or compress a ContextBundle to fit.

    The Retriever over-fetches deliberately (more candidates than the budget
    allows) so the Optimizer has room to make good trade-offs.  The Optimizer
    is the last step before the bundle is returned to the caller.

    Built-in strategies (each is a separate concrete class):
    * ``TruncationOptimizer``   — drop lowest-scored items until budget fits.
    * ``CompressionOptimizer``  — shorten item content via LLMlingua-style
                                  prompt compression, keeping all items.
    * ``HybridOptimizer``       — compress low-priority items, keep high-priority
                                  verbatim; falls back to truncation if needed.

    Contract
    --------
    Implementations must *never* alter the semantic meaning of retained items.
    They may only remove items from ``ctx.items`` or shorten ``item.content``.
    The returned ``debug_info`` should record every drop / compression decision
    so callers can audit the Optimizer's choices.
    """

    async def apply(
        self,
        ctx: ContextBundle,
        budget: TokenBudget,
    ) -> ContextBundle:
        """
        Trim or compress *ctx* until ``tokens_used ≤ budget.context_available``.

        Parameters
        ----------
        ctx:
            The over-fetched bundle from the Retriever.  ``ctx.tokens_used``
            may exceed ``budget.context_available`` on entry.
        budget:
            The target token envelope.

        Returns
        -------
        ContextBundle
            A new bundle (never mutate *ctx* in place) where:
            * ``tokens_used ≤ budget.context_available``
            * ``debug_info`` records what was dropped or compressed
            * ``tier_breakdown`` is recalculated to reflect actual usage
        """
        ...


# ============================================================================
# RerankerProtocol
# ============================================================================


@runtime_checkable
class RerankerProtocol(Protocol):
    """
    Cross-encoder reranking — a precision second pass over a recalled set.

    The Retriever's first stage optimises *recall* (bi-encoder vector search
    + lexical) and deliberately over-fetches. A reranker then optimises
    *precision*: it scores each ``(query, item.content)`` pair jointly with a
    cross-encoder (e.g. ``BAAI/bge-reranker-v2-m3``) and re-sorts.

    Contract
    --------
    * Return the same items, re-ordered best-first — never drop or mutate
      the underlying ``MemoryItem`` content.
    * Be a safe no-op on a too-small set (reranking a handful of items is
      not worth the model latency).
    * Degrade gracefully: on model failure, return the input order rather
      than raising into the retrieval path.
    """

    async def rerank(
        self,
        query: str,
        items: Sequence[ScoredItem],
    ) -> Sequence[ScoredItem]:
        """
        Re-order *items* by cross-encoder relevance to *query*.

        Parameters
        ----------
        query:
            The raw query text (not an embedding — the cross-encoder reads
            the text pair directly).
        items:
            The recalled candidates from stage one, in recall order.

        Returns
        -------
        Sequence[ScoredItem]
            The same items, best-first by cross-encoder score.
        """
        ...


# ============================================================================
# PersistenceProtocol
# ============================================================================


@runtime_checkable
class PersistenceProtocol(Protocol):
    """
    Swappable raw storage backend for any memory tier.

    This is the lowest-level abstraction: it knows nothing about scoring,
    bi-temporal validity, or sessions — it simply stores and retrieves
    MemoryItem objects by id.

    Concrete implementations:
    * ``PostgresPersistence``   — production (psycopg2 + pgvector)
    * ``SQLitePersistence``     — local development / CI
    * ``InMemoryPersistence``   — unit tests (plain dict)

    Higher-level components (MTMProtocol, LTMProtocol) compose this protocol
    rather than inheriting from it, so the storage backend can be swapped
    without touching any business logic.
    """

    async def save(self, item: MemoryItem) -> str:
        """
        Persist *item* and return its storage id.

        Upsert semantics: if an item with ``item.id`` already exists, overwrite
        it.  Implementations that generate their own ids should return the new
        id even when it differs from ``item.id``.

        Parameters
        ----------
        item:
            The memory item to persist.  Must have ``tier`` set.

        Returns
        -------
        str
            The ``id`` under which the item was stored.
        """
        ...

    async def load(self, item_id: str) -> MemoryItem | None:
        """
        Load a single item by its id.

        Returns ``None`` (rather than raising) when the item does not exist, so
        callers can distinguish "not found" from storage errors cleanly.
        """
        ...

    async def delete(self, item_id: str) -> bool:
        """
        Permanently delete the item with *item_id*.

        Returns
        -------
        bool
            ``True`` if an item was deleted, ``False`` if it did not exist.
        """
        ...

    async def exists(self, item_id: str) -> bool:
        """Return ``True`` if an item with *item_id* is currently stored."""
        ...


# ============================================================================
# Public surface
# ============================================================================

__all__ = [
    # Protocols (dependency order)
    "ScorerProtocol",
    "STMProtocol",
    "MTMProtocol",
    "LTMProtocol",
    "RetrieverProtocol",
    "PromoterProtocol",
    "OptimizerProtocol",
    "RerankerProtocol",
    "PersistenceProtocol",
    # Types re-exported for one-stop import
    "MemoryItem",
    "MemoryTier",
    "ProcessingState",
    "BiTemporalRange",
    "Query",
    "TokenBudget",
    "ContextBundle",
    "ScoreBreakdown",
    "ScoredItem",
    "PromotionReport",
]
