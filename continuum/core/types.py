"""
continuum/core/types.py
=======================
Shared data types for the Continuum memory framework.

All Protocol interfaces and concrete implementations speak this common
vocabulary.  The module is intentionally dependency-free (stdlib only +
dataclasses) so it can be imported anywhere without pulling in optional
extras such as psycopg2, sentence-transformers, or litellm.

Type map
--------
Enumerations
    MemoryTier          Which layer a record lives in (STM / MTM / LTM)
    ProcessingState     MTM work-queue status (UNPROCESSED / PROCESSED / FAILED)

Core record
    BiTemporalRange     LTM valid-time interval (valid_from … valid_to)
    MemoryItem          The canonical cross-tier memory record

Retrieval I/O
    Query               What an agent wants to retrieve
    TokenBudget         How many tokens each tier and the response may use
    ContextBundle       The assembled, budget-fitting prompt returned to callers

Scoring
    ScoreBreakdown      Per-dimension scores + composite (the canonical formula)
    ScoredItem          MemoryItem paired with its ScoreBreakdown

Promoter
    PromotionCandidate  MTM item flagged as a potential LTM fact
    PromotionDecision   Accepted / rejected outcome for one candidate
    PromotionReport     Full summary of a Promoter.promote() run

MTM
    SummaryBlock        An MTM summary block (id, text, tokens, embedding …)

LTM graph
    Edge                A knowledge-graph edge between two LTM nodes

Promotion
    PromotionScope      What the Promoter should sweep (session / since / all)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MemoryTier(str, Enum):
    """The storage layer a memory item belongs to."""

    STM = "stm"  # Short-term  — recent turns, session-local, in-process
    MTM = "mtm"  # Mid-term    — project context, session-persisted, vector-indexed
    LTM = "ltm"  # Long-term   — durable facts, bi-temporal, cross-session


class ProcessingState(str, Enum):
    """
    Lifecycle state of an MTM item with respect to LTM promotion.

    The Promoter reads UNPROCESSED items, evaluates them, then writes
    PROCESSED (or FAILED on error) so the same items are never re-evaluated.
    """

    UNPROCESSED = "unprocessed"
    PROCESSED = "processed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Bi-temporal support (LTM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiTemporalRange:
    """
    Tracks *when a fact was true in the world* (valid time), independently of
    when Continuum recorded it (transaction time = MemoryItem.created_at).

    ``valid_to=None`` means the fact is currently open-ended / still true.

    Example
    -------
    A user changed their preferred language from English to French on 2025-03-01.
    The old record has valid_to=2025-03-01; the new record has valid_from=2025-03-01
    and valid_to=None.  Both rows remain in storage for historical queries.
    """

    valid_from: datetime
    valid_to: datetime | None = None

    def is_valid_at(self, point: datetime) -> bool:
        """Return True if *point* falls within this validity window."""
        if point < self.valid_from:
            return False
        if self.valid_to is not None and point >= self.valid_to:
            return False
        return True

    def is_currently_valid(self) -> bool:
        """Shorthand for is_valid_at(now)."""
        return self.is_valid_at(datetime.now(UTC))


# ---------------------------------------------------------------------------
# Core memory record
# ---------------------------------------------------------------------------


@dataclass
class MemoryItem:
    """
    The canonical memory record shared across STM, MTM, and LTM.

    Design notes
    ------------
    * ``importance`` and ``confidence`` are normalised floats in [0, 1].
      Concrete STM code uses the four-level ``Importance`` enum
      (LOW=1 … CRITICAL=4); callers should map it as ``(value - 1) / 3``
      before populating this field so that the Scorer treats all tiers
      uniformly.

    * ``embedding`` is populated lazily by whichever component first needs
      the vector (usually the Retriever or the Persistence backend).

    * ``valid_range`` is only meaningful for LTM items; STM/MTM items leave
      it as ``None``.

    * ``related_ids`` stores the ids of neighbouring LTM nodes in the
      knowledge graph, enabling the LTMProtocol.neighbors() traversal.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    tier: MemoryTier = MemoryTier.STM

    # Scoring dimensions (all normalised 0–1)
    importance: float = 0.5
    confidence: float = 1.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Ownership / routing
    session_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None

    # Vector representation (None until embedded)
    embedding: list[float] | None = None

    # MTM lifecycle — drives Promoter work queue
    processing_state: ProcessingState = ProcessingState.UNPROCESSED

    # LTM bi-temporal validity
    valid_range: BiTemporalRange | None = None

    # LTM knowledge graph edges
    related_ids: list[str] = field(default_factory=list)

    # Access statistics — drive the Scorer's reinforcement-aware recency.
    # ``last_access`` falls back to ``created_at`` when never re-read.
    last_access: datetime | None = None
    access_count: int = 0

    # Arbitrary key-value annotations (role, source, tags, …)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval request
# ---------------------------------------------------------------------------


@dataclass
class Query:
    """
    A retrieval request issued by an agent or the Retriever.

    ``embedding`` is typically ``None`` when first constructed and filled in
    by the Retriever before fan-out to each tier — compute once, reuse everywhere.

    ``as_of`` enables point-in-time queries into LTM bi-temporal history:
    only facts whose BiTemporalRange contains *as_of* are returned.
    Pass ``None`` to query the current state (most common).
    """

    text: str

    # Routing / scoping
    session_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None

    # Filled lazily by the Retriever
    embedding: list[float] | None = None

    # Which tiers to fan out to
    top_k: int = 10
    tiers: list[MemoryTier] = field(
        default_factory=lambda: [MemoryTier.STM, MemoryTier.MTM, MemoryTier.LTM]
    )

    # Optional key-value filter pushed down to each tier's backend
    metadata_filter: dict[str, Any] = field(default_factory=dict)

    # Bi-temporal point-in-time for LTM; None → current
    as_of: datetime | None = None


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


@dataclass
class TokenBudget:
    """
    Token envelope for a single Retriever call.

    ``total`` is the model's full context window.  The three per-tier
    reserves are the *minimum* allocation for each layer; any slack
    (= context_available − memory_reserved) can be redistributed by the
    Optimizer.

    Invariant (not enforced here, validated by Optimizer):
        stm_reserved + mtm_reserved + ltm_reserved + response_reserved ≤ total
    """

    total: int  # full context window
    stm_reserved: int  # minimum tokens for STM items
    mtm_reserved: int  # minimum tokens for MTM items
    ltm_reserved: int  # minimum tokens for LTM items
    response_reserved: int  # tokens held back for the model's reply

    @property
    def context_available(self) -> int:
        """Tokens available for memory content (total minus response reserve)."""
        return max(self.total - self.response_reserved, 0)

    @property
    def memory_reserved(self) -> int:
        """Sum of the three per-tier reserves."""
        return self.stm_reserved + self.mtm_reserved + self.ltm_reserved

    @property
    def slack(self) -> int:
        """
        Unallocated tokens the Optimizer may freely redistribute across tiers
        when one tier returns fewer items than reserved for.
        """
        return max(self.context_available - self.memory_reserved, 0)


# ---------------------------------------------------------------------------
# Context bundle (output of Retriever / Optimizer)
# ---------------------------------------------------------------------------


@dataclass
class ContextBundle:
    """
    The assembled, token-budgeted context handed to the LLM.

    ``messages`` is a list of ``{"role": …, "content": …}`` dicts, ready
    for any OpenAI-compatible chat API.

    ``items`` is the unflattened source list for callers that need to
    inspect, re-rank, or log individual records after retrieval.

    ``tier_breakdown`` maps tier name → tokens consumed so callers can
    observe where the budget was spent:
        {"stm": 512, "mtm": 1024, "ltm": 256}
    """

    items: list[MemoryItem]
    messages: list[dict[str, str]]
    tokens_used: int
    budget: TokenBudget
    tier_breakdown: dict[str, int] = field(default_factory=dict)
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_remaining(self) -> int:
        """How much of the context window is still unused."""
        return max(self.budget.total - self.tokens_used, 0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Canonical scoring weights — change here to affect the whole framework.
#   composite = W_REL·relevance + W_IMP·importance + W_REC·recency + W_CONF·confidence
W_REL, W_IMP, W_REC, W_CONF = 0.45, 0.25, 0.20, 0.10


@dataclass(frozen=True)
class ScoreBreakdown:
    """
    Per-dimension scores for a single MemoryItem against a specific Query.
    All dimensions and the composite are in [0, 1].

    Canonical formula
    -----------------
        composite = 0.45·relevance + 0.25·importance + 0.20·recency + 0.10·confidence

    Keeping the breakdown separate from the scalar lets callers inspect which
    dimension drove retrieval (useful for debugging, re-ranking UI, and A/B
    testing alternate weight sets).
    """

    relevance: float  # cosine similarity or BM25 overlap with the query
    importance: float  # MemoryItem.importance (normalised 0–1)
    recency: float  # exponential decay over age; 1.0 = just created
    confidence: float  # MemoryItem.confidence (lower for inferred facts)
    composite: float  # weighted sum per the formula above

    @classmethod
    def compute(
        cls,
        relevance: float,
        importance: float,
        recency: float,
        confidence: float,
    ) -> ScoreBreakdown:
        """
        Compute the composite from four dimensions using the canonical weights.

        All inputs are assumed to be in [0, 1]; the composite is clamped to
        [0, 1] to absorb floating-point overshoot.
        """
        composite = W_REL * relevance + W_IMP * importance + W_REC * recency + W_CONF * confidence
        return cls(
            relevance=relevance,
            importance=importance,
            recency=recency,
            confidence=confidence,
            composite=min(max(composite, 0.0), 1.0),
        )


@dataclass
class ScoredItem:
    """A MemoryItem paired with its ScoreBreakdown for a specific query."""

    item: MemoryItem
    scores: ScoreBreakdown

    @property
    def score(self) -> float:
        """Shortcut to the composite score."""
        return self.scores.composite


# ---------------------------------------------------------------------------
# Promoter types
# ---------------------------------------------------------------------------


@dataclass
class PromotionCandidate:
    """
    An MTM item identified by the Promoter as a potential LTM fact.

    ``reason`` is a natural-language explanation produced by the LLM
    evaluator step inside PromoterProtocol.promote().

    ``suggested_importance`` is the Promoter's recommendation for the
    resulting LTM item's importance field; concrete implementations may
    override it based on domain policy.
    """

    item: MemoryItem
    reason: str
    suggested_importance: float  # 0–1


@dataclass
class PromotionDecision:
    """
    The final outcome for a single PromotionCandidate.

    When ``promoted=True``, ``new_id`` holds the id of the newly created LTM
    record.  When ``promoted=False``, ``rejection_reason`` explains why the
    Promoter decided the item should stay in MTM (or be discarded).
    """

    candidate: PromotionCandidate
    promoted: bool
    new_id: str | None = None  # set on success
    rejection_reason: str | None = None  # set on skip


@dataclass
class PromotionReport:
    """
    Summary of one PromoterProtocol.promote() invocation.

    ``scope`` identifies what was scanned — convention:
        "session:<session_id>"  — one conversation's MTM items
        "agent:<agent_id>"      — all sessions for one agent
        "global"                — all UNPROCESSED items across all agents

    ``elapsed_ms`` is informational; implementations should populate it for
    observability dashboards.
    """

    scope: str
    promoted_count: int
    skipped_count: int
    candidates: list[PromotionCandidate]
    decisions: list[PromotionDecision]
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# MTM summary block
# ---------------------------------------------------------------------------


@dataclass
class SummaryBlock:
    """
    A mid-term-memory summary block.

    MTM stores LLM-generated summaries / compressed episodes evicted from
    STM.  Each block carries its own pre-computed ``tokens`` count so the
    Retriever can pack ``recent()`` results into a budget without
    re-tokenising, and a ``processed`` flag mirroring the Promoter
    work-queue state (a block is *processed* once the Promoter has evaluated
    it for LTM promotion).

    Relationship to :class:`MemoryItem`
    -----------------------------------
    ``SummaryBlock`` is the ergonomic MTM-specific view; ``MemoryItem`` is
    the canonical cross-tier record persisted in ``memory_nodes``.  Use
    :meth:`to_memory_item` / :meth:`from_memory_item` to convert at the
    storage boundary (``PostgresMTM`` does this for you).
    """

    text: str
    tokens: int = 0
    id: UUID = field(default_factory=uuid4)
    embedding: list[float] | None = None
    session_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── conversions ──────────────────────────────────────────────────────

    def to_memory_item(self) -> MemoryItem:
        """Project this block onto the canonical ``MemoryItem`` record."""
        meta = dict(self.metadata)
        meta.setdefault("tokens", self.tokens)
        return MemoryItem(
            id=str(self.id),
            content=self.text,
            tier=MemoryTier.MTM,
            created_at=self.created_at,
            session_id=self.session_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            embedding=self.embedding,
            processing_state=(
                ProcessingState.PROCESSED if self.processed else ProcessingState.UNPROCESSED
            ),
            metadata=meta,
        )

    @classmethod
    def from_memory_item(
        cls,
        item: MemoryItem,
        *,
        tokens: int | None = None,
    ) -> SummaryBlock:
        """
        Build a ``SummaryBlock`` from a ``MemoryItem``.

        ``tokens`` falls back to ``item.metadata['tokens']`` then 0; callers
        that have a real tokenizer should pass an explicit count.
        """
        tok = tokens if tokens is not None else int(item.metadata.get("tokens", 0) or 0)
        try:
            block_id = UUID(str(item.id))
        except (ValueError, AttributeError):
            block_id = uuid4()
        return cls(
            text=item.content,
            tokens=tok,
            id=block_id,
            embedding=item.embedding,
            session_id=item.session_id,
            agent_id=item.agent_id,
            user_id=item.user_id,
            created_at=item.created_at,
            processed=item.processing_state is ProcessingState.PROCESSED,
            metadata=dict(item.metadata),
        )


# ---------------------------------------------------------------------------
# LTM knowledge-graph edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """
    A directed edge in the LTM knowledge graph (``memory_edges`` row).

    ``predicate`` is a SCREAMING_SNAKE_CASE relation (CALLS, SUPPORTS,
    SUPERSEDES, …).  ``weight`` is the edge's confidence/strength in [0, 1]
    (``None`` when unscored).  ``depth`` is the hop distance from the seed
    node in a :meth:`LTMProtocol.neighbors` traversal (1 = direct neighbour).

    ``target`` is the optionally-hydrated destination :class:`MemoryItem`,
    populated by ``PostgresLTM.neighbors`` so callers get edges *and* nodes
    in one round-trip; ``None`` if the caller asked for edges only.

    Bi-temporal: only edges (and target nodes) with
    ``invalidated_at IS NULL`` are ever returned.
    """

    source_id:  UUID
    target_id:  UUID
    id:         UUID = field(default_factory=uuid4)
    predicate:  str | None = None
    weight:     float | None = None
    depth:      int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    target:     MemoryItem | None = None


# ---------------------------------------------------------------------------
# Promotion scope
# ---------------------------------------------------------------------------

@dataclass
class PromotionScope:
    """
    Selects which MTM blocks a Promoter run should consider.

    * ``session_id`` set  → only that conversation's blocks.
    * ``since`` set       → only blocks created at/after this instant.
    * both ``None``       → every unprocessed block (global sweep).

    :meth:`parse` accepts the legacy string scope used by
    ``PromoterProtocol`` / ``ContinuumSession.checkpoint`` so the orchestrator
    is a drop-in for both call styles:

    * ``"session:<uuid>"`` → ``PromotionScope(session_id=<uuid>)``
    * ``"global"`` / ``""`` / ``None`` → ``PromotionScope()`` (all)
    * ``"agent:<id>"`` → ``PromotionScope()`` (agent scoping not modelled here)
    """

    session_id: UUID | None = None
    since: datetime | None = None

    @classmethod
    def parse(cls, scope: PromotionScope | str | None) -> PromotionScope:
        if scope is None:
            return cls()
        if isinstance(scope, PromotionScope):
            return scope
        text = str(scope).strip()
        if not text or text.lower() == "global":
            return cls()
        if text.lower().startswith("session:"):
            raw = text.split(":", 1)[1]
            try:
                return cls(session_id=UUID(raw))
            except (ValueError, AttributeError):
                return cls()
        # "agent:<id>" and anything else → global sweep.
        return cls()

    def as_str(self) -> str:
        """Render back to the protocol's string scope (for audit/report)."""
        if self.session_id is not None:
            return f"session:{self.session_id}"
        return "global"
