"""
continuum/policies/models.py
============================
The vocabulary of the Policy Engine — enums + Pydantic v2 data carriers.

These types are intentionally **storage-agnostic**: a ``MemoryCandidate`` is
what arrives from extraction; a ``MemoryHandlingPlan`` is what the engine
produces; a ``MemoryDecisionTrace`` is the auditable record of *why*. The
``StorageRouter`` is the only thing that knows how to project a plan into
concrete rows.

All weights / scores are normalised to ``[0, 1]``; all enums are string
enums so they serialise round-trip to JSONB cleanly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enums
# =============================================================================


class MemoryCandidateType(str, Enum):
    """What *kind* of memory the extractor thinks this candidate is."""

    FACT = "fact"
    USER_PREFERENCE = "user_preference"
    TASK = "task"
    DEADLINE = "deadline"
    DECISION = "decision"
    MEETING_EPISODE = "meeting_episode"
    PROCEDURE = "procedure"
    CODE_SYMBOL = "code_symbol"
    CODE_DECISION = "code_decision"
    DOCUMENT_KNOWLEDGE = "document_knowledge"
    CORRECTION = "correction"
    SENSITIVE_DATA = "sensitive_data"
    TEMPORARY_CONTEXT = "temporary_context"
    UNKNOWN = "unknown"


class MemoryAction(str, Enum):
    """What the engine decides to *do* with the candidate."""

    STORE = "store"
    IGNORE = "ignore"
    ASK_USER = "ask_user"
    UPDATE = "update"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    EXPIRE = "expire"
    COMPACT = "compact"
    DELETE = "delete"
    NOOP = "noop"


class MemoryScope(str, Enum):
    """Where the canonical record lives."""

    STM = "stm"
    MTM = "mtm"
    LTM = "ltm"
    TASK_STORE = "task_store"
    CODE_INDEX = "code_index"
    EPISODE_STORE = "episode_store"


class Urgency(str, Enum):
    HOT = "hot"
    SOON = "soon"
    NORMAL = "normal"
    ARCHIVAL = "archival"


class Volatility(str, Enum):
    TEMPORARY = "temporary"
    CHANGING = "changing"
    STABLE = "stable"
    HISTORICAL = "historical"


class Sensitivity(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


#: Order from least → most sensitive, used by conflict resolution to pick
#: the more restrictive value.
_SENSITIVITY_ORDER: dict[str, int] = {
    Sensitivity.PUBLIC.value: 0,
    Sensitivity.PRIVATE.value: 1,
    Sensitivity.CONFIDENTIAL.value: 2,
    Sensitivity.RESTRICTED.value: 3,
}


def more_sensitive(a: Sensitivity, b: Sensitivity) -> Sensitivity:
    """Return the more-restrictive of two sensitivities (ties → ``a``)."""
    return a if _SENSITIVITY_ORDER[a.value] >= _SENSITIVITY_ORDER[b.value] else b


class SourceAuthority(str, Enum):
    USER_EXPLICIT = "user_explicit"
    USER_IMPLICIT = "user_implicit"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
    CODEBASE = "codebase"
    CALENDAR = "calendar"
    EMAIL = "email"
    INFERRED = "inferred"
    LOW_CONFIDENCE = "low_confidence"


class RetrievalMode(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    GRAPH = "graph"
    HYBRID = "hybrid"
    PROCEDURAL = "procedural"
    TASK_AWARE = "task_aware"
    CODE_SYMBOL = "code_symbol"


class UpdateStrategy(str, Enum):
    APPEND_ONLY = "append_only"
    OVERWRITE = "overwrite"
    MERGE = "merge"
    VERSIONED = "versioned"
    SUPERSEDE = "supersede"
    IGNORE_DUPLICATE = "ignore_duplicate"


class ConflictStrategy(str, Enum):
    KEEP_BOTH = "keep_both"
    MARK_CONFLICT = "mark_conflict"
    SUPERSEDE_OLD = "supersede_old"
    ASK_USER = "ask_user"
    TRUST_NEWER = "trust_newer"
    TRUST_HIGHER_AUTHORITY = "trust_higher_authority"


class StorageProjection(str, Enum):
    POSTGRES_CANONICAL = "postgres_canonical"
    VECTOR_INDEX = "vector_index"
    GRAPH_INDEX = "graph_index"
    TEMPORAL_INDEX = "temporal_index"
    TASK_QUEUE = "task_queue"
    CODE_SYMBOL_INDEX = "code_symbol_index"
    RAW_EVIDENCE_STORE = "raw_evidence_store"


# =============================================================================
# Pydantic data carriers
# =============================================================================


class MemoryCandidate(BaseModel):
    """
    A memory candidate produced by extraction, ready for policy evaluation.

    The behavioural attributes (urgency, volatility, sensitivity,
    source_authority) are *hints* from extraction. Policies may override
    them in the returned plan.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    text: str
    candidate_type: MemoryCandidateType
    labels: list[str] = Field(default_factory=list)

    source_ref: str | None = None
    source_span: str | None = None
    session_id: UUID | None = None
    turn_id: UUID | None = None
    speaker: str | None = None

    entities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    urgency: Urgency = Urgency.NORMAL
    volatility: Volatility = Volatility.STABLE
    sensitivity: Sensitivity = Sensitivity.PRIVATE
    source_authority: SourceAuthority = SourceAuthority.INFERRED

    occurred_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetentionPolicy(BaseModel):
    """How long the row should live and what happens at expiry."""

    model_config = ConfigDict(extra="forbid")

    ttl_seconds: int | None = Field(default=None, ge=0)
    expire_at: datetime | None = None
    on_expiry: Literal["delete", "archive", "compact", "mark_expired"] = (
        "mark_expired"
    )
    retain_raw_evidence: bool = True


class RetrievalPolicy(BaseModel):
    """Which retrieval modes to enable and how to weight this row."""

    model_config = ConfigDict(extra="forbid")

    modes: list[RetrievalMode]
    boost_recent: bool = False
    boost_importance: bool = True
    require_exact_match: bool = False
    max_context_tokens: int | None = Field(default=None, ge=1)
    retrieval_priority: float = Field(default=0.5, ge=0.0, le=1.0)


class PrivacyPolicy(BaseModel):
    """Sensitivity-driven access + at-rest controls."""

    model_config = ConfigDict(extra="forbid")

    requires_user_approval: bool = False
    redact_before_storage: bool = False
    encrypt_at_rest: bool = False
    allow_cross_session_retrieval: bool = True
    allow_team_scope: bool = False
    allow_org_scope: bool = False


class UpdatePolicy(BaseModel):
    """How write-conflicts are handled."""

    model_config = ConfigDict(extra="forbid")

    strategy: UpdateStrategy
    conflict_strategy: ConflictStrategy
    allow_llm_merge: bool = True
    preserve_versions: bool = True


class CompactionPolicy(BaseModel):
    """Whether and how the engine may compact this row later."""

    model_config = ConfigDict(extra="forbid")

    allow_compaction: bool = True
    preserve_source_span: bool = True
    preserve_entities: bool = True
    max_summary_tokens: int | None = Field(default=None, ge=1)


class MemoryHandlingPlan(BaseModel):
    """
    A policy's verdict for one candidate.

    Plans are *declarative*: they don't touch storage. The ``StorageRouter``
    turns the plan into concrete writes; the ``trace`` records what
    happened either way.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: UUID
    action: MemoryAction
    target_scope: MemoryScope

    retention: RetentionPolicy
    retrieval: RetrievalPolicy
    privacy: PrivacyPolicy
    update: UpdatePolicy
    compaction: CompactionPolicy

    storage_projections: list[StorageProjection]

    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str
    policy_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryDecisionTrace(BaseModel):
    """Forensic record of one candidate's trip through the engine."""

    model_config = ConfigDict(extra="forbid")

    trace_id: UUID = Field(default_factory=uuid4)
    candidate_id: UUID
    candidate_text: str
    selected_action: MemoryAction
    selected_scope: MemoryScope
    applied_policies: list[str] = Field(default_factory=list)
    rejected_policies: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    final_plan: MemoryHandlingPlan
    created_at: datetime


class PolicyEvaluationContext(BaseModel):
    """
    Side-channel inputs every policy may consult.

    ``existing_neighbors`` is intentionally typed as ``list[Any]`` so we
    don't pull retrieval types into this module — the engine accepts
    whatever shape the caller provides (typically ``ScoredItem``).
    """

    model_config = ConfigDict(extra="forbid")

    now: datetime
    user_id: UUID | None = None
    session_id: UUID | None = None
    project_id: UUID | None = None

    existing_neighbors: list[Any] = Field(default_factory=list)
    current_task_context: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    # Enums
    "MemoryCandidateType",
    "MemoryAction",
    "MemoryScope",
    "Urgency",
    "Volatility",
    "Sensitivity",
    "SourceAuthority",
    "RetrievalMode",
    "UpdateStrategy",
    "ConflictStrategy",
    "StorageProjection",
    # Models
    "MemoryCandidate",
    "RetentionPolicy",
    "RetrievalPolicy",
    "PrivacyPolicy",
    "UpdatePolicy",
    "CompactionPolicy",
    "MemoryHandlingPlan",
    "MemoryDecisionTrace",
    "PolicyEvaluationContext",
    # Helpers
    "more_sensitive",
]
