"""
continuum.policies
==================
Policy-Based Memory Lifecycle Engine.

The pipeline this package replaces / augments::

    OLD:   fact → ADD/UPDATE/DELETE/NOOP → LTM
    NEW:   MemoryCandidate → PolicyEngine → MemoryHandlingPlan → StorageRouter

Public API
----------
* :class:`MemoryPolicy` (Protocol) — implement to add a new policy.
* :class:`PolicyEngine` — orchestrates the policy cascade + conflict resolution.
* :class:`StorageRouter` — turns plans into traceable writes.
* :func:`default_policies` — the eight built-in policies, in registration order.
* :class:`TraceWriter` — persists ``MemoryDecisionTrace``s to Postgres.

The deeper imports (heavy psycopg / extraction) are lazy where it matters.
"""

from __future__ import annotations

from continuum.policies.base import MemoryPolicy
from continuum.policies.code_policy import CodeMemoryPolicy
from continuum.policies.conflict_policy import CorrectionPolicy
from continuum.policies.decision_policy import DecisionPolicy
from continuum.policies.default_policies import DefaultFactPolicy
from continuum.policies.engine import PolicyEngine
from continuum.policies.meeting_policy import MeetingPolicy
from continuum.policies.models import (
    CompactionPolicy,
    ConflictStrategy,
    MemoryAction,
    MemoryCandidate,
    MemoryCandidateType,
    MemoryDecisionTrace,
    MemoryHandlingPlan,
    MemoryScope,
    PolicyEvaluationContext,
    PrivacyPolicy,
    RetentionPolicy,
    RetrievalMode,
    RetrievalPolicy,
    Sensitivity,
    SourceAuthority,
    StorageProjection,
    UpdatePolicy,
    UpdateStrategy,
    Urgency,
    Volatility,
)
from continuum.policies.procedural_policy import ProceduralWorkflowPolicy
from continuum.policies.registry import default_policies
from continuum.policies.sensitivity_policy import SensitivityPolicy
from continuum.policies.storage_router import StorageRouter
from continuum.policies.task_urgency_policy import TaskUrgencyPolicy
from continuum.policies.trace import TraceWriter
from continuum.policies.user_preference_policy import UserPreferencePolicy

__all__ = [
    # Protocol + orchestration
    "MemoryPolicy",
    "PolicyEngine",
    "StorageRouter",
    "TraceWriter",
    "default_policies",
    # The 8 built-in policies
    "DefaultFactPolicy",
    "UserPreferencePolicy",
    "TaskUrgencyPolicy",
    "DecisionPolicy",
    "MeetingPolicy",
    "ProceduralWorkflowPolicy",
    "CodeMemoryPolicy",
    "SensitivityPolicy",
    "CorrectionPolicy",
    # Models / enums
    "MemoryCandidate",
    "MemoryHandlingPlan",
    "MemoryDecisionTrace",
    "PolicyEvaluationContext",
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
    "RetentionPolicy",
    "RetrievalPolicy",
    "PrivacyPolicy",
    "UpdatePolicy",
    "CompactionPolicy",
]
