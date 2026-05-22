"""
continuum/policies/default_policies.py
======================================
``DefaultFactPolicy`` — the catch-all for plain facts.

It is the lowest-priority policy (priority 10) so any specialised policy
(preferences, decisions, tasks, secrets, …) wins. When *nothing* else
applies the engine still produces a sensible plan: store the fact in LTM,
hybrid retrieval, versioned updates so contradictions are caught later.
"""

from __future__ import annotations

from datetime import timedelta

from continuum.policies.models import (
    CompactionPolicy,
    ConflictStrategy,
    MemoryAction,
    MemoryCandidate,
    MemoryCandidateType,
    MemoryHandlingPlan,
    MemoryScope,
    PolicyEvaluationContext,
    PrivacyPolicy,
    RetentionPolicy,
    RetrievalMode,
    RetrievalPolicy,
    StorageProjection,
    UpdatePolicy,
    UpdateStrategy,
)


class DefaultFactPolicy:
    """Fall-through: plain ``FACT`` candidates go to LTM, versioned."""

    policy_id = "default_fact_v1"
    version = "1.0.0"
    priority = 10  # lowest — any specialised policy wins

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type in (
            MemoryCandidateType.FACT,
            MemoryCandidateType.DOCUMENT_KNOWLEDGE,
            MemoryCandidateType.UNKNOWN,
        )

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan | None:
        ttl_days = context.config.get("default_fact_ttl_days")
        ttl_seconds = (
            int(timedelta(days=int(ttl_days)).total_seconds())
            if ttl_days is not None
            else None
        )
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(
                ttl_seconds=ttl_seconds,
                on_expiry="mark_expired",
            ),
            retrieval=RetrievalPolicy(
                modes=[RetrievalMode.HYBRID, RetrievalMode.SEMANTIC],
                boost_importance=True,
                retrieval_priority=0.5,
            ),
            privacy=PrivacyPolicy(allow_cross_session_retrieval=True),
            update=UpdatePolicy(
                strategy=UpdateStrategy.VERSIONED,
                conflict_strategy=ConflictStrategy.MARK_CONFLICT,
            ),
            compaction=CompactionPolicy(allow_compaction=True),
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
            ],
            priority=0.4,
            reason="Plain fact → LTM, versioned with conflict marking.",
            policy_ids=[self.policy_id],
        )


__all__ = ["DefaultFactPolicy"]
