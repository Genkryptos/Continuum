"""
continuum/policies/conflict_policy.py
=====================================
``CorrectionPolicy`` — explicit corrections ("actually it's X, not Y").

Corrections are treated as authoritative supersedes: the new statement
replaces the old one but bi-temporal history is preserved. The
``conflict_strategy`` is ``SUPERSEDE_OLD`` (with retention of versions)
because a correction explicitly invalidates the previous value.
"""

from __future__ import annotations

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


class CorrectionPolicy:
    """Corrections supersede prior values with versioned history."""

    policy_id = "correction_v1"
    version = "1.0.0"
    priority = 85

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type == MemoryCandidateType.CORRECTION

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan:
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.SUPERSEDE,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(
                ttl_seconds=None,
                retain_raw_evidence=True,
            ),
            retrieval=RetrievalPolicy(
                modes=[RetrievalMode.SEMANTIC, RetrievalMode.HYBRID],
                boost_recent=True,
                retrieval_priority=0.85,
            ),
            privacy=PrivacyPolicy(allow_cross_session_retrieval=True),
            update=UpdatePolicy(
                strategy=UpdateStrategy.SUPERSEDE,
                conflict_strategy=ConflictStrategy.SUPERSEDE_OLD,
                allow_llm_merge=False,
                preserve_versions=True,
            ),
            compaction=CompactionPolicy(allow_compaction=False),
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
                StorageProjection.TEMPORAL_INDEX,
            ],
            priority=0.9,
            reason=(
                "Correction: supersedes prior value, full version history "
                "retained for forensic queries."
            ),
            policy_ids=[self.policy_id],
        )


__all__ = ["CorrectionPolicy"]
