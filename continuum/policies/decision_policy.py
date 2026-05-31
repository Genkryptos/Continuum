"""
continuum/policies/decision_policy.py
=====================================
``DecisionPolicy`` — explicit choices ("we decided to use Postgres").

Decisions are high-authority and append-only: a new decision **supersedes**
the previous one but the old one is preserved (bi-temporal history matters
for "why did we pick this?"). Cross-session retrieval is on so the next
session inherits the choice.
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


class DecisionPolicy:
    """Decisions → LTM, append-only versioned, high authority."""

    policy_id = "decision_v1"
    version = "1.0.0"
    priority = 80

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type in (
            MemoryCandidateType.DECISION,
            MemoryCandidateType.CODE_DECISION,
        )

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan:
        preserve = bool(context.config.get("preserve_decision_versions", True))
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(
                ttl_seconds=None,  # indefinite
                retain_raw_evidence=True,
            ),
            retrieval=RetrievalPolicy(
                modes=[
                    RetrievalMode.SEMANTIC,
                    RetrievalMode.HYBRID,
                    RetrievalMode.TEMPORAL,
                ],
                boost_importance=True,
                retrieval_priority=0.95,
            ),
            privacy=PrivacyPolicy(allow_cross_session_retrieval=True),
            update=UpdatePolicy(
                strategy=UpdateStrategy.SUPERSEDE,
                conflict_strategy=ConflictStrategy.SUPERSEDE_OLD,
                preserve_versions=preserve,
                allow_llm_merge=False,
            ),
            compaction=CompactionPolicy(allow_compaction=False),
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
                StorageProjection.TEMPORAL_INDEX,
            ],
            priority=0.95,
            reason=(
                "Decision: high-authority, append-only via supersede; old "
                "versions preserved for forensic queries."
            ),
            policy_ids=[self.policy_id],
        )


__all__ = ["DecisionPolicy"]
