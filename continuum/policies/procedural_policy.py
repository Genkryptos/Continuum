"""
continuum/policies/procedural_policy.py
=======================================
``ProceduralWorkflowPolicy`` — recipes / workflows ("whenever X, do Y").

Procedural memory is stable, frequently re-queried for "how do I…" tasks,
and benefits from a dedicated retrieval mode (``PROCEDURAL``) so retrieval
can match by *intent* rather than by surface text.
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


class ProceduralWorkflowPolicy:
    """Procedures → LTM with PROCEDURAL retrieval; merge on update."""

    policy_id = "procedural_v1"
    version = "1.0.0"
    priority = 65

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type == MemoryCandidateType.PROCEDURE

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan | None:
        if not context.config.get("enable_procedural_policy", True):
            return None
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(ttl_seconds=None),
            retrieval=RetrievalPolicy(
                modes=[
                    RetrievalMode.PROCEDURAL,
                    RetrievalMode.SEMANTIC,
                    RetrievalMode.HYBRID,
                ],
                boost_importance=True,
                retrieval_priority=0.8,
            ),
            privacy=PrivacyPolicy(allow_cross_session_retrieval=True),
            update=UpdatePolicy(
                strategy=UpdateStrategy.MERGE,
                conflict_strategy=ConflictStrategy.TRUST_NEWER,
                allow_llm_merge=True,
                preserve_versions=True,
            ),
            compaction=CompactionPolicy(
                allow_compaction=True,
                preserve_entities=True,
            ),
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
            ],
            priority=0.8,
            reason=(
                "Procedure / workflow: stable, retrieved via PROCEDURAL "
                "intent matching, merged on update."
            ),
            policy_ids=[self.policy_id],
        )


__all__ = ["ProceduralWorkflowPolicy"]
