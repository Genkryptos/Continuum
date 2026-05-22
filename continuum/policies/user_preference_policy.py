"""
continuum/policies/user_preference_policy.py
============================================
``UserPreferencePolicy`` — explicit user choices ("I prefer X").

Preferences are stable and highly retrieval-relevant for personalisation,
so they go to LTM with high ``retrieval_priority`` and a ``VERSIONED``
update strategy. Conflicts default to ``KEEP_BOTH`` because preferences
are usually domain-scoped (a user can prefer Python *for backend* and Rust
*for systems work* — both are valid).
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


class UserPreferencePolicy:
    """Persistent, highly-retrievable; domain-aware versioning."""

    policy_id = "user_preference_v1"
    version = "1.0.0"
    priority = 70  # high — wins over plain facts

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type == MemoryCandidateType.USER_PREFERENCE

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan:
        keep_versions = bool(
            context.config.get("preserve_preference_versions", True)
        )
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(
                ttl_seconds=None,  # indefinite
                on_expiry="mark_expired",
            ),
            retrieval=RetrievalPolicy(
                modes=[RetrievalMode.SEMANTIC, RetrievalMode.EXACT],
                boost_importance=True,
                retrieval_priority=0.9,
            ),
            privacy=PrivacyPolicy(
                allow_cross_session_retrieval=True,
            ),
            update=UpdatePolicy(
                strategy=UpdateStrategy.VERSIONED,
                conflict_strategy=ConflictStrategy.KEEP_BOTH,
                preserve_versions=keep_versions,
            ),
            compaction=CompactionPolicy(allow_compaction=False),
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
            ],
            priority=0.9,
            reason=(
                "User preference: persistent, high retrieval priority, "
                "domain-scoped versions kept."
            ),
            policy_ids=[self.policy_id],
        )


__all__ = ["UserPreferencePolicy"]
