"""
continuum/policies/meeting_policy.py
====================================
``MeetingPolicy`` — meeting episodes ("standup with Alice and Bob").

The raw episode goes to the **episode store** with a bounded TTL (the
transcript is heavyweight). The interesting *content* — decisions and
tasks the meeting produced — is extracted by other extractors and flows
through their own policies independently.
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


class MeetingPolicy:
    """Meeting episodes → EPISODE_STORE with bounded raw-transcript TTL."""

    policy_id = "meeting_v1"
    version = "1.0.0"
    priority = 60

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type == MemoryCandidateType.MEETING_EPISODE

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan:
        ttl_days = int(context.config.get("meeting_raw_transcript_ttl_days", 30))
        expire_at = context.now + timedelta(days=ttl_days)
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.EPISODE_STORE,
            retention=RetentionPolicy(
                ttl_seconds=int(timedelta(days=ttl_days).total_seconds()),
                expire_at=expire_at,
                on_expiry="compact",
                retain_raw_evidence=True,
            ),
            retrieval=RetrievalPolicy(
                modes=[RetrievalMode.TEMPORAL, RetrievalMode.SEMANTIC],
                boost_recent=True,
                retrieval_priority=0.55,
            ),
            privacy=PrivacyPolicy(
                allow_cross_session_retrieval=True,
                allow_team_scope=True,
            ),
            update=UpdatePolicy(
                strategy=UpdateStrategy.APPEND_ONLY,
                conflict_strategy=ConflictStrategy.KEEP_BOTH,
            ),
            compaction=CompactionPolicy(
                allow_compaction=True,
                preserve_source_span=True,
            ),
            storage_projections=[
                StorageProjection.RAW_EVIDENCE_STORE,
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.TEMPORAL_INDEX,
            ],
            priority=0.55,
            reason=(
                f"Meeting episode: raw transcript kept in evidence store, "
                f"compacted after {ttl_days} days. Decisions and tasks "
                f"extracted separately follow their own policies."
            ),
            policy_ids=[self.policy_id],
        )


__all__ = ["MeetingPolicy"]
