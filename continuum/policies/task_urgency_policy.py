"""
continuum/policies/task_urgency_policy.py
=========================================
``TaskUrgencyPolicy`` — actionable tasks, reminders, deadlines.

Tasks belong in the **task store** (``MemoryScope.TASK_STORE``), not LTM:
they expire (deadline + grace period), they need task-aware retrieval, and
they're append-only (don't merge "remind me to call Alice" with another
reminder).
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


class TaskUrgencyPolicy:
    """Tasks and deadlines route to TASK_STORE with a TTL."""

    policy_id = "task_urgency_v1"
    version = "1.0.0"
    priority = 75

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type in (
            MemoryCandidateType.TASK,
            MemoryCandidateType.DEADLINE,
        )

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan:
        grace_hours = int(context.config.get("task_grace_period_hours", 24))
        # If the candidate carries a deadline, expire = deadline + grace;
        # otherwise default to 7 days from now.
        if candidate.valid_until is not None:
            expire_at = candidate.valid_until + timedelta(hours=grace_hours)
        else:
            expire_at = context.now + timedelta(days=7)
        ttl_seconds = max(0, int((expire_at - context.now).total_seconds()))

        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.TASK_STORE,
            retention=RetentionPolicy(
                ttl_seconds=ttl_seconds,
                expire_at=expire_at,
                on_expiry="compact",  # finished task → keep a digest
            ),
            retrieval=RetrievalPolicy(
                modes=[RetrievalMode.TASK_AWARE, RetrievalMode.TEMPORAL],
                boost_recent=True,
                retrieval_priority=0.85,
            ),
            privacy=PrivacyPolicy(allow_cross_session_retrieval=True),
            update=UpdatePolicy(
                strategy=UpdateStrategy.APPEND_ONLY,
                conflict_strategy=ConflictStrategy.KEEP_BOTH,
            ),
            compaction=CompactionPolicy(
                allow_compaction=True,
                preserve_entities=True,
            ),
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.TASK_QUEUE,
                StorageProjection.TEMPORAL_INDEX,
            ],
            priority=0.85,
            reason=(
                f"Task / deadline: TASK_STORE, expires {expire_at.isoformat()} "
                f"(deadline + {grace_hours}h grace)."
            ),
            policy_ids=[self.policy_id],
            metadata={"expire_at_iso": expire_at.isoformat()},
        )


__all__ = ["TaskUrgencyPolicy"]
