"""
continuum/policies/sensitivity_policy.py
========================================
``SensitivityPolicy`` — **highest priority**, always wins on conflict.

Detects sensitive payloads (API keys, tokens, secrets, restricted PII) and
forces:
* ``MemoryAction.ASK_USER`` for ``RESTRICTED`` (when approval is required),
* ``redact_before_storage`` + ``encrypt_at_rest`` for ``CONFIDENTIAL``,
* ``allow_cross_session_retrieval = False`` for anything above ``PRIVATE``.

The engine's conflict-resolver takes the *more restrictive* of any two
privacy / retention values, so this policy can be safely *merged* with
others — its restrictions always survive.
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
    Sensitivity,
    StorageProjection,
    UpdatePolicy,
    UpdateStrategy,
)


class SensitivityPolicy:
    """Safety override: redact, encrypt, and/or defer for approval."""

    policy_id = "sensitivity_v1"
    version = "1.0.0"
    priority = 100  # highest — never overridden

    def supports(self, candidate: MemoryCandidate) -> bool:
        return (
            candidate.candidate_type == MemoryCandidateType.SENSITIVE_DATA
            or candidate.sensitivity
            in (Sensitivity.CONFIDENTIAL, Sensitivity.RESTRICTED)
        )

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan:
        require_approval = bool(
            context.config.get("require_approval_for_restricted", True)
        )
        is_restricted = candidate.sensitivity == Sensitivity.RESTRICTED or (
            candidate.candidate_type == MemoryCandidateType.SENSITIVE_DATA
        )

        if is_restricted and require_approval:
            action = MemoryAction.ASK_USER
            reason = (
                "Restricted / sensitive payload detected — deferred to "
                "memory_pending_approvals for explicit user sign-off."
            )
        else:
            action = MemoryAction.STORE
            reason = (
                "Confidential content stored with redaction + at-rest "
                "encryption; cross-session retrieval disabled."
            )

        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=action,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(
                ttl_seconds=None,
                on_expiry="archive",
                retain_raw_evidence=False,
            ),
            retrieval=RetrievalPolicy(
                modes=[RetrievalMode.EXACT],
                require_exact_match=True,
                boost_importance=False,
                retrieval_priority=0.3,
            ),
            privacy=PrivacyPolicy(
                requires_user_approval=is_restricted,
                redact_before_storage=True,
                encrypt_at_rest=True,
                allow_cross_session_retrieval=False,
                allow_team_scope=False,
                allow_org_scope=False,
            ),
            update=UpdatePolicy(
                strategy=UpdateStrategy.IGNORE_DUPLICATE,
                conflict_strategy=ConflictStrategy.ASK_USER,
                allow_llm_merge=False,
                preserve_versions=False,
            ),
            compaction=CompactionPolicy(
                allow_compaction=False,
                preserve_source_span=False,
                preserve_entities=False,
            ),
            storage_projections=[StorageProjection.POSTGRES_CANONICAL],
            priority=1.0,
            reason=reason,
            policy_ids=[self.policy_id],
        )


__all__ = ["SensitivityPolicy"]
