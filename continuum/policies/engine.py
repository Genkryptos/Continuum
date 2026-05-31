"""
continuum/policies/engine.py
============================
``PolicyEngine`` — the orchestrator.

For each ``MemoryCandidate`` the engine:

1. Finds the policies whose ``supports`` returns True (cheap filter).
2. Runs ``evaluate`` on each, collecting non-``None`` plans.
3. **Merges** the plans deterministically: the highest-priority plan is the
   base, and *more-restrictive* privacy / retention values from any other
   matching plan are layered on top (so the safety policies' restrictions
   always survive even when a higher-priority business policy fired).
4. Emits a :class:`MemoryDecisionTrace` recording which policies applied
   and why.

Conflict resolution rules (deterministic, documented)
-----------------------------------------------------
* Plans sorted by ``policy.priority`` desc → highest is the *base*.
* ``privacy``: choose the per-field maximum (more sensitive wins).
* ``retention``: take the *shorter* TTL between base and any other; if any
  policy requires approval, that survives.
* ``storage_projections``: the *union* (deduped, stable order) — any
  policy that wants the vector index gets it.
* ``action``: a ``SensitivityPolicy``-style ``ASK_USER`` is preserved
  whenever any matching policy returned it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from continuum.policies.base import MemoryPolicy
from continuum.policies.models import (
    MemoryAction,
    MemoryCandidate,
    MemoryDecisionTrace,
    MemoryHandlingPlan,
    PolicyEvaluationContext,
    PrivacyPolicy,
    RetentionPolicy,
    Sensitivity,
    StorageProjection,
    more_sensitive,
)

log = logging.getLogger(__name__)


class PolicyEngine:
    """Evaluate candidates against a stack of policies."""

    def __init__(self, policies: list[MemoryPolicy]) -> None:
        # Highest priority first — the *base* plan on conflict.
        self.policies: list[MemoryPolicy] = sorted(policies, key=lambda p: p.priority, reverse=True)

    # ── single-candidate path ───────────────────────────────────────────────

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> tuple[MemoryHandlingPlan, MemoryDecisionTrace]:
        """Run the policy cascade, resolve conflicts, emit a trace."""
        matching: list[MemoryPolicy] = [p for p in self.policies if p.supports(candidate)]
        plans: list[tuple[MemoryPolicy, MemoryHandlingPlan]] = []
        rejected: list[str] = []
        for policy in matching:
            try:
                plan = await policy.evaluate(candidate, context)
            except Exception:
                log.exception("policy %s.evaluate failed — skipping", policy.policy_id)
                rejected.append(policy.policy_id)
                continue
            if plan is None:
                rejected.append(policy.policy_id)
                continue
            plans.append((policy, plan))

        if not plans:
            # Defensive: DefaultFactPolicy should always match plain facts,
            # but for an unexpected type return an explicit IGNORE so the
            # candidate is *recorded* (via trace) but never written.
            final_plan = self._ignore_plan(candidate, "no policy matched")
            applied_ids: list[str] = []
            reasons = [final_plan.reason]
        else:
            final_plan = self._resolve(plans)
            applied_ids = [p.policy_id for p, _ in plans]
            reasons = [pl.reason for _, pl in plans]

        trace = MemoryDecisionTrace(
            trace_id=uuid4(),
            candidate_id=candidate.id,
            candidate_text=candidate.text,
            selected_action=final_plan.action,
            selected_scope=final_plan.target_scope,
            applied_policies=applied_ids,
            rejected_policies=rejected,
            reasons=reasons,
            final_plan=final_plan,
            created_at=context.now,
        )
        return final_plan, trace

    # ── batch path ──────────────────────────────────────────────────────────

    async def evaluate_many(
        self,
        candidates: list[MemoryCandidate],
        context: PolicyEvaluationContext,
    ) -> list[tuple[MemoryCandidate, MemoryHandlingPlan, MemoryDecisionTrace]]:
        results: list[tuple[MemoryCandidate, MemoryHandlingPlan, MemoryDecisionTrace]] = []
        for c in candidates:
            plan, trace = await self.evaluate(c, context)
            results.append((c, plan, trace))
        return results

    # ── conflict resolution ─────────────────────────────────────────────────

    def _resolve(
        self,
        plans: list[tuple[MemoryPolicy, MemoryHandlingPlan]],
    ) -> MemoryHandlingPlan:
        """
        Deterministic merge.

        The plan from the highest-priority policy is the base; restrictions
        from every other matching plan are layered on. See the module
        docstring for the per-field rules.
        """
        plans = sorted(plans, key=lambda p: p[0].priority, reverse=True)
        _base_policy, base_plan = plans[0]
        privacy = base_plan.privacy
        retention = base_plan.retention
        projections: list[StorageProjection] = list(base_plan.storage_projections)
        action = base_plan.action
        policy_ids: list[str] = list(base_plan.policy_ids)
        # Highest matching sensitivity drives the *displayed* sensitivity.
        # (We don't change the model field on the candidate here; the
        # privacy block carries the operational restrictions.)
        derived_sensitivity = Sensitivity.PRIVATE

        for policy, plan in plans[1:]:
            privacy = self._strictest_privacy(privacy, plan.privacy)
            retention = self._tightest_retention(retention, plan.retention)
            for proj in plan.storage_projections:
                if proj not in projections:
                    projections.append(proj)
            # If any matching policy demanded ASK_USER, that survives.
            if plan.action == MemoryAction.ASK_USER:
                action = MemoryAction.ASK_USER
            policy_ids.append(policy.policy_id)
            derived_sensitivity = more_sensitive(
                derived_sensitivity, _infer_sensitivity(plan.privacy)
            )

        merged = base_plan.model_copy(
            update={
                "action": action,
                "privacy": privacy,
                "retention": retention,
                "storage_projections": projections,
                "policy_ids": policy_ids,
            }
        )
        merged.metadata.setdefault("derived_sensitivity", derived_sensitivity.value)
        return merged

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _strictest_privacy(a: PrivacyPolicy, b: PrivacyPolicy) -> PrivacyPolicy:
        """Per-field maximum on the *restrictive* axis."""
        return PrivacyPolicy(
            requires_user_approval=(a.requires_user_approval or b.requires_user_approval),
            redact_before_storage=(a.redact_before_storage or b.redact_before_storage),
            encrypt_at_rest=a.encrypt_at_rest or b.encrypt_at_rest,
            allow_cross_session_retrieval=(
                a.allow_cross_session_retrieval and b.allow_cross_session_retrieval
            ),
            allow_team_scope=a.allow_team_scope and b.allow_team_scope,
            allow_org_scope=a.allow_org_scope and b.allow_org_scope,
        )

    @staticmethod
    def _tightest_retention(a: RetentionPolicy, b: RetentionPolicy) -> RetentionPolicy:
        """Shorter TTL wins; earlier expire_at wins."""

        def _min_or_none(x: int | None, y: int | None) -> int | None:
            if x is None:
                return y
            if y is None:
                return x
            return min(x, y)

        def _earliest(a_t: datetime | None, b_t: datetime | None) -> datetime | None:
            if a_t is None:
                return b_t
            if b_t is None:
                return a_t
            return min(a_t, b_t)

        return RetentionPolicy(
            ttl_seconds=_min_or_none(a.ttl_seconds, b.ttl_seconds),
            expire_at=_earliest(a.expire_at, b.expire_at),
            on_expiry=a.on_expiry,
            retain_raw_evidence=(a.retain_raw_evidence and b.retain_raw_evidence),
        )

    @staticmethod
    def _ignore_plan(candidate: MemoryCandidate, reason: str) -> MemoryHandlingPlan:
        """Build an explicit IGNORE plan for un-policied candidates."""
        from continuum.policies.models import (
            CompactionPolicy,
            ConflictStrategy,
            MemoryScope,
            RetrievalMode,
            RetrievalPolicy,
            UpdatePolicy,
            UpdateStrategy,
        )

        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.IGNORE,
            target_scope=MemoryScope.LTM,
            retention=RetentionPolicy(),
            retrieval=RetrievalPolicy(modes=[RetrievalMode.SEMANTIC]),
            privacy=PrivacyPolicy(),
            update=UpdatePolicy(
                strategy=UpdateStrategy.IGNORE_DUPLICATE,
                conflict_strategy=ConflictStrategy.KEEP_BOTH,
            ),
            compaction=CompactionPolicy(),
            storage_projections=[],
            priority=0.0,
            reason=reason,
            policy_ids=[],
        )


def _infer_sensitivity(privacy: PrivacyPolicy) -> Sensitivity:
    if privacy.encrypt_at_rest or privacy.requires_user_approval:
        return Sensitivity.RESTRICTED
    if privacy.redact_before_storage:
        return Sensitivity.CONFIDENTIAL
    if not privacy.allow_cross_session_retrieval:
        return Sensitivity.CONFIDENTIAL
    return Sensitivity.PRIVATE


__all__ = ["PolicyEngine"]
