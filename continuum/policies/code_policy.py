"""
continuum/policies/code_policy.py
=================================
``CodeMemoryPolicy`` — code symbols, files, code decisions.

Code memory needs *both* a code-symbol index (precise file/symbol lookup)
and the canonical LTM (semantic recall of "what did we say about this
function"). Versioning is on so we can answer "what did this look like at
commit X".
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


class CodeMemoryPolicy:
    """Code symbols → CODE_INDEX + LTM; versioned per commit."""

    policy_id = "code_v1"
    version = "1.0.0"
    priority = 65

    def supports(self, candidate: MemoryCandidate) -> bool:
        return candidate.candidate_type in (
            MemoryCandidateType.CODE_SYMBOL,
            MemoryCandidateType.CODE_DECISION,
        )

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan | None:
        if not context.config.get("enable_code_policy", True):
            return None
        return MemoryHandlingPlan(
            candidate_id=candidate.id,
            action=MemoryAction.STORE,
            target_scope=MemoryScope.CODE_INDEX,
            retention=RetentionPolicy(
                ttl_seconds=None,
                retain_raw_evidence=True,
            ),
            retrieval=RetrievalPolicy(
                modes=[
                    RetrievalMode.CODE_SYMBOL,
                    RetrievalMode.EXACT,
                    RetrievalMode.SEMANTIC,
                ],
                require_exact_match=False,
                boost_importance=True,
                retrieval_priority=0.8,
            ),
            privacy=PrivacyPolicy(allow_cross_session_retrieval=True),
            update=UpdatePolicy(
                strategy=UpdateStrategy.VERSIONED,
                conflict_strategy=ConflictStrategy.TRUST_NEWER,
                preserve_versions=True,
            ),
            compaction=CompactionPolicy(allow_compaction=False),
            storage_projections=[
                StorageProjection.CODE_SYMBOL_INDEX,
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
            ],
            priority=0.8,
            reason=(
                "Code memory: dual-indexed (symbol + canonical), versioned "
                "per commit, exact and semantic retrieval enabled."
            ),
            policy_ids=[self.policy_id],
        )


__all__ = ["CodeMemoryPolicy"]
