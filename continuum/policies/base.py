"""
continuum/policies/base.py
==========================
The ``MemoryPolicy`` runtime Protocol.

Policies are async, declarative, and storage-blind: they read the
candidate + context and return a ``MemoryHandlingPlan`` (or ``None`` if the
policy doesn't apply). The :class:`~continuum.policies.engine.PolicyEngine`
runs every matching policy and merges the plans deterministically.

Contract
--------
* ``policy_id`` is stable; new versions bump ``version`` and (often)
  ``policy_id`` (e.g. ``user_preference_v2``) so audit traces remain
  unambiguous.
* ``priority`` orders policies on conflict — higher wins (see
  :meth:`PolicyEngine._resolve_plans`). Sensitivity-style policies should
  use a high number (≥ 80) so safety overrides convenience.
* ``supports`` is a cheap, synchronous *gate* — it must not touch I/O.
* ``evaluate`` is async (a policy may consult the LLM or LTM) and **must
  not** write to storage; it only returns a plan.
* Plans always include a human-readable ``reason`` — the audit-trail story
  for why this policy fired.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from continuum.policies.models import (
    MemoryCandidate,
    MemoryHandlingPlan,
    PolicyEvaluationContext,
)


@runtime_checkable
class MemoryPolicy(Protocol):
    """Structural Protocol every concrete policy implements."""

    #: Stable identifier (e.g. ``"user_preference_v1"``). Surfaces in the
    #: ``policy_ids`` array of every audit row this policy contributes to.
    policy_id: str
    #: Semver-ish; bump when behaviour materially changes.
    version: str
    #: Higher → stronger. Sensitivity policies should sit at the top.
    priority: int

    def supports(self, candidate: MemoryCandidate) -> bool:
        """Cheap synchronous gate — does this policy apply at all?"""
        ...

    async def evaluate(
        self,
        candidate: MemoryCandidate,
        context: PolicyEvaluationContext,
    ) -> MemoryHandlingPlan | None:
        """Produce a plan, or ``None`` to abstain."""
        ...


__all__ = ["MemoryPolicy"]
