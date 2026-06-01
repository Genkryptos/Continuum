"""
continuum/policies/registry.py
==============================
Registration helpers for the Policy Engine.

The canonical default set is wired in :func:`default_policies`. Production
callers can subclass / replace any individual policy and pass their own
list to :class:`PolicyEngine`.
"""

from __future__ import annotations

from continuum.policies.base import MemoryPolicy
from continuum.policies.code_policy import CodeMemoryPolicy
from continuum.policies.conflict_policy import CorrectionPolicy
from continuum.policies.decision_policy import DecisionPolicy
from continuum.policies.default_policies import DefaultFactPolicy
from continuum.policies.meeting_policy import MeetingPolicy
from continuum.policies.procedural_policy import ProceduralWorkflowPolicy
from continuum.policies.sensitivity_policy import SensitivityPolicy
from continuum.policies.task_urgency_policy import TaskUrgencyPolicy
from continuum.policies.user_preference_policy import UserPreferencePolicy


def default_policies() -> list[MemoryPolicy]:
    """
    Return the 8 built-in policies in registration order.

    Order here is informational only — :class:`PolicyEngine` sorts by
    ``priority`` descending internally.
    """
    return [
        SensitivityPolicy(),  # 100 — safety override
        CorrectionPolicy(),  # 85
        DecisionPolicy(),  # 80
        TaskUrgencyPolicy(),  # 75
        UserPreferencePolicy(),  # 70
        ProceduralWorkflowPolicy(),  # 65
        CodeMemoryPolicy(),  # 65
        MeetingPolicy(),  # 60
        DefaultFactPolicy(),  # 10 — fall-through
    ]


__all__ = ["default_policies"]
