"""
continuum.promotion
====================
MTM → LTM promotion strategies.

Exports
-------
Promoter                — full MTM→LTM orchestrator (PromoterProtocol).
PromotionRunReport      — added/updated/deleted/noop/errors/tokens/cost.
TriggerManager          — decides when promotion should run (auto + forced).
ContinuumPromoter       — the four-operation (ADD/UPDATE/DELETE/NOOP)
                          decision engine over candidate facts.
Decision                — (op, target_id, rationale, merged_text) + audit fields.
MEMORY_OP_SCHEMA        — the LLM function-calling tool schema.
make_postgres_audit_sink— default audit sink writing to memory_promotions.

litellm / psycopg3 are imported lazily; unit tests inject ``completion_fn``
and ``audit_sink`` and need neither.
"""

from __future__ import annotations

from continuum.promotion.continuum_promoter import (
    MEMORY_OP_SCHEMA,
    ContinuumPromoter,
    Decision,
    make_postgres_audit_sink,
)
from continuum.promotion.promoter import Promoter, PromotionRunReport
from continuum.promotion.triggers import TriggerManager

__all__ = [
    "Promoter",
    "PromotionRunReport",
    "TriggerManager",
    "ContinuumPromoter",
    "Decision",
    "MEMORY_OP_SCHEMA",
    "make_postgres_audit_sink",
]


def __getattr__(name: str) -> object:
    # The pre-2.2 name, kept so code written against the published 2.x API
    # still imports. Remove in 3.0.
    if name == "Mem0Promoter":
        import warnings

        warnings.warn(
            "continuum.promotion.Mem0Promoter is renamed ContinuumPromoter; "
            "the old name will be removed in 3.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return ContinuumPromoter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
