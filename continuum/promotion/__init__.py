"""
continuum.promotion
====================
MTM → LTM promotion strategies.

Exports
-------
Promoter                — full MTM→LTM orchestrator (PromoterProtocol).
PromotionRunReport      — added/updated/deleted/noop/errors/tokens/cost.
TriggerManager          — decides when promotion should run (auto + forced).
Mem0Promoter            — the Mem0 four-operation (ADD/UPDATE/DELETE/NOOP)
                          decision engine over candidate facts.
Decision                — (op, target_id, rationale, merged_text) + audit fields.
MEMORY_OP_SCHEMA        — the LLM function-calling tool schema.
make_postgres_audit_sink— default audit sink writing to memory_promotions.

litellm / psycopg3 are imported lazily; unit tests inject ``completion_fn``
and ``audit_sink`` and need neither.
"""
from __future__ import annotations

from continuum.promotion.mem0_promoter import (
    MEMORY_OP_SCHEMA,
    Decision,
    Mem0Promoter,
    make_postgres_audit_sink,
)
from continuum.promotion.promoter import Promoter, PromotionRunReport
from continuum.promotion.triggers import TriggerManager

__all__ = [
    "Promoter",
    "PromotionRunReport",
    "TriggerManager",
    "Mem0Promoter",
    "Decision",
    "MEMORY_OP_SCHEMA",
    "make_postgres_audit_sink",
]
