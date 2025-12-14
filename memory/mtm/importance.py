"""
Shared helpers for converting and normalizing importance across STM and MTM.

Importance levels are stored as integers based on the ConversationSTM enum:
LOW=1, NORMAL=2, HIGH=3, CRITICAL=4. These helpers clamp any external values
into that range and provide a normalized 0-1 score for retrieval weighting.
"""

from typing import Any

from memory.stm.ConversationSTM import Importance

IMPORTANCE_MIN = Importance.LOW.value
IMPORTANCE_MAX = Importance.CRITICAL.value
IMPORTANCE_SPAN = max(IMPORTANCE_MAX - IMPORTANCE_MIN, 1)


def normalize_importance_value(value: Any) -> int:
    """Return an integer importance clamped to the configured range."""

    try:
        raw = value.value if hasattr(value, "value") else int(value)
    except Exception:  # noqa: BLE001
        return Importance.NORMAL.value

    try:
        raw_int = int(raw)
    except Exception:  # noqa: BLE001
        return Importance.NORMAL.value

    return max(IMPORTANCE_MIN, min(IMPORTANCE_MAX, raw_int))


def importance_to_score(value: Any) -> float:
    """Normalize importance to a 0-1 score for ranking."""

    normalized_value = normalize_importance_value(value)
    return (normalized_value - IMPORTANCE_MIN) / IMPORTANCE_SPAN
