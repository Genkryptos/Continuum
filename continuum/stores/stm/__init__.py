"""
continuum.stores.stm
====================
Concrete STMProtocol implementations.

Exports
-------
InMemorySTM   — multi-session in-memory STM backed by ConversationSTM engines.
                Ideal for single-process agents and unit tests.
AsyncSafeSTM  — asyncio.Lock wrapper for any STMProtocol; guards concurrent
                coroutines sharing one STM instance.
PostgresSTM   — psycopg3 async STM persisting messages to PostgreSQL.
                Suitable for distributed agents and multi-process deployments.

Conversion helpers (re-exported for callers that bridge the old Message API)
-------
message_to_item  — memory.stm.Message  → continuum.core.types.MemoryItem
item_to_message  — MemoryItem → memory.stm.Message
"""

from __future__ import annotations

from continuum.stores.stm.conversation_stm import (
    InMemorySTM,
    item_to_message,
    message_to_item,
)
from continuum.stores.stm.thread_safe_stm import AsyncSafeSTM

__all__ = [
    "InMemorySTM",
    "AsyncSafeSTM",
    "message_to_item",
    "item_to_message",
    # PostgresSTM is imported lazily to avoid hard-dep on psycopg3 at import time.
    # Use: from continuum.stores.stm.postgres_stm import PostgresSTM
]
