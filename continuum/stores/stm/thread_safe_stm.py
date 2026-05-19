"""
continuum/stores/stm/thread_safe_stm.py
========================================
``AsyncSafeSTM`` — an ``asyncio.Lock`` guard around any ``STMProtocol``.

Why you need this
-----------------
``InMemorySTM`` is safe for *concurrent asyncio tasks* because it holds its
own internal lock around engine access.  You need ``AsyncSafeSTM`` only when:

* Multiple independent coroutines share a *single* STMProtocol instance that
  is **not** internally lock-guarded (e.g. a custom implementation or
  ``PostgresSTM`` where the underlying DB may not support concurrent writes
  from one connection).

* You want a single, observable lock boundary for profiling or testing.

Usage
-----
    from continuum.stores.stm import AsyncSafeSTM, InMemorySTM

    inner = InMemorySTM(config, tokenizer)
    stm   = AsyncSafeSTM(inner)

    # Both coroutines are now serialised through one asyncio.Lock:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(stm.append(item_a))
        tg.create_task(stm.append(item_b))

Migration from ThreadSafeSTM
------------------------------
Replace::

    from memory.stm.ThreadSafeSTM import ThreadSafeSTM
    stm = ThreadSafeSTM(base_stm)

With::

    from continuum.stores.stm import AsyncSafeSTM, InMemorySTM
    stm = AsyncSafeSTM(InMemorySTM(config, tokenizer))

The async surface replaces ``add_user_message`` / ``get_messages`` with
``append`` / ``window``.  Keep the legacy sync path alive by calling
``stm.inner.add_user_message(...)`` during the migration window.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from continuum.core.protocols import MTMProtocol, STMProtocol
from continuum.core.types import MemoryItem


class AsyncSafeSTM:
    """
    Serialise all ``STMProtocol`` calls through a single ``asyncio.Lock``.

    Parameters
    ----------
    inner:
        The wrapped ``STMProtocol`` implementation.  It is accessible via
        ``self.inner`` for callers that need to reach the sync backward-compat
        API during migration.
    """

    def __init__(self, inner: STMProtocol) -> None:
        self.inner = inner
        self._lock = asyncio.Lock()

    # ── STMProtocol — all six methods ─────────────────────────────────────

    async def append(self, item: MemoryItem) -> None:
        """Serialised ``append`` — acquires the lock then delegates."""
        async with self._lock:
            await self.inner.append(item)

    async def window(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> Sequence[MemoryItem]:
        """Serialised ``window``."""
        async with self._lock:
            return await self.inner.window(session_id, max_tokens)

    async def get_recent(
        self,
        session_id: str,
        n: int = 10,
    ) -> Sequence[MemoryItem]:
        """Serialised ``get_recent``."""
        async with self._lock:
            return await self.inner.get_recent(session_id, n)

    async def flush_to(
        self,
        session_id: str,
        target: MTMProtocol,
    ) -> int:
        """Serialised ``flush_to``."""
        async with self._lock:
            return await self.inner.flush_to(session_id, target)

    async def clear(self, session_id: str) -> None:
        """Serialised ``clear``."""
        async with self._lock:
            await self.inner.clear(session_id)

    async def stats(self, session_id: str) -> dict[str, Any]:
        """Serialised ``stats``."""
        async with self._lock:
            return await self.inner.stats(session_id)

    # ── Introspection ──────────────────────────────────────────────────────

    def is_locked(self) -> bool:
        """Return ``True`` if the lock is currently held by a coroutine."""
        return self._lock.locked()
