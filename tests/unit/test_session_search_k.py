"""
tests/unit/test_session_search_k.py
===================================
`ContinuumSession.search` bounds on k. Found by adversarial testing: `k=-5`
returned results, because `items[:k]` with a negative k is a Python slice
meaning "all but the last five" rather than "none".
"""

from __future__ import annotations

from typing import Any

import pytest

from continuum.core.config import ContinuumConfig
from continuum.core.session import ContinuumSession
from continuum.core.types import MemoryItem, MemoryTier

pytestmark = pytest.mark.unit


class _STM:
    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items

    async def get_recent(self, session_id: str, n: int = 10) -> list[MemoryItem]:
        return self._items[:n]

    async def append(self, item: Any) -> None: ...


def _items(n: int) -> list[MemoryItem]:
    return [MemoryItem(content=f"fact {i}", tier=MemoryTier.STM) for i in range(n)]


@pytest.mark.parametrize("k", [-5, -1, 0])
async def test_non_positive_k_returns_nothing(k: int) -> None:
    s = ContinuumSession(ContinuumConfig(), stm=_STM(_items(7)))
    assert await s.search("anything", k=k) == []


async def test_positive_k_still_bounded() -> None:
    s = ContinuumSession(ContinuumConfig(), stm=_STM(_items(7)))
    assert len(await s.search("anything", k=3)) == 3
