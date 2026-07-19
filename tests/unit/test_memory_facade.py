"""
tests/unit/test_memory_facade.py
================================
The public `continuum.Memory` facade. Hermetic — a fake ContinuumSession
(fake stm/ltm/search) so the facade's delegation, supersession filtering, and
timeline ordering are validated without a DB, embedder, or LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from continuum import Memory, MemoryItem, MemoryTier
from continuum.core.types import BiTemporalRange

pytestmark = pytest.mark.unit


class _FakeSTM:
    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    async def append(self, item: MemoryItem) -> None:
        self.items.append(item)


class _FakeLTM:
    def __init__(self) -> None:
        self.upserts: list[MemoryItem] = []

    async def upsert(self, item: MemoryItem) -> Any:
        self.upserts.append(item)
        return item.id


class _FakeSession:
    """Minimal ContinuumSession stand-in for the facade."""

    def __init__(self, search_result: list[MemoryItem] | None = None) -> None:
        self.stm = _FakeSTM()
        self.ltm = _FakeLTM()
        self.session_id = "default"
        self._search_result = search_result or []

    async def search(self, query: str, k: int = 10) -> list[MemoryItem]:
        return list(self._search_result[:k])


def _item(
    content: str, *, valid_from: datetime | None = None, valid_to: datetime | None = None
) -> MemoryItem:
    vr = None
    if valid_from is not None or valid_to is not None:
        vr = BiTemporalRange(valid_from=valid_from or datetime.now(UTC), valid_to=valid_to)
    return MemoryItem(content=content, tier=MemoryTier.LTM, valid_range=vr)


# ── construction + exports ────────────────────────────────────────────────────


def test_public_exports() -> None:
    import continuum

    assert hasattr(continuum, "Memory")
    assert {"Memory", "MemoryItem", "MemoryTier", "Query"} <= set(continuum.__all__)


def test_in_memory_factory_constructs() -> None:
    mem = Memory.in_memory()
    assert mem.session is not None
    assert mem.session.ltm is not None  # in-memory LTM attached


def test_is_durable_distinguishes_the_backends() -> None:
    # Callers report success to humans; "stored" is a lie when the store
    # evaporates at exit, and that failure is otherwise invisible.
    assert Memory.in_memory().is_durable is False
    pytest.importorskip("psycopg", reason="psycopg required for the Postgres backend")
    mem = Memory.from_postgres("postgresql://u:p@localhost:5432/none", embeddings=False)
    assert mem.is_durable is True


async def test_is_durable_false_without_long_term_storage() -> None:
    s = _FakeSession()
    s.ltm = None
    assert Memory(s).is_durable is False  # type: ignore[arg-type]


def test_from_postgres_wires_hybrid_retriever() -> None:
    # Construction only — no .start(), so no DB connection / no model download
    # (embeddings=False keeps it dependency-light). Proves the production stack
    # gets a real retriever (dense/sparse), unlike in_memory() which has none.
    pytest.importorskip("psycopg", reason="psycopg required for the Postgres backend")
    from continuum.retrieval.embedding_query import EmbeddingQueryRetriever

    mem = Memory.from_postgres(
        "postgresql://u:p@localhost:5432/none", embeddings=False, session_id="s1"
    )
    assert mem.session.retriever is not None
    assert isinstance(mem.session.retriever, EmbeddingQueryRetriever)
    assert mem.session.session_id == "s1"


# ── write path ────────────────────────────────────────────────────────────────


async def test_add_writes_stm_and_ltm() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("I moved to Boston")
    assert [i.content for i in s.stm.items] == ["I moved to Boston"]
    assert [i.content for i in s.ltm.upserts] == ["I moved to Boston"]
    assert s.ltm.upserts[0].tier is MemoryTier.LTM


async def test_remember_is_add() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.remember("favorite rice is Japanese short-grain")
    assert len(s.stm.items) == 1


# ── recall ────────────────────────────────────────────────────────────────────


async def test_recall_delegates_to_search() -> None:
    hits = [_item("Boston"), _item("NYC")]
    mem = Memory(_FakeSession(hits))  # type: ignore[arg-type]
    out = await mem.recall("where do I live?", k=2)
    assert [h.content for h in out] == ["Boston", "NYC"]


# ── current() — supersession-resolved ─────────────────────────────────────────


async def test_current_prefers_live_over_superseded() -> None:
    old = _item(
        "Boston",
        valid_from=datetime(2023, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 6, 1, tzinfo=UTC),
    )
    new = _item("NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))  # open (current)
    mem = Memory(_FakeSession([old, new]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "NYC"


class _TagLTM(_FakeLTM):
    """LTM that supports the exact attribute lookup (like PostgresLTM)."""

    def __init__(self, rows: list[MemoryItem] | None = None) -> None:
        super().__init__()
        self.rows = rows or []
        self.calls: list[tuple[dict[str, Any], Any]] = []

    async def by_tags(
        self, tags: dict[str, Any], *, key: str | None = None, as_of: Any = None, **_kw: Any
    ) -> list[MemoryItem]:
        self.calls.append((tags, as_of))
        rows = self.rows
        if key is not None:  # "does any row carry this tag key at all?"
            rows = [r for r in rows if key in (r.metadata or {})]
        if tags:
            want = tags.get("attribute")
            rows = [r for r in rows if (r.metadata or {}).get("attribute") == want]
        return rows


async def test_add_tags_the_attribute() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("I moved to NYC", attribute="residence")
    assert s.ltm.upserts[0].metadata.get("attribute") == "residence"


async def test_current_prefers_exact_attribute_lookup() -> None:
    # The exact lookup must win over fuzzy retrieval — and over a NEWER
    # unrelated fact, which is what broke the retrieval-only version.
    old = _item("Boston", valid_from=datetime(2026, 1, 10, tzinfo=UTC))
    new = _item("New York City", valid_from=datetime(2026, 6, 15, tzinfo=UTC))
    for it in (old, new):
        it.metadata["attribute"] = "residence"
    s = _FakeSession([_item("bought a laptop", valid_from=datetime(2026, 7, 20, tzinfo=UTC))])
    s.ltm = _TagLTM([old, new])
    mem = Memory(s)  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "New York City"
    assert s.ltm.calls[0][0] == {"attribute": "residence"}  # exact filter used


async def test_current_returns_none_rather_than_guessing() -> None:
    # The corpus DOES use attribute tags, just not this attribute. That makes the
    # store authoritative — including "no such fact". Falling back to retrieval
    # here would invent an employer out of an unrelated memory.
    tagged = _item("Boston", valid_from=datetime(2026, 1, 1, tzinfo=UTC))
    tagged.metadata["attribute"] = "residence"
    s = _FakeSession([_item("I studied at IIT", valid_from=datetime(2026, 7, 20, tzinfo=UTC))])
    s.ltm = _TagLTM([tagged])
    mem = Memory(s)  # type: ignore[arg-type]
    assert await mem.current("user", "employer") is None


async def test_current_falls_back_when_corpus_has_no_attribute_tags() -> None:
    # Nothing is tagged at all → the store isn't using attributes, so the
    # relevance-ranked fallback is the right behaviour (not an empty answer).
    hits = [_item("residence: NYC", valid_from=datetime(2026, 6, 1, tzinfo=UTC))]
    s = _FakeSession(hits)
    s.ltm = _TagLTM([])  # lookup works, but corpus carries no attribute tags
    mem = Memory(s)  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "residence: NYC"


async def test_current_as_of_is_passed_to_the_store() -> None:
    s = _FakeSession()
    s.ltm = _TagLTM([])
    mem = Memory(s)  # type: ignore[arg-type]
    when = datetime(2026, 3, 1, tzinfo=UTC)
    await mem.current("user", "residence", as_of=when)
    assert s.ltm.calls[0][1] == when  # bi-temporal point-in-time honoured


async def test_current_falls_back_when_store_cannot_look_up() -> None:
    # _FakeLTM has no by_tags → retrieval fallback still works (in-memory store).
    hits = [_item("residence: NYC", valid_from=datetime(2026, 6, 1, tzinfo=UTC))]
    mem = Memory(_FakeSession(hits))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "residence: NYC"


async def test_current_ignores_newer_but_irrelevant_hits() -> None:
    # Regression: `current` used to take the newest of ALL hits, so an unrelated
    # fact recorded today outranked the real answer. `recall` is relevance-ranked,
    # so only the topical head counts; valid time breaks ties inside it.
    hits = [
        _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC)),
        _item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC)),
        _item("filler a", valid_from=datetime(2022, 1, 1, tzinfo=UTC)),
        _item("filler b", valid_from=datetime(2022, 1, 1, tzinfo=UTC)),
        # Newest by far, but ranked outside the topical window — must NOT win.
        _item("bought a laptop", valid_from=datetime(2025, 1, 1, tzinfo=UTC)),
    ]
    mem = Memory(_FakeSession(hits))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "residence: NYC"


async def test_current_none_when_empty() -> None:
    mem = Memory(_FakeSession([]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") is None


async def test_current_falls_back_to_latest_when_all_superseded() -> None:
    a = _item(
        "Boston",
        valid_from=datetime(2023, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 3, 1, tzinfo=UTC),
    )
    b = _item(
        "Chicago",
        valid_from=datetime(2023, 3, 1, tzinfo=UTC),
        valid_to=datetime(2023, 6, 1, tzinfo=UTC),
    )
    mem = Memory(_FakeSession([a, b]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "Chicago"  # latest of the closed ones


# ── timeline() — bi-temporal history, ordered ─────────────────────────────────


async def test_timeline_orders_oldest_to_newest() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    b = _item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([b, a]))  # type: ignore[arg-type]  # search returns unordered
    hist = await mem.timeline("residence")
    assert [h.content for h in hist] == ["residence: Boston", "residence: NYC"]


async def test_timeline_filters_by_since() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    b = _item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([a, b]))  # type: ignore[arg-type]
    hist = await mem.timeline("residence", since=datetime(2023, 3, 1, tzinfo=UTC))
    assert [h.content for h in hist] == ["residence: NYC"]


async def test_timeline_only_matching_entity() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    noise = _item("bought a tank", valid_from=datetime(2023, 2, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([a, noise]))  # type: ignore[arg-type]
    hist = await mem.timeline("residence")
    assert [h.content for h in hist] == ["residence: Boston"]


# ── sync wrapper ──────────────────────────────────────────────────────────────


def test_recall_sync_outside_loop() -> None:
    hits = [_item("Boston")]
    mem = Memory(_FakeSession(hits))  # type: ignore[arg-type]
    out = mem.recall_sync("where?", k=1)
    assert [h.content for h in out] == ["Boston"]


def test_sync_wrapper_refuses_inside_running_loop() -> None:
    import asyncio

    async def _inner() -> None:
        mem = Memory(_FakeSession([]))  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="cannot be called from inside"):
            mem.recall_sync("x")

    asyncio.run(_inner())


def test_add_sync_and_current_sync_outside_loop() -> None:
    s = _FakeSession([_item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))])
    mem = Memory(s)  # type: ignore[arg-type]
    mem.add_sync("I moved to NYC", occurred_at=datetime(2023, 6, 1, tzinfo=UTC))
    assert [i.content for i in s.stm.items] == ["I moved to NYC"]
    assert mem.current_sync("user", "residence") == "residence: NYC"


# ── write path: no-LTM session ────────────────────────────────────────────────


async def test_add_without_ltm_only_writes_stm() -> None:
    s = _FakeSession()
    s.ltm = None  # session with no long-term store attached
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("ephemeral note")
    assert [i.content for i in s.stm.items] == ["ephemeral note"]  # STM only, no crash


async def test_remember_passes_occurred_at() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    when = datetime(2022, 3, 4, tzinfo=UTC)
    await mem.remember("older fact", occurred_at=when)
    assert s.stm.items[0].created_at == when


# ── timeline: until bound + effective-time fallback to created_at ─────────────


async def test_timeline_filters_by_until() -> None:
    a = _item("residence: Boston", valid_from=datetime(2023, 1, 1, tzinfo=UTC))
    b = _item("residence: NYC", valid_from=datetime(2023, 6, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([a, b]))  # type: ignore[arg-type]
    hist = await mem.timeline("residence", until=datetime(2023, 3, 1, tzinfo=UTC))
    assert [h.content for h in hist] == ["residence: Boston"]


async def test_timeline_orders_by_created_at_when_no_valid_range() -> None:
    # No bi-temporal range → _effective_time falls back to created_at ordering.
    old = MemoryItem(
        content="residence: Boston",
        tier=MemoryTier.LTM,
        created_at=datetime(2021, 1, 1, tzinfo=UTC),
    )
    new = MemoryItem(
        content="residence: NYC", tier=MemoryTier.LTM, created_at=datetime(2024, 1, 1, tzinfo=UTC)
    )
    mem = Memory(_FakeSession([new, old]))  # type: ignore[arg-type]  # unordered in
    hist = await mem.timeline("residence")
    assert [h.content for h in hist] == ["residence: Boston", "residence: NYC"]


# ── lifecycle: start / aclose / async context manager ─────────────────────────


async def test_lifecycle_start_aclose_and_context_manager() -> None:
    calls: list[str] = []

    class _LifecycleSession(_FakeSession):
        async def start(self) -> None:
            calls.append("start")

        async def aclose(self) -> None:
            calls.append("aclose")

    mem = Memory(_LifecycleSession())  # type: ignore[arg-type]
    await mem.start()
    await mem.aclose()
    assert calls == ["start", "aclose"]

    calls.clear()
    async with Memory(_LifecycleSession()) as m:  # type: ignore[arg-type]
        assert m is not None
    assert calls == ["start", "aclose"]  # __aenter__ / __aexit__ drive lifecycle


_ = timedelta  # keep import referenced for future date-math tests
