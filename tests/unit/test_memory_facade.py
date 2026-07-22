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
from continuum.core.types import BiTemporalRange, PruneReport

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

    async def search_hybrid(self, q: Any, k: int = 8) -> list[Any]:
        return []

    async def update(self, target: Any, patch: Any) -> None: ...

    async def invalidate(self, target: Any) -> None: ...


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


async def test_add_input_hygiene() -> None:
    from continuum.memory import MAX_FACT_CHARS

    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("")  # empty
    await mem.add("   ")  # whitespace
    assert s.stm.items == [] and s.ltm.upserts == []  # both no-ops, no junk rows

    await mem.add("  padded fact  ")  # stripped
    assert s.stm.items[0].content == "padded fact"

    await mem.add("x" * (MAX_FACT_CHARS + 5000))  # oversized → truncated, not embedded whole
    assert len(s.ltm.upserts[-1].content or "") == MAX_FACT_CHARS


async def test_auto_attribute_tags_on_write() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("I live in Boston.", auto_attribute=True)
    assert s.ltm.upserts[0].metadata.get("attribute") == "residence"


async def test_explicit_attribute_beats_auto() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("I live in Boston.", attribute="home_city", auto_attribute=True)
    assert s.ltm.upserts[0].metadata.get("attribute") == "home_city"


async def test_auto_attribute_off_by_default() -> None:
    s = _FakeSession()
    mem = Memory(s)  # type: ignore[arg-type]
    await mem.add("I live in Boston.")  # no auto_attribute
    assert "attribute" not in (s.ltm.upserts[0].metadata or {})


async def test_recall_clamps_k() -> None:
    captured: dict[str, int] = {}

    class _S(_FakeSession):
        async def search(self, query: str, k: int = 10) -> list[MemoryItem]:
            captured["k"] = k
            return []

    from continuum.memory import MAX_RECALL_K

    mem = Memory(_S())  # type: ignore[arg-type]
    await mem.recall("q", k=1_000_000)
    assert captured["k"] == MAX_RECALL_K
    await mem.recall("q", k=-5)
    assert captured["k"] == 0


async def test_no_data_loss_when_the_decider_raises() -> None:
    # Phase 0.2: with supersession on but the LLM flaky (erroring/timing out),
    # the write must still land — never silently dropped. _decided_write catches
    # any decider failure and falls back to a plain upsert.
    class _FlakyDecider:
        async def decide_operation(self, fact: Any, neighbors: Any) -> Any:
            raise TimeoutError("LLM timed out")

    s = _FakeSession()
    mem = Memory(s, decider=_FlakyDecider())  # type: ignore[arg-type]
    await mem.add("I moved to New York City.")
    assert [i.content for i in s.ltm.upserts] == ["I moved to New York City."]


async def test_no_data_loss_on_unadjudicated_noop() -> None:
    # An un-short-circuited NOOP means "could not decide" (typically no LLM), not
    # "this is a duplicate" — honouring it would discard the fact. Falls back.
    class _NoopDecider:
        async def decide_operation(self, fact: Any, neighbors: Any) -> Any:
            from continuum.promotion.mem0_promoter import Decision

            return Decision(op="NOOP", target_id=None, rationale="no llm", short_circuited=False)

    s = _FakeSession()
    mem = Memory(s, decider=_NoopDecider())  # type: ignore[arg-type]
    await mem.add("I moved to New York City.")
    assert [i.content for i in s.ltm.upserts] == ["I moved to New York City."]


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


async def test_current_fallback_honours_as_of() -> None:
    # Found by adversarial testing: the exact tag lookup applies the bi-temporal
    # window in SQL, but the retrieval fallback ignored `as_of` entirely and
    # answered "what was true in year 1?" with a fact recorded in 2026.
    fact = _item("residence: Bhilai", valid_from=datetime(2026, 1, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([fact]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence") == "residence: Bhilai"
    assert await mem.current("user", "residence", as_of=datetime(1, 1, 1, tzinfo=UTC)) is None
    assert (
        await mem.current("user", "residence", as_of=datetime(2027, 1, 1, tzinfo=UTC))
        == "residence: Bhilai"
    )


async def test_current_accepts_a_naive_as_of() -> None:
    # The as_of filter compared the store's AWARE timestamps against a NAIVE
    # datetime, raising TypeError — which reached the client as a failed tool
    # call. The MCP layer always produces naive datetimes
    # (`datetime.fromisoformat("2027-01-01")`), so every explicit as_of crashed.
    fact = _item("residence: Bhilai", valid_from=datetime(2026, 1, 1, tzinfo=UTC))
    mem = Memory(_FakeSession([fact]))  # type: ignore[arg-type]
    assert await mem.current("user", "residence", as_of=datetime(2027, 1, 1)) == "residence: Bhilai"
    assert await mem.current("user", "residence", as_of=datetime(1, 1, 1)) is None


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


# ── forget (store-level pruning) ──────────────────────────────────────────────


class _PruningLTM(_FakeLTM):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Any, bool]] = []

    async def prune(self, policy: Any, *, dry_run: bool = True) -> PruneReport:
        self.calls.append((policy, dry_run))
        return PruneReport(matched=3, pruned=0 if dry_run else 3, dry_run=dry_run)


async def test_forget_is_a_dry_run_unless_told_otherwise() -> None:
    # Forgetting is the one operation memory cannot undo. A caller who forgets
    # to say dry_run=False must get a report, not an empty store.
    session = _FakeSession()
    session.ltm = _PruningLTM()  # type: ignore[assignment]
    mem = Memory(session)  # type: ignore[arg-type]

    report = await mem.forget(unused_for=timedelta(days=90))

    assert report.dry_run is True and report.pruned == 0 and report.matched == 3
    _, dry_run = session.ltm.calls[0]  # type: ignore[attr-defined]
    assert dry_run is True


async def test_forget_passes_the_policy_through() -> None:
    session = _FakeSession()
    session.ltm = _PruningLTM()  # type: ignore[assignment]
    mem = Memory(session)  # type: ignore[arg-type]

    report = await mem.forget(
        unused_for=timedelta(days=30),
        superseded_only=False,
        max_importance=0.25,
        max_access_count=0,
        limit=10,
        dry_run=False,
    )

    policy, dry_run = session.ltm.calls[0]  # type: ignore[attr-defined]
    assert policy.unused_for == timedelta(days=30)
    assert policy.superseded_only is False
    assert policy.max_importance == 0.25 and policy.max_access_count == 0
    assert policy.limit == 10
    assert dry_run is False and report.pruned == 3


async def test_forget_says_so_when_the_store_cannot() -> None:
    # The in-memory demo store has no prune; silently reporting "0 forgotten"
    # would read as "nothing to forget" rather than "this cannot forget".
    mem = Memory(_FakeSession())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Postgres"):
        await mem.forget(unused_for=timedelta(days=1))


# ── duplicate writes reinforce instead of re-inserting ────────────────────────


class _DedupingLTM(_FakeLTM):
    def __init__(self, known: bool) -> None:
        super().__init__()
        self._known = known
        self.touched: list[tuple[str, Any, Any]] = []

    async def touch_duplicate(
        self, text: str, *, valid_from: Any = None, attribute: Any = None
    ) -> Any:
        self.touched.append((text, valid_from, attribute))
        return "existing-id" if self._known else None


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2] for _ in texts]


async def test_a_restated_fact_is_not_embedded_or_stored_again() -> None:
    # The embedding is the expensive part (~77ms); re-embedding text we already
    # know is the entire waste this avoids.
    session = _FakeSession()
    session.ltm = _DedupingLTM(known=True)  # type: ignore[assignment]
    emb = _CountingEmbedder()
    mem = Memory(session, embedder=emb)  # type: ignore[arg-type]

    await mem.add("I cycle to work every day.")

    assert session.ltm.upserts == []  # type: ignore[attr-defined]
    assert emb.calls == 0  # never embedded


async def test_a_new_fact_still_takes_the_normal_path() -> None:
    session = _FakeSession()
    session.ltm = _DedupingLTM(known=False)  # type: ignore[assignment]
    emb = _CountingEmbedder()
    mem = Memory(session, embedder=emb)  # type: ignore[arg-type]

    await mem.add("I ride a blue Brompton.")

    assert len(session.ltm.upserts) == 1  # type: ignore[attr-defined]
    assert emb.calls == 1


async def test_the_restatement_passes_along_what_it_adds() -> None:
    from datetime import datetime

    session = _FakeSession()
    session.ltm = _DedupingLTM(known=True)  # type: ignore[assignment]
    mem = Memory(session)  # type: ignore[arg-type]

    when = datetime(2025, 4, 1, tzinfo=UTC)
    await mem.add("I work at Nimbus.", occurred_at=when, attribute="employer")

    text, valid_from, attribute = session.ltm.touched[0]  # type: ignore[attr-defined]
    assert text == "I work at Nimbus."
    assert valid_from == when and attribute == "employer"


async def test_a_broken_duplicate_check_never_loses_the_write() -> None:
    # Degrading to a duplicate row is survivable; dropping the fact is not.
    class _Exploding(_FakeLTM):
        async def touch_duplicate(self, text: str, **_kw: Any) -> Any:
            raise RuntimeError("db hiccup")

    session = _FakeSession()
    session.ltm = _Exploding()  # type: ignore[assignment]
    mem = Memory(session)  # type: ignore[arg-type]

    await mem.add("I cycle to work.")

    assert len(session.ltm.upserts) == 1  # type: ignore[attr-defined]


# ── the current version of an attribute speaks for its group ──────────────────


def _tagged(
    content: str,
    attribute: str,
    *,
    valid_from: datetime | None = None,
    created_at: datetime | None = None,
) -> MemoryItem:
    item = _item(content, valid_from=valid_from)
    item.metadata = {"attribute": attribute}
    if created_at is not None:
        item.created_at = created_at
    return item


def test_a_correction_takes_the_group_best_position() -> None:
    """Relevance has no idea a later fact replaces an earlier one, and without
    an LLM decider nothing marks the old one superseded — so recall put the
    stale fact first and the correction four places below it."""
    from continuum.memory import _prefer_current_versions

    old = _tagged("I live in Porto.", "residence", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    noise = _item("I own a dog named Bolt.")
    new = _tagged("I moved to Berlin.", "residence", valid_from=datetime(2026, 7, 20, tzinfo=UTC))

    out = _prefer_current_versions([old, noise, new])

    assert out[0].content == "I moved to Berlin."  # correction leads
    assert out[1] is noise  # untagged neighbours do not move
    assert out[2].content == "I live in Porto."  # history kept, just below


def test_nothing_is_dropped_or_reordered_beyond_the_group() -> None:
    from continuum.memory import _prefer_current_versions

    items = [_item("a"), _item("b"), _item("c")]
    assert _prefer_current_versions(items) == items


def test_two_attributes_are_resolved_independently() -> None:
    from continuum.memory import _prefer_current_versions

    r_old = _tagged("I live in Porto.", "residence", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    e_old = _tagged("I work at Nimbus.", "employer", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    r_new = _tagged("I moved to Berlin.", "residence", valid_from=datetime(2026, 7, 20, tzinfo=UTC))
    e_new = _tagged("I joined Stripe.", "employer", valid_from=datetime(2026, 7, 12, tzinfo=UTC))

    out = [i.content for i in _prefer_current_versions([r_old, e_old, r_new, e_new])]

    assert out[0] == "I moved to Berlin." and out[1] == "I joined Stripe."
    assert set(out[2:]) == {"I live in Porto.", "I work at Nimbus."}


def test_an_undated_original_still_loses_to_a_dated_correction() -> None:
    # Same comparable-clock rule `current` uses: valid time only counts when
    # every candidate states one, else transaction time decides.
    from continuum.memory import _prefer_current_versions

    old = _tagged("I live in Porto.", "residence", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    new = _tagged(
        "I moved to Berlin.",
        "residence",
        valid_from=datetime(2025, 7, 20, tzinfo=UTC),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert _prefer_current_versions([old, new])[0].content == "I moved to Berlin."


async def test_recall_applies_it_end_to_end() -> None:
    old = _tagged("I live in Porto.", "residence", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    new = _tagged("I moved to Berlin.", "residence", valid_from=datetime(2026, 7, 20, tzinfo=UTC))
    mem = Memory(_FakeSession(search_result=[old, new]))  # type: ignore[arg-type]
    assert (await mem.recall("where do I live?"))[0].content == "I moved to Berlin."


# ── untagged replacements: "I switched from X to Y" ───────────────────────────


def test_a_replacement_outranks_what_it_replaced_without_any_tag() -> None:
    """The case tagged reordering could not reach.

    "I switched from Neovim to Zed" carries no attribute, so it used to rank
    below "I use Neovim with a tmux setup" — the stale fact winning again, just
    without a tag to catch it.
    """
    from continuum.memory import _prefer_current_versions

    old = _item("I use Neovim with a tmux setup.")
    noise = _item("I prefer pytest over unittest.")
    new = _item("I switched from Neovim to Zed.")

    out = [i.content for i in _prefer_current_versions([old, noise, new])]

    assert out[0] == "I switched from Neovim to Zed."
    assert out[1] == "I prefer pytest over unittest."  # unrelated, unmoved
    assert out[2] == "I use Neovim with a tmux setup."


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("I switched from Neovim to Zed.", "neovim"),
        ("I moved from Porto to Berlin.", "porto"),
        ("I migrated from MySQL to PostgreSQL.", "mysql"),
        # Capitalisation on BOTH sides is the precision lever — without it these
        # everyday sentences would start reordering memories.
        ("I switched from the kitchen to the office.", None),
        ("I moved the file from src to lib.", None),
        ("I switched branches.", None),
        ("I use Neovim with a tmux setup.", None),
    ],
)
def test_only_a_named_replacement_counts(text: str, entity: str | None) -> None:
    from continuum.memory import _superseded_entity

    assert _superseded_entity(_item(text)) == entity


def test_a_replacement_with_nothing_to_displace_changes_nothing() -> None:
    from continuum.memory import _prefer_current_versions

    items = [_item("I like Berlin techno."), _item("I switched from Neovim to Zed.")]
    assert _prefer_current_versions(items) == items


def test_two_replacements_do_not_reorder_each_other() -> None:
    # A later switch does not "displace" an earlier one just by naming a tool.
    from continuum.memory import _prefer_current_versions

    a = _item("I switched from Neovim to Zed.")
    b = _item("I switched from Zed to Helix.")
    assert _prefer_current_versions([a, b]) == [a, b]
