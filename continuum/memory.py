"""
continuum.memory
================
The public, high-level Continuum API — one import, natural verbs.

``Memory`` is a thin facade over :class:`continuum.core.session.ContinuumSession`
(it wraps, never reimplements: retrieval, supersession and bi-temporal history
all live in the session + stores). It gives you five verbs:

    from continuum import Memory

    mem = Memory.in_memory()                       # zero-config, no DB
    await mem.add("I moved to Boston")
    await mem.add("Actually I'm in NYC now")
    hits = await mem.recall("where do I live?")    # hybrid retrieval
    where = await mem.current("user", "residence") # resolved, not stale
    hist = await mem.timeline("residence")         # bi-temporal history

For production, construct a full Postgres-backed session (with the supersession
decider) and wrap it: ``Memory(my_session)``. Supersession/bi-temporal reads are
only as strong as the store behind the session — the in-memory factory is for
demos and tests; Postgres is where valid-time history and the ADD/UPDATE/DELETE
decider live.

Sync variants (``add_sync`` etc.) are provided for non-async callers; they refuse
to run inside a live event loop (use the ``await`` form there).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from continuum.core.session import ContinuumSession
from continuum.core.types import MemoryItem, MemoryTier

if TYPE_CHECKING:
    from continuum.core.config import ContinuumConfig
    from continuum.core.protocols import LTMProtocol, RetrieverProtocol

__all__ = ["Memory"]


class Memory:
    """High-level memory API for AI agents. Wraps a :class:`ContinuumSession`."""

    def __init__(self, session: ContinuumSession, *, session_id: str | None = None) -> None:
        """Wrap an existing session. Use :meth:`in_memory` for a zero-config one."""
        self._session = session
        if session_id is not None:
            self._session.session_id = session_id

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def in_memory(
        cls,
        *,
        session_id: str = "default",
        config: ContinuumConfig | None = None,
    ) -> Memory:
        """A zero-config, in-process memory (InMemory STM + LTM, no Postgres, no
        model download). Ideal for the quickstart, demos and tests. Retrieval is
        lexical/recency unless you attach a retriever+embedder."""
        from continuum.core.config import ContinuumConfig as _Cfg
        from continuum.stores.in_memory.ltm import InMemoryLTM
        from continuum.stores.stm import InMemorySTM

        cfg = config or _Cfg()
        session = ContinuumSession(
            cfg,
            stm=InMemorySTM(),
            ltm=cast("LTMProtocol", InMemoryLTM()),
            session_id=session_id,
        )
        return cls(session, session_id=session_id)

    @classmethod
    def from_postgres(
        cls,
        dsn: str | None = None,
        *,
        embeddings: bool = True,
        session_id: str = "default",
        config: ContinuumConfig | None = None,
    ) -> Memory:
        """A Postgres-backed memory with the full hybrid retriever — the
        production stack, where **dense (semantic) recall** and durable
        valid-time history actually work.

        Requires a reachable, migrated database (``make db-up && make
        db-migrate``). *dsn* overrides the configured DSN (else it comes from
        ``continuum.yaml`` / ``CONTINUUM_DB_DSN``). *embeddings* attaches the
        local bge-m3 embedder so the LTM dense channel fires — it downloads
        ~2.3 GB on first use; pass ``embeddings=False`` for a sparse-only setup
        with no model download. Call :meth:`start` (or use ``async with``)
        before reading — that is when the connection pool opens."""
        from continuum.core.config import ContinuumConfig as _Cfg
        from continuum.retrieval import Retriever
        from continuum.retrieval.embedding_query import EmbeddingQueryRetriever
        from continuum.stores.postgres.ltm import PostgresLTM
        from continuum.stores.stm.postgres_stm import PostgresSTM

        cfg = config or _Cfg.load()
        resolved_dsn = dsn or str(cfg.database.dsn)

        embedder = None
        if embeddings:
            from continuum.embeddings import EmbeddingService

            embedder = EmbeddingService(cfg.embedding)

        stm = PostgresSTM(dsn=resolved_dsn)
        ltm = PostgresLTM(dsn=resolved_dsn)
        inner = Retriever(ltm=ltm, stm=stm, session_id=session_id)
        retriever = EmbeddingQueryRetriever(inner, embedder)

        session = ContinuumSession(
            cfg,
            stm=stm,
            ltm=cast("LTMProtocol", ltm),
            retriever=cast("RetrieverProtocol", retriever),
            session_id=session_id,
        )
        return cls(session, session_id=session_id)

    @property
    def session(self) -> ContinuumSession:
        """The wrapped :class:`ContinuumSession` (for advanced use / lifecycle)."""
        return self._session

    async def start(self) -> None:
        """Bring the session online (bootstraps schemas / background workers)."""
        await self._session.start()

    async def aclose(self) -> None:
        """Flush and shut the session down cleanly."""
        await self._session.aclose()

    async def __aenter__(self) -> Memory:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # ── write ────────────────────────────────────────────────────────────────

    async def add(
        self,
        text: str,
        *,
        occurred_at: datetime | None = None,
        importance: float = 0.5,
    ) -> None:
        """Remember a fact/turn. Appends to short-term memory and, when the
        session has long-term storage, upserts it there (supersession/dedup are
        applied by the store's promoter where configured)."""
        sid = self._session.session_id
        when = occurred_at or datetime.now(UTC)
        await self._session.stm.append(
            MemoryItem(
                content=text,
                tier=MemoryTier.STM,
                session_id=sid,
                created_at=when,
                importance=importance,
            )
        )
        ltm = self._session.ltm
        if ltm is not None:
            await ltm.upsert(
                MemoryItem(
                    content=text,
                    tier=MemoryTier.LTM,
                    session_id=sid,
                    created_at=when,
                    importance=importance,
                )
            )

    async def remember(self, text: str, *, occurred_at: datetime | None = None) -> None:
        """Explicit alias for :meth:`add` — reads well when storing a known fact."""
        await self.add(text, occurred_at=occurred_at)

    # ── read ─────────────────────────────────────────────────────────────────

    async def recall(self, query: str, *, k: int = 8) -> list[MemoryItem]:
        """Retrieve up to *k* memories relevant to *query*, best-first."""
        return await self._session.search(query, k=k)

    async def current(self, subject: str, attribute: str) -> str | None:
        """The user's CURRENT value for an attribute, after supersession — e.g.
        ``current("user", "residence")`` → the latest non-superseded value.
        Returns ``None`` if nothing relevant is stored."""
        hits = await self.recall(f"{subject} {attribute}", k=8)
        if not hits:
            return None
        live = [h for h in hits if not _is_superseded(h)]
        chosen = max(live or hits, key=_effective_time)
        return chosen.content or None

    async def timeline(
        self,
        entity: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[MemoryItem]:
        """Every stored version relevant to *entity*, oldest→newest (bi-temporal
        history). Optionally bounded by ``since``/``until`` on valid time."""
        hits = await self.recall(entity, k=50)
        ent = entity.lower()
        items = [h for h in hits if ent in (h.content or "").lower()]
        items.sort(key=_effective_time)
        if since is not None:
            items = [h for h in items if _effective_time(h) >= since]
        if until is not None:
            items = [h for h in items if _effective_time(h) <= until]
        return items

    # ── sync wrappers ─────────────────────────────────────────────────────────

    def add_sync(self, text: str, *, occurred_at: datetime | None = None) -> None:
        ContinuumSession._run_sync(
            self.add(text, occurred_at=occurred_at),
            label="add",
        )

    def recall_sync(self, query: str, *, k: int = 8) -> list[MemoryItem]:
        return ContinuumSession._run_sync(self.recall(query, k=k), label="recall")

    def current_sync(self, subject: str, attribute: str) -> str | None:
        return ContinuumSession._run_sync(
            self.current(subject, attribute),
            label="current",
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_superseded(item: MemoryItem) -> bool:
    """True when a bi-temporal valid_to has closed the item (no longer current)."""
    vr = item.valid_range
    return bool(vr is not None and vr.valid_to is not None)


def _effective_time(item: MemoryItem) -> datetime:
    """Valid-from when present (bi-temporal), else creation time — for ordering."""
    vr = item.valid_range
    if vr is not None and vr.valid_from is not None:
        return vr.valid_from
    return item.created_at
