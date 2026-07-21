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

import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from continuum.core.session import ContinuumSession
from continuum.core.types import (
    BiTemporalRange,
    MemoryItem,
    MemoryTier,
    PrunePolicy,
    PruneReport,
)

if TYPE_CHECKING:
    from continuum.core.config import ContinuumConfig
    from continuum.core.protocols import LTMProtocol, RetrieverProtocol
    from continuum.promotion.mem0_promoter import CompletionFn
    from continuum.retrieval.embedding_query import EmbedderProtocol

log = logging.getLogger(__name__)

#: How many of the most-relevant hits `current` considers topical. Valid time
#: breaks ties inside this window; beyond it, a hit is about something else.
_CURRENT_TOPICAL_WINDOW = 3

#: Longest text a single memory keeps. One embedding covers the whole string, so
#: a huge blob is one meaningless vector and a bloated row; truncate instead.
MAX_FACT_CHARS = 8000

#: Upper bound on recall k, so an absurd request cannot ask for a giant result.
MAX_RECALL_K = 100

__all__ = ["Memory"]


class Memory:
    """High-level memory API for AI agents. Wraps a :class:`ContinuumSession`."""

    def __init__(
        self,
        session: ContinuumSession,
        *,
        session_id: str | None = None,
        embedder: EmbedderProtocol | None = None,
        decider: Any | None = None,
    ) -> None:
        """Wrap an existing session. Use :meth:`in_memory` for a zero-config one.

        *embedder*, when given, is used to embed text **on write** so the LTM
        dense channel has vectors to match against — the read side alone is not
        enough (a stored item with no embedding is invisible to dense search).

        *decider*, when given, routes each write through the Mem0
        ADD/UPDATE/DELETE decision so a contradicting fact **supersedes** the one
        it replaces instead of piling up beside it. See
        :meth:`from_postgres` — it requires an LLM and is off by default."""
        self._session = session
        self._embedder = embedder
        self._decider = decider
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
        supersession_completion_fn: CompletionFn | None = None,
        supersession_model: str = "openai/gpt-4o-mini",
        namespace: str = "default",
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
        before reading — that is when the connection pool opens.

        *supersession_completion_fn* enables true supersession: writes are routed
        through the Mem0 ADD/UPDATE/DELETE decider so a contradicting fact
        retires the one it replaces (which is what makes :meth:`current`
        meaningful). It is **off by default and requires an LLM**: contradiction
        pairs land in the decider's ambiguous similarity band — e.g.
        ``cosine("I live in Boston", "I moved from Boston to New York") = 0.81``,
        between ``add_threshold`` 0.5 and ``noop_threshold`` 0.97 — where only
        the model can adjudicate. With no LLM the decider returns ``NOOP`` and
        the new fact would be **dropped**, so we never enable it implicitly.

        *namespace* is the tenant scope. Facts are written under it and every
        read is filtered to it, so two ``Memory.from_postgres(..., namespace=…)``
        on the same database never see each other's memories. Defaults to
        ``"default"`` — one shared store — which is what a single user wants."""
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

        decider = None
        if supersession_completion_fn is not None:
            from continuum.core.config import PromoterConfig
            from continuum.promotion.mem0_promoter import Mem0Promoter

            # The model id must match the provider's namespace — PromoterConfig
            # defaults to "gpt-4o-mini", which OpenRouter rejects without the
            # "openai/" prefix.
            decider = Mem0Promoter(
                PromoterConfig(llm_model=supersession_model),
                completion_fn=supersession_completion_fn,
            )

        stm = PostgresSTM(dsn=resolved_dsn)
        # namespace scopes LTM: one database can hold isolated stores, so a
        # recall/current can never surface another tenant's facts. STM is already
        # scoped per session_id, a finer (ephemeral) axis.
        ltm = PostgresLTM(dsn=resolved_dsn, namespace=namespace)
        inner = Retriever(ltm=ltm, stm=stm, session_id=session_id)
        retriever = EmbeddingQueryRetriever(inner, embedder)

        session = ContinuumSession(
            cfg,
            stm=stm,
            ltm=cast("LTMProtocol", ltm),
            retriever=cast("RetrieverProtocol", retriever),
            session_id=session_id,
        )
        return cls(session, session_id=session_id, embedder=embedder, decider=decider)

    @property
    def session(self) -> ContinuumSession:
        """The wrapped :class:`ContinuumSession` (for advanced use / lifecycle)."""
        return self._session

    @property
    def is_durable(self) -> bool:
        """Do writes survive this process?

        False for the in-memory store, where :meth:`add` succeeds and the data
        then disappears when the process exits. Callers that report success to a
        human should say which of the two happened — "stored" is a lie if the
        store evaporates, and the failure is otherwise invisible.
        """
        ltm = self._session.ltm
        if ltm is None:
            return False
        from continuum.stores.in_memory.ltm import InMemoryLTM

        return not isinstance(ltm, InMemoryLTM)

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
        attribute: str | None = None,
        auto_attribute: bool = False,
        split: bool = False,
    ) -> None:
        """Remember a fact/turn. Appends to short-term memory and, when the
        session has long-term storage, upserts it there (supersession/dedup are
        applied by the store's promoter where configured).

        *attribute* names what the fact is **about** (``"residence"``,
        ``"employer"``, …). Tagging it turns :meth:`current` from a fuzzy search
        for an attribute *label* into an exact lookup — which is the difference
        between reliably answering "where do I live now?" and hoping the phrasing
        happens to embed close to the word "residence".

        *split* stores a compound statement as separate atomic facts. One
        embedding covers the whole string, so "We're going with Postgres 16,
        pgvector needs 0.8 or newer" sits between two topics and matches
        "what database are we using?" *worse* than an unrelated memory
        (cos 0.479 vs 0.522). Split, the same content scores 0.614 and ranks
        first. The splitter is conservative and returns the text unchanged when
        a split would be unsafe, so it is safe to pass unconditionally.

        Input hygiene: an empty/whitespace string is a no-op (a zero-length row
        is noise that recall would still return), and text past
        :data:`MAX_FACT_CHARS` is truncated rather than embedded whole — a
        100k-char blob is one useless vector and a large row."""
        text = text.strip()
        if not text:
            return
        if len(text) > MAX_FACT_CHARS:
            text = text[:MAX_FACT_CHARS]
        if split:
            from continuum.promotion.clause_split import split_facts

            facts = split_facts(text)
            if len(facts) > 1:
                for fact in facts:
                    await self.add(
                        fact,
                        occurred_at=occurred_at,
                        importance=importance,
                        attribute=attribute,
                        auto_attribute=auto_attribute,
                    )
                return
        # Infer the attribute when the caller did not give one, so `current` can
        # answer it exactly. Conservative — extract_attribute returns None unless
        # confident, and an explicit *attribute* always wins.
        if attribute is None and auto_attribute:
            from continuum.promotion.attribute_extract import extract_attribute

            attribute = extract_attribute(text)
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
        if ltm is None:
            return
        # Already know this, word for word? Reinforce it instead of storing a
        # second copy. Checked BEFORE embedding, because the embedding is the
        # expensive part (~77ms) and re-embedding known text is the whole waste.
        touch = getattr(ltm, "touch_duplicate", None)
        if touch is not None:
            try:
                if await touch(text, valid_from=occurred_at, attribute=attribute) is not None:
                    return
            except Exception:
                log.debug("duplicate check failed — writing normally", exc_info=True)
        vec = await self._embed(text)
        node = MemoryItem(
            content=text,
            tier=MemoryTier.LTM,
            session_id=sid,
            created_at=when,
            importance=importance,
            embedding=vec,
            metadata=({"attribute": attribute} if attribute else {}),
            # Valid time is recorded ONLY when the caller stated it. Defaulting
            # it to the write time invents a claim we cannot support, and the
            # invention inverts supersession: state "I live in Lisbon" today
            # (valid_from := now), then later "I moved to Porto on July 1st"
            # (valid_from := July 1st), and the correction now looks *older*
            # than the fact it replaces — so `current` keeps answering Lisbon.
            # A NULL here means "no stated valid time"; ordering then falls back
            # to transaction time, which is the honest signal we do have.
            valid_range=(BiTemporalRange(valid_from=occurred_at) if occurred_at else None),
        )
        if self._decider is not None and await self._decided_write(ltm, node, vec):
            return
        await ltm.upsert(node)

    async def _decided_write(self, ltm: Any, node: MemoryItem, vec: list[float] | None) -> bool:
        """Route the write through the Mem0 decider. Returns True if it handled
        the write, False to fall back to a plain upsert.

        Never drops the fact: any failure — decider error, an UPDATE/DELETE with
        no target, or a NOOP that came from an unavailable LLM rather than a real
        duplicate — returns False so the caller stores it normally.
        """
        import uuid as _uuid

        from continuum.extraction.fact_extractor import Fact

        decider = self._decider
        if decider is None:
            return False
        text = node.content or ""
        try:
            neighbors = await self._cosine_neighbors(ltm, text, vec)
            decision = await decider.decide_operation(
                Fact(
                    text=text,
                    confidence=0.9,
                    entities_mentioned=[],
                    source_block_id=_uuid.uuid4(),
                ),
                neighbors,
            )
        except Exception:
            log.debug("supersession decider unavailable — storing as a plain add", exc_info=True)
            return False

        op, target = decision.op, decision.target_id
        try:
            if op == "UPDATE" and target is not None:
                merged = decision.merged_text or text
                await ltm.update(target, {"text": merged, "embedding": await self._embed(merged)})
                return True
            if op == "DELETE" and target is not None:
                await ltm.invalidate(target)  # retire the contradicted fact…
                return False  # …and still store the new one
            if op == "NOOP" and not decision.short_circuited:
                # A genuine duplicate short-circuits. An un-short-circuited NOOP
                # means the decider could not adjudicate (typically no LLM), and
                # honouring it would silently discard the fact.
                return False
        except Exception:
            log.debug("applying decision %s failed — falling back to add", op, exc_info=True)
            return False
        return bool(op == "NOOP")

    async def _cosine_neighbors(self, ltm: Any, text: str, vec: list[float] | None) -> list[Any]:
        """Nearest stored facts, re-scored by true cosine.

        ``search_hybrid`` ranks by RRF (~0.03) and returns items *without* their
        embedding, but the decider's add/noop thresholds are calibrated for
        cosine — so re-embed the top candidates and score them properly.
        """
        from continuum.core.types import Query, ScoreBreakdown, ScoredItem

        q = Query(
            text=text,
            embedding=vec,
            top_k=8,
            tiers=[MemoryTier.LTM],
            session_id=self._session.session_id,
        )
        hits = list(await ltm.search_hybrid(q, 8))
        if vec is None:
            return hits
        rescored: list[Any] = []
        for si in hits[:6]:
            cos = _cosine(vec, await self._embed(si.item.content or ""))
            rescored.append(
                ScoredItem(
                    item=si.item,
                    scores=ScoreBreakdown(
                        relevance=cos, importance=0.0, recency=0.0, confidence=1.0, composite=cos
                    ),
                )
            )
        return rescored

    async def _embed(self, text: str) -> list[float] | None:
        """Dense vector for *text*, or None when no embedder is attached.

        Degrades to None on failure — the item is still stored, just invisible
        to the dense channel (sparse + recency still find it)."""
        if self._embedder is None:
            return None
        try:
            return list((await self._embedder.embed([text]))[0])
        except Exception:
            return None

    async def remember(self, text: str, *, occurred_at: datetime | None = None) -> None:
        """Explicit alias for :meth:`add` — reads well when storing a known fact."""
        await self.add(text, occurred_at=occurred_at)

    # ── forget ───────────────────────────────────────────────────────────────

    async def forget(
        self,
        *,
        unused_for: timedelta = timedelta(0),
        superseded_only: bool = True,
        contains: str | None = None,
        max_importance: float | None = None,
        max_access_count: int | None = None,
        limit: int = 1000,
        dry_run: bool = True,
    ) -> PruneReport:
        """Retire memories nobody reads that are no longer true.

        Memory that only grows is memory that gets slower and noisier forever,
        but forgetting is the one operation a memory system cannot take back —
        so it is **explicit, scoped and dry-run by default**. Nothing prunes on
        a timer; you call this.

        By default only *superseded* facts are eligible: those whose valid-time
        window has closed because a newer fact replaced them. Pass
        ``superseded_only=False`` to also forget facts that are still true.

        For the other kind of forgetting — "that one is wrong, delete it" —
        pass *contains*, a case-insensitive substring of the memory you mean::

            await mem.forget(contains="dentist appointment",
                             superseded_only=False, dry_run=False)

        Without it, targeted removal is impossible: `superseded_only=True`
        matches nothing until an LLM decider has retired something, and turning
        it off makes age the only lever, which reaches the whole store.

        Scope: this prunes **long-term** memory. A turn still sitting in the
        current session's short-term buffer keeps surfacing in that session's
        `recall` until the buffer rolls over — STM is session-scoped and
        transient, so the next session sees the fact gone. This is maintenance,
        not "delete this thing right now".

        Returns a :class:`~continuum.stores.postgres.ltm.PruneReport`; with
        ``dry_run=True`` (the default) nothing is written and ``pruned`` is 0.

        Raises
        ------
        NotImplementedError:
            If the backing store cannot prune (the in-memory demo store).
        """
        prune = getattr(self._session.ltm, "prune", None)
        if prune is None:
            raise NotImplementedError(
                "This store cannot forget — Memory.forget() needs a Postgres-backed "
                "store (Memory.from_postgres(...) or CONTINUUM_DB_DSN)."
            )
        policy = PrunePolicy(
            unused_for=unused_for,
            superseded_only=superseded_only,
            contains=contains,
            max_importance=max_importance,
            max_access_count=max_access_count,
            limit=limit,
        )
        report: PruneReport = await prune(policy, dry_run=dry_run)
        return report

    # ── read ─────────────────────────────────────────────────────────────────

    async def recall(self, query: str, *, k: int = 8) -> list[MemoryItem]:
        """Retrieve up to *k* memories relevant to *query*, best-first.

        *k* is clamped to ``[0, MAX_RECALL_K]`` — a caller-supplied ``k`` of a
        million is not a reason to build a million-row result."""
        k = min(max(k, 0), MAX_RECALL_K)
        return await self._session.search(query, k=k)

    async def current(
        self,
        subject: str,
        attribute: str,
        *,
        as_of: datetime | None = None,
    ) -> str | None:
        """The CURRENT value for an attribute, after supersession — e.g.
        ``current("user", "residence")`` → the latest non-superseded value.
        ``None`` if nothing relevant is stored.

        Facts written with ``add(..., attribute=...)`` are answered by an **exact
        tag lookup** honouring valid time — deterministic, and immune to whether
        the wording happens to embed near the attribute's name. Untagged facts
        fall back to relevance-ranked retrieval, which is best-effort: an
        attribute label ("user residence") is a poor semantic probe for a
        sentence ("I moved from Boston to New York City").

        *as_of* answers it **as of a past date** ("where did I live in March?")
        rather than today, using the store's bi-temporal window.
        """
        when = as_of or datetime.now(UTC)
        looked_up, exact = await self._current_by_tag(attribute, when)
        if looked_up:
            # The store can answer attributes exactly, so its answer is final —
            # including "nothing". Guessing from fuzzy retrieval here would
            # invent a value for an attribute we simply have no fact about,
            # which is the confidently-wrong failure this API exists to avoid.
            return exact
        hits = await self.recall(f"{subject} {attribute}", k=8)
        # Honour the bi-temporal window here too. The exact lookup applies it in
        # SQL, but this fallback previously ignored *as_of* entirely and happily
        # answered "what was true in year 1?" with a fact recorded in 2026 —
        # silently wrong in exactly the query time-travel exists to serve.
        hits = [h for h in hits if _as_utc(_effective_time(h)) <= _as_utc(when)]
        if not hits:
            return None
        live = [h for h in hits if not _is_superseded(h)] or hits
        # Relevance first, then valid time — NOT valid time alone. This asks
        # about one attribute, so the answer must be *about* it; ranking the
        # whole result set by time lets an unrelated fact recorded today outrank
        # the real answer. `recall` is relevance-ranked, so the head of `live` is
        # the topical set and valid time only breaks ties inside it.
        head = live[:_CURRENT_TOPICAL_WINDOW]
        chosen = max(head, key=_order_key(head))
        return chosen.content or None

    async def _current_by_tag(self, attribute: str, when: datetime) -> tuple[bool, str | None]:
        """Exact attribute lookup → ``(did_lookup, value)``.

        ``did_lookup`` is False only when the store cannot answer attributes at
        all (no ``by_tags``, or it errored) — that is the one case where the
        caller should fall back to retrieval. A successful lookup that matched
        nothing returns ``(True, None)``: an honest "no such fact".
        """
        by_tags = getattr(self._session.ltm, "by_tags", None)
        if by_tags is None:
            return False, None
        try:
            rows = await by_tags({"attribute": attribute}, as_of=when)
        except Exception:
            log.debug("attribute lookup failed for %r — falling back", attribute, exc_info=True)
            return False, None
        live = [r for r in rows if not _is_superseded(r)] or list(rows)
        if live:
            return True, (max(live, key=_order_key(live)).content or None)
        # Nothing carries this attribute. Whether that is an honest "no such
        # fact" or simply a corpus that never tags attributes decides between
        # answering None and falling back to retrieval — so ask.
        try:
            tagged = await by_tags({}, key="attribute", limit=1)
        except TypeError:
            return False, None  # store predates the `key` argument
        return bool(tagged), None

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
        items.sort(key=_order_key(items))
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


def _as_utc(dt: datetime) -> datetime:
    """Make *dt* timezone-aware (assuming UTC) so comparisons never raise.

    Callers mix both kinds: the stores return aware timestamps, while an ISO
    date parsed from a tool argument (``datetime.fromisoformat("2027-01-01")``)
    is naive. Comparing the two raises TypeError, which surfaces to the client
    as a failed tool call rather than an answer.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity; 0.0 when either vector is missing."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


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


def _order_key(items: list[MemoryItem]) -> Callable[[MemoryItem], datetime]:
    """Pick a comparable clock for *items*.

    Valid time and transaction time answer different questions, and mixing them
    silently is how a correction ends up ranked before the thing it corrects.
    So compare valid time only when **every** candidate actually states one;
    otherwise compare when we were told, which all of them have.
    """
    if items and all(
        i.valid_range is not None and i.valid_range.valid_from is not None for i in items
    ):
        return _effective_time
    return lambda i: _as_utc(i.created_at)
