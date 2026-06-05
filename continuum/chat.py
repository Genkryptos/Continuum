"""
continuum.chat
==============
An interactive, **Postgres-backed** chat REPL on top of a real
:class:`ContinuumSession` — the genuine production stack, not the in-memory
teaching demo under ``examples/chat_agent``. Use it to drive the memory layer
by hand after ``make db-up && make db-migrate``::

    python -m continuum.chat                       # full Postgres + OpenRouter
    python -m continuum.chat --session alice        # a named, persistent session
    python -m continuum.chat --in-memory            # no DB (quick smoke)
    python -m continuum.chat --mock                 # no LLM call (deterministic)
    python -m continuum.chat --no-embeddings        # skip the embedder download

What it wires (the real components):
  • ``PostgresSTM``  — every turn persists to ``stm_messages`` and survives
    restarts (re-run with the same ``--session`` and it remembers).
  • ``PostgresLTM``  — each user turn is routed through the **Mem0 decider**
    (ADD / UPDATE / DELETE / NOOP) against its nearest LTM neighbors, so you can
    SEE supersession and retraction happen: "I moved to Y" retires the old
    location and stores the new; "I was never in Y" *retracts* — retires the
    contradicted memory without storing the negation. (Light stand-in for the
    full pipeline: one turn = one fact, no entity/atomic-fact extraction.)
  • ``Retriever``    — the full hybrid pipeline (LTM dense+sparse ⊕ STM recency),
    with the query embedded on the way in.
  • responder        — OpenRouter (your ``OPENROUTER_API_KEY``), default model
    ``openai/gpt-4o-mini``; override with ``--model``.

Commands inside the REPL: ``/help``, ``/search <q>``, ``/stats``,
``/session <id>``, ``/exit``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from continuum.core.config import ContinuumConfig
from continuum.core.session import ContinuumSession
from continuum.core.types import ContextBundle, MemoryItem, MemoryTier, Query
from continuum.doctor import load_dotenv

DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_SYSTEM_PREAMBLE = (
    "You are Continuum, a helpful assistant with persistent long-term memory. "
    "Use the remembered context below when relevant; do not invent memories. "
    "IMPORTANT: when the RECENT CONVERSATION conflicts with CURRENT MEMORY, "
    "trust CURRENT MEMORY — the user may have corrected or retracted earlier "
    "statements. Treat EARLIER (PAST) facts as no longer true now; use them "
    "only to answer questions about the past (e.g. 'where did I used to live')."
)


# ── context formatting ────────────────────────────────────────────────────────


def format_context(ctx: ContextBundle | None, *, skip: str = "", limit: int = 12) -> str:
    """
    Render retrieved memory for the system prompt, partitioned into CURRENT
    MEMORY (live LTM facts — authoritative) vs RECENT CONVERSATION (raw STM
    turns, which may contain statements the user later changed or retracted).
    The split lets the reader prefer current structured memory over stale chat.
    """
    items = list(getattr(ctx, "items", []) or []) if ctx is not None else []
    current: list[str] = []
    recent: list[str] = []
    for it in items:
        content = (it.content or "").strip()
        if not content or content == skip:
            continue
        if it.tier == MemoryTier.LTM:
            current.append(f"  • {content}")
        else:
            role = str(it.metadata.get("role", it.tier.value if it.tier else "msg"))
            recent.append(f"  - [{role}] {content}")

    blocks: list[str] = []
    if current:
        blocks.append(
            "CURRENT MEMORY (the user's latest known facts — authoritative):\n"
            + "\n".join(current[:limit])
        )
    if recent:
        blocks.append("RECENT CONVERSATION:\n" + "\n".join(recent[:limit]))
    return "\n\n".join(blocks) if blocks else "Remembered context: (nothing relevant yet)"


# ── responders ─────────────────────────────────────────────────────────────────


def build_openrouter_responder(api_key: str, model: str, history_fn: Any = None) -> Any:
    """
    A ``Responder`` that calls OpenRouter with the retrieved context.

    ``history_fn`` (optional ``async () -> list[str]``) supplies *superseded but
    not retracted* past facts — valid past states the user has since moved on
    from — so the reader can answer "where did I used to live?" / "which city
    did I move from?" without resurrecting them as current truth.
    """
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Genkryptos/Continuum",
        "X-Title": "Continuum chat",
    }

    async def respond(user_message: str, ctx: ContextBundle | None) -> str:
        system = f"{_SYSTEM_PREAMBLE}\n\n{format_context(ctx, skip=user_message)}"
        if history_fn is not None:
            try:
                past, retracted = await history_fn()
            except Exception:
                past, retracted = [], []
            if past:
                system += (
                    "\n\nEARLIER (PAST) facts — true before, now superseded "
                    "(use ONLY for questions about the past, e.g. a previous city):\n"
                    + "\n".join(f"  · {p}" for p in past)
                )
            if retracted:
                system += (
                    "\n\nRETRACTED — the user said these NEVER happened. Treat them "
                    "as false and IGNORE them entirely; never name a retracted place "
                    "as somewhere the user lived, visited, or moved from:\n"
                    + "\n".join(f"  ✗ {r}" for r in retracted)
                )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
        }
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except httpx.HTTPStatusError as exc:
            return f"[openrouter error {exc.response.status_code}: {exc.response.text[:200]}]"
        except Exception as exc:  # network, JSON shape, …
            return f"[openrouter call failed: {exc}]"

    return respond


def build_mock_responder() -> Any:
    """Deterministic, zero-cost responder — echoes what it remembered."""

    async def respond(user_message: str, ctx: ContextBundle | None) -> str:
        n = len(getattr(ctx, "items", []) or []) if ctx is not None else 0
        return f"(mock) heard: {user_message!r} · recalled {n} memory item(s)"

    return respond


def build_decider_completion(api_key: str) -> Any:
    """
    A litellm-style ``completion_fn`` for the Mem0 decider, backed by OpenRouter.

    The decider calls it with ``model/messages/tools/tool_choice`` and reads
    ``resp["choices"][0]["message"]["tool_calls"]`` — OpenRouter's chat API is
    OpenAI-shaped, so returning the raw JSON dict is exactly what it expects.
    """
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Genkryptos/Continuum",
        "X-Title": "Continuum chat",
    }

    async def complete(
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **_: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    return complete


_FACT_EXTRACT_SYS = (
    "Extract atomic, self-contained facts about the USER from their message. "
    "Rewrite each as a short third-person statement, e.g. 'User lives in Paris', "
    "'User's name is Sam', 'User never visited Tokyo'. Resolve pronouns and "
    "normalize verbs ('I moved to X' -> 'User lives in X'). PRESERVE negations "
    "and retractions exactly ('User was never in X', 'User never went to X'). "
    "Output one fact per line, no bullets or numbering. If the message is a "
    "question or contains no durable fact about the user, output exactly: NONE"
)


def build_fact_splitter(api_key: str, model: str) -> Any:
    """
    ``async (text) -> list[str] | None`` — LLM atomic-fact extraction.

    Returns the extracted facts (``[]`` if the turn is a question / has none),
    or ``None`` on error so the caller falls back to the regex clause split.
    Normalizing to "User <verb> X" form is what makes multi-fact turns split,
    the decider's cosine matching reliable, and questions not get stored.
    """
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Genkryptos/Continuum",
        "X-Title": "Continuum chat",
    }

    async def split(text: str) -> list[str] | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _FACT_EXTRACT_SYS},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
                r.raise_for_status()
                out = str(r.json()["choices"][0]["message"]["content"]).strip()
        except Exception:
            return None  # → caller falls back to regex split
        if not out or out.upper().strip(" .") == "NONE":
            return []
        facts = [
            ln.strip(" -•\t0123456789.")
            for ln in out.splitlines()
            if ln.strip() and ln.strip().upper().strip(" .") != "NONE"
        ]
        return [f for f in facts if len(f.split()) >= 2]

    return split


# ── embedding-aware retriever wrapper ──────────────────────────────────────────


class EmbeddingRetriever:
    """Embeds ``query.text`` before delegating, so the LTM dense channel fires."""

    def __init__(self, inner: Any, embedder: Any) -> None:
        self._inner = inner
        self._embedder = embedder

    async def retrieve(self, query: Query, budget: Any) -> ContextBundle:
        if self._embedder is not None and query.embedding is None and query.text:
            try:
                vec = (await self._embedder.embed([query.text]))[0]
                query = replace(query, embedding=vec)
            except Exception:
                pass  # degrade to sparse-only retrieval
        bundle: ContextBundle = await self._inner.retrieve(query, budget)
        return bundle


# ── component wiring ───────────────────────────────────────────────────────────


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _neighbor_text(neighbors: list[Any], tid: Any) -> str:
    for si in neighbors:
        if str(si.item.id) == str(tid):
            return (si.item.content or "").strip()
    return str(tid)


_CLAUSE_SPLIT_RE = re.compile(r"(?i)(?:[.!?;]+|\bactually\b|\bbut\b|\bhowever\b)")


def _split_clauses(text: str) -> list[str]:
    """
    Split a turn into clauses so a combined statement is handled as multiple
    facts — e.g. "I'm in Bhilai, actually I never went to Bangalore" becomes
    ["I'm in Bhilai", "I never went to Bangalore"], letting the first ADD and
    the second retract. Conservative (sentence punctuation + a few discourse
    markers); short fragments are dropped. Light stand-in for atomic-fact
    extraction — falls back to the whole text if nothing splits cleanly.
    """
    parts = [p.strip(" ,.;:-") for p in _CLAUSE_SPLIT_RE.split(text)]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts or [text.strip()]


async def _remember(info: dict[str, Any], session_id: str, text: str) -> str:
    """
    Break the turn into atomic facts and run each through the decider; join
    notes. Prefers LLM extraction (info['fact_splitter']) — which normalizes
    phrasing, separates multi-fact turns, and drops questions — and falls back
    to the regex clause split when there's no LLM or it errors.
    """
    facts: list[str] | None = None
    splitter = info.get("fact_splitter")
    if splitter is not None:
        facts = await splitter(text)  # [] = no durable fact; None = error → fallback
    if facts is None:
        facts = _split_clauses(text)
    if not facts:
        return ""  # nothing worth storing (e.g. a question)
    notes = [n for c in facts if (n := await _remember_one(info, session_id, c))]
    return "  ".join(notes)


async def _remember_one(info: dict[str, Any], session_id: str, text: str) -> str:
    """
    Decider-driven LTM write for one clause. Returns a short human note of what
    memory did: ADD (new), UPDATE (revise), SUPERSEDE / DELETE-retraction
    (retire a contradicted fact), or NOOP. Best-effort — never raises.
    """
    import uuid as _uuid

    from continuum.extraction.fact_extractor import Fact

    ltm, embedder, decider = info["ltm"], info["embedder"], info["decider"]
    vec = await _embed_one(embedder, text)

    neighbors: list[Any] = []
    try:
        q = Query(text=text, embedding=vec, top_k=8,
                  tiers=[MemoryTier.LTM], session_id=session_id)
        neighbors = list(await ltm.search_hybrid(q, 8))
    except Exception:
        pass

    # search_hybrid scores neighbors by RRF (~0.03), but the decider's
    # ADD/NOOP/retraction thresholds are calibrated for cosine. Re-score the
    # top neighbors by true cosine (query vec ⊕ neighbor embedding) so those
    # thresholds mean what they should. Needs embeddings on (`make run-full`).
    if vec is not None and neighbors:
        from continuum.core.types import ScoreBreakdown, ScoredItem

        rescored: list[Any] = []
        for si in neighbors[:6]:
            nvec = await _embed_one(embedder, si.item.content or "")
            cos = _cosine(vec, nvec)
            rescored.append(
                ScoredItem(
                    item=si.item,
                    scores=ScoreBreakdown(
                        relevance=cos, importance=0.0, recency=0.0,
                        confidence=1.0, composite=cos,
                    ),
                )
            )
        neighbors = rescored

    def _node(content: str, embedding: list[float] | None) -> MemoryItem:
        return MemoryItem(
            content=content, tier=MemoryTier.LTM, embedding=embedding,
            session_id=session_id, metadata={"role": "user", "source": "chat"},
        )

    try:
        fact = Fact(text=text, confidence=0.9, entities_mentioned=[],
                    source_block_id=_uuid.uuid4())
        decision = await decider.decide_operation(fact, neighbors)
    except Exception:
        # Decider unavailable → fall back to a plain ADD so memory still accrues.
        try:
            await ltm.upsert(_node(text, vec))
            return "[memory] ADD (fallback)"
        except Exception as exc:
            return f"[memory] skipped: {exc}"

    op, tid = decision.op, decision.target_id
    try:
        if op == "ADD":
            await ltm.upsert(_node(text, vec))
            return "[memory] ADD → stored new memory"
        if op == "UPDATE" and tid is not None:
            merged = decision.merged_text or text
            await ltm.update(tid, {"text": merged, "embedding": await _embed_one(embedder, merged)})
            return f'[memory] UPDATE → revised "{_neighbor_text(neighbors, tid)[:60]}"'
        if op == "DELETE" and tid is not None:
            old = _neighbor_text(neighbors, tid)[:60]
            if decision.metadata.get("retraction"):
                # Retraction: the prior claim was never true. Mark it erroneous
                # so /history and the reader EXCLUDE it (a retracted fact is not
                # a valid past state), then retire it. The bare negation is not
                # stored.
                with contextlib.suppress(Exception):
                    await ltm.update(tid, {"tags": {"retracted": True}})
                await ltm.invalidate(tid)
                return f'[memory] DELETE (retraction) → retired "{old}"'
            # Non-retraction contradiction = supersession: retire the old fact
            # (kept as a valid PAST state for history) AND store the new one.
            await ltm.invalidate(tid)
            await ltm.upsert(_node(text, vec))
            return f'[memory] SUPERSEDE → retired "{old}", stored new'
        # An ambiguous-band decision needs the LLM tie-breaker. Without one
        # (mock / no key), the decider degrades to NOOP — but that silently
        # drops the turn, so store it instead (don't lose data offline).
        if op == "NOOP" and not decision.short_circuited and not info.get("decider_llm"):
            await ltm.upsert(_node(text, vec))
            return "[memory] ADD → stored new memory (no LLM tie-breaker)"
        return "[memory] NOOP → already known / nothing to change"
    except Exception as exc:
        return f"[memory] {op} skipped: {exc}"


def _fetch_superseded_sync(dsn: str, *, include_retracted: bool, limit: int) -> list[str]:
    """Invalidated LTM facts. By default excludes retracted ones (never-true)."""
    import psycopg

    where = "layer='LTM' AND invalidated_at IS NOT NULL"
    if not include_retracted:
        where += " AND COALESCE(tags->>'retracted','') <> 'true'"
    out: list[str] = []
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            rows = conn.execute(
                f'SELECT "text" FROM memory_nodes WHERE {where} '
                f"ORDER BY invalidated_at DESC LIMIT {int(limit)}"
            ).fetchall()
            out = [str(r[0]) for r in rows]
    except Exception:
        pass
    return out


def make_history_fn(dsn: str, *, limit: int = 6) -> Any:
    """
    ``async () -> tuple[past, retracted]`` where *past* are superseded-but-valid
    past states (e.g. a former city) and *retracted* are facts the user said
    were never true (so the reader can be told to ignore them outright).
    """

    async def history() -> tuple[list[str], list[str]]:
        all_gone = await asyncio.to_thread(
            _fetch_superseded_sync, dsn, include_retracted=True, limit=limit * 3
        )
        past = await asyncio.to_thread(
            _fetch_superseded_sync, dsn, include_retracted=False, limit=limit
        )
        retracted = [g for g in all_gone if g not in past][:limit]
        return past, retracted

    return history


async def _embed_one(embedder: Any, text: str) -> list[float] | None:
    if embedder is None:
        return None
    try:
        return list((await embedder.embed([text]))[0])
    except Exception:
        return None


def _build_components(cfg: ContinuumConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Construct stores + retriever + responder per the flags."""
    # The embedder is a *local* model (no network), independent of the
    # responder — so --mock can still use dense embeddings (retraction/
    # supersession matching needs them). Only --no-embeddings turns it off.
    embedder = None
    if not args.no_embeddings:
        from continuum.embeddings import EmbeddingService

        embedder = EmbeddingService(cfg.embedding)

    stm: Any
    ltm: Any = None
    retriever: Any = None

    if args.in_memory:
        from continuum.stores.stm import InMemorySTM

        stm = InMemorySTM()
    else:
        from continuum.retrieval import Retriever
        from continuum.stores.postgres.ltm import PostgresLTM
        from continuum.stores.stm.postgres_stm import PostgresSTM

        dsn = str(cfg.database.dsn)
        stm = PostgresSTM(dsn=dsn)
        ltm = PostgresLTM(dsn=dsn)
        inner = Retriever(ltm=ltm, stm=stm, session_id=args.session)
        retriever = EmbeddingRetriever(inner, embedder)

    # Resolve the OpenRouter key once (used by both the responder and the
    # memory decider). .env wins nothing over the shell; shell wins.
    env = {**load_dotenv(Path.cwd()), **os.environ}
    key = (env.get("OPENROUTER_API_KEY") or "").strip()
    have_key = bool(key) and not key.endswith("...") and not args.mock

    # History provider (superseded-but-not-retracted past facts) — Postgres only.
    history_fn = make_history_fn(str(cfg.database.dsn)) if ltm is not None else None

    # Responder selection.
    if have_key:
        responder = build_openrouter_responder(key, args.model, history_fn)
        responder_label = f"openrouter:{args.model}"
    else:
        responder = build_mock_responder()
        responder_label = "mock" if args.mock else "mock (no OPENROUTER_API_KEY found)"

    # Memory decider (Mem0 ADD/UPDATE/DELETE/NOOP) — only on the Postgres path.
    # Routes each turn's LTM write through conflict resolution + retraction
    # instead of a blind upsert. The LLM tie-breaker (for the ambiguous
    # similarity band) uses OpenRouter when a key is present; without one it
    # degrades to the deterministic ops (ADD / NOOP / retraction-DELETE).
    decider = None
    fact_splitter = None
    if ltm is not None:
        from continuum.core.config import PromoterConfig
        from continuum.promotion.mem0_promoter import Mem0Promoter

        decider_fn = build_decider_completion(key) if have_key else None
        decider = Mem0Promoter(PromoterConfig(llm_model=args.model), completion_fn=decider_fn)
        fact_splitter = build_fact_splitter(key, args.model) if have_key else None
        if not have_key:
            # No tie-breaker LLM → the decider logs an expected failure for
            # ambiguous-band turns (we ADD-fallback those). Hush that noise.
            import logging

            logging.getLogger("continuum.promotion.mem0_promoter").setLevel(logging.CRITICAL)

    return {
        "embedder": embedder,
        "stm": stm,
        "ltm": ltm,
        "retriever": retriever,
        "responder": responder,
        "responder_label": responder_label,
        "decider": decider,
        "decider_llm": have_key,
        "fact_splitter": fact_splitter,
        "dsn": str(cfg.database.dsn) if ltm is not None else None,
    }


# ── REPL ───────────────────────────────────────────────────────────────────────


_HELP = """\
commands:
  /help              show this help
  /search <query>    search current memory (hybrid) and print the top hits
  /history           show retired memory — superseded (past) + retracted (never-true)
  /stats             show session + backend info
  /session <id>      switch to a different (persistent) session
  /exit              quit
anything else is sent as a chat turn (persisted + answered with memory).
"""


async def _handle_command(line: str, session: ContinuumSession, info: dict[str, Any]) -> bool:
    """Return True if the line was a command (handled), False otherwise."""
    if not line.startswith("/"):
        return False
    head, _, rest = line.partition(" ")
    rest = rest.strip()
    if head in ("/help", "/?"):
        print(_HELP)
    elif head in ("/exit", "/quit"):
        raise SystemExit(0)
    elif head == "/stats":
        backend = "in-memory" if info["ltm"] is None else "postgres"
        print(f"  session   : {session.session_id}")
        print(f"  backend   : {backend}")
        print(f"  responder : {info['responder_label']}")
        print(f"  embedder  : {'off' if info['embedder'] is None else info['embedder'].config.model_name}")
        if info.get("decider") is not None:
            tie = "LLM" if info["responder_label"].startswith("openrouter") else "deterministic-only"
            print(f"  memory    : Mem0 decider (ADD/UPDATE/DELETE/NOOP; tie-break={tie})")
    elif head == "/search":
        if not rest:
            print("  usage: /search <query>")
        else:
            hits = await session.search(rest, k=8)
            if not hits:
                print("  (no matches)")
            for h in hits:
                role = h.metadata.get("role", h.tier.value if h.tier else "?")
                print(f"  · [{role}] {(h.content or '').strip()[:140]}")
    elif head == "/history":
        dsn = info.get("dsn")
        if not dsn:
            print("  (history needs the Postgres backend)")
        else:
            past = await asyncio.to_thread(
                _fetch_superseded_sync, dsn, include_retracted=False, limit=20
            )
            gone = await asyncio.to_thread(
                _fetch_superseded_sync, dsn, include_retracted=True, limit=40
            )
            retracted = [g for g in gone if g not in past]
            if not past and not retracted:
                print("  (no retired memory yet)")
            for p in past:
                print(f"  · past (superseded): {p[:120]}")
            for r in retracted:
                print(f"  ✗ retracted (never true): {r[:120]}")
    elif head == "/session":
        if not rest:
            print("  usage: /session <id>")
        else:
            session.session_id = rest
            print(f"  switched to session '{rest}'")
    else:
        print(f"  unknown command: {head} (try /help)")
    return True


async def _amain(argv: list[str]) -> int:
    args = _parse_args(argv)
    cfg = ContinuumConfig.load()
    info = _build_components(cfg, args)

    session = ContinuumSession(
        cfg,
        stm=info["stm"],
        ltm=info["ltm"],
        retriever=info["retriever"],
        responder=info["responder"],
        session_id=args.session,
    )

    print("=" * 64)
    print("  Continuum chat  —  real ContinuumSession")
    print(f"  session: {args.session}   responder: {info['responder_label']}")
    print(f"  backend: {'in-memory' if info['ltm'] is None else f'postgres ({cfg.database.dsn})'}")
    print("  type /help for commands, /exit to quit")
    if info["embedder"] is not None:
        print("  (first turn may pause while the embedder model loads)")
    print("=" * 64)

    async with session:
        while True:
            try:
                line = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                return 0
            if not line:
                continue
            try:
                if await _handle_command(line, session, info):
                    continue
            except SystemExit:
                print("bye.")
                return 0

            reply = await session.process_turn(line)
            print(f"\nbot> {reply}")

            # Route the turn's LTM write through the memory decider so the user
            # can SEE supersession / retraction happen (not a blind upsert).
            if info["ltm"] is not None and info["decider"] is not None:
                note = await _remember(info, session.session_id, line)
                if note:
                    print(f"  {note}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m continuum.chat",
        description="Interactive Postgres-backed chat REPL on a real ContinuumSession.",
    )
    p.add_argument("--session", default="default", help="persistent session id (default: 'default')")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    p.add_argument("--in-memory", action="store_true", help="use InMemorySTM, no Postgres")
    p.add_argument("--mock", action="store_true", help="deterministic responder, no LLM call")
    p.add_argument("--no-embeddings", action="store_true",
                   help="skip the embedder (LTM dense channel off; sparse + STM still work)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(list(argv) if argv is not None else sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "main",
    "format_context",
    "build_openrouter_responder",
    "build_mock_responder",
    "EmbeddingRetriever",
]
