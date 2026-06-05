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
import math
import os
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
    "When the remembered context below is relevant to the user's message, use "
    "it; otherwise answer normally. Do not invent memories."
)


# ── context formatting ────────────────────────────────────────────────────────


def format_context(ctx: ContextBundle | None, *, skip: str = "", limit: int = 12) -> str:
    """Render retrieved memory into a compact block for the system prompt."""
    items = list(getattr(ctx, "items", []) or []) if ctx is not None else []
    lines: list[str] = []
    for it in items:
        content = (it.content or "").strip()
        if not content or content == skip:
            continue
        role = str(it.metadata.get("role", it.tier.value if it.tier else "memory"))
        lines.append(f"- [{role}] {content}")
        if len(lines) >= limit:
            break
    if not lines:
        return "Remembered context: (nothing relevant yet)"
    return "Remembered context:\n" + "\n".join(lines)


# ── responders ─────────────────────────────────────────────────────────────────


def build_openrouter_responder(api_key: str, model: str) -> Any:
    """A ``Responder`` that calls OpenRouter with the retrieved context."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Genkryptos/Continuum",
        "X-Title": "Continuum chat",
    }

    async def respond(user_message: str, ctx: ContextBundle | None) -> str:
        system = f"{_SYSTEM_PREAMBLE}\n\n{format_context(ctx, skip=user_message)}"
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


async def _remember(info: dict[str, Any], session_id: str, text: str) -> str:
    """
    Decider-driven LTM write. Returns a short human note of what memory did:
    ADD (new), UPDATE (revise), DELETE (retire/retract a contradicted fact),
    or NOOP (already known). Best-effort — never raises into the REPL.
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
            await ltm.invalidate(tid)
            if decision.metadata.get("retraction"):
                # Retraction: the prior claim was never true → retire only,
                # do NOT store the bare negation.
                return f'[memory] DELETE (retraction) → retired "{old}"'
            # Non-retraction contradiction = supersession: retire the old fact
            # AND store the new one (single-turn stand-in for extract-then-add).
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

    # Responder selection.
    if have_key:
        responder = build_openrouter_responder(key, args.model)
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
    if ltm is not None:
        from continuum.core.config import PromoterConfig
        from continuum.promotion.mem0_promoter import Mem0Promoter

        decider_fn = build_decider_completion(key) if have_key else None
        decider = Mem0Promoter(PromoterConfig(llm_model=args.model), completion_fn=decider_fn)
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
    }


# ── REPL ───────────────────────────────────────────────────────────────────────


_HELP = """\
commands:
  /help              show this help
  /search <query>    search memory (hybrid) and print the top hits
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
