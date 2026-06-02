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
  • ``PostgresLTM``  — each user turn is embedded + indexed into ``memory_nodes``
    for cross-session semantic recall. (This is a deliberately simple "index
    every turn" write — production uses extraction + the promotion pipeline,
    not raw turns.)
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


async def _embed_one(embedder: Any, text: str) -> list[float] | None:
    if embedder is None:
        return None
    try:
        return list((await embedder.embed([text]))[0])
    except Exception:
        return None


def _build_components(cfg: ContinuumConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Construct stores + retriever + responder per the flags."""
    embedder = None
    if not args.no_embeddings and not args.mock:
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

    # Responder selection.
    if args.mock:
        responder = build_mock_responder()
        responder_label = "mock"
    else:
        env = {**load_dotenv(Path.cwd()), **os.environ}
        key = (env.get("OPENROUTER_API_KEY") or "").strip()
        if key and not key.endswith("..."):
            responder = build_openrouter_responder(key, args.model)
            responder_label = f"openrouter:{args.model}"
        else:
            responder = build_mock_responder()
            responder_label = "mock (no OPENROUTER_API_KEY found)"

    return {
        "embedder": embedder,
        "stm": stm,
        "ltm": ltm,
        "retriever": retriever,
        "responder": responder,
        "responder_label": responder_label,
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

            # Best-effort: index the user turn into LTM for cross-session recall.
            if info["ltm"] is not None:
                vec = await _embed_one(info["embedder"], line)
                try:
                    await info["ltm"].upsert(
                        MemoryItem(
                            content=line,
                            tier=MemoryTier.LTM,
                            embedding=vec,
                            session_id=session.session_id,
                            metadata={"role": "user", "source": "chat"},
                        )
                    )
                except Exception as exc:
                    print(f"  [note: LTM index skipped: {exc}]")


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
