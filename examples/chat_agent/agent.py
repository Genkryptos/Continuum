"""
examples/chat_agent/agent.py
============================
A minimal CLI chat agent that demonstrates Continuum's three pillars
without any infrastructure:

* **Tiered memory** — STM (raw turns), MTM (session summaries — stubbed
  here), LTM (extracted facts with supersession edges).
* **Supersession** — when a new fact contradicts an existing one on
  the same (entity, attribute), the old fact is marked
  ``superseded_by`` the new one. Retrieval filters to current facts.
* **Inspectable state** — slash-commands let the user dump each tier
  and run a retrieval query against the wiki-style synthesis layer.

This demo is deliberately self-contained: in-memory stores, mock LLM,
no Postgres, no API keys. It mirrors Continuum's *architecture*, not
its production deployment — every concept (STM/MTM/LTM tiering,
supersession via ``superseded_by`` edges, fact extraction at ingest)
maps one-to-one to the framework's real components.

Run
---
::

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
        -m examples.chat_agent.agent

Then type chat messages, or use the slash commands:

* ``/show stm``  — print recent turns
* ``/show ltm``  — print current (non-superseded) facts
* ``/show ltm --all`` — print every fact, including superseded ones
* ``/query <text>`` — run a retrieval against the current LTM
* ``/help`` / ``/exit``

Scripted demo: see ``examples/chat_agent/demo.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("chat_agent")


# ---------------------------------------------------------------------------
# Memory model — in-memory simulation of Continuum's tiered architecture
# ---------------------------------------------------------------------------


@dataclass
class _STMTurn:
    role: str
    content: str
    timestamp: dt.datetime


@dataclass
class _Fact:
    id: str
    text: str
    attribute: str            # e.g. "user.location"
    value: str                # e.g. "Boston"
    valid_from: dt.datetime
    recorded_at: dt.datetime
    superseded_by: str | None = None
    raw_turn: str = ""


@dataclass
class DemoMemory:
    """
    The full demo memory state. Maps to Continuum like this:

    * ``stm``  → :class:`continuum.stores.stm.InMemorySTM` content
    * ``ltm``  → :class:`continuum.stores.postgres.PostgresLTM` rows
                 (in real Continuum, with the ``superseded_by`` column
                 from migration 004 doing what we do in
                 :meth:`_apply_supersession`)
    """

    stm: list[_STMTurn] = field(default_factory=list)
    ltm: list[_Fact] = field(default_factory=list)

    # ── STM ────────────────────────────────────────────────────────────
    def append_turn(self, role: str, content: str) -> None:
        self.stm.append(_STMTurn(
            role=role, content=content,
            timestamp=dt.datetime.now(dt.UTC),
        ))

    # ── LTM with supersession ─────────────────────────────────────────
    def add_fact(self, attribute: str, value: str, source_turn: str) -> _Fact:
        now = dt.datetime.now(dt.UTC)
        fact = _Fact(
            id=str(uuid.uuid4())[:8],
            text=f"{attribute} = {value}",
            attribute=attribute,
            value=value,
            valid_from=now,
            recorded_at=now,
            raw_turn=source_turn,
        )
        self._apply_supersession(fact)
        self.ltm.append(fact)
        return fact

    def _apply_supersession(self, new_fact: _Fact) -> None:
        """Mark any existing live fact on the same attribute as superseded."""
        for f in self.ltm:
            if f.attribute == new_fact.attribute and f.superseded_by is None:
                f.superseded_by = new_fact.id

    def current_facts(self) -> list[_Fact]:
        return [f for f in self.ltm if f.superseded_by is None]

    def all_facts(self) -> list[_Fact]:
        return list(self.ltm)

    # ── Retrieval ─────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 4) -> list[_Fact]:
        """
        Trivial substring-overlap retrieval over *current* facts only.
        Production Continuum uses cosine similarity on a sentence-
        transformer embedder + a composite scorer; the *filter* — only
        non-superseded facts — is what this demo is showing.
        """
        q = query.lower()
        scored = []
        for f in self.current_facts():
            score = sum(1 for tok in q.split() if tok in f.text.lower())
            scored.append((score, f))
        scored.sort(key=lambda x: -x[0])
        return [f for s, f in scored[:k] if s > 0] or self.current_facts()[:k]


# ---------------------------------------------------------------------------
# Fact extraction — rule-based stand-in for Continuum's LLM extractors.
# Mirrors what continuum.extraction.FactExtractor + LLMEntityExtractor
# would produce on the same turn, deterministically + cheaply.
# ---------------------------------------------------------------------------


#: All rules are case-insensitive — natural input is "I live in NYC",
#: not "i live in nyc". Each rule extracts ONE value-bearing capture.
_RULE_FLAGS = re.IGNORECASE
_RULES: list[tuple[re.Pattern[str], str]] = [
    # location: "I live in NYC", "I moved to Boston", "I'm in Seattle",
    # "moved to Boston for a new job"
    (re.compile(
        r"\b(?:i\s+(?:live|moved|relocated)\s+(?:in|to)|"
        r"i'?m\s+(?:in|from)|"
        r"moved\s+to|relocated\s+to)\s+"
        r"([A-Z][A-Za-z\- ]*?[A-Za-z])(?=\s+(?:for|with|to|and|but|after|"
        r"because|since|this|last|next|now|—|-|in|on|via)|[.,!;?]|$)",
        _RULE_FLAGS,
    ), "user.location"),
    # dog name: "my dog Rex", "dog named Rex", "puppy Luna", "adopted a dog named Rex"
    (re.compile(
        r"\b(?:my\s+dog'?s?\s+name\s+is|my\s+dog\s+is\s+called|"
        r"(?:our|my|the)\s+(?:new\s+)?(?:dog|puppy)\s+(?:named\s+|called\s+)?|"
        r"adopted\s+a\s+(?:dog|puppy)\s+(?:named|called)\s+)"
        r"([A-Z][A-Za-z\-]+)",
        _RULE_FLAGS,
    ), "user.pets.dog"),
    # employer: "I work at Acme", "joined Globex", "new job at Foo"
    (re.compile(
        r"\b(?:i\s+work\s+(?:at|for)|my\s+employer\s+is|"
        r"i\s+joined|joined|new\s+job\s+at)\s+"
        r"([A-Z][A-Za-z0-9\- ]*?[A-Za-z0-9])(?=\s+(?:as|in|for|on|—|-)|"
        r"[.,!;?]|$)",
        _RULE_FLAGS,
    ), "user.employer"),
    # vehicle
    (re.compile(
        r"\b(?:i\s+drive\s+a|my\s+car\s+is\s+a|switched\s+to\s+a|"
        r"upgraded\s+to\s+a)\s+"
        r"([A-Z][A-Za-z0-9\- ]*?[A-Za-z0-9])(?=[.,!;?]|$)",
        _RULE_FLAGS,
    ), "user.vehicle"),
    # marital status — the matched word itself is the value
    (re.compile(
        r"\bi'?m\s+(single|married|divorced|engaged|widowed)\b",
        _RULE_FLAGS,
    ), "user.marital_status"),
]


def extract_facts(text: str) -> list[tuple[str, str]]:
    """Return ``[(attribute, value), …]`` mined from a single turn."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pattern, attr in _RULES:
        for m in pattern.finditer(text):
            # Every rule's value lives in capture group 1.
            value = m.group(1).strip().rstrip(".,!;?").strip()
            if not value:
                continue
            # Normalise marital-status case so supersession compares cleanly.
            if attr == "user.marital_status":
                value = value.lower()
            key = (attr, value.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((attr, value))
    return out


# ---------------------------------------------------------------------------
# Responder — canned (default) or OpenAI (--llm openai if key is set)
# ---------------------------------------------------------------------------


async def _canned_responder(user_msg: str, ctx_facts: list[_Fact]) -> str:
    if ctx_facts:
        cur = "; ".join(f.text for f in ctx_facts[:3])
        return f"Got it. (recalled: {cur})"
    return "Got it."


async def _openai_responder(user_msg: str, ctx_facts: list[_Fact]) -> str:
    import os

    import httpx
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return await _canned_responder(user_msg, ctx_facts)
    ctx_text = "\n".join(f"- {f.text}" for f in ctx_facts) or "(none)"
    prompt = (
        "You are a helpful assistant with persistent memory. "
        "Use the recalled facts when relevant, but stay conversational.\n\n"
        f"Recalled facts:\n{ctx_text}\n\nUser: {user_msg}\nAssistant:"
    )
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0.0,
            },
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# CLI / REPL
# ---------------------------------------------------------------------------


_HELP = """\
Slash commands:
  /help                 — this message
  /show stm             — recent turns
  /show ltm             — current (non-superseded) facts
  /show ltm --all       — all facts including superseded
  /query <text>         — retrieve current facts matching <text>
  /exit                 — quit
Anything else is treated as a user turn.
"""


def _fmt_fact(f: _Fact) -> str:
    sup = f"  ← superseded by {f.superseded_by}" if f.superseded_by else ""
    age = (dt.datetime.now(dt.UTC) - f.valid_from).total_seconds()
    return (
        f"  [{f.id}] {f.text:<40s} "
        f"valid_from={f.valid_from.strftime('%H:%M:%S')} "
        f"({age:.0f}s ago){sup}"
    )


def _fmt_turn(t: _STMTurn) -> str:
    return (
        f"  [{t.timestamp.strftime('%H:%M:%S')}] "
        f"{t.role:<9s} {t.content}"
    )


async def _handle_command(memory: DemoMemory, cmd: str) -> bool:
    """Return True if cmd was a slash command, False otherwise."""
    if not cmd.startswith("/"):
        return False
    parts = cmd.strip().split()
    head = parts[0]
    if head in ("/help", "/?"):
        print(_HELP)
    elif head == "/exit" or head == "/quit":
        print("bye.")
        raise SystemExit(0)
    elif head == "/show" and len(parts) >= 2 and parts[1] == "stm":
        if not memory.stm:
            print("  (STM empty)")
        for t in memory.stm[-10:]:
            print(_fmt_turn(t))
    elif head == "/show" and len(parts) >= 2 and parts[1] == "ltm":
        show_all = len(parts) >= 3 and parts[2] == "--all"
        facts = memory.all_facts() if show_all else memory.current_facts()
        label = "ALL FACTS" if show_all else "CURRENT FACTS"
        print(f"  {label} (n={len(facts)})")
        if not facts:
            print("  (none)")
        for f in facts:
            print(_fmt_fact(f))
    elif head == "/query" and len(parts) >= 2:
        q = cmd.split(" ", 1)[1]
        hits = memory.retrieve(q, k=4)
        print(f"  Retrieval for {q!r} — top {len(hits)} of "
              f"{len(memory.current_facts())} current facts")
        if not hits:
            print("  (no hits)")
        for f in hits:
            print(_fmt_fact(f))
    else:
        print(f"  unknown command: {head}; try /help")
    return True


async def _process_turn(
    memory: DemoMemory, user_msg: str, responder,
) -> None:
    memory.append_turn("user", user_msg)
    facts_extracted = extract_facts(user_msg)
    for attr, value in facts_extracted:
        f = memory.add_fact(attr, value, user_msg)
        sup = (" (superseded prior)" if any(
            x.superseded_by == f.id for x in memory.ltm) else "")
        print(f"  → extracted {f.text}{sup}")
    ctx = memory.retrieve(user_msg, k=4)
    reply = await responder(user_msg, ctx)
    memory.append_turn("assistant", reply)
    print(f"  agent> {reply}")


def _select_responder(name: str):
    if name == "openai":
        return _openai_responder
    return _canned_responder


async def _amain(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm", choices=["mock", "openai"], default="mock",
                   help="Response generator. 'mock' (default) is "
                        "deterministic + zero-cost; 'openai' uses "
                        "gpt-4o-mini if OPENAI_API_KEY is set.")
    p.add_argument("--script", type=Path, default=None,
                   help="Read input lines from this file instead of "
                        "stdin (used by demo.sh). One command/turn per line.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    memory = DemoMemory()
    responder = _select_responder(args.llm)

    print()
    print("=" * 64)
    print("  Continuum chat-agent demo")
    print("  responder: " + args.llm)
    print("  type /help for commands, /exit to quit")
    print("=" * 64)

    # Either replay a script or do an interactive REPL.
    lines: list[str]
    if args.script is not None:
        lines = [
            ln.rstrip("\n") for ln in args.script.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        for ln in lines:
            print(f"\nyou> {ln}")
            try:
                if not await _handle_command(memory, ln):
                    await _process_turn(memory, ln, responder)
            except SystemExit:
                return 0
        return 0

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not line:
            continue
        try:
            if not await _handle_command(memory, line):
                await _process_turn(memory, line, responder)
        except SystemExit:
            return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(list(argv) if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "DemoMemory", "extract_facts"]
