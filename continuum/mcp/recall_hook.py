"""
continuum.mcp.recall_hook
=========================
A Claude Code ``UserPromptSubmit`` hook that makes memory **automatic**: it
recalls against the actual prompt and injects the top hits as context, so memory
fires every turn instead of only when Claude thinks to call ``recall``.

    python -m continuum.mcp.recall_hook      # reads the hook JSON on stdin

Design constraints, all load-bearing:

* **Never block or slow the prompt.** The hook runs synchronously before Claude
  sees the turn. Any failure — no DB, bad JSON, timeout — prints nothing and
  exits 0, so the prompt proceeds untouched.
* **Fresh process per prompt**, so it must not load the ~2.3GB embedder (that is
  ~7s every turn). It uses **sparse** recall by default (no model). Point it at
  the always-on HTTP server for dense recall later; for now sparse is what keeps
  it sub-second.
* **Scoped.** Honours ``CONTINUUM_MCP_NAMESPACE`` like the server, so it never
  injects another tenant's memories.

Environment:
  CONTINUUM_DB_DSN            required — no DSN, nothing to recall, silent no-op.
  CONTINUUM_MCP_NAMESPACE     tenant scope (default 'default').
  CONTINUUM_RECALL_HOOK_K     how many memories to inject (default 5).
  CONTINUUM_RECALL_HOOK_EMBEDDINGS  '1' to use the embedder (slow; needs a warm
                             process — off by default).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

__all__ = ["build_context", "main"]

_HEADER = "Relevant memories (Continuum) — use if pertinent, ignore otherwise:"


def build_context(prompt: str, memories: list[str]) -> str:
    """The additionalContext block, or '' when there is nothing worth injecting."""
    seen: set[str] = set()
    facts: list[str] = []
    for m in memories:  # dedup, preserving rank order (recall returns LTM+STM copies)
        f = (m or "").strip()
        if f and f.lower() not in seen:
            seen.add(f.lower())
            facts.append(f)
    if not prompt.strip() or not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts)
    return f"{_HEADER}\n{lines}"


def _read_prompt(stdin_text: str) -> str:
    try:
        payload = json.loads(stdin_text)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    # UserPromptSubmit delivers the text as `prompt`; tolerate a couple of aliases.
    for key in ("prompt", "user_prompt", "message"):
        val = payload.get(key)
        if isinstance(val, str):
            return val
    return ""


async def _recall(prompt: str) -> list[str]:
    dsn = os.environ.get("CONTINUUM_DB_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        return []
    from continuum import Memory

    namespace = (os.environ.get("CONTINUUM_MCP_NAMESPACE") or "default").strip() or "default"
    embeddings = (os.environ.get("CONTINUUM_RECALL_HOOK_EMBEDDINGS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        k = max(1, int(os.environ.get("CONTINUUM_RECALL_HOOK_K") or "5"))
    except ValueError:
        k = 5

    import contextlib

    mem = Memory.from_postgres(
        dsn, embeddings=embeddings, namespace=namespace, session_id=namespace
    )
    try:
        await mem.start()
        hits = await mem.recall(prompt, k=k)
        return [h.content or "" for h in hits]
    finally:
        with contextlib.suppress(Exception):
            await mem.aclose()


def main(argv: list[str] | None = None) -> int:
    """Read the hook JSON on stdin, print additionalContext, never fail loudly."""
    try:
        prompt = _read_prompt(sys.stdin.read())
        if not prompt.strip():
            return 0
        memories = asyncio.run(_recall(prompt))
        context = build_context(prompt, memories)
        if context:
            out: dict[str, Any] = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            }
            sys.stdout.write(json.dumps(out))
    except Exception:
        # A memory hook must never break the user's prompt. Swallow everything.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
