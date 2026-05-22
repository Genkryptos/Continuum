"""
evals/longmemeval/adapter.py
============================
:class:`ContinuumAdapter` — bridges Continuum's session/retriever loop
onto the LongMemEval benchmark's expected adapter interface.

LongMemEval's harness expects two coroutines per system-under-test:

* ``process_conversation(messages)`` — ingest a multi-turn history.
* ``answer_question(question) -> str`` — answer one question given
  whatever the system has ingested so far.

The adapter is intentionally *decoupled* from a concrete LLM client.
Wire the responder explicitly via the ``llm`` collaborator (any object
with ``async complete(prompt: str, max_tokens: int) -> str``) so the
harness can swap models without touching framework code.

Failure handling mirrors the rest of Continuum: retrieval failures
degrade to no-context responses, LLM failures surface a short error
string, and nothing in this adapter raises into LongMemEval's runner.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from continuum.core.types import (
    ContextBundle,
    MemoryItem,
    MemoryTier,
    Query,
    TokenBudget,
)

log = logging.getLogger(__name__)


_DEFAULT_BUDGET = TokenBudget(
    total=4_000,
    stm_reserved=1_000,
    mtm_reserved=1_000,
    ltm_reserved=2_000,
    response_reserved=200,
)


class ContinuumAdapter:
    """
    Adapt a :class:`continuum.core.session.ContinuumSession` to
    LongMemEval's adapter protocol.

    Parameters
    ----------
    session:
        A live ``ContinuumSession``. The adapter does not manage its
        lifecycle — start / stop it from your runner.
    llm:
        Required collaborator exposing
        ``async complete(prompt: str, max_tokens: int) -> str``.
        Wire your preferred client here (litellm, OpenAI, Anthropic,
        local-server, …).
    budget:
        :class:`TokenBudget` used for every retrieval. Defaults to a
        4 K / 1 K / 1 K / 2 K layout that LongMemEval's published
        baselines use.
    answer_max_tokens:
        Max tokens the LLM should produce per answer. Default 100 —
        LongMemEval scoring tolerates short answers.
    """

    def __init__(
        self,
        *,
        session: Any,
        llm: Any,
        budget: TokenBudget | None = None,
        answer_max_tokens: int = 100,
    ) -> None:
        self.session = session
        self.llm = llm
        self.budget = budget or _DEFAULT_BUDGET
        self.answer_max_tokens = answer_max_tokens
        #: Last :class:`ContextBundle` returned by retrieve(); the
        #: baseline harness reads it to compute retrieval recall.
        self.last_ctx: ContextBundle | None = None

    # ── LongMemEval interface ───────────────────────────────────────────────

    async def process_conversation(
        self, messages: Iterable[Mapping[str, Any]]
    ) -> None:
        """
        Ingest a multi-turn message list into the session.

        Each *message* is a mapping with at least ``role`` and
        ``content`` keys (the OpenAI chat format LongMemEval uses).
        User turns go through ``session.process_turn`` so the full
        Continuum pipeline (STM append → retrieve → respond → STM
        append → background bookkeeping) runs. Non-user messages are
        appended directly to STM so the assistant's prior replies are
        retained for retrieval.
        """
        for msg in messages:
            role = str(msg.get("role", "user")).lower()
            content = str(msg.get("content", ""))
            if not content.strip():
                continue
            try:
                if role == "user":
                    await self.session.process_turn(
                        content, context_budget=self.budget
                    )
                else:
                    # Assistant / system turns: persist verbatim so
                    # retrieval sees them, but don't trigger a response.
                    await self._append_assistant(content, role=role)
            except Exception:
                log.exception(
                    "process_conversation: failed on message %r — "
                    "continuing", content[:80],
                )

    async def answer_question(self, question: str) -> str:
        """
        Answer *question* using the retrieved context.

        Pipeline: ``retriever.retrieve(question)`` → format prompt →
        ``llm.complete(prompt)``. Retrieval failures degrade to
        no-context prompts; LLM failures surface a short ``[error: …]``
        string so the harness can still score the row.
        """
        retriever = getattr(self.session, "retriever", None)
        ctx: ContextBundle | None = None
        if retriever is not None:
            try:
                ctx = await retriever.retrieve(
                    Query(text=question), self.budget
                )
            except Exception:
                log.exception("retrieve failed for %r", question[:80])

        self.last_ctx = ctx
        prompt = self.format_prompt(question, ctx)
        try:
            answer = await self.llm.complete(
                prompt=prompt, max_tokens=self.answer_max_tokens
            )
        except Exception as exc:
            log.exception("LLM completion failed for %r", question[:80])
            return f"[error: {exc!r}]"
        return str(answer).strip()

    # ── prompt formatting ───────────────────────────────────────────────────

    def format_prompt(
        self, question: str, ctx: ContextBundle | None
    ) -> str:
        """
        Render the retrieved bundle into a single string prompt.

        Sections are emitted only when they have content — keeping the
        prompt token-tight even when one tier returned empty. The order
        mirrors Continuum's prompt convention: durable knowledge first,
        recent context last.
        """
        stm_text, mtm_text, ltm_text = _bucket_by_tier(ctx)
        sections: list[str] = []
        if ltm_text:
            sections.append(f"Long-term knowledge:\n{ltm_text}")
        if mtm_text:
            sections.append(f"Project summary:\n{mtm_text}")
        if stm_text:
            sections.append(f"Recent conversation:\n{stm_text}")
        sections.append(f"Question: {question}\nAnswer:")
        header = (
            "Given the following context, answer the question. "
            "Use only facts that appear in the context; say "
            "\"I don't know\" if the context doesn't contain the answer.\n"
        )
        return header + "\n\n".join(sections)

    # ── internals ───────────────────────────────────────────────────────────

    async def _append_assistant(self, content: str, *, role: str) -> None:
        """
        Push an assistant / system message into STM without running a
        full turn. Uses whichever low-level STM API the session exposes;
        falls back to ``session.stm.append`` if present.
        """
        stm = getattr(self.session, "stm", None)
        if stm is None:
            return
        item = MemoryItem(
            content=content,
            tier=MemoryTier.STM,
            metadata={"role": role},
        )
        # The STMProtocol surface has ``append`` everywhere in this
        # codebase; degrade silently if a custom store differs.
        appender = getattr(stm, "append", None)
        if appender is None:
            return
        try:
            result = appender(item)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            log.exception("STM append failed for %s role", role)


# ---------------------------------------------------------------------------
# Helpers (module-level for testability)
# ---------------------------------------------------------------------------


def _bucket_by_tier(
    ctx: ContextBundle | None,
) -> tuple[str, str, str]:
    """Return (stm_text, mtm_text, ltm_text) joined by newlines."""
    if ctx is None:
        return "", "", ""
    stm: list[str] = []
    mtm: list[str] = []
    ltm: list[str] = []
    for it in ctx.items:
        if it.tier == MemoryTier.STM:
            stm.append(it.content)
        elif it.tier == MemoryTier.MTM:
            mtm.append(it.content)
        elif it.tier == MemoryTier.LTM:
            ltm.append(it.content)
    return "\n".join(stm), "\n".join(mtm), "\n".join(ltm)


__all__ = ["ContinuumAdapter"]
