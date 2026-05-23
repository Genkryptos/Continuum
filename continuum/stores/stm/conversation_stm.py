"""
continuum/stores/stm/conversation_stm.py
=========================================
``InMemorySTM`` — a multi-session, in-memory implementation of
``STMProtocol`` built on top of the existing ``ConversationSTM`` engine.

Design
------
Each session gets a dedicated ``ConversationSTM`` engine that owns the
eviction/compression machinery.  ``InMemorySTM`` is the session manager:
it creates engines lazily, routes calls by ``session_id``, and exposes
the async ``STMProtocol`` surface alongside the legacy sync API.

Backward compatibility
----------------------
The legacy sync API (``add_user_message``, ``get_messages``, etc.) is
preserved on ``InMemorySTM`` with a *session_id* prefix argument.  Existing
call sites that created a single ``ConversationSTM`` can migrate by:

    # Before
    stm = ConversationSTM(config, tokenizer)
    stm.add_user_message("hello")

    # After  (drop-in — same engine, Protocol-compliant wrapper)
    stm = InMemorySTM(config, tokenizer, default_session_id="sess-1")
    stm.add_user_message("sess-1", "hello")   # or use async append()

Conversion helpers
------------------
``message_to_item`` and ``item_to_message`` bridge the ``Message`` ↔
``MemoryItem`` boundary for callers that mix the two APIs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

from continuum.core.protocols import MTMProtocol
from continuum.core.types import MemoryItem, MemoryTier, ProcessingState

# ---------------------------------------------------------------------------
# Re-import legacy types so callers can do:
#   from continuum.stores.stm.conversation_stm import Importance, STMConfig
# ---------------------------------------------------------------------------
from memory.stm.ConversationSTM import (
    ConversationSTM as _ConversationSTMEngine,
)
from memory.stm.ConversationSTM import (
    Importance,
    Message,
    STMCallbacks,
    STMConfig,
    SummarizerFn,
    TokenizerFn,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importance ↔ float conversion
# (canonical mapping: LOW=0.25, NORMAL=0.50, HIGH=0.75, CRITICAL=1.00)
# ---------------------------------------------------------------------------

_IMP_TO_FLOAT: dict[Importance, float] = {
    Importance.LOW: 0.25,
    Importance.NORMAL: 0.50,
    Importance.HIGH: 0.75,
    Importance.CRITICAL: 1.00,
}

# Descending thresholds so the first match wins
_FLOAT_TO_IMP: list[tuple[float, Importance]] = [
    (0.875, Importance.CRITICAL),
    (0.625, Importance.HIGH),
    (0.375, Importance.NORMAL),
    (0.000, Importance.LOW),
]


def _float_to_importance(value: float) -> Importance:
    for threshold, imp in _FLOAT_TO_IMP:
        if value >= threshold:
            return imp
    return Importance.LOW


# ---------------------------------------------------------------------------
# Message ↔ MemoryItem conversion helpers (public API)
# ---------------------------------------------------------------------------


def message_to_item(
    msg: Message,
    session_id: str,
    agent_id: str | None = None,
    user_id: str | None = None,
) -> MemoryItem:
    """
    Convert a legacy ``Message`` to a ``MemoryItem``.

    The message's ``role``, ``tokens``, and ``summary`` flag are carried
    in ``MemoryItem.metadata`` so the reverse conversion is lossless.

    Parameters
    ----------
    msg:
        Source message from ``ConversationSTM.get_messages()``.
    session_id:
        The session this message belongs to.
    agent_id, user_id:
        Optional ownership fields forwarded to ``MemoryItem``.
    """
    meta: dict[str, Any] = dict(msg.meta)  # unfreeze MappingProxyType → dict
    owner_agent_id = meta.pop("_agent_id", agent_id)
    owner_user_id = meta.pop("_user_id", user_id)
    meta["role"] = msg.role
    meta["tokens"] = msg.tokens
    return MemoryItem(
        content=msg.content,
        tier=MemoryTier.STM,
        importance=_IMP_TO_FLOAT[msg.importance],
        confidence=1.0,
        created_at=msg.timestamp,
        session_id=session_id,
        agent_id=owner_agent_id,
        user_id=owner_user_id,
        metadata=meta,
    )


def item_to_message(item: MemoryItem, tokenizer: TokenizerFn | None = None) -> Message:
    """
    Convert a ``MemoryItem`` back to a legacy ``Message``.

    Parameters
    ----------
    item:
        Source item, typically produced by ``message_to_item`` or retrieved
        from the Protocol surface.
    tokenizer:
        Callable mapping content → token count.  Used only when
        ``item.metadata["tokens"]`` is absent.  Falls back to whitespace
        split when ``None``.
    """
    _tok: TokenizerFn = tokenizer or (lambda s: len(s.split()))
    tokens = int(item.metadata.get("tokens") or _tok(item.content))
    role = str(item.metadata.get("role", "user"))
    imp = _float_to_importance(item.importance)
    meta = {k: v for k, v in item.metadata.items() if k not in ("role", "tokens")}
    if item.agent_id is not None:
        meta["_agent_id"] = item.agent_id
    if item.user_id is not None:
        meta["_user_id"] = item.user_id
    return Message(
        role=role,
        content=item.content,
        tokens=tokens,
        timestamp=item.created_at,
        importance=imp,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# InMemorySTM — multi-session, Protocol-compliant in-memory STM
# ---------------------------------------------------------------------------


class InMemorySTM:
    """
    Multi-session, in-memory STMProtocol implementation.

    Each unique ``session_id`` gets a dedicated ``ConversationSTM`` engine.
    Engines are created lazily on first ``append`` and destroyed by
    ``clear()`` or ``flush_to()``.

    Parameters
    ----------
    config:
        Shared ``STMConfig`` applied to every new session engine.
        Defaults to a 4096-token / 100-message budget.
    tokenizer_fn:
        Callable mapping content → token count.  Defaults to whitespace
        split (replace with tiktoken in production).
    agent_id:
        Forwarded to ``MemoryItem.agent_id`` in ``window`` / ``get_recent``.
    callbacks_factory:
        Called with ``(session_id)`` when a new engine is created so
        per-session eviction / compression hooks can be wired up.

    Thread / task safety
    --------------------
    ``InMemorySTM`` is safe for concurrent asyncio tasks because it holds an
    ``asyncio.Lock`` around every engine access.  For multi-thread safety,
    wrap with ``AsyncSafeSTM`` and call from a single event loop.
    """

    def __init__(
        self,
        config: STMConfig | None = None,
        tokenizer_fn: TokenizerFn | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        callbacks_factory: Callable[[str], STMCallbacks] | None = None,
    ) -> None:
        self._config = config or STMConfig(max_tokens=4096, reserved_for_response=512)
        self._tokenizer: TokenizerFn = tokenizer_fn or (lambda s: len(s.split()))
        self._agent_id = agent_id
        self._user_id = user_id
        self._callbacks_factory = callbacks_factory
        self._sessions: dict[str, _ConversationSTMEngine] = {}
        self._lock = asyncio.Lock()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_or_create(self, session_id: str) -> _ConversationSTMEngine:
        """Return the engine for *session_id*, creating it if absent."""
        if session_id not in self._sessions:
            callbacks = (
                self._callbacks_factory(session_id) if self._callbacks_factory is not None else None
            )
            self._sessions[session_id] = _ConversationSTMEngine(
                config=self._config,
                tokenizer_fn=self._tokenizer,
                callbacks=callbacks,
            )
        return self._sessions[session_id]

    def _msgs_to_items(
        self,
        messages: list[Message],
        session_id: str,
    ) -> list[MemoryItem]:
        return [message_to_item(m, session_id, self._agent_id, self._user_id) for m in messages]

    # ── STMProtocol — async methods ────────────────────────────────────────

    async def append(self, item: MemoryItem) -> None:
        """
        Add *item* to the engine for ``item.session_id``.

        The item's ``metadata["role"]`` selects the message role; falls back
        to ``"user"``.  ``item.importance`` is mapped to the nearest
        ``Importance`` level before being handed to the engine.
        """
        session_id = item.session_id or "default"
        item.tier = MemoryTier.STM
        role = str(item.metadata.get("role", "user"))
        imp = _float_to_importance(item.importance)
        meta = {k: v for k, v in item.metadata.items() if k not in ("role",)}
        if item.agent_id is not None:
            meta["_agent_id"] = item.agent_id
        if item.user_id is not None:
            meta["_user_id"] = item.user_id

        async with self._lock:
            engine = self._get_or_create(session_id)
            engine._add_message(role, item.content, imp, meta)

    async def window(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> Sequence[MemoryItem]:
        """
        Return the conversation window for *session_id*, oldest-first.

        When *max_tokens* is ``None``, the engine's effective budget
        (``max_tokens − reserved_for_response``) is used.  When specified,
        items are accumulated newest-first then reversed so the most-recent
        messages are kept when the budget is tight.
        """
        async with self._lock:
            engine = self._sessions.get(session_id)
            if engine is None:
                return []
            msgs = engine.get_messages()

        if max_tokens is None:
            return self._msgs_to_items(msgs, session_id)

        # Trim from newest → oldest, keeping most-recent within budget
        result: list[MemoryItem] = []
        used = 0
        for msg in reversed(msgs):
            if used + msg.tokens > max_tokens:
                break
            result.append(message_to_item(msg, session_id, self._agent_id, self._user_id))
            used += msg.tokens
        result.reverse()
        return result

    async def get_recent(
        self,
        session_id: str,
        n: int = 10,
    ) -> Sequence[MemoryItem]:
        """Return the *n* most-recent messages, oldest-first, ignoring token budget."""
        async with self._lock:
            engine = self._sessions.get(session_id)
            if engine is None:
                return []
            msgs = engine.get_messages()

        tail = msgs[-n:] if len(msgs) > n else msgs
        return self._msgs_to_items(tail, session_id)

    async def flush_to(
        self,
        session_id: str,
        target: MTMProtocol,
    ) -> int:
        """
        Transfer all messages for *session_id* to *target* (MTM) then clear.

        Items are promoted one by one.  If the target raises for an item,
        the flush aborts and only items transferred so far are cleared from
        the engine — no data is lost.
        """
        async with self._lock:
            engine = self._sessions.get(session_id)
            if engine is None:
                return 0
            items = self._msgs_to_items(engine.get_messages(), session_id)

        if not items:
            return 0

        transferred = 0
        for item in items:
            item.processing_state = ProcessingState.UNPROCESSED
            try:
                await target.add_summary(item)
                transferred += 1
            except Exception:
                log.exception(
                    "flush_to: failed to transfer item %s to MTM — aborting after %d/%d",
                    item.id,
                    transferred,
                    len(items),
                )
                break

        async with self._lock:
            if transferred == len(items):
                self._sessions.pop(session_id, None)
            elif transferred > 0:
                # Keep only the items that didn't make it
                kept = engine.get_messages()[transferred:]
                engine._reset_messages(kept)

        log.info(
            "flush_to: transferred %d/%d items for session %s",
            transferred,
            len(items),
            session_id,
        )
        return transferred

    async def clear(self, session_id: str) -> None:
        """Discard the engine for *session_id* without flushing to MTM."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def stats(self, session_id: str) -> dict[str, Any]:
        """
        Return utilisation statistics for *session_id*.

        Keys: ``session_id``, ``message_count``, ``total_tokens``,
        ``max_tokens``, ``utilization``, ``budget_remaining``.
        """
        async with self._lock:
            engine = self._sessions.get(session_id)
            if engine is None:
                eff = max(self._config.max_tokens - self._config.reserved_for_response, 0)
                return {
                    "session_id": session_id,
                    "message_count": 0,
                    "total_tokens": 0,
                    "max_tokens": self._config.max_tokens,
                    "utilization": 0.0,
                    "budget_remaining": eff,
                }
            raw = engine.stats()

        eff = max(self._config.max_tokens - self._config.reserved_for_response, 0)
        return {
            "session_id": session_id,
            "message_count": raw["messages"],
            "total_tokens": raw["tokens"],
            "max_tokens": raw["max_tokens"],
            "utilization": raw["utilization"],
            "budget_remaining": max(eff - raw["tokens"], 0),
        }

    # ── Backward-compat sync API ───────────────────────────────────────────
    # These mirror the ConversationSTM method signatures but add session_id
    # as the first argument.  Existing code can migrate one call site at a time.

    def add_user_message(
        self,
        session_id: str,
        content: str,
        importance: Importance = Importance.NORMAL,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Sync backward-compat wrapper. Prefer ``await stm.append(item)``."""
        self._get_or_create(session_id).add_user_message(content, importance, meta)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        importance: Importance = Importance.NORMAL,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Sync backward-compat wrapper. Prefer ``await stm.append(item)``."""
        self._get_or_create(session_id).add_assistant_message(content, importance, meta)

    def add_system_message(
        self,
        session_id: str,
        content: str,
        importance: Importance = Importance.HIGH,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Sync backward-compat wrapper. Prefer ``await stm.append(item)``."""
        self._get_or_create(session_id).add_system_message(content, importance, meta)

    def add_tool_message(
        self,
        session_id: str,
        content: str,
        importance: Importance = Importance.NORMAL,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Sync backward-compat wrapper. Prefer ``await stm.append(item)``."""
        self._get_or_create(session_id).add_tool_message(content, importance, meta)

    def get_messages(self, session_id: str) -> list[Message]:
        """Sync backward-compat wrapper — returns raw ``Message`` objects."""
        engine = self._sessions.get(session_id)
        return engine.get_messages() if engine else []

    def get_prompt_messages(
        self,
        session_id: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Sync backward-compat wrapper — returns chat-API-ready dicts."""
        engine = self._sessions.get(session_id)
        if engine is None:
            return [{"role": "system", "content": system_prompt}] if system_prompt else []
        return engine.get_prompt_messages(system_prompt=system_prompt)

    def rollback_last_user_message(self, session_id: str) -> Message | None:
        """Sync backward-compat wrapper — undo the last user turn."""
        engine = self._sessions.get(session_id)
        return engine.rollback_last_user_message() if engine else None

    def maybe_compress(
        self,
        session_id: str,
        summarizer_fn: SummarizerFn,
        min_messages_to_compress: int = 10,
    ) -> None:
        """Sync backward-compat wrapper — trigger compression if threshold met."""
        engine = self._sessions.get(session_id)
        if engine:
            engine.maybe_compress(summarizer_fn, min_messages_to_compress)
