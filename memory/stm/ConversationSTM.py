from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import logging
from typing import Any, Callable, Deque, Dict, List, Optional, Mapping
from datetime import datetime, timezone
from types import MappingProxyType

TokenizerFn = Callable[[str], int]
SummarizerFn = Callable[[List["Message"], int], str]

class Importance(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: Importance = Importance.NORMAL
    meta: Mapping[str, Any] = field(default_factory=dict)

    def _deep_freeze(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            frozen_mapping = {k: self._deep_freeze(v) for k, v in value.items()}
            return MappingProxyType(frozen_mapping)
        if isinstance(value, (list, tuple)):
            return tuple(self._deep_freeze(v) for v in value)
        if isinstance(value, set):
            return frozenset(self._deep_freeze(v) for v in value)
        return value

    def with_meta(self, updates: Optional[Mapping[str, Any]] = None) -> "Message":
        """Return a new Message with merged metadata.

        The original message remains unchanged, preserving immutability. Passing
        ``updates`` overlays the existing metadata, enabling workflows such as
        marking a message as answered without mutating in-place.
        """

        merged_meta: Dict[str, Any] = dict(self.meta)
        if updates:
            merged_meta.update(updates)

        return Message(
            role=self.role,
            content=self.content,
            tokens=self.tokens,
            timestamp=self.timestamp,
            importance=self.importance,
            meta=merged_meta,
        )
    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", self._deep_freeze(self.meta))

@dataclass
class STMCallbacks:
    on_evict: Optional[Callable[[Message], None]] = None
    on_compress: Optional[Callable[[List[Message], Message], None]] = None

@dataclass
class STMConfig:
    max_tokens: int
    max_messages: int = 100
    compress_threshold_ratio: float = 0.8
    compress_fraction: float = 0.5
    max_summary_tokens: int = 512
    reserved_for_response: int = 1024
    name: str = "default_conversation"

class ConversationSTM:

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        config: STMConfig,
        tokenizer_fn: TokenizerFn,
        callbacks: Optional[STMCallbacks] = None,
    ):
        self.config = config
        self._tokenizer_fn = tokenizer_fn
        self._messages: Deque[Message] = deque()
        self._total_tokens: int = 0
        self._callbacks = callbacks or STMCallbacks()

    def add_user_message(self, content: str,
                         importance: Importance = Importance.NORMAL,
                         meta: Optional[Dict[str, Any]] = None) -> None:
        self._add_message("user", content, importance, meta)

    def add_assistant_message(self, content: str,
                              importance: Importance = Importance.NORMAL,
                              meta: Optional[Dict[str, Any]] = None) -> None:
        self._add_message("assistant", content, importance, meta)

    def add_system_message(self, content: str,
                           importance: Importance = Importance.HIGH,
                           meta: Optional[Dict[str, Any]] = None) -> None:
        self._add_message("system", content, importance, meta)

    def add_tool_message(self, content: str,
                         importance: Importance = Importance.NORMAL,
                         meta: Optional[Dict[str, Any]] = None) -> None:
        self._add_message("tool", content, importance, meta)

    def get_messages(self) -> List[Message]:
        return list(self._messages)

    def get_prompt_messages(
        self,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        prompt_msgs: List[Dict[str, str]] = []

        if system_prompt:
            prompt_msgs.append({"role": "system", "content": system_prompt})

        for m in self._messages:
            prompt_msgs.append({"role": m.role, "content": m.content})

        return prompt_msgs

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "messages": len(self._messages),
            "tokens": self._total_tokens,
            "max_tokens": self.config.max_tokens,
            "utilization": self._total_tokens / self.config.max_tokens
            if self.config.max_tokens > 0 else 0.0,
        }

    def rollback_last_user_message(self) -> Optional[Message]:
        """Remove the most recent user message, if present.

        This is intended for undoing a turn when downstream steps fail so
        subsequent prompts do not include unanswered user queries.
        """

        if not self._messages or self._messages[-1].role != "user":
            return None

        msg = self._messages.pop()
        self._total_tokens = max(self._total_tokens - msg.tokens, 0)
        return msg

    def maybe_compress(
        self,
        summarizer_fn: SummarizerFn,
        min_messages_to_compress: int = 10,
    ) -> None:

        if not self._messages:
            return

        utilization = self.stats()["utilization"]
        if utilization < self.config.compress_threshold_ratio:
            return

        if len(self._messages) < min_messages_to_compress:
            return

        effective_budget = self._effective_budget()
        current_messages = list(self._messages)

        compress_fraction = self._compute_compress_fraction(utilization)
        split_idx = max(int(len(current_messages) * compress_fraction), 1)
        messages_to_summarize = current_messages[:split_idx]
        messages_to_keep = current_messages[split_idx:]

        summary_budget = min(self.config.max_summary_tokens, effective_budget)
        summary = self._generate_summary_with_retry(
            summarizer_fn,
            messages_to_summarize,
            summary_budget,
            effective_budget,
        )

        if summary is None:
            return

        summary_tokens = self._tokenizer_fn(summary)

        if summary_tokens > effective_budget or summary_tokens > summary_budget:
            return

        self._messages.clear()
        self._total_tokens = 0

        summary_message = Message(
            role="system",
            content=(
                f"[Conversation summary up to {datetime.now(timezone.utc).isoformat()}]\n"
                f"{summary}"
            ),
            tokens=summary_tokens,
            importance=Importance.HIGH,
            meta={"summary": True},
        )

        if self._callbacks.on_compress:
            try:
                self._callbacks.on_compress(messages_to_summarize, summary_message)
            except Exception:
                self._logger.exception("STM on_compress callback failed; continuing compression")

        self._reset_messages([summary_message, *messages_to_keep])

    def _compute_compress_fraction(self, utilization: float) -> float:
        base_fraction = min(max(self.config.compress_fraction, 0.05), 0.95)
        if utilization <= self.config.compress_threshold_ratio:
            return base_fraction

        extra_fraction = min(utilization, 0.9)
        return max(base_fraction, extra_fraction)

    def _generate_summary_with_retry(
            self,
            summarizer_fn: SummarizerFn,
            messages_to_summarize: List[Message],
            initial_budget: int,
            effective_budget: int,
            max_attempts: int = 3,
            budget_decay: float = 0.8,
    ) -> Optional[str]:
        attempts = 0
        current_budget = min(initial_budget, effective_budget)
        while attempts < max_attempts and current_budget > 0:
            try:
                summary = summarizer_fn(messages_to_summarize, current_budget)
            except Exception:
                summary = None

            if summary:
                summary_tokens = self._tokenizer_fn(summary)
                if summary_tokens <= current_budget:
                    return summary

            attempts += 1
            current_budget = int(current_budget * budget_decay)

        return None

    def _add_message(
        self,
        role: str,
        content: str,
        importance: Importance,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        tokens = self._tokenizer_fn(content)
        new_msg = Message(
            role=role,
            content=content,
            tokens=tokens,
            importance=importance,
            meta=meta or {},
        )
        self._make_room_for(tokens)
        self._append_message_raw(new_msg)

    def _append_message_raw(self, msg: Message) -> None:
        self._messages.append(msg)
        self._total_tokens += msg.tokens
        self._enforce_limits()

    def _make_room_for(self, new_tokens: int) -> None:

        effective_budget = self._effective_budget()
        allow_summary_eviction = self._summary_exceeds_budget(
            additional_tokens=new_tokens
        )

        while self._messages and (
            self._total_tokens + new_tokens > effective_budget
            or len(self._messages) >= self.config.max_messages
        ):
            idx = self._find_eviction_index(
                allow_summary_eviction=allow_summary_eviction
            )
            if idx is None:
                if allow_summary_eviction:
                    break
                allow_summary_eviction = True
                continue
            self._evict_index(idx)

            if not allow_summary_eviction:
                allow_summary_eviction = self._summary_exceeds_budget(
                    additional_tokens=new_tokens
                )

    def _enforce_limits(self) -> None:
        effective_budget = self._effective_budget()
        allow_summary_eviction = self._summary_exceeds_budget()
        while self._messages and (
            self._total_tokens > effective_budget
            or len(self._messages) > self.config.max_messages
        ):
            idx = self._find_eviction_index(
                allow_summary_eviction=allow_summary_eviction
            )
            if idx is None:
                if allow_summary_eviction:
                    break
                allow_summary_eviction = True
                continue
            self._evict_index(idx)
            if not allow_summary_eviction:
                allow_summary_eviction = self._summary_exceeds_budget()


    def _find_eviction_index(self, allow_summary_eviction: bool = False) -> Optional[int]:
        msgs = list(self._messages)

        def eligible(message: Message) -> bool:
            return allow_summary_eviction or not message.meta.get("summary")

        for i, m in enumerate(msgs):
            if eligible(m) and m.importance in (Importance.LOW, Importance.NORMAL):
                return i

        for i, m in enumerate(msgs):
            if eligible(m) and m.importance == Importance.HIGH:
                return i

        for i, m in enumerate(msgs):
            if eligible(m):
                return i

        return None

    def _reset_messages(self, messages: List[Message]) -> None:
        self._messages = deque(messages)
        self._total_tokens = sum(m.tokens for m in messages)
        self._enforce_limits()

    def _evict_index(self, idx: int) -> None:
        msg = self._messages[idx]
        self._total_tokens -= msg.tokens
        del self._messages[idx]

        if self._callbacks.on_evict:
            try:
                self._callbacks.on_evict(msg)
            except Exception:
                self._logger.exception("STM on_evict callback failed; continuing eviction")

    def _summary_exceeds_budget(self, additional_tokens: int = 0) -> bool:
        budget = self._effective_budget()
        summary = next(
            (m for m in self._messages if m.meta.get("summary")), None
        )
        if summary is None:
            return False
        return summary.tokens + additional_tokens > budget

    def _effective_budget(self) -> int:
        return max(self.config.max_tokens - self.config.reserved_for_response, 0)

