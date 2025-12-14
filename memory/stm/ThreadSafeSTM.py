"""
Thread-safe wrapper that serializes access to a ConversationSTM instance.

This keeps the STM state consistent when accessed from multiple agent threads
without changing the underlying STM semantics.
"""

from memory.stm import ConversationSTM
from threading import Lock
from typing import Any, Dict, List, Optional
from memory.stm.ConversationSTM import Message


class ThreadSafeSTM:
    """Delegate ConversationSTM operations under a shared mutex."""

    def __init__(self, stm: ConversationSTM):
        self._stm = stm
        self._lock = Lock()

    def add_user_message(self, *args, **kwargs) -> None:
        with self._lock:
            self._stm.add_user_message(*args, **kwargs)

    def add_assistant_message(self, *args, **kwargs) -> None:
        with self._lock:
            self._stm.add_assistant_message(*args, **kwargs)

    def add_system_message(self, *args, **kwargs) -> None:
        with self._lock:
            self._stm.add_system_message(*args, **kwargs)

    def add_tool_message(self, *args, **kwargs) -> None:
        with self._lock:
            self._stm.add_tool_message(*args, **kwargs)

    def get_messages(self) -> List[Message]:
        with self._lock:
            return self._stm.get_messages()

    def get_prompt_messages(self, *args, **kwargs) -> List[Dict[str, str]]:
        with self._lock:
            return self._stm.get_prompt_messages(*args, **kwargs)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._stm.stats()

    def maybe_compress(self, *args, **kwargs) -> None:
        with self._lock:
            self._stm.maybe_compress(*args, **kwargs)

    def rollback_last_user_message(self, *args, **kwargs) -> Optional[Message]:
        with self._lock:
            return self._stm.rollback_last_user_message(*args, **kwargs)
