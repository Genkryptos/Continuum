from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from memory.stm.ConversationSTM import Importance, Message
from memory.stm.ThreadSafeSTM import ThreadSafeSTM


class SharedSTM:
    """Provide filtered views over a shared, thread-safe STM instance.

    Notes:
        - Callers should add messages through :meth:`add_message_for_all` to keep
          the shared conversation consistent across agent views.
        - Any ``filter_fn`` supplied to :meth:`get_view_for_agent` is expected to
          be fast and exception-free; errors from the filter will propagate to
          the caller.
    """

    def __init__(self, stm: ThreadSafeSTM):
        self._stm = stm

    def add_message_for_all(
        self,
        role: str,
        content: str,
        importance: Importance = Importance.NORMAL,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if role == "user":
            self._stm.add_user_message(content, importance, meta)
        elif role == "assistant":
            self._stm.add_assistant_message(content, importance, meta)
        elif role == "system":
            self._stm.add_system_message(content, importance, meta)
        else:
            self._stm.add_tool_message(content, importance, meta)

    def get_view_for_agent(
        self,
        agent_id: str,
        filter_fn: Optional[Callable[[Message, str], bool]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        all_msgs = self._stm.get_messages()
        if filter_fn:
            msgs = [m for m in all_msgs if filter_fn(m, agent_id)]
        else:
            msgs = all_msgs
        prompt_msgs: List[Dict[str, str]] = []
        if system_prompt:
            prompt_msgs.append({"role": "system", "content": system_prompt})

        for m in msgs:
            prompt_msgs.append({"role": m.role, "content": m.content})

        return prompt_msgs
