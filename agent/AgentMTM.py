"""
Agent loop that combines short-term memory (STM) with mid-term memory (MTM).

The class keeps STM state small via token-aware compression and augments each
turn with MTM retrieval before calling the chosen LLM provider.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

from agent.agent import LocalAgent
from helper.tknCounter import TknCounter
from helper.web_search import WebSearchResult, WebSearchService
from memory.mtm.contextAugmenter import ContextAugmenter, ContextResult
from memory.mtm.repository.mtmRetriever import MTMRetriever
from memory.stm.ConversationSTM import STMConfig, STMCallbacks, ConversationSTM, Importance, Message
from settings import (
    LLM_MODEL,
    MAX_CONTEXT_TOKENS,
    ANSWER_FRACTION,
    MTM_TOP_K,
)


class TextTokenCounter:
    """
    Adapter: use TknCounter (chat-aware) but expose a simple str -> int
    for ConversationSTM.tokenizer_fn.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or LLM_MODEL
        self._tkn = TknCounter()

    def __call__(self, text: str) -> int:
        # Wrap raw text into a single fake message so we can reuse TknCounter logic.
        return self._tkn.count_request_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )


class AgentMTM:
    """
    Agent loop that uses:
      - STM for short-term conversation state
      - MTM for retrieved episodic/summary memories
      - TknCounter-based budgeting to build the final LLM prompt

    All model / budget config is read from environment via settings.py.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[LocalAgent] = None,
        stm_tokenizer: Optional[Callable[[str], int]] = None,
        stm_config: Optional[STMConfig] = None,
        stm_callbacks: Optional[STMCallbacks] = None,
        mtm_retriever: Optional[MTMRetriever] = None,
        context_augmenter: Optional[ContextAugmenter] = None,
        max_context_tokens: Optional[int] = None,
        answer_fraction: Optional[float] = None,
        mtm_top_k: Optional[int] = None,
        min_messages_to_compress: int = 4,
        system_prompt: str = "You are a helpful assistant.",
        web_search_service: Optional[WebSearchService] = None,
        auto_web_search: bool = True,
        web_search_max_results: int = 3,
        max_web_context_tokens: int = 512,
        web_search_triggers: Optional[List[str]] = None,
    ):
        self.logger = logging.getLogger("AgentMTM")

        # ---- Load from env / settings ----
        self.model_name = model_name or LLM_MODEL
        self.max_context_tokens = max_context_tokens or MAX_CONTEXT_TOKENS
        self.answer_fraction = answer_fraction or ANSWER_FRACTION
        self.mtm_top_k = mtm_top_k or MTM_TOP_K
        self.system_prompt = system_prompt
        self.min_messages_to_compress = min_messages_to_compress
        self.web_search_service = web_search_service
        self.auto_web_search = auto_web_search
        self.web_search_max_results = web_search_max_results
        self.max_web_context_tokens = max_web_context_tokens
        self.web_search_triggers = tuple(
            web_search_triggers
            or ("@web", "/web", "web search", "search the web")
        )

        # ---- LLM + token counters ----
        self.llm = llm or LocalAgent(model=self.model_name)
        self._tkn = TknCounter()
        self.stm_tokenizer: Callable[[str], int] = (
            stm_tokenizer or TextTokenCounter(model=self.model_name)
        )

        # ---- STM config ----
        stm_config = stm_config or STMConfig(
            max_tokens=400,
            max_messages=20,
            compress_threshold_ratio=0.5,
            max_summary_tokens=128,
            reserved_for_response=128,
            name="mtm_agent_session",
        )

        # ---- STM callbacks (can inject MTMCallbacks here) ----
        user_callbacks = stm_callbacks or STMCallbacks()

        def _on_evict(self, msg: Message) -> None:
            """Default on_evict callback: just log."""
            self.logger.info(
                f"[STM] Evicted message role={msg.role} "
                f"tokens={msg.tokens} importance={msg.importance}"
            )

        def _on_compress(self, old_messages: List[Message], summary: Message) -> None:
            """Default on_compress callback: just log compression ratio."""
            total_old = sum(m.tokens for m in old_messages)
            ratio = summary.tokens / max(total_old, 1)
            self.logger.info(
                f"[STM] Compression: {len(old_messages)} msgs, "
                f"{total_old} → {summary.tokens} tokens ({ratio:.1%})"
            )
        def wrap_on_evict(user_cb, default_cb):
            if user_cb is None:
                return default_cb

            def combined(msg: Message):
                user_cb(msg)
                default_cb(msg)

            return combined

        def wrap_on_compress(user_cb, default_cb):
            if user_cb is None:
                return default_cb

            def combined(old_messages: List[Message], summary: Message):
                user_cb(old_messages, summary)
                default_cb(old_messages, summary)

            return combined

        user_on_evict = getattr(user_callbacks, "on_evict", None)
        user_on_compress = getattr(user_callbacks, "on_compress", None)

        stm_callbacks = STMCallbacks(
            on_evict=wrap_on_evict(user_on_evict, self._on_evict),
            on_compress=wrap_on_compress(user_on_compress, self._on_compress),
        )

        self.stm = ConversationSTM(
            config=stm_config,
            tokenizer_fn=self.stm_tokenizer,
            callbacks=stm_callbacks,
        )

        self.mtm_retriever = mtm_retriever
        if self.mtm_retriever is None:
            raise ValueError("AgentMTM requires an MTMRetriever instance")

        self.context_augmenter: ContextAugmenter = context_augmenter or ContextAugmenter(
            stm=self.stm,
            mtm_retriever=self.mtm_retriever,
            tokenizer=self._tkn,
            model_name=self.model_name,
            max_context_tokens=self.max_context_tokens,
            answer_fraction=self.answer_fraction,
            mtm_top_k=self.mtm_top_k,
        )

        self.turn = 0

    # --- STM callbacks, token counting, context building, etc. ---

    def _on_evict(self, msg: "Message") -> None:
        self.logger.info(
            f"[STM] Evicted message role={msg.role} "
            f"tokens={msg.tokens} importance={msg.importance}"
        )

    def _on_compress(self, old_messages: List["Message"], summary: "Message") -> None:
        total_old = sum(m.tokens for m in old_messages)
        ratio = summary.tokens / max(total_old, 1)
        self.logger.info(
            f"[STM] Compression: {len(old_messages)} msgs, "
            f"{total_old} → {summary.tokens} tokens ({ratio:.1%})"
        )

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return self._tkn.count_request_tokens(model=self.model_name, messages=messages)

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "No additional context from MTM."

        parts = []
        for idx, mem in enumerate(memories, start=1):
            scope = mem.get("scope", "unknown")
            source = mem.get("source", "unknown")
            content = mem.get("content", "")
            parts.append(f"[{idx}] ({scope} via {source}) {content}")
        return "\n".join(parts)

    def _build_prompt(self, context: str) -> List[Dict[str, str]]:
        prompt: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\nRelevant context:\n{context}",
            }
        ]

        for m in self.stm.get_messages():
            prompt.append({"role": m.role, "content": m.content})
        return prompt

    def handle_user_input(
        self,
        user_input: str,
        user_id: str,
        agent_id: str,
        session_key: str,
    ) -> Dict[str, Any]:
        self.turn += 1
        self.logger.info(f"════════════════ Turn {self.turn} ════════════════")
        self.logger.info(f"User: {user_input}")
        response_budget = max(int(self.max_context_tokens * self.answer_fraction), 1)

        self.stm.add_user_message(
            content=user_input,
            importance=Importance.NORMAL,
            meta={"turn": self.turn, "timestamp": datetime.now().isoformat()},
        )

        stats_before = self.stm.stats()
        self.logger.info(
            f"[STM Before] messages={stats_before['messages']} "
            f"tokens={stats_before['tokens']} "
            f"util={stats_before['utilization']:.1%}"
        )

        compression_error: Optional[str] = None
        try:
            self.stm.maybe_compress(
                summarizer_fn=self._naive_summarizer,
                min_messages_to_compress=self.min_messages_to_compress,
            )
        except Exception as exc:  # noqa: BLE001
            compression_error = str(exc)
            self.logger.exception("STM compression failed; continuing without compression")

        context_result: Optional[ContextResult] = None
        prompt_messages: List[Dict[str, str]]
        memories: List[Dict[str, Any]] = []
        mtm_error: Optional[str] = None
        web_results: List[WebSearchResult] = []
        web_error: Optional[str] = None
        web_tokens_used = 0
        try:
            context_result = self.context_augmenter.build_context(
                user_id=user_id,
                agent_id=agent_id,
                session_key=session_key,
                user_query=user_input,
                system_prompt=self.system_prompt,
            )
            prompt_messages = context_result.prompt_messages
            memories = context_result.mtm_memories
            self.logger.debug(
                "Context debug info: %s", getattr(context_result, "debug_info", {})
            )
        except Exception as exc:  # noqa: BLE001
            mtm_error = str(exc)
            self.logger.exception(
                "Context augmentation failed; proceeding with STM-only prompt"
            )
            prompt_messages = self.stm.get_prompt_messages(system_prompt=self.system_prompt)

        prompt_messages = self._inject_capabilities_prompt(prompt_messages)
        prompt_messages = self._inject_time_prompt(prompt_messages)

        if self.web_search_service and self._should_search_web(user_input, context_result):
            (
                prompt_messages,
                web_results,
                web_tokens_used,
                web_error,
            ) = self._attach_web_context(
                prompt_messages=prompt_messages,
                user_query=user_input,
                response_budget=response_budget,
                )
            if web_results:
                self.logger.info(
                    "Added web context: %s results (+%s tokens)",
                    len(web_results),
                    web_tokens_used,
                )
            if web_error:
                self.logger.warning("Web search error: %s", web_error)

        web_results_payload = [r.as_dict() for r in web_results]
        prompt_tokens = self._count_tokens(prompt_messages)
        if prompt_tokens + response_budget > self.max_context_tokens:
            self.logger.warning(
                "Prompt exceeds budget; continuing but response may be truncated "
                f"(prompt={prompt_tokens}, budget={self.max_context_tokens}, "
                f"reserve={response_budget})"
            )

        try:
            result = self.llm.call_model(prompt_messages)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("LLM exception", exc_info=True)
            rolled_back = self.stm.rollback_last_user_message()
            if rolled_back:
                self.logger.info(
                    "Rolled back user message after LLM exception to avoid "
                    "unanswered prompts",
                )

            stats_after_error = self.stm.stats()
            return {
                "response": None,
                "error": str(exc),
                "success": False,
                "stm_stats_before": stats_before,
                "stm_stats_after": stats_after_error,
                "mtm_error": mtm_error,
                "compression_error": compression_error,
                "rolled_back_user_message": bool(rolled_back),
                "web_results": web_results_payload,
                "web_error": web_error,
            }

        if not result.get("success"):
            self.logger.error(f"LLM error: {result.get('error')}")
            rolled_back = self.stm.rollback_last_user_message()
            if rolled_back:
                self.logger.info(
                    "Rolled back user message after failed LLM response to avoid "
                    "unanswered prompts",
                )

            stats_after_error = self.stm.stats()
            return {
                "response": None,
                "error": result.get("error"),
                "success": False,
                "stm_stats_before": stats_before,
                "stm_stats_after": stats_after_error,
                "mtm_error": mtm_error,
                "compression_error": compression_error,
                "rolled_back_user_message": bool(rolled_back),
                "web_results": web_results_payload,
                "web_error": web_error,
            }

        assistant_reply = result.get("response", "")
        self.logger.info(
            f"Assistant: {assistant_reply[:200]}"
            f"{'...' if len(assistant_reply) > 200 else ''}"
        )
        self.logger.info(
            f"LLM latency={result.get('latency', 0):.2f}s "
            f"tokens={result.get('tokens_used')}"
        )

        self.stm.add_assistant_message(
            content=assistant_reply,
            importance=Importance.NORMAL,
            meta={
                "turn": self.turn,
                "timestamp": datetime.now().isoformat(),
                "model": self.llm.model,
                "mtm_memories": memories,
                "context_debug_info": getattr(context_result, "debug_info", None),
                "web_results": web_results_payload,
                "web_error": web_error,
            },
        )

        stats_after = self.stm.stats()
        self.logger.info(
            f"[STM After] messages={stats_after['messages']} "
            f"tokens={stats_after['tokens']} "
            f"util={stats_after['utilization']:.1%}"
        )
        self._debug_print_messages()
        return {
            "response": assistant_reply,
            "success": True,
            "error": None,
            "latency": result.get("latency"),
            "tokens_used": result.get("tokens_used"),
            "stm_stats_before": stats_before,
            "stm_stats_after": stats_after,
            "mtm_memories": memories,
            "mtm_error": mtm_error,
            "compression_error": compression_error,
            "web_results": web_results_payload,
            "web_error": web_error,
        }

    def _should_search_web(
        self,
        user_input: str,
        context_result: Optional[ContextResult],
    ) -> bool:
        """
        Decide whether to trigger a web search.

        Manual triggers (keywords) override the auto flag so users can request
        a fetch ad-hoc without enabling it globally.
        """
        if self.web_search_service is None:
            return False

        lowered = user_input.lower()
        if any(trigger in lowered for trigger in self.web_search_triggers):
            return True

        if not self.auto_web_search:
            return False

        if context_result is not None and not getattr(context_result, "mtm_memories", None):
            return True

        return True

    def _attach_web_context(
        self,
        prompt_messages: List[Dict[str, str]],
        user_query: str,
        response_budget: int,
    ) -> Tuple[List[Dict[str, str]], List[WebSearchResult], int, Optional[str]]:
        """Fetch web results and attempt to prepend them within the token budget."""
        try:
            results = self.web_search_service.search(
                user_query,
                max_results=self.web_search_max_results,
            )
        except Exception as exc:  # noqa: BLE001
            return prompt_messages, [], 0, str(exc)

        if not results:
            return prompt_messages, [], 0, None

        base_tokens = self._count_tokens(prompt_messages)
        web_context = self.web_search_service.format_results(results)
        web_msg = {
            "role": "system",
            "content": f"Live web search results:\n{web_context}",
        }

        def _inject(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
            return msgs[:1] + [web_msg] + msgs[1:] if msgs else [web_msg]

        candidate_prompt = _inject(prompt_messages)
        total_tokens = self._count_tokens(candidate_prompt)

        if total_tokens + response_budget > self.max_context_tokens:
            try:
                truncate_budget = max(int(self.max_web_context_tokens or 0), 0)
                if truncate_budget > 0:
                    web_msg["content"] = self._tkn.truncate_text(
                        self.model_name, web_msg["content"], truncate_budget
                    )
                    candidate_prompt = _inject(prompt_messages)
                    total_tokens = self._count_tokens(candidate_prompt)
            except Exception as exc:  # noqa: BLE001
                return prompt_messages, [], 0, f"Failed to budget web context: {exc}"

        if total_tokens + response_budget > self.max_context_tokens:
            return prompt_messages, [], 0, "web_context_exceeds_budget"

        tokens_added = max(total_tokens - base_tokens, 0)
        return candidate_prompt, results, tokens_added, None

    def _inject_capabilities_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        If web search is available, add a capability hint so the model does not
        claim it cannot browse when results are present or fail silently.
        """
        if self.web_search_service is None:
            return messages

        capability_msg = {
            "role": "system",
            "content": (
                "You have access to live web search results injected as system messages "
                "labeled 'Live web search results'. Use them for the freshest facts. "
                "If no results are present or the search fails, say that the search "
                "returned nothing instead of claiming you cannot browse."
            ),
        }

        if not messages:
            return [capability_msg]

        if messages[0].get("role") == "system":
            return [messages[0], capability_msg] + messages[1:]

        return [capability_msg] + messages

    def _inject_time_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Insert a system message with current datetime to ground time-sensitive questions."""
        time_msg = self._current_datetime_message()
        if not messages:
            return [time_msg]

        if messages[0].get("role") == "system":
            return [messages[0], time_msg] + messages[1:]

        return [time_msg] + messages

    def _current_datetime_message(self) -> Dict[str, str]:
        now = datetime.now()
        human = now.strftime("%A, %B %d, %Y %H:%M:%S")
        return {
            "role": "system",
            "content": f"Current datetime (system clock): {human} (ISO: {now.isoformat()})",
        }

    def _naive_summarizer(self, messages: List["Message"], budget: int) -> str:
        parts: List[str] = []
        used = 0
        for msg in messages:
            snippet = f"{msg.role}: {msg.content}"
            tokens = self.stm_tokenizer(snippet)
            if used + tokens > budget:
                remaining = max(budget - used, 0)
                truncated = self._truncate_to_budget(snippet, remaining)
                if truncated:
                    parts.append(truncated)
                break
            parts.append(snippet)
            used += tokens
        return "\n".join(parts)

    def _truncate_to_budget(self, snippet: str, remaining: int) -> str:
        if remaining <= 0:
            return ""

        if self.stm_tokenizer(snippet) <= remaining:
            return snippet

        overhead = max(self.stm_tokenizer(""), 0)
        content_budget = max(remaining - overhead, 1)

        truncated = self._tkn.truncate_text(self.model_name, snippet, content_budget)
        while truncated and self.stm_tokenizer(truncated) > remaining:
            content_budget -= 1
            if content_budget <= 0:
                return ""
            truncated = self._tkn.truncate_text(self.model_name, truncated, content_budget)

        return truncated

    def _debug_print_messages(self) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        msgs = self.stm.get_messages()
        lines = ["===== STM MESSAGES ====="]
        for idx, m in enumerate(msgs):
            imp = m.importance.name if hasattr(m.importance, "name") else str(m.importance)
            preview = (m.content[:80] + "…") if len(m.content) > 80 else m.content
            lines.append(
                f"[{idx:03}] role={m.role:<9} "
                f"importance={imp:<8} "
                f"tokens={m.tokens:<4} "
                f"summary={str(m.meta.get('summary', False)):5} "
                f"content={preview}"
            )
        lines.append("========================")
        self.logger.debug("\n".join(lines))
