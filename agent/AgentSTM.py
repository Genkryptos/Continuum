import logging
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional

from agent.agent import LocalAgent

from helper.tknCounter import TknCounter
from memory.stm.ConversationSTM import STMConfig, STMCallbacks, ConversationSTM, Importance
from settings import LLM_MODEL


class TextTokenCounter:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self._tkn = TknCounter()

    def __call__(self, text: str) -> int:
        return self._tkn.count_tokens(self.model, [{"role": "user", "content": text}])


class AgentSTM:
    """
    Agent loop that uses ConversationSTM + local Ollama model.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        llm: Optional[LocalAgent] = None,
        tokenizer: Optional[Callable[[str], int]] = None,
        config: Optional[STMConfig] = None,
        callbacks: Optional[STMCallbacks] = None,
        min_messages_to_compress: int = 4,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.logger = logging.getLogger("AgentSTM")
        self.model_name = model_name
        self.llm = llm or LocalAgent(model=model_name)
        self.tokenizer: Callable[[str], int] = tokenizer or TextTokenCounter(model=model_name)
        self._tkn = TknCounter()
        config = config or STMConfig(
            max_tokens=500,              # overall STM budget
            max_messages=10,
            compress_threshold_ratio=0.8, # trigger maybe_compress() >= 80%
            max_summary_tokens=128,
            reserved_for_response=64,
            name="local_agent_session",
        )

        callbacks = callbacks or STMCallbacks()
        callbacks = STMCallbacks(
            on_evict=callbacks.on_evict or self._on_evict,
            on_compress=callbacks.on_compress or self._on_compress,
        )
        self.system_prompt = system_prompt
        self.stm = ConversationSTM(
            config=config,
            tokenizer_fn=self.tokenizer,
            callbacks=callbacks,
        )
        self.min_messages_to_compress = min_messages_to_compress
        self.turn = 0

    # --- callbacks ------------------------------------------------------------

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

    # --- core loop ------------------------------------------------------------

    def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        self.turn += 1
        self.logger.info(f"════════════════ Turn {self.turn} ════════════════")
        self.logger.info(f"User: {user_input}")

        # 1) add user message
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

        # 2) build prompt
        prompt_messages = self.stm.get_prompt_messages(system_prompt=self.system_prompt)

        # 3) call local model
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
                "rolled_back_user_message": bool(rolled_back),
            }

        if not result["success"]:
            self.logger.error(f"LLM error: {result['error']}")
            rolled_back = self.stm.rollback_last_user_message()
            if rolled_back:
                self.logger.info(
                    "Rolled back user message after failed LLM response to avoid "
                    "unanswered prompts",
                )

            stats_after_error = self.stm.stats()
            return {
                "response": None,
                "error": result["error"],
                "success": False,
                "stm_stats_before": stats_before,
                "stm_stats_after": stats_after_error,
                "rolled_back_user_message": bool(rolled_back),
            }

        assistant_reply = result["response"]
        self.logger.info(f"Assistant: {assistant_reply[:200]}{'...' if len(assistant_reply) > 200 else ''}")
        self.logger.info(
            f"LLM latency={result['latency']:.2f}s tokens={result['tokens_used']}"
        )

        # 4) add assistant reply
        self.stm.add_assistant_message(
            content=assistant_reply,
            importance=Importance.NORMAL,
            meta={
                "turn": self.turn,
                "timestamp": datetime.now().isoformat(),
                "model": self.llm.model,
            },
        )

        stats_after = self.stm.stats()
        self.logger.info(
            f"[STM After] messages={stats_after['messages']} "
            f"tokens={stats_after['tokens']} "
            f"util={stats_after['utilization']:.1%}"
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
        self._debug_print_messages()
        return {
            "response": assistant_reply,
            "success": True,
            "error": None,
            "latency": result["latency"],
            "tokens_used": result["tokens_used"],
            "stm_stats_before": stats_before,
            "stm_stats_after": stats_after,
            "compression_error": compression_error,
        }

    def _naive_summarizer(self, messages: List["Message"], budget: int) -> str:
        parts: List[str] = []
        used = 0
        for msg in messages:
            snippet = f"{msg.role}: {msg.content}"
            tokens = self.tokenizer(snippet)
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

        if self.tokenizer(snippet) <= remaining:
            return snippet

        overhead = max(self.tokenizer(""), 0)
        content_budget = max(remaining - overhead, 1)

        truncated = self._tkn.truncate_text(self.model_name, snippet, content_budget)
        while truncated and self.tokenizer(truncated) > remaining:
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