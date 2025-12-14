"""
Builds a prompt that combines STM messages with mid-term memory (MTM) results.

The augmenter is responsible for budgeting tokens between STM context, MTM
retrieved memories, and the reserved response allowance so the final prompt
stays within the target model context window.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from helper.tknCounter import TknCounter
from memory.mtm.repository.mtmRetriever import MTMRetriever
from memory.stm.ConversationSTM import ConversationSTM


@dataclass
class ContextResult:
    """Bundle of the final prompt, included memories, and debug metrics."""
    prompt_messages: List[Dict[str, str]]
    mtm_memories: List[Dict]
    debug_info: Dict[str, object]


class ContextAugmenter:
    """Orchestrates MTM retrieval and prompt assembly within a token budget."""
    def __init__(
        self,
        stm: ConversationSTM,
        mtm_retriever: MTMRetriever,
        tokenizer: TknCounter,
        model_name: str,
        max_context_tokens: int = 8000,
        answer_fraction: float = 0.25,
        min_mtm_budget: int = 256,
        mtm_top_k: int = 5,
    ):
        self.stm = stm
        self.mtm_retriever = mtm_retriever
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.answer_fraction = answer_fraction
        self.min_mtm_budget = min_mtm_budget
        self.mtm_top_k = mtm_top_k

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return self.tokenizer.count_request_tokens(
            model=self.model_name,
            messages=messages,
        )

    def build_context(
        self,
        user_id: str,
        agent_id: str,
        session_key: str,
        user_query: str,
        system_prompt: Optional[str] = None,
    ) -> ContextResult:
        if system_prompt is None:
            system_prompt = "You are a helpful, intelligent assistant."

        system_msg = {"role": "system", "content": system_prompt}

        stm_msgs = self.stm.get_prompt_messages(system_prompt=None)

        # Baseline prompt without MTM
        base_prompt: List[Dict[str, str]] = [system_msg] + stm_msgs
        base_tokens = self._count_tokens(base_prompt)

        # Reserve tokens for model output
        reserved_for_answer = int(self.max_context_tokens * self.answer_fraction)

        # Tokens left for *everything* (we've already used base_tokens)
        remaining_for_context = self.max_context_tokens - reserved_for_answer - base_tokens

        # If no room for MTM, return baseline
        if remaining_for_context <= self.min_mtm_budget:
            debug_info = {
                "system+stm+user_tokens": base_tokens,
                "mtm_used": 0,
                "mtm_memories_used": 0,
                "total_messages": len(base_prompt),
                "reason": "no_budget_for_mtm",
            }
            print(f"[ContextAugmenter] {debug_info}")
            return ContextResult(
                prompt_messages=base_prompt,
                mtm_memories=[],
                debug_info=debug_info,
            )

        # -------- 2) Retrieve MTM memories --------
        mtm_results = self.mtm_retriever.search(
            user_id=user_id,
            agent_id=agent_id,
            session_key=session_key,
            query=user_query,
            top_k=self.mtm_top_k,
        )

        mtm_msgs: List[Dict[str, str]] = []
        mtm_tokens_used = 0
        memories_used = 0

        current_tokens = base_tokens

        for mem in mtm_results:
            mem_msg = {
                "role": "system",
                "content": f"[Memory: {mem['scope']}] {mem['content']}",
            }

            candidate_prompt = [system_msg] + mtm_msgs + [mem_msg] + stm_msgs
            candidate_tokens = self._count_tokens(candidate_prompt)

            if candidate_tokens + reserved_for_answer > self.max_context_tokens:
                break

            mtm_msgs.append(mem_msg)
            mtm_tokens_used = candidate_tokens - current_tokens
            current_tokens = candidate_tokens
            memories_used += 1

        final_prompt = [system_msg] + mtm_msgs + stm_msgs

        debug_info = {
            "system+stm+user_tokens": base_tokens,
            "total_tokens_with_mtm": current_tokens,
            "mtm_tokens_used": mtm_tokens_used,
            "mtm_memories_retrieved": len(mtm_results),
            "mtm_memories_used": memories_used,
            "total_messages": len(final_prompt),
        }
        print(f"[ContextAugmenter] {debug_info}")

        return ContextResult(
            prompt_messages=final_prompt,
            mtm_memories=mtm_results,
            debug_info=debug_info,
        )
