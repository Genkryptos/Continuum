from typing import Dict, List, Optional
from LLMManager import LLM
from memory.mtm.service.summarizationService import SummarizationService
from settings import OPENAI_API_KEY, SUMMARIZER_MODEL


class LLMSummarizerService(SummarizationService):
    """Summarization service that delegates to an LLM chat completion."""
    def __init__(self, client: Optional[LLM] = None, model_name: Optional[str] = None):
        self._model_name = model_name or SUMMARIZER_MODEL
        self._client = client or LLM(api_key=OPENAI_API_KEY)

    @property
    def model_name(self) -> str:
        return self._model_name

    def summarize_message(
        self,
        messages: List[Dict[str, str]],
        max_token: int,
        context: Optional[str] = None,
    ) -> str:
        """Condense a list of chat messages into a concise summary via chat completions."""
        conversation_text = "\n".join(
            [f"{message['role'].upper()}:{message['content']}" for message in messages]
        )

        summary_prompt = (
            "You are an expert at summarizing conversations "
            "Preserve Key facts, decisions and context "
            "Be concise and objective"
        )

        user_prompt = (
            f"Summarize the following conversation in at most {max_token} tokens. "
            f"Conversation:\n {conversation_text}"
        )

        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_token,
            temperature=0.3,
        )

        return response.choices[0].message.content
