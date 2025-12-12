from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class SummarizationService(ABC):
    @abstractmethod
    def summarize_message(
        self,
        messages: List[Dict[str, str]],
        max_token: int,
        context: Optional[str] = None,
    ) -> str:
        """Summarize a list of chat messages."""
