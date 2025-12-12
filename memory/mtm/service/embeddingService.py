from abc import ABC, abstractmethod
from typing import List


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails to produce embeddings."""


class EmbeddingService(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a vector representation."""

    @abstractmethod
    def embed_batch(self, text: List[str]) -> List[List[float]]:
        """Embed a batch of strings into vector representations."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the underlying embedding model name."""
