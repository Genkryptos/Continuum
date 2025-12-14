import logging
from typing import List, Optional

from LLMManager import LLM

from memory.mtm.service.embeddingService import EmbeddingError, EmbeddingService
from settings import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding provider that calls the OpenAI embeddings API via LLM wrapper."""
    def __init__(self, model_name: Optional[str] = None, client: Optional[LLM] = None):
        self._model_name = model_name or OPENAI_EMBEDDING_MODEL
        self._client = client or LLM(api_key=OPENAI_API_KEY)
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, text: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed text in batches to reduce per-request overhead."""
        try:
            all_vectors: List[List[float]] = []
            effective_batch_size = max(batch_size, 1)

            for start in range(0, len(text), effective_batch_size):
                batch = text[start : start + effective_batch_size]
                response = self._client.embeddings.create(
                    model=self._model_name, input=batch
                )
                embeddings = sorted(response.data, key=lambda item: item.index)
                vectors = [item.embedding for item in embeddings]

                if vectors and self._embedding_dim is None:
                    self._embedding_dim = len(vectors[0])

                all_vectors.extend(vectors)

            return all_vectors
        except Exception as exc:  # pragma: no cover - exercised via tests
            logging.getLogger(__name__).exception(
                "OpenAI embedding request failed for model %s", self._model_name
            )
            raise EmbeddingError(
                f"Failed to embed batch via OpenAI model {self._model_name}: {exc}"
            ) from exc
