import logging
from typing import List, Optional

from threading import Lock

from sentence_transformers import SentenceTransformer

from memory.mtm.service.embeddingService import EmbeddingError, EmbeddingService
from settings import LOCAL_EMBEDDING_MODEL


class LocalEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
    ):
        self._model_name = model_name or LOCAL_EMBEDDING_MODEL
        self._lock = Lock()
        try:
            self._model = model or SentenceTransformer(self._model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        except Exception as exc:  # pragma: no cover - exercised via tests
            logging.getLogger(__name__).exception(
                "Failed to load local embedding model %s", self._model_name
            )
            raise EmbeddingError(
                f"Failed to load local embedding model {self._model_name}: {exc}"
            ) from exc

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, text: List[str], batch_size: int = 32) -> List[List[float]]:
        try:
            with self._lock:
                embeddings = self._model.encode(
                    text,
                    batch_size=batch_size,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                )
            return [vector.tolist() for vector in embeddings]
        except Exception as exc:  # pragma: no cover - exercised via tests
            logging.getLogger(__name__).exception(
                "Local embedding request failed for model %s", self._model_name
            )
            raise EmbeddingError(
                f"Failed to embed batch via local model {self._model_name}: {exc}"
            ) from exc
