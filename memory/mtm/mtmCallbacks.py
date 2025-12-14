"""
Callbacks that persist STM events (evictions/compressions) into MTM storage.

The callbacks translate STM messages into MTM records, handling embedding
creation, per-user pruning, and error logging so STM callers can remain thin.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from memory.mtm.repository.mtmRepository import MTMRepository
from memory.mtm.service.embeddingService import EmbeddingService
from memory.stm.ConversationSTM import Importance, Message
from memory.mtm.importance import normalize_importance_value


class MTMCallbacks:
    """Bridge STM lifecycle events to persisted MTM memories."""
    ALLOWED_ROLES = frozenset({"user", "assistant"})

    def __init__(
        self,
        mtm_repo: MTMRepository,
        embedding_service: EmbeddingService,
        user_id: str,
        agent_id: str,
        session_key: str,
        max_memories_per_user: Optional[int] = None,
    ):
        self.mtm_repo = mtm_repo
        self.embedding_service = embedding_service
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_key = session_key
        self.session_id = None
        self._logger = logging.getLogger(__name__)
        self._warned_missing_session = False
        self._max_memories_per_user = max_memories_per_user

    def _ensure_session(self) -> Optional[int]:
        if self.session_id is None:
            self.session_id = self.mtm_repo.create_or_get_session(
                self.user_id,
                self.agent_id,
                self.session_key,
            )
            if self.session_id is None and not self._warned_missing_session:
                self._warned_missing_session = True
                self._logger.warning(
                    "Failed to create or fetch MTM session for user=%s agent=%s "
                    "session_key=%s; memories will not be persisted",
                    self.user_id,
                    self.agent_id,
                    self.session_key,
                )
        return self.session_id

    def _convert_importance(self, importance: Any) -> int:
        """Convert enum/int importance to the clamped numeric scale used in MTM."""
        return normalize_importance_value(importance)

    def _prune_if_configured(self) -> None:
        if not self._max_memories_per_user:
            return
        try:
            self.mtm_repo.prune_per_user_caps(
                user_id=self.user_id, max_per_user=self._max_memories_per_user
            )
        except Exception:
            self._logger.exception("Failed to prune MTM memories for user=%s", self.user_id)

    def _base_tags(self, msg: Message) -> Dict[str, Any]:
        tags = {
            "type": "episode",
            "role": msg.role,
        }
        if hasattr(msg, "importance") and hasattr(msg.importance, "name"):
            tags["original_importance"] = msg.importance.name
        return tags

    def on_evict(self, msg: Message) -> None:
        """Store evicted conversational messages as episodic MTM memories."""
        # Skip non-conversational messages
        if msg.role not in self.ALLOWED_ROLES:
            return

        session_id = self._ensure_session()
        if session_id is None:
            return

        start = time.perf_counter()
        try:
            msg_embedding = self.embedding_service.embed_text(msg.content)
            importance_value = self._convert_importance(msg.importance)

            result = self.mtm_repo.add_memory(
                session_id=session_id,
                scope="episode",
                source="stm_evict",
                content=msg.content,
                importance=importance_value,
                tags=self._base_tags(msg),
                embeddings=msg_embedding,
                expires_at=None,
            )

            if result is None:
                self._logger.warning(
                    "Failed to persist MTM memory for session_id=%s (role=%s)",
                    session_id,
                    msg.role,
                )
            self._prune_if_configured()
        except Exception:
            self._logger.exception(
                "Failed to store evicted message for session_id=%s role=%s",
                session_id,
                msg.role,
            )
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            self._logger.debug(
                "Eviction persisted for session_id=%s role=%s in %.1fms",
                session_id,
                msg.role,
                duration_ms,
            )

    def on_compress(self, old_messages: List[Message], summary: Message) -> None:
        """Persist summaries plus the compressed messages they replaced."""
        # Lazily create or fetch MTM session
        session_id = self._ensure_session()
        if session_id is None:
            return

        start = time.perf_counter()
        embed_duration_ms: Optional[float] = None
        summary_group_id = str(uuid.uuid4())

        # Attempt batch embedding to minimize network calls
        embed_messages: List[Message] = [summary] + [
            m for m in old_messages if m.role in self.ALLOWED_ROLES
        ]
        embeddings: Optional[List[Any]] = None

        if embed_messages:
            try:
                contents = [m.content for m in embed_messages]
                embeddings = self.embedding_service.embed_batch(contents)
                if len(embeddings) != len(embed_messages):
                    self._logger.error(
                        "Batch embedding length mismatch for session_id=%s: expected %s got %s",
                        session_id,
                        len(embed_messages),
                        len(embeddings),
                    )
                    embeddings = None
                else:
                    embed_duration_ms = (time.perf_counter() - start) * 1000
            except Exception:
                self._logger.exception(
                    "Failed to batch embed compressed messages for session_id=%s", session_id
                )

        memories_to_store: List[Dict[str, Any]] = []

        def _append_summary(embedding_value: Any) -> None:
            summary_tags = {
                "type": "conversation_summary",
                "num_messages_compressed": len(old_messages),
                "summary_group_id": summary_group_id,
            }
            if hasattr(summary, "importance") and hasattr(summary.importance, "name"):
                summary_tags["original_importance"] = summary.importance.name

            memories_to_store.append(
                {
                    "session_id": session_id,
                    "scope": "stm_summary",
                    "source": "stm_compress",
                    "content": summary.content,
                    "importance": Importance.HIGH.value,
                    "tags": summary_tags,
                    "embedding": embedding_value,
                    "embedding_model": getattr(self.embedding_service, "model_name", None),
                    "expires_at": None,
                }
            )

        def _append_message(msg: Message, embedding_value: Any) -> None:
            importance_value = self._convert_importance(msg.importance)
            msg_tags = self._base_tags(msg)
            msg_tags["summary_group_id"] = summary_group_id

            memories_to_store.append(
                {
                    "session_id": session_id,
                    "scope": "episode",
                    "source": "stm_compress",
                    "content": msg.content,
                    "importance": importance_value,
                    "tags": msg_tags,
                    "embedding": embedding_value,
                    "embedding_model": getattr(self.embedding_service, "model_name", None),
                    "expires_at": None,
                }
            )

        def _fallback_embed_and_append(msg: Message, is_summary: bool = False) -> None:
            try:
                embedding_val = self.embedding_service.embed_text(msg.content)
            except Exception:
                self._logger.exception(
                    "Failed to store compressed message for session_id=%s role=%s",
                    session_id,
                    msg.role,
                )
                return

            if is_summary:
                _append_summary(embedding_val)
            else:
                _append_message(msg, embedding_val)

        # Prefer batch embeddings when available, otherwise fall back to per-message calls
        fallback_needed = embeddings is None
        if embeddings is not None:
            embedding_iter = iter(embeddings)
            summary_embedding = next(embedding_iter, None)
            if summary_embedding is None:
                fallback_needed = True
            else:
                _append_summary(summary_embedding)
                for msg in old_messages:
                    if msg.role not in self.ALLOWED_ROLES:
                        continue
                    try:
                        _append_message(msg, next(embedding_iter))
                    except StopIteration:
                        self._logger.error(
                            "Batch embeddings exhausted early for session_id=%s", session_id
                        )
                        fallback_needed = True
                        break

        if fallback_needed:
            memories_to_store.clear()
            _fallback_embed_and_append(summary, is_summary=True)
            for msg in old_messages:
                if msg.role not in self.ALLOWED_ROLES:
                    continue
                _fallback_embed_and_append(msg)

            if embed_duration_ms is None:
                embed_duration_ms = (time.perf_counter() - start) * 1000

        if memories_to_store:
            try:
                db_start = time.perf_counter()
                result_ids = self.mtm_repo.add_memory_batch(memories_to_store)
                if not result_ids:
                    self._logger.warning(
                        "Failed to persist MTM compressed batch for session_id=%s", session_id
                    )
            except Exception:
                self._logger.exception(
                    "Failed to persist MTM compressed batch for session_id=%s", session_id
                )
            else:
                db_duration_ms = (time.perf_counter() - db_start) * 1000
                total_ms = (time.perf_counter() - start) * 1000
                self._logger.debug(
                    "Compressed MTM batch persisted for session_id=%s: %s memories (embed=%.1fms, db=%.1fms, total=%.1fms)",
                    session_id,
                    len(memories_to_store),
                    embed_duration_ms if embed_duration_ms is not None else -1,
                    db_duration_ms,
                    total_ms,
                )

        self._prune_if_configured()
