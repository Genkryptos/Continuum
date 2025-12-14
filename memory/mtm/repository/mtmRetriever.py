"""
Retriever for mid-term memories that blends vector similarity with heuristics.

The retriever queries embeddings from the repository, re-ranks them with
recency/importance/session affinity, and updates access stats for the chosen
memories.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List

from psycopg2.extras import RealDictCursor

from memory.mtm.repository.mtmRepository import MTMRepository
from memory.mtm.service.embeddingService import EmbeddingService
from memory.mtm.importance import normalize_importance_value, importance_to_score


logger = logging.getLogger(__name__)


class MTMRetriever:
    """Coordinate similarity search and scoring for MTM memories."""
    def __init__(self, mtm_repo: MTMRepository, embedding_service: EmbeddingService):
        self.mtm_repo = mtm_repo
        self.embedding_service = embedding_service

    def search(
        self,
        user_id: str,
        agent_id: str,
        session_key: str,
        query: str,
        top_k: int = 10,
        importance_weight: float = 0.2,
        recency_weight: float = 0.1,
        session_match_boost: float = 0.05,
    ) -> List[Dict]:
        """Retrieve memories relevant to the query, ranked by composite score."""
        try:
            query_embedding = self.embedding_service.embed_text(query)
            now = datetime.now(timezone.utc)

            with self.mtm_repo.connection() as db:
                candidate_limit = max(top_k * 3, 20)
                with db.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        WITH base AS (
                            SELECT
                                m.id,
                                m.content,
                                m.scope,
                                m.source,
                                m.importance,
                                m.created_at,
                                m.access_count,
                                (m.embedding <=> %s::vector) AS distance,
                                CASE WHEN s.session_key = %s THEN TRUE ELSE FALSE END AS session_match
                            FROM mtm_memories m
                            JOIN mtm_sessions s ON m.session_id = s.id
                            WHERE (s.session_key = %s OR (s.user_id = %s AND s.agent_id = %s))
                              AND (m.expires_at IS NULL OR m.expires_at > now())
                              AND m.embedding IS NOT NULL
                        )
                        SELECT
                            id,
                            content,
                            scope,
                            source,
                            importance,
                            created_at,
                            access_count,
                            distance,
                            session_match
                        FROM base
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (query_embedding, session_key, session_key, user_id, agent_id, candidate_limit),
                    )
                    rows = cur.fetchall()
        except Exception:
            logger.exception(
                "MTM retrieval failed; falling back to STM only",
                extra={
                    "agent_id": agent_id,
                    "session_key": session_key,
                    "user_id": user_id,
                },
            )
            return []

        results: List[Dict] = []
        for row in rows:
            distance = float(row["distance"]) if row["distance"] is not None else 1.0
            similarity = 1.0 - distance

            importance_raw = normalize_importance_value(row.get("importance"))
            importance_norm = importance_to_score(importance_raw)

            created_at = row.get("created_at")
            if isinstance(created_at, datetime):
                age_days = max(
                    0.0,
                    (now - created_at).total_seconds() / 86400.0,
                )
            else:
                age_days = 0.0

            HALF_LIFE_DAYS = 7.0
            recency_score = math.exp(-age_days * math.log(2) / HALF_LIFE_DAYS)
            session_match = bool(row.get("session_match"))

            composite_score = similarity
            composite_score += importance_weight * importance_norm
            composite_score += recency_weight * recency_score
            if session_match:
                composite_score += session_match_boost

            results.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "scope": row["scope"],
                    "source": row["source"],
                    "importance": importance_raw,
                    "created_at": created_at,
                    "access_count": row["access_count"],
                    "distance": distance,
                    "similarity": similarity,
                    "recency_score": recency_score,
                    "session_match": session_match,
                    "composite_score": composite_score,
                }
            )

        results.sort(key=lambda r: r["composite_score"], reverse=True)
        top_results = results[:top_k]

        if top_results:
            self.mtm_repo.update_access_stats([r["id"] for r in top_results])

        return top_results

    def search_by_tags(
        self,
        user_id: str,
        tag_filter: Dict[str, str],
        top_k: int = 10,
    ) -> List[Dict]:
        """Return most recent memories that match the provided tag filters."""
        try:
            with self.mtm_repo.connection() as db:
                with db.cursor(cursor_factory=RealDictCursor) as cur:
                    conditions = ["s.user_id = %s"]
                    params = [user_id]
                    for key, value in tag_filter.items():
                        conditions.append("m.tags ->> %s = %s")
                        params.extend([key, value])
                    conditions.append("(m.expires_at IS NULL OR m.expires_at > now())")

                    where_sql = " AND ".join(conditions)

                    sql = f"""
                        SELECT
                            m.id,
                            m.content,
                            m.importance,
                            m.scope,
                            m.source,
                            m.created_at,
                            m.tags
                        FROM mtm_memories m
                        JOIN mtm_sessions s ON m.session_id = s.id
                        WHERE {where_sql}
                        ORDER BY m.created_at DESC
                        LIMIT %s;
                    """
                    params.append(top_k)

                    cur.execute(sql, params)
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            logger.exception(
                "MTM search_by_tags failed; returning empty results",
                extra={"user_id": user_id, "tag_filter": tag_filter},
            )
            return []
