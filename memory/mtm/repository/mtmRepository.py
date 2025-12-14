"""
PostgreSQL-backed repository for mid-term memory (MTM) sessions and memories.

The repository isolates connection pooling concerns and provides small helpers
for inserting, retrieving, and pruning conversation memories.
"""

from contextlib import contextmanager
from datetime import datetime
import logging
from typing import Any, Optional, Dict, List
from types import SimpleNamespace
try:
    import psycopg2
    from psycopg2 import pool as psycopg2_pool
    from psycopg2.extras import RealDictCursor, Json
except ImportError:  # pragma: no cover - psycopg2 may be optional in tests
    psycopg2 = SimpleNamespace(connect=None)
    psycopg2_pool = None
    RealDictCursor = None
    Json = None

logger = logging.getLogger(__name__)


class MTMRepository:
    """Lightweight data-access layer around the MTM Postgres schema."""

    def __init__(
        self,
        db_url: str,
        *,
        use_pool: bool = True,
        pool_minconn: int = 1,
        pool_maxconn: int = 5,
        connection_pool: Optional[Any] = None,
    ):
        self.db_url = db_url
        self._pool: Optional[Any] = connection_pool
        if use_pool and self._pool is None:
            if psycopg2_pool is None:
                raise ImportError("psycopg2 is required for pooled connections")
            try:
                self._pool = psycopg2_pool.ThreadedConnectionPool(
                    pool_minconn, pool_maxconn, dsn=self.db_url
                )
            except Exception:
                logger.exception("Failed to initialize MTM connection pool")
                raise

    def get_connection(self):
        """Fetch a database connection from the pool or create one directly."""
        if self._pool:
            return self._pool.getconn()
        if psycopg2 is None:
            raise ImportError("psycopg2 is required for direct connections")
        return psycopg2.connect(self.db_url)

    def release_connection(self, conn):
        """Return a pooled connection or close a direct one."""
        if conn is None:
            return
        if self._pool:
            self._pool.putconn(conn)
        else:
            conn.close()

    @contextmanager
    def connection(self):
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)

    def close_pool(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None

    @staticmethod
    def _rollback_safely(db) -> None:
        try:
            db.rollback()
        except Exception:
            logger.exception("Failed to rollback MTM transaction")

    def create_or_get_session(
            self,
            user_id: str,
            agent_id: str,
            session_key: str,
            metadata: Optional[Dict] = None,
    ) -> Optional[int]:
        """Idempotently create a session row and return its id."""
        with self.connection() as db:
            try:
                with db.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO mtm_sessions (user_id, agent_id, session_key, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, agent_id, session_key)
                        DO UPDATE SET last_accessed_at = now()
                        RETURNING id;
                        """,
                        (user_id, agent_id, session_key, Json(metadata) if metadata is not None else None),
                    )
                    result = cur.fetchone()
                    db.commit()
                    return result["id"] if result else None
            except Exception:
                logger.exception("Error in create_or_get_session")
                self._rollback_safely(db)
                return None




    def add_memory(
        self,
        session_id: int,
        scope: str,
        source: str,
        content: str,
        importance: int,
        tags: Dict,
        embeddings: Optional[List[float]] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """Insert a single MTM memory record and return its identifier."""
        ##Inserting memory in memory table
        with self.connection() as db:
            try:
                with db.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO mtm_memories
                        (session_id, "scope", "source", "content",
                        importance, tags, embedding, expires_at)
                        VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (session_id, scope, source,
                         content, importance, Json(tags),
                         embeddings if embeddings else None
                         , expires_at))
                    result = cur.fetchone()
                    db.commit()
                    return result['id']
            except Exception:
                logger.exception("Error inserting memory")
                self._rollback_safely(db)
                return None

    def add_memory_batch(self, memories: List[Dict]) -> List[int]:
        """Insert multiple memory rows efficiently in a single transaction."""
        with self.connection() as db:
            try:
                with db.cursor(cursor_factory=RealDictCursor) as cur:
                    ids = []
                    for memory in memories:
                        cur.execute(
                            """
                            INSERT INTO mtm_memories
                              (session_id, "scope", "source", "content",
                               importance, tags, embedding, embedding_model, expires_at)
                            VALUES
                              (%(session_id)s, %(scope)s, %(source)s, %(content)s,
                               %(importance)s, %(tags)s, %(embedding)s, %(embedding_model)s, %(expires_at)s)
                            RETURNING id;
                            """,
                            {
                                **memory,
                                "tags": Json(memory.get("tags", {})),
                                "embedding": memory.get("embedding"),  # not 'embeddings'
                                "embedding_model": memory.get("embedding_model"),
                            },
                        )
                        row = cur.fetchone()
                        ids.append(row["id"])
                    db.commit()
                    return ids
            except Exception:
                logger.exception("Error in add_memory_batch")
                self._rollback_safely(db)
                return []

    def update_access_stats(self, memory_ids: List[int]) -> None:
        """Increment access counters for the provided memory IDs."""
        if not memory_ids:
            return
        with self.connection() as db:
            try:
                with db.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(memory_ids))
                    cur.execute(
                        f"""
                        UPDATE mtm_memories
                        SET last_accessed_at = now(),
                            access_count = access_count + 1
                        WHERE id IN ({placeholders});
                        """,
                        memory_ids,
                    )
                    db.commit()
            except Exception:
                logger.exception("Error in update_access_stats")
                self._rollback_safely(db)

    def prune_per_user_caps(self, user_id: str, max_per_user: int = 2000) -> int:
        """Delete least important/oldest memories once a per-user cap is exceeded."""
        with self.connection() as db:
            try:
                with db.cursor() as cur:
                    cur.execute(
                        """
                        WITH to_delete AS (
                            SELECT m.id
                            FROM mtm_memories m
                            JOIN mtm_sessions s ON m.session_id = s.id
                            WHERE s.user_id = %s
                            ORDER BY m.importance ASC, m.created_at ASC
                            OFFSET %s
                        )
                        DELETE FROM mtm_memories
                        WHERE id IN (SELECT id FROM to_delete);
                        """,
                        (user_id, max_per_user),
                    )
                    db.commit()
                    return cur.rowcount
            except Exception:
                logger.exception("Error in prune_per_user_caps")
                self._rollback_safely(db)
                return 0

