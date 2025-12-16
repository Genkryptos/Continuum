import unittest
from datetime import datetime, timezone

from memory.mtm.repository.mtmRetriever import MTMRetriever


class _FakeEmbeddingService:
    def embed_text(self, text: str):
        return [0.1, 0.2]


class _Cursor:
    def __init__(self, rows, executed):
        self.rows = rows
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows, executed):
        self.rows = rows
        self.executed = executed
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    def cursor(self, cursor_factory=None):
        return _Cursor(self.rows, self.executed)


class _Repo:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []
        self.updated_ids = None

    def connection(self):
        return _Connection(self.rows, self.executed)

    def update_access_stats(self, ids):
        self.updated_ids = ids


class _FailingEmbeddingService:
    def embed_text(self, text: str):
        raise RuntimeError("embedding failed")


class _FailingCursor(_Cursor):
    def execute(self, sql, params=None):
        raise RuntimeError("db blew up")


class _FailingConnection(_Connection):
    def cursor(self, cursor_factory=None):
        return _FailingCursor(self.rows, self.executed)


class _FailingRepo(_Repo):
    def connection(self):
        return _FailingConnection(self.rows, self.executed)


class _FailingTagRepo(_Repo):
    def connection(self):
        raise RuntimeError("cannot connect")


class TestMTMRetriever(unittest.TestCase):
    def test_search_prioritizes_session_matches_and_uses_scoped_filters(self):
        now = datetime.now(timezone.utc)
        rows = [
            {
                "id": 1,
                "content": "other session",
                "scope": "episode",
                "source": "stm",
                "importance": 1,
                "created_at": now,
                "access_count": 0,
                "distance": 0.2,
                "session_match": False,
            },
            {
                "id": 2,
                "content": "current session",
                "scope": "episode",
                "source": "stm",
                "importance": 1,
                "created_at": now,
                "access_count": 0,
                "distance": 0.2,
                "session_match": True,
            },
        ]

        repo = _Repo(rows)
        retriever = MTMRetriever(repo, _FakeEmbeddingService())

        results = retriever.search(
            user_id="user-123",
            agent_id="agent-abc",
            session_key="session-key",
            query="hello",
            top_k=2,
            session_match_boost=0.2,
        )

        self.assertTrue(repo.executed)
        _, params = repo.executed[0]
        self.assertEqual(
            params,
            (
                [0.1, 0.2],
                "session-key",
                "session-key",
                "user-123",
                "agent-abc",
                20,
            ),
        )

        self.assertEqual([r["id"] for r in results], [2, 1])
        self.assertTrue(results[0]["session_match"])
        self.assertEqual(repo.updated_ids, [2, 1])

    def test_search_returns_empty_on_embedding_failure(self):
        repo = _Repo([])
        retriever = MTMRetriever(repo, _FailingEmbeddingService())

        with self.assertLogs("memory.mtm.repository.mtmRetriever", level="ERROR") as log_ctx:
            results = retriever.search(
                user_id="user-123",
                agent_id="agent-abc",
                session_key="session-key",
                query="hello",
            )

        self.assertEqual(results, [])
        self.assertFalse(repo.executed)
        self.assertTrue(any("MTM retrieval failed" in message for message in log_ctx.output))

    def test_search_returns_empty_on_query_failure(self):
        repo = _FailingRepo([])
        retriever = MTMRetriever(repo, _FakeEmbeddingService())

        with self.assertLogs("memory.mtm.repository.mtmRetriever", level="ERROR") as log_ctx:
            results = retriever.search(
                user_id="user-123",
                agent_id="agent-abc",
                session_key="session-key",
                query="hello",
            )

        self.assertEqual(results, [])
        self.assertTrue(any("MTM retrieval failed" in message for message in log_ctx.output))

    def test_search_by_tags_logs_and_returns_empty_on_failure(self):
        repo = _FailingTagRepo([])
        retriever = MTMRetriever(repo, _FakeEmbeddingService())

        with self.assertLogs("memory.mtm.repository.mtmRetriever", level="ERROR") as log_ctx:
            results = retriever.search_by_tags(user_id="user-123", tag_filter={"topic": "x"})

        self.assertEqual(results, [])
        self.assertTrue(any("search_by_tags failed" in msg for msg in log_ctx.output))


if __name__ == "__main__":
    unittest.main()
