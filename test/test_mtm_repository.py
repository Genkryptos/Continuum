import pytest

from memory.mtm.repository.mtmRepository import MTMRepository


class _DummyConnection:
    def __init__(self):
        self.closed = False

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True


class _DummyPool:
    def __init__(self):
        self.conn = _DummyConnection()
        self.get_calls = 0
        self.put_calls = []
        self.closed = False

    def getconn(self):
        self.get_calls += 1
        return self.conn

    def putconn(self, conn):
        self.put_calls.append(conn)

    def closeall(self):
        self.closed = True


def test_pool_connections_are_reused_and_released():
    pool = _DummyPool()
    repo = MTMRepository("postgres://example", connection_pool=pool)

    conn = repo.get_connection()

    assert conn is pool.conn
    assert pool.get_calls == 1

    repo.release_connection(conn)

    assert pool.put_calls == [conn]

    repo.close_pool()
    assert pool.closed is True


def test_context_manager_closes_connection_without_pool(monkeypatch):
    captured = {}

    def fake_connect(dsn):
        captured["conn"] = _DummyConnection()
        return captured["conn"]

    monkeypatch.setattr("memory.mtm.repository.mtmRepository.psycopg2.connect", fake_connect)

    repo = MTMRepository("postgres://example", use_pool=False)

    with repo.connection() as conn:
        assert conn is captured["conn"]

    assert captured["conn"].closed is True


def test_add_memory_rolls_back_on_failure(monkeypatch):
    class _FailingCursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def execute(self, *args, **kwargs):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    class _RollbackConnection(_DummyConnection):
        def __init__(self):
            super().__init__()
            self.rollback_called = False
            self.commit_called = False

        def cursor(self, cursor_factory=None):
            return _FailingCursor(self)

        def rollback(self):
            self.rollback_called = True

        def commit(self):
            self.commit_called = True

    captured = {}

    def fake_connect(dsn):
        captured["conn"] = _RollbackConnection()
        return captured["conn"]

    monkeypatch.setattr("memory.mtm.repository.mtmRepository.psycopg2.connect", fake_connect)

    repo = MTMRepository("postgres://example", use_pool=False)

    result = repo.add_memory(1, "scope", "source", "content", 1, {"a": 1}, embeddings=None)

    assert result is None
    assert captured["conn"].rollback_called is True
    assert captured["conn"].commit_called is False


def test_add_memory_batch_sets_embedding_model(monkeypatch):
    executed_params = []

    class _Cursor:
        def __init__(self, conn):
            self.conn = conn
            self.call_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            executed_params.append(params)
            self.call_count += 1

        def fetchone(self):
            return {"id": self.call_count}

    class _Conn(_DummyConnection):
        def __init__(self):
            super().__init__()
            self.commit_called = False
            self.cursor_calls = 0

        def cursor(self, cursor_factory=None):
            self.cursor_calls += 1
            return _Cursor(self)

        def commit(self):
            self.commit_called = True

    captured = {}

    def fake_connect(dsn):
        captured["conn"] = _Conn()
        return captured["conn"]

    monkeypatch.setattr("memory.mtm.repository.mtmRepository.psycopg2.connect", fake_connect)
    monkeypatch.setattr("memory.mtm.repository.mtmRepository.Json", lambda v: v)

    repo = MTMRepository("postgres://example", use_pool=False)

    result = repo.add_memory_batch(
        [
            {
                "session_id": 1,
                "scope": "episode",
                "source": "stm",
                "content": "a",
                "importance": 2,
                "tags": {},
                "embedding": [0.1, 0.2],
                "embedding_model": "test-model",
                "expires_at": None,
            },
            {
                "session_id": 2,
                "scope": "episode",
                "source": "stm",
                "content": "b",
                "importance": 3,
                "tags": {"foo": "bar"},
                "embedding": None,
                "embedding_model": None,
                "expires_at": None,
            },
        ]
    )

    assert result == [1, 2]
    assert captured["conn"].commit_called is True
    assert [params["embedding_model"] for params in executed_params] == [
        "test-model",
        None,
    ]

