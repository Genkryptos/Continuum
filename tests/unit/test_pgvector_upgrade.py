"""
tests/unit/test_pgvector_upgrade.py
===================================
Unit tests for ``continuum.db.pgvector_upgrade``.

* Pure helpers (version parsing, recall math, query builder) need no DB.
* The async orchestration (`verify_upgrade`) is driven by a stateful
  ``MockConnection`` injected via ``conn_factory`` — no psycopg3, no Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from continuum.db.pgvector_upgrade import (
    MIN_PGVECTOR,
    UpgradeReport,
    build_knn_query,
    meets_minimum,
    parse_pgvector_version,
    recall_at_k,
    to_halfvec_literal,
    verify_upgrade,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParseVersion:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("0.8.0", (0, 8, 0)),
            ("0.8", (0, 8, 0)),
            ("0.10.2", (0, 10, 2)),
            ("0.7.0-dev", (0, 7, 0)),
            ("0.8.0+meta", (0, 8, 0)),
            ("1.0.0", (1, 0, 0)),
            ("  0.8.0  ", (0, 8, 0)),
        ],
    )
    def test_parses(self, raw: str, expected: tuple[int, int, int]) -> None:
        assert parse_pgvector_version(raw) == expected

    @pytest.mark.parametrize("bad", ["", "abc", "x.y.z", None])
    def test_rejects_garbage(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            parse_pgvector_version(bad)


class TestMeetsMinimum:
    def test_default_minimum_is_0_8_0(self) -> None:
        assert MIN_PGVECTOR == (0, 8, 0)

    @pytest.mark.parametrize(
        "version, ok",
        [
            ((0, 8, 0), True),
            ((0, 8, 1), True),
            ((0, 10, 0), True),
            ((1, 0, 0), True),
            ((0, 7, 9), False),
            ((0, 5, 0), False),
        ],
    )
    def test_boundary(self, version: tuple[int, int, int], ok: bool) -> None:
        assert meets_minimum(version) is ok

    def test_custom_minimum(self) -> None:
        assert meets_minimum((0, 7, 0), (0, 7, 0)) is True
        assert meets_minimum((0, 6, 9), (0, 7, 0)) is False


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k([1, 2, 3], [1, 2, 3]) == 1.0

    def test_partial_recall(self) -> None:
        # 2 of 4 ground-truth ids found → 0.5
        assert recall_at_k([1, 2, 9, 8], [1, 2, 3, 4]) == 0.5

    def test_zero_recall(self) -> None:
        assert recall_at_k([7, 8], [1, 2]) == 0.0

    def test_empty_truth_is_one(self) -> None:
        assert recall_at_k([1, 2], []) == 1.0

    def test_only_top_k_of_approx_counted(self) -> None:
        # truth has 2 ids; only approx[:2] inspected, so the trailing
        # correct id (3) past position k must NOT count.
        assert recall_at_k([9, 8, 3], [3, 4]) == 0.0


class TestLiteralAndQuery:
    def test_halfvec_literal_format(self) -> None:
        assert to_halfvec_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"

    def test_literal_coerces_to_float(self) -> None:
        assert to_halfvec_literal([1, 2]) == "[1.0,2.0]"

    def test_build_knn_query_shape(self) -> None:
        sql = build_knn_query(k=5)
        assert "<=>" in sql  # cosine-distance operator
        assert "%(q)s::halfvec" in sql  # halfvec-cast bind param
        assert "%(k)s" in sql  # limit bind param
        assert "ORDER  BY" in sql
        assert "cosine_similarity" in sql
        assert "memory_nodes" in sql

    def test_build_knn_query_custom_columns(self) -> None:
        sql = build_knn_query(table="t", column="emb", select_columns=("id",), k=3)
        assert "FROM   t" in sql
        assert "emb <=> %(q)s::halfvec" in sql


# ---------------------------------------------------------------------------
# Mock DB layer
# ---------------------------------------------------------------------------


class MockCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class MockConnection:
    """
    Stateful fake connection.

    Distinguishes the exact (seq-scan) KNN from the ANN KNN by remembering
    which ``SET LOCAL`` preceded the otherwise-identical SELECT.
    """

    def __init__(
        self,
        *,
        version: str = "0.8.0",
        column_type: str = "halfvec(1024)",
        index: tuple[str | None, int, str] = ("hnsw", 2048, "2048 bytes"),
        row_count: int = 3,
        n_samples: int = 3,
        exact_ids: list[Any] | None = None,
        approx_ids: list[Any] | None = None,
    ) -> None:
        self.version = version
        self.column_type = column_type
        self.index = index
        self.row_count = row_count
        self.n_samples = n_samples
        self.exact_ids = exact_ids if exact_ids is not None else [1, 2, 3]
        self.approx_ids = approx_ids if approx_ids is not None else [1, 2, 3]
        self._mode: str | None = None
        self.executed: list[str] = []

    async def execute(self, sql: str, params: Any = None) -> MockCursor:
        self.executed.append(sql)

        if (
            "enable_indexscan" in sql
            or "enable_bitmapscan" in sql
            or ("enable_indexonlyscan" in sql)
        ):
            self._mode = "exact"
            return MockCursor()
        if "hnsw.ef_search" in sql:
            self._mode = "approx"
            return MockCursor()

        if "pg_extension" in sql:
            return MockCursor([] if self.version is None else [{"v": self.version}])
        if "format_type" in sql:
            return MockCursor([{"t": self.column_type}])
        if "pg_relation_size" in sql:
            method, nbytes, pretty = self.index
            if method is None:
                return MockCursor([])
            return MockCursor([{"method": method, "bytes": nbytes, "pretty": pretty}])
        if "COUNT(*)" in sql:
            return MockCursor([{"n": self.row_count}])
        if "random()" in sql:
            return MockCursor([{"v": f"[0.{i}]"} for i in range(self.n_samples)])
        if "<=>" in sql and "ORDER  BY" in sql:
            ids = self.exact_ids if self._mode == "exact" else self.approx_ids
            return MockCursor([{"id": i} for i in ids])

        return MockCursor()


def _factory(conn: MockConnection) -> Any:
    @asynccontextmanager
    async def _f() -> AsyncIterator[MockConnection]:
        yield conn

    return _f


# ---------------------------------------------------------------------------
# verify_upgrade orchestration
# ---------------------------------------------------------------------------


class TestVerifyUpgrade:
    async def test_happy_path_halfvec_perfect_recall(self) -> None:
        conn = MockConnection(
            version="0.8.0",
            column_type="halfvec(1024)",
            index=("hnsw", 4096, "4096 bytes"),
            row_count=3,
            n_samples=3,
            exact_ids=[1, 2, 3],
            approx_ids=[1, 2, 3],
        )
        report = await verify_upgrade(conn_factory=_factory(conn), k=3)

        assert isinstance(report, UpgradeReport)
        assert report.version_ok is True
        assert report.is_halfvec is True
        assert report.column_type == "halfvec(1024)"
        assert report.index_method == "hnsw"
        assert report.index_size_bytes == 4096
        assert report.row_count == 3
        assert report.recall_at_k == 1.0

    async def test_recall_below_one_is_measured(self) -> None:
        # exact top-3 = {1,2,3}; ANN returns {1,2,9} → recall 2/3 each probe
        conn = MockConnection(
            row_count=2,
            n_samples=2,
            exact_ids=[1, 2, 3],
            approx_ids=[1, 2, 9],
        )
        report = await verify_upgrade(conn_factory=_factory(conn), k=3)
        assert report.recall_at_k == pytest.approx(2 / 3)
        assert report.sample_size == 2

    async def test_pre_upgrade_state_detected(self) -> None:
        # Old pgvector + still vector(1024) + no index at all.
        conn = MockConnection(
            version="0.6.0",
            column_type="vector(1024)",
            index=(None, 0, "0 bytes"),
            row_count=0,
        )
        report = await verify_upgrade(conn_factory=_factory(conn))

        assert report.version_ok is False
        assert report.is_halfvec is False
        assert report.column_type == "vector(1024)"
        assert report.index_name is None
        assert report.index_method is None
        assert report.recall_at_k is None  # empty table → not measured
        assert report.sample_size == 0

    async def test_empty_table_skips_recall(self) -> None:
        conn = MockConnection(row_count=0)
        report = await verify_upgrade(conn_factory=_factory(conn))
        assert report.recall_at_k is None
        assert report.sample_size == 0

    async def test_missing_extension_raises(self) -> None:
        conn = MockConnection(version=None)  # pg_extension returns no row
        with pytest.raises(RuntimeError, match="not installed"):
            await verify_upgrade(conn_factory=_factory(conn))

    async def test_summary_is_one_line(self) -> None:
        conn = MockConnection()
        report = await verify_upgrade(conn_factory=_factory(conn), k=3)
        s = report.summary()
        assert "\n" not in s
        assert "recall@3=" in s
        assert "pgvector=0.8.0" in s

    async def test_exact_and_ann_paths_both_invoked(self) -> None:
        conn = MockConnection(row_count=1, n_samples=1)
        await verify_upgrade(conn_factory=_factory(conn), k=2)
        joined = "\n".join(conn.executed)
        assert "enable_indexscan = off" in joined  # exact path ran
        assert "hnsw.ef_search" in joined  # ANN path ran

    async def test_requires_dsn_or_factory(self) -> None:
        with pytest.raises(ValueError, match="dsn or conn_factory"):
            await verify_upgrade()
