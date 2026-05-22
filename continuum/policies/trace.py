"""
continuum/policies/trace.py
===========================
``TraceWriter`` — persists :class:`MemoryDecisionTrace` rows to
``memory_decision_traces``.

The writer is intentionally injectable / duck-typed: production callers
pass a ``dsn`` or a pre-built ``conn_factory``; unit tests inject the
``record_sink`` to capture rows without psycopg.

Failures are logged and swallowed — tracing is best-effort, never blocks a
promotion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from continuum.policies.models import MemoryDecisionTrace

log = logging.getLogger(__name__)

#: Optional in-memory sink used by tests: ``async (rows) -> None``.
RecordSink = Callable[[list[dict[str, Any]]], Awaitable[None]]


class TraceWriter:
    """
    Persist ``MemoryDecisionTrace`` rows.

    Parameters
    ----------
    dsn:
        ``postgresql://…``; lazily opens psycopg3 connections per write.
    conn_factory:
        Pre-built async ``conn_factory()`` context manager (tests).
    record_sink:
        Async callable that receives a list of dicts (also tests / custom
        sinks). Mutually exclusive with the DB paths.
    table:
        Override the target table name (default ``memory_decision_traces``).
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        conn_factory: Any | None = None,
        record_sink: RecordSink | None = None,
        table: str = "memory_decision_traces",
    ) -> None:
        if dsn is None and conn_factory is None and record_sink is None:
            raise ValueError(
                "TraceWriter needs one of dsn, conn_factory, or record_sink"
            )
        self._dsn = dsn
        self._conn_factory = conn_factory
        self._record_sink = record_sink
        self._table = table

    @asynccontextmanager
    async def _connect(self) -> Any:
        if self._conn_factory is not None:
            async with self._conn_factory() as conn:
                yield conn
            return
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - via conn_factory
            raise ImportError(
                "psycopg3 is required for TraceWriter's DB path.\n"
                "Install it with:  pip install 'psycopg[binary]>=3.1'"
            ) from exc
        assert self._dsn is not None
        conn = await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True
        )
        try:
            yield conn
        finally:
            await conn.close()

    @staticmethod
    def _trace_row(trace: MemoryDecisionTrace) -> dict[str, Any]:
        return {
            "id": str(trace.trace_id),
            "candidate_id": str(trace.candidate_id),
            "candidate_text": trace.candidate_text,
            "selected_action": trace.selected_action.value,
            "selected_scope": trace.selected_scope.value,
            "applied_policies": list(trace.applied_policies),
            "rejected_policies": list(trace.rejected_policies),
            "reasons": json.dumps(list(trace.reasons)),
            "final_plan": trace.final_plan.model_dump_json(),
            "created_at": trace.created_at,
        }

    async def write(self, traces: list[MemoryDecisionTrace]) -> None:
        """Persist *traces* (best-effort; failures are logged, not raised)."""
        if not traces:
            return
        rows = [self._trace_row(t) for t in traces]
        try:
            if self._record_sink is not None:
                await self._record_sink(rows)
                return
            sql = (
                f"INSERT INTO {self._table} "
                f"(id, candidate_id, candidate_text, selected_action, "
                f" selected_scope, applied_policies, rejected_policies, "
                f" reasons, final_plan, created_at) "
                f"VALUES (%(id)s, %(candidate_id)s, %(candidate_text)s, "
                f"        %(selected_action)s, %(selected_scope)s, "
                f"        %(applied_policies)s, %(rejected_policies)s, "
                f"        %(reasons)s::jsonb, %(final_plan)s::jsonb, "
                f"        %(created_at)s)"
            )
            async with self._connect() as conn:
                for r in rows:
                    await conn.execute(sql, r)
        except Exception:
            log.exception("TraceWriter.write failed — %d trace(s) dropped",
                          len(traces))


__all__ = ["TraceWriter", "RecordSink"]
