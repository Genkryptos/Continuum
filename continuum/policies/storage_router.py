"""
continuum/policies/storage_router.py
====================================
``StorageRouter`` — turn a :class:`MemoryHandlingPlan` into traceable writes.

v1 strategy (pragmatic — see the module note in the README): everything
canonical lives in ``memory_nodes`` with the policy columns added by
migration 004; ``TASK_QUEUE`` / ``CODE_SYMBOL_INDEX`` / ``RAW_EVIDENCE_STORE``
are tagged via ``tags`` JSONB so downstream code can filter by them today,
and dedicated tables can be added later without breaking this interface.

Two outcomes are not "writes":

* ``MemoryAction.ASK_USER`` → row inserted into ``memory_pending_approvals``
* ``MemoryAction.IGNORE`` → no-op (only the trace is persisted)

Failures are logged and **never raised** — the trace still gets written by
the Promoter so audit history is complete even if a projection failed.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from continuum.policies.models import (
    MemoryAction,
    MemoryCandidate,
    MemoryHandlingPlan,
    StorageProjection,
)

log = logging.getLogger(__name__)


class StorageRouter:
    """
    Execute the storage projections of a plan.

    Parameters
    ----------
    dsn / conn_factory:
        Pick one; both routes use psycopg3 ``AsyncConnection``. Tests
        inject ``conn_factory``.
    ltm:
        Optional :class:`PostgresLTM` — used as a fallback when no DSN /
        factory is given (calls ``ltm.upsert(MemoryItem)`` and then
        UPDATEs the policy columns via the same connection mechanism).
    nodes_table / approvals_table:
        Override target tables (defaults match migration 004).
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        conn_factory: Any | None = None,
        ltm: Any | None = None,
        nodes_table: str = "memory_nodes",
        approvals_table: str = "memory_pending_approvals",
    ) -> None:
        if dsn is None and conn_factory is None and ltm is None:
            raise ValueError(
                "StorageRouter needs one of dsn, conn_factory, or ltm."
            )
        self._dsn = dsn
        self._conn_factory = conn_factory
        self._ltm = ltm
        self._nodes = nodes_table
        self._approvals = approvals_table

    # ── public API ──────────────────────────────────────────────────────────

    async def execute(
        self,
        candidate: MemoryCandidate,
        plan: MemoryHandlingPlan,
    ) -> list[UUID]:
        """Run all projections in *plan*; return the ids written."""
        if plan.action == MemoryAction.IGNORE:
            log.debug("plan IGNORE for %s — no projection written",
                      candidate.id)
            return []
        if plan.action == MemoryAction.ASK_USER:
            await self._write_pending_approval(candidate, plan)
            return []

        written: list[UUID] = []
        for projection in plan.storage_projections:
            try:
                new_id = await self._run_projection(
                    projection, candidate, plan
                )
                if new_id is not None and new_id not in written:
                    written.append(new_id)
            except Exception:
                log.exception(
                    "projection %s failed for candidate %s — continuing",
                    projection.value, candidate.id,
                )
        return written

    # ── per-projection dispatch ─────────────────────────────────────────────

    async def _run_projection(
        self,
        projection: StorageProjection,
        candidate: MemoryCandidate,
        plan: MemoryHandlingPlan,
    ) -> UUID | None:
        if projection in (
            StorageProjection.POSTGRES_CANONICAL,
            StorageProjection.VECTOR_INDEX,        # absorbed by canonical
            StorageProjection.TEMPORAL_INDEX,      # absorbed by canonical
        ):
            return await self._write_canonical(candidate, plan)
        if projection == StorageProjection.TASK_QUEUE:
            return await self._tag_projection(
                candidate, plan, tag="task_queue", value=True
            )
        if projection == StorageProjection.CODE_SYMBOL_INDEX:
            return await self._tag_projection(
                candidate, plan, tag="code_symbol_index", value=True
            )
        if projection == StorageProjection.RAW_EVIDENCE_STORE:
            return await self._tag_projection(
                candidate, plan, tag="evidence_store", value=True
            )
        if projection == StorageProjection.GRAPH_INDEX:
            # v1: nothing to do — edges are written by domain code that
            # actually knows the relationships; this projection is a
            # capability flag for the moment.
            return None
        log.debug("unhandled projection %s — no-op", projection)
        return None

    # ── canonical write ─────────────────────────────────────────────────────

    @staticmethod
    def _maybe_redacted(candidate: MemoryCandidate, plan: MemoryHandlingPlan) -> str:
        """If the privacy block asks for redaction, return a placeholder."""
        if plan.privacy.redact_before_storage:
            return f"[redacted {candidate.candidate_type.value}]"
        return candidate.text

    async def _write_canonical(
        self,
        candidate: MemoryCandidate,
        plan: MemoryHandlingPlan,
    ) -> UUID:
        node_id = candidate.id
        text = self._maybe_redacted(candidate, plan)
        tags: dict[str, Any] = {
            "kind": candidate.candidate_type.value,
            "policy_ids": list(plan.policy_ids),
        }
        if candidate.labels:
            tags["labels"] = list(candidate.labels)
        params: dict[str, Any] = {
            "id": str(node_id),
            "layer": plan.target_scope.value.upper(),
            "text": text,
            "kind": candidate.candidate_type.value,
            "confidence": candidate.confidence,
            "importance": candidate.importance,
            "tags": json.dumps(tags),
            "candidate_type": candidate.candidate_type.value,
            "urgency": candidate.urgency.value,
            "volatility": candidate.volatility.value,
            "sensitivity": candidate.sensitivity.value,
            "source_authority": candidate.source_authority.value,
            "policy_ids": list(plan.policy_ids),
            "retention": plan.retention.model_dump_json(),
            "retrieval_policy": plan.retrieval.model_dump_json(),
            "privacy_policy": plan.privacy.model_dump_json(),
            "update_policy": plan.update.model_dump_json(),
            "source_ref": candidate.source_ref,
            "source_span": candidate.source_span,
            "speaker": candidate.speaker,
            "expires_at": plan.retention.expire_at,
            "status": "pending_approval"
            if plan.privacy.requires_user_approval
            else "active",
            "valid_from": candidate.valid_from,
            "valid_to": candidate.valid_until,
        }
        sql = self._canonical_upsert_sql()
        async with self._connect() as conn:
            await conn.execute(sql, params)
        return node_id

    def _canonical_upsert_sql(self) -> str:
        return (
            f"INSERT INTO {self._nodes} ("
            "  id, layer, \"text\", kind, confidence, importance, tags,"
            "  candidate_type, urgency, volatility, sensitivity,"
            "  source_authority, policy_ids,"
            "  retention, retrieval_policy, privacy_policy, update_policy,"
            "  source_ref, source_span, speaker, expires_at, status,"
            "  valid_from, valid_to, created_at, updated_at"
            ") VALUES ("
            "  %(id)s, %(layer)s, %(text)s, %(kind)s, %(confidence)s,"
            "  %(importance)s, %(tags)s::jsonb,"
            "  %(candidate_type)s, %(urgency)s, %(volatility)s,"
            "  %(sensitivity)s, %(source_authority)s, %(policy_ids)s,"
            "  %(retention)s::jsonb, %(retrieval_policy)s::jsonb,"
            "  %(privacy_policy)s::jsonb, %(update_policy)s::jsonb,"
            "  %(source_ref)s, %(source_span)s, %(speaker)s,"
            "  %(expires_at)s, %(status)s,"
            "  %(valid_from)s, %(valid_to)s, now(), now()"
            ") ON CONFLICT (id) DO UPDATE SET"
            "  \"text\"           = EXCLUDED.\"text\","
            "  kind             = EXCLUDED.kind,"
            "  confidence       = EXCLUDED.confidence,"
            "  importance       = EXCLUDED.importance,"
            "  tags             = EXCLUDED.tags,"
            "  candidate_type   = EXCLUDED.candidate_type,"
            "  urgency          = EXCLUDED.urgency,"
            "  volatility       = EXCLUDED.volatility,"
            "  sensitivity      = EXCLUDED.sensitivity,"
            "  source_authority = EXCLUDED.source_authority,"
            "  policy_ids       = EXCLUDED.policy_ids,"
            "  retention        = EXCLUDED.retention,"
            "  retrieval_policy = EXCLUDED.retrieval_policy,"
            "  privacy_policy   = EXCLUDED.privacy_policy,"
            "  update_policy    = EXCLUDED.update_policy,"
            "  source_ref       = EXCLUDED.source_ref,"
            "  source_span      = EXCLUDED.source_span,"
            "  speaker          = EXCLUDED.speaker,"
            "  expires_at       = EXCLUDED.expires_at,"
            "  status           = EXCLUDED.status,"
            "  valid_from       = EXCLUDED.valid_from,"
            "  valid_to         = EXCLUDED.valid_to,"
            "  updated_at       = now()"
        )

    # ── tag projection (v1 stub for task_queue / code_index / evidence) ────

    async def _tag_projection(
        self,
        candidate: MemoryCandidate,
        plan: MemoryHandlingPlan,
        *,
        tag: str,
        value: Any,
    ) -> UUID | None:
        """
        Set a marker in ``tags`` JSONB on the canonical row.

        v1 keeps task/code/evidence "projections" as JSONB tags on the
        canonical row. Downstream code filters by ``tags->>tag``. A real
        dedicated table can be introduced later without touching the
        Policy Engine / Promoter contract.
        """
        sql = (
            f"UPDATE {self._nodes} "
            f"SET tags = COALESCE(tags, '{{}}'::jsonb) "
            f"          || jsonb_build_object(%(k)s, %(v)s::jsonb) "
            f"WHERE id = %(id)s"
        )
        async with self._connect() as conn:
            await conn.execute(
                sql,
                {
                    "k": tag,
                    "v": json.dumps(value),
                    "id": str(candidate.id),
                },
            )
        return None  # canonical row already counted

    # ── ASK_USER path ───────────────────────────────────────────────────────

    async def _write_pending_approval(
        self,
        candidate: MemoryCandidate,
        plan: MemoryHandlingPlan,
    ) -> None:
        sql = (
            f"INSERT INTO {self._approvals} "
            "(id, candidate_id, candidate_text, proposed_plan, reason, "
            " created_at, status) "
            "VALUES (%(id)s, %(cand)s, %(text)s, %(plan)s::jsonb, "
            "        %(reason)s, %(now)s, 'pending')"
        )
        async with self._connect() as conn:
            await conn.execute(
                sql,
                {
                    "id": str(uuid4()),
                    "cand": str(candidate.id),
                    "text": candidate.text,
                    "plan": plan.model_dump_json(),
                    "reason": plan.reason,
                    "now": datetime.now(UTC),
                },
            )

    # ── connection helper ───────────────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self) -> Any:
        if self._conn_factory is not None:
            async with self._conn_factory() as conn:
                yield conn
            return
        if self._ltm is not None and hasattr(self._ltm, "_connect"):
            async with self._ltm._connect() as conn:
                yield conn
            return
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - via conn_factory in tests
            raise ImportError(
                "psycopg3 is required for StorageRouter.\n"
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


__all__ = ["StorageRouter"]
