"""
tests/unit/policies/test_policies.py
====================================
Unit coverage for the Policy Engine stack:

* every concrete policy's ``supports`` gate + ``evaluate`` plan,
* :class:`PolicyEngine` cascade / conflict-merge / trace emission,
* :func:`default_policies` registry,
* :class:`StorageRouter` projection dispatch (fake async conn),
* :class:`TraceWriter` persistence (record-sink + fake conn).

All hermetic — no DB, no network. The Postgres-touching classes
(``StorageRouter`` / ``TraceWriter``) are exercised through an injected
async ``conn_factory`` / ``record_sink`` so the SQL-building and dispatch
logic is covered without a live database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from continuum.policies.code_policy import CodeMemoryPolicy
from continuum.policies.conflict_policy import CorrectionPolicy
from continuum.policies.decision_policy import DecisionPolicy
from continuum.policies.default_policies import DefaultFactPolicy
from continuum.policies.engine import PolicyEngine, _infer_sensitivity
from continuum.policies.meeting_policy import MeetingPolicy
from continuum.policies.models import (
    CompactionPolicy,
    ConflictStrategy,
    MemoryAction,
    MemoryCandidate,
    MemoryCandidateType,
    MemoryHandlingPlan,
    MemoryScope,
    PolicyEvaluationContext,
    PrivacyPolicy,
    RetentionPolicy,
    RetrievalMode,
    RetrievalPolicy,
    Sensitivity,
    StorageProjection,
    UpdatePolicy,
    UpdateStrategy,
    more_sensitive,
)
from continuum.policies.procedural_policy import ProceduralWorkflowPolicy
from continuum.policies.registry import default_policies
from continuum.policies.sensitivity_policy import SensitivityPolicy
from continuum.policies.storage_router import StorageRouter
from continuum.policies.task_urgency_policy import TaskUrgencyPolicy
from continuum.policies.trace import TraceWriter
from continuum.policies.user_preference_policy import UserPreferencePolicy

pytestmark = pytest.mark.unit


# ───────────────────────────── helpers ──────────────────────────────────────


def _candidate(
    ctype: MemoryCandidateType = MemoryCandidateType.FACT,
    *,
    text: str = "the sky is blue",
    sensitivity: Sensitivity = Sensitivity.PRIVATE,
    valid_until: datetime | None = None,
    labels: list[str] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        text=text,
        candidate_type=ctype,
        sensitivity=sensitivity,
        valid_until=valid_until,
        labels=labels or [],
    )


def _ctx(**config: Any) -> PolicyEvaluationContext:
    return PolicyEvaluationContext(now=datetime(2026, 5, 31, tzinfo=UTC), config=config)


def _plan(**overrides: Any) -> MemoryHandlingPlan:
    """A minimal valid plan, overridable per field."""
    base: dict[str, Any] = {
        "candidate_id": uuid4(),
        "action": MemoryAction.STORE,
        "target_scope": MemoryScope.LTM,
        "retention": RetentionPolicy(),
        "retrieval": RetrievalPolicy(modes=[RetrievalMode.SEMANTIC]),
        "privacy": PrivacyPolicy(),
        "update": UpdatePolicy(
            strategy=UpdateStrategy.VERSIONED,
            conflict_strategy=ConflictStrategy.MARK_CONFLICT,
        ),
        "compaction": CompactionPolicy(),
        "storage_projections": [StorageProjection.POSTGRES_CANONICAL],
        "reason": "t",
    }
    base.update(overrides)
    return MemoryHandlingPlan(**base)


class _FakeConn:
    """Records every (sql, params) execute call."""

    def __init__(self, *, raise_on_execute: bool = False) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._raise = raise_on_execute

    async def execute(self, sql: str, params: Any = None) -> None:
        if self._raise:
            raise RuntimeError("db down")
        self.calls.append((sql, params))


def _conn_factory(conn: _FakeConn) -> Any:
    """Build an async-context-manager factory yielding *conn*."""

    class _CM:
        async def __aenter__(self) -> _FakeConn:
            return conn

        async def __aexit__(self, *exc: Any) -> None:
            return None

    def factory() -> _CM:
        return _CM()

    return factory


# ───────────────────────── concrete policies ────────────────────────────────


async def test_default_fact_policy_supports_and_plan() -> None:
    p = DefaultFactPolicy()
    assert p.supports(_candidate(MemoryCandidateType.FACT))
    assert p.supports(_candidate(MemoryCandidateType.DOCUMENT_KNOWLEDGE))
    assert p.supports(_candidate(MemoryCandidateType.UNKNOWN))
    assert not p.supports(_candidate(MemoryCandidateType.DECISION))

    plan = await p.evaluate(_candidate(), _ctx())
    assert plan is not None
    assert plan.action == MemoryAction.STORE
    assert plan.target_scope == MemoryScope.LTM
    assert plan.retention.ttl_seconds is None  # no ttl configured
    assert StorageProjection.VECTOR_INDEX in plan.storage_projections


async def test_default_fact_policy_ttl_from_config() -> None:
    plan = await DefaultFactPolicy().evaluate(_candidate(), _ctx(default_fact_ttl_days=2))
    assert plan is not None
    assert plan.retention.ttl_seconds == int(timedelta(days=2).total_seconds())


async def test_user_preference_policy() -> None:
    p = UserPreferencePolicy()
    assert p.supports(_candidate(MemoryCandidateType.USER_PREFERENCE))
    assert not p.supports(_candidate(MemoryCandidateType.FACT))
    plan = await p.evaluate(_candidate(MemoryCandidateType.USER_PREFERENCE), _ctx())
    assert plan.update.strategy == UpdateStrategy.VERSIONED
    assert plan.update.conflict_strategy == ConflictStrategy.KEEP_BOTH
    assert plan.retrieval.retrieval_priority == 0.9
    # config flag flows into preserve_versions
    plan2 = await p.evaluate(
        _candidate(MemoryCandidateType.USER_PREFERENCE),
        _ctx(preserve_preference_versions=False),
    )
    assert plan2.update.preserve_versions is False


async def test_decision_policy() -> None:
    p = DecisionPolicy()
    assert p.supports(_candidate(MemoryCandidateType.DECISION))
    assert p.supports(_candidate(MemoryCandidateType.CODE_DECISION))
    plan = await p.evaluate(_candidate(MemoryCandidateType.DECISION), _ctx())
    assert plan.update.strategy == UpdateStrategy.SUPERSEDE
    assert StorageProjection.TEMPORAL_INDEX in plan.storage_projections


async def test_task_urgency_policy_with_deadline_and_default() -> None:
    p = TaskUrgencyPolicy()
    assert p.supports(_candidate(MemoryCandidateType.TASK))
    assert p.supports(_candidate(MemoryCandidateType.DEADLINE))
    assert not p.supports(_candidate(MemoryCandidateType.FACT))

    # explicit deadline → expire = deadline + grace
    deadline = datetime(2026, 6, 10, tzinfo=UTC)
    plan = await p.evaluate(
        _candidate(MemoryCandidateType.DEADLINE, valid_until=deadline),
        _ctx(task_grace_period_hours=12),
    )
    assert plan.target_scope == MemoryScope.TASK_STORE
    assert plan.retention.expire_at == deadline + timedelta(hours=12)
    assert plan.metadata["expire_at_iso"] == (deadline + timedelta(hours=12)).isoformat()

    # no deadline → default 7 days from now
    plan2 = await p.evaluate(_candidate(MemoryCandidateType.TASK), _ctx())
    assert plan2.retention.ttl_seconds == int(timedelta(days=7).total_seconds())


async def test_meeting_policy() -> None:
    p = MeetingPolicy()
    assert p.supports(_candidate(MemoryCandidateType.MEETING_EPISODE))
    plan = await p.evaluate(
        _candidate(MemoryCandidateType.MEETING_EPISODE),
        _ctx(meeting_raw_transcript_ttl_days=5),
    )
    assert plan.target_scope == MemoryScope.EPISODE_STORE
    assert plan.retention.ttl_seconds == int(timedelta(days=5).total_seconds())
    assert StorageProjection.RAW_EVIDENCE_STORE in plan.storage_projections


async def test_procedural_policy_enabled_and_disabled() -> None:
    p = ProceduralWorkflowPolicy()
    assert p.supports(_candidate(MemoryCandidateType.PROCEDURE))
    plan = await p.evaluate(_candidate(MemoryCandidateType.PROCEDURE), _ctx())
    assert plan is not None
    assert RetrievalMode.PROCEDURAL in plan.retrieval.modes
    # disabled via config → abstains (None)
    none_plan = await p.evaluate(
        _candidate(MemoryCandidateType.PROCEDURE),
        _ctx(enable_procedural_policy=False),
    )
    assert none_plan is None


async def test_code_policy_enabled_and_disabled() -> None:
    p = CodeMemoryPolicy()
    assert p.supports(_candidate(MemoryCandidateType.CODE_SYMBOL))
    assert p.supports(_candidate(MemoryCandidateType.CODE_DECISION))
    plan = await p.evaluate(_candidate(MemoryCandidateType.CODE_SYMBOL), _ctx())
    assert plan is not None
    assert plan.target_scope == MemoryScope.CODE_INDEX
    assert StorageProjection.CODE_SYMBOL_INDEX in plan.storage_projections
    none_plan = await p.evaluate(
        _candidate(MemoryCandidateType.CODE_SYMBOL),
        _ctx(enable_code_policy=False),
    )
    assert none_plan is None


async def test_correction_policy() -> None:
    p = CorrectionPolicy()
    assert p.supports(_candidate(MemoryCandidateType.CORRECTION))
    plan = await p.evaluate(_candidate(MemoryCandidateType.CORRECTION), _ctx())
    assert plan.action == MemoryAction.SUPERSEDE
    assert plan.update.conflict_strategy == ConflictStrategy.SUPERSEDE_OLD


async def test_sensitivity_policy_restricted_asks_user() -> None:
    p = SensitivityPolicy()
    # gate fires on type OR on a confidential/restricted sensitivity
    assert p.supports(_candidate(MemoryCandidateType.SENSITIVE_DATA))
    assert p.supports(_candidate(sensitivity=Sensitivity.CONFIDENTIAL))
    assert p.supports(_candidate(sensitivity=Sensitivity.RESTRICTED))
    assert not p.supports(_candidate(sensitivity=Sensitivity.PRIVATE))

    plan = await p.evaluate(_candidate(MemoryCandidateType.SENSITIVE_DATA), _ctx())
    assert plan.action == MemoryAction.ASK_USER
    assert plan.privacy.requires_user_approval is True
    assert plan.privacy.allow_cross_session_retrieval is False


async def test_sensitivity_policy_confidential_stores_redacted() -> None:
    p = SensitivityPolicy()
    plan = await p.evaluate(
        _candidate(sensitivity=Sensitivity.CONFIDENTIAL),
        _ctx(require_approval_for_restricted=True),
    )
    # confidential (not restricted) → STORE with redaction + encryption
    assert plan.action == MemoryAction.STORE
    assert plan.privacy.redact_before_storage is True
    assert plan.privacy.encrypt_at_rest is True


async def test_sensitivity_policy_restricted_without_approval_requirement() -> None:
    p = SensitivityPolicy()
    plan = await p.evaluate(
        _candidate(sensitivity=Sensitivity.RESTRICTED),
        _ctx(require_approval_for_restricted=False),
    )
    assert plan.action == MemoryAction.STORE


# ───────────────────────────── registry ─────────────────────────────────────


def test_default_policies_registry() -> None:
    policies = default_policies()
    assert len(policies) == 9
    ids = {p.policy_id for p in policies}
    assert "sensitivity_v1" in ids
    assert "default_fact_v1" in ids
    # priorities span the documented range
    assert max(p.priority for p in policies) == 100
    assert min(p.priority for p in policies) == 10


# ─────────────────────────── PolicyEngine ───────────────────────────────────


async def test_engine_sorts_by_priority_descending() -> None:
    engine = PolicyEngine(default_policies())
    priorities = [p.priority for p in engine.policies]
    assert priorities == sorted(priorities, reverse=True)


async def test_engine_single_fact_routes_to_default() -> None:
    engine = PolicyEngine(default_policies())
    plan, trace = await engine.evaluate(_candidate(MemoryCandidateType.FACT), _ctx())
    assert plan.action == MemoryAction.STORE
    assert "default_fact_v1" in trace.applied_policies
    assert trace.selected_action == plan.action
    assert trace.candidate_text == "the sky is blue"


async def test_engine_no_policy_matches_yields_ignore() -> None:
    # An engine with only the preference policy can't handle a plain FACT.
    engine = PolicyEngine([UserPreferencePolicy()])
    plan, trace = await engine.evaluate(_candidate(MemoryCandidateType.FACT), _ctx())
    assert plan.action == MemoryAction.IGNORE
    assert trace.applied_policies == []
    assert plan.reason == "no policy matched"


async def test_engine_skips_policy_that_raises() -> None:
    class _Boom:
        policy_id = "boom_v1"
        version = "1.0.0"
        priority = 50

        def supports(self, candidate: MemoryCandidate) -> bool:
            return True

        async def evaluate(self, candidate: MemoryCandidate, context: Any) -> Any:
            raise RuntimeError("kaboom")

    engine = PolicyEngine([_Boom(), DefaultFactPolicy()])
    plan, trace = await engine.evaluate(_candidate(), _ctx())
    assert "boom_v1" in trace.rejected_policies
    assert "default_fact_v1" in trace.applied_policies
    assert plan.action == MemoryAction.STORE


async def test_engine_records_abstaining_policy_as_rejected() -> None:
    class _Abstain:
        policy_id = "abstain_v1"
        version = "1.0.0"
        priority = 50

        def supports(self, candidate: MemoryCandidate) -> bool:
            return True

        async def evaluate(self, candidate: MemoryCandidate, context: Any) -> None:
            return None

    engine = PolicyEngine([_Abstain(), DefaultFactPolicy()])
    _plan_out, trace = await engine.evaluate(_candidate(), _ctx())
    assert "abstain_v1" in trace.rejected_policies


async def test_engine_merge_prefers_high_priority_and_layers_restrictions() -> None:
    # Sensitive data + a (hypothetically also-matching) fact-like policy:
    # the sensitivity policy (priority 100) is the base, and its
    # ASK_USER + restrictive privacy survive the merge.
    engine = PolicyEngine(default_policies())
    cand = _candidate(MemoryCandidateType.SENSITIVE_DATA)
    plan, trace = await engine.evaluate(cand, _ctx())
    assert plan.action == MemoryAction.ASK_USER
    assert plan.privacy.allow_cross_session_retrieval is False
    assert "sensitivity_v1" in trace.applied_policies
    assert "derived_sensitivity" in plan.metadata


async def test_engine_evaluate_many() -> None:
    engine = PolicyEngine(default_policies())
    cands = [
        _candidate(MemoryCandidateType.FACT),
        _candidate(MemoryCandidateType.USER_PREFERENCE),
        _candidate(MemoryCandidateType.DECISION),
    ]
    results = await engine.evaluate_many(cands, _ctx())
    assert len(results) == 3
    actions = {c.candidate_type: plan.action for c, plan, _ in results}
    assert actions[MemoryCandidateType.FACT] == MemoryAction.STORE


def test_engine_strictest_privacy_takes_max_restriction() -> None:
    a = PrivacyPolicy(redact_before_storage=False, allow_org_scope=True)
    b = PrivacyPolicy(redact_before_storage=True, allow_org_scope=False)
    merged = PolicyEngine._strictest_privacy(a, b)
    assert merged.redact_before_storage is True  # OR of the two
    assert merged.allow_org_scope is False  # AND of the two


def test_engine_tightest_retention_picks_shorter() -> None:
    a = RetentionPolicy(ttl_seconds=100, expire_at=datetime(2026, 6, 1, tzinfo=UTC))
    b = RetentionPolicy(ttl_seconds=50, expire_at=datetime(2026, 5, 1, tzinfo=UTC))
    merged = PolicyEngine._tightest_retention(a, b)
    assert merged.ttl_seconds == 50
    assert merged.expire_at == datetime(2026, 5, 1, tzinfo=UTC)
    # None handling: a None ttl yields the other; both None stays None
    assert (
        PolicyEngine._tightest_retention(
            RetentionPolicy(ttl_seconds=None), RetentionPolicy(ttl_seconds=7)
        ).ttl_seconds
        == 7
    )
    assert (
        PolicyEngine._tightest_retention(
            RetentionPolicy(ttl_seconds=None), RetentionPolicy(ttl_seconds=None)
        ).ttl_seconds
        is None
    )


def test_infer_sensitivity_ladder() -> None:
    assert _infer_sensitivity(PrivacyPolicy(encrypt_at_rest=True)) == Sensitivity.RESTRICTED
    assert _infer_sensitivity(PrivacyPolicy(requires_user_approval=True)) == Sensitivity.RESTRICTED
    assert _infer_sensitivity(PrivacyPolicy(redact_before_storage=True)) == Sensitivity.CONFIDENTIAL
    assert (
        _infer_sensitivity(PrivacyPolicy(allow_cross_session_retrieval=False))
        == Sensitivity.CONFIDENTIAL
    )
    assert _infer_sensitivity(PrivacyPolicy()) == Sensitivity.PRIVATE


def test_more_sensitive_helper() -> None:
    assert more_sensitive(Sensitivity.PUBLIC, Sensitivity.RESTRICTED) == Sensitivity.RESTRICTED
    assert more_sensitive(Sensitivity.CONFIDENTIAL, Sensitivity.PRIVATE) == Sensitivity.CONFIDENTIAL
    # tie → first arg
    assert more_sensitive(Sensitivity.PRIVATE, Sensitivity.PRIVATE) == Sensitivity.PRIVATE


# ─────────────────────────── StorageRouter ──────────────────────────────────


def test_storage_router_requires_a_target() -> None:
    with pytest.raises(ValueError):
        StorageRouter()


async def test_storage_router_ignore_writes_nothing() -> None:
    conn = _FakeConn()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    written = await router.execute(_candidate(), _plan(action=MemoryAction.IGNORE))
    assert written == []
    assert conn.calls == []


async def test_storage_router_ask_user_writes_pending_approval() -> None:
    conn = _FakeConn()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    written = await router.execute(_candidate(), _plan(action=MemoryAction.ASK_USER))
    assert written == []
    assert len(conn.calls) == 1
    sql, params = conn.calls[0]
    assert "memory_pending_approvals" in sql
    assert "'pending'" in sql  # status is a literal in the INSERT
    assert params["reason"]  # the plan's reason is carried through


async def test_storage_router_canonical_write_returns_node_id() -> None:
    conn = _FakeConn()
    cand = _candidate()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    written = await router.execute(
        cand, _plan(storage_projections=[StorageProjection.POSTGRES_CANONICAL])
    )
    assert written == [cand.id]
    sql, params = conn.calls[0]
    assert "INSERT INTO memory_nodes" in sql
    assert params["id"] == str(cand.id)


async def test_storage_router_dedup_canonical_projections() -> None:
    # CANONICAL + VECTOR_INDEX + TEMPORAL_INDEX all hit the canonical
    # writer with the same node id → returned once.
    conn = _FakeConn()
    cand = _candidate()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    written = await router.execute(
        cand,
        _plan(
            storage_projections=[
                StorageProjection.POSTGRES_CANONICAL,
                StorageProjection.VECTOR_INDEX,
                StorageProjection.TEMPORAL_INDEX,
            ]
        ),
    )
    assert written == [cand.id]


async def test_storage_router_tag_projections() -> None:
    conn = _FakeConn()
    cand = _candidate()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    written = await router.execute(
        cand,
        _plan(
            storage_projections=[
                StorageProjection.TASK_QUEUE,
                StorageProjection.CODE_SYMBOL_INDEX,
                StorageProjection.RAW_EVIDENCE_STORE,
            ]
        ),
    )
    # tag projections return None (canonical row already counted elsewhere)
    assert written == []
    assert len(conn.calls) == 3
    assert all("UPDATE memory_nodes" in sql for sql, _ in conn.calls)


async def test_storage_router_graph_index_is_noop() -> None:
    conn = _FakeConn()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    written = await router.execute(
        _candidate(), _plan(storage_projections=[StorageProjection.GRAPH_INDEX])
    )
    assert written == []
    assert conn.calls == []


async def test_storage_router_redacts_when_privacy_demands() -> None:
    conn = _FakeConn()
    cand = _candidate(text="my secret token")
    router = StorageRouter(conn_factory=_conn_factory(conn))
    await router.execute(
        cand,
        _plan(
            privacy=PrivacyPolicy(redact_before_storage=True),
            storage_projections=[StorageProjection.POSTGRES_CANONICAL],
        ),
    )
    _sql, params = conn.calls[0]
    assert params["text"].startswith("[redacted")
    assert "my secret token" not in params["text"]


async def test_storage_router_projection_failure_is_swallowed() -> None:
    conn = _FakeConn(raise_on_execute=True)
    router = StorageRouter(conn_factory=_conn_factory(conn))
    # the canonical write raises inside _run_projection; execute logs and
    # continues, returning no ids rather than raising.
    written = await router.execute(
        _candidate(), _plan(storage_projections=[StorageProjection.POSTGRES_CANONICAL])
    )
    assert written == []


async def test_storage_router_uses_ltm_connect_fallback() -> None:
    conn = _FakeConn()

    class _FakeLTM:
        def _connect(self) -> Any:
            return _conn_factory(conn)()

    router = StorageRouter(ltm=_FakeLTM())
    cand = _candidate()
    written = await router.execute(
        cand, _plan(storage_projections=[StorageProjection.POSTGRES_CANONICAL])
    )
    assert written == [cand.id]
    assert conn.calls


async def test_storage_router_status_pending_when_approval_required() -> None:
    conn = _FakeConn()
    router = StorageRouter(conn_factory=_conn_factory(conn))
    await router.execute(
        _candidate(),
        _plan(
            privacy=PrivacyPolicy(requires_user_approval=True),
            storage_projections=[StorageProjection.POSTGRES_CANONICAL],
        ),
    )
    _sql, params = conn.calls[0]
    assert params["status"] == "pending_approval"


# ──────────────────────────── TraceWriter ───────────────────────────────────


def _trace() -> Any:
    from continuum.policies.models import MemoryDecisionTrace

    return MemoryDecisionTrace(
        candidate_id=uuid4(),
        candidate_text="hi",
        selected_action=MemoryAction.STORE,
        selected_scope=MemoryScope.LTM,
        applied_policies=["default_fact_v1"],
        rejected_policies=[],
        reasons=["stored"],
        final_plan=_plan(),
        created_at=datetime(2026, 5, 31, tzinfo=UTC),
    )


def test_trace_writer_requires_a_target() -> None:
    with pytest.raises(ValueError):
        TraceWriter()


async def test_trace_writer_record_sink_path() -> None:
    captured: list[list[dict[str, Any]]] = []

    async def sink(rows: list[dict[str, Any]]) -> None:
        captured.append(rows)

    writer = TraceWriter(record_sink=sink)
    await writer.write([_trace()])
    assert len(captured) == 1
    assert captured[0][0]["selected_action"] == "store"
    assert captured[0][0]["candidate_text"] == "hi"


async def test_trace_writer_empty_is_noop() -> None:
    called = False

    async def sink(rows: list[dict[str, Any]]) -> None:
        nonlocal called
        called = True

    await TraceWriter(record_sink=sink).write([])
    assert called is False


async def test_trace_writer_conn_factory_path() -> None:
    conn = _FakeConn()
    writer = TraceWriter(conn_factory=_conn_factory(conn))
    await writer.write([_trace(), _trace()])
    assert len(conn.calls) == 2
    sql, _params = conn.calls[0]
    assert "memory_decision_traces" in sql


async def test_trace_writer_swallows_failures() -> None:
    conn = _FakeConn(raise_on_execute=True)
    writer = TraceWriter(conn_factory=_conn_factory(conn))
    # should not raise even though execute blows up
    await writer.write([_trace()])
