"""
tests/unit/test_mem0_promoter.py
================================
Unit tests for ``continuum.promotion.mem0_promoter.Mem0Promoter``.

A fake litellm-shaped ``completion_fn`` (tool-calls) and a recording
``audit_sink`` are injected — no litellm, no DB, no network.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from continuum.core.config import PromoterConfig
from continuum.core.types import MemoryItem, MemoryTier, ScoreBreakdown, ScoredItem
from continuum.extraction.fact_extractor import Fact
from continuum.promotion import (
    MEMORY_OP_SCHEMA,
    Decision,
    Mem0Promoter,
    make_postgres_audit_sink,
)

pytestmark = pytest.mark.unit

BLOCK = uuid.UUID("00000000-0000-0000-0000-0000000000b1")
N_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c1")


def _fact(text: str = "Alice works at Acme Corp") -> Fact:
    return Fact(text=text, confidence=0.9, entities_mentioned=[],
                source_block_id=BLOCK)


def _si(text: str, cos: float, nid: uuid.UUID = N_ID) -> ScoredItem:
    return ScoredItem(
        item=MemoryItem(id=str(nid), content=text, tier=MemoryTier.LTM),
        scores=ScoreBreakdown(
            relevance=cos, importance=0.0, recency=0.0,
            confidence=1.0, composite=cos,
        ),
    )


def _tool_resp(args_list: list[dict], *, tin: int = 120, tout: int = 30) -> Any:
    calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="memory_operation", arguments=json.dumps(a)
            )
        )
        for a in args_list
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout),
    )


class FakeLLM:
    def __init__(self, behaviours: list[Any]) -> None:
        self._b = list(behaviours)
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls += 1
        self.kwargs.append(kw)
        b = self._b.pop(0) if self._b else self._b
        if isinstance(b, BaseException):
            raise b
        return b


class RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def __call__(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(records)


def _promoter(fake: Any = None, sink: Any = None, **cfg: Any) -> Mem0Promoter:
    p = Mem0Promoter(
        PromoterConfig(**cfg), completion_fn=fake, audit_sink=sink
    )
    p._backoff_initial = 0.0
    p._backoff_max = 0.0
    return p


# ---------------------------------------------------------------------------
# Schema / dataclass
# ---------------------------------------------------------------------------


def test_schema_shape() -> None:
    assert MEMORY_OP_SCHEMA["name"] == "memory_operation"
    props = MEMORY_OP_SCHEMA["parameters"]["properties"]
    assert props["operation"]["enum"] == ["ADD", "UPDATE", "DELETE", "NOOP"]
    assert MEMORY_OP_SCHEMA["parameters"]["required"] == ["operation", "rationale"]


def test_decision_audit_record() -> None:
    d = Decision(op="UPDATE", target_id=N_ID, rationale="why",
                 merged_text="m", candidate_text="c", model="gpt-4o-mini",
                 tokens_in=10, tokens_out=2)
    rec = d.audit_record()
    assert rec == {
        "op": "UPDATE", "candidate_text": "c", "target_id": str(N_ID),
        "llm_model": "gpt-4o-mini", "llm_rationale": "why",
        "tokens_in": 10, "tokens_out": 2,
    }


# ---------------------------------------------------------------------------
# Short-circuit rules (no LLM)
# ---------------------------------------------------------------------------


class TestShortCircuit:
    async def test_low_cosine_short_circuits_to_add(self) -> None:
        fake = FakeLLM([])
        p = _promoter(fake, RecordingSink())
        d = await p.decide_operation(_fact(), [_si("unrelated", 0.30)])
        assert d.op == "ADD"
        assert d.target_id is None
        assert d.short_circuited is True
        assert fake.calls == 0                      # LLM skipped

    async def test_no_neighbors_short_circuits_to_add(self) -> None:
        fake = FakeLLM([])
        d = await _promoter(fake).decide_operation(_fact(), [])
        assert d.op == "ADD" and d.short_circuited is True
        assert fake.calls == 0

    async def test_high_cosine_short_circuits_to_noop(self) -> None:
        fake = FakeLLM([])
        d = await _promoter(fake).decide_operation(
            _fact(), [_si("Alice is employed by Acme Corp", 0.985)]
        )
        assert d.op == "NOOP"
        assert d.target_id == N_ID                  # the near-duplicate
        assert d.short_circuited is True
        assert fake.calls == 0

    async def test_custom_thresholds(self) -> None:
        fake = FakeLLM([])
        # add_threshold lowered → 0.45 no longer "definitely new"; it would
        # go to the LLM, so give a NOOP-high instead to keep it model-free.
        p = _promoter(fake, noop_threshold=0.80)
        d = await p.decide_operation(_fact(), [_si("dup", 0.85)])
        assert d.op == "NOOP"


# ---------------------------------------------------------------------------
# LLM operations
# ---------------------------------------------------------------------------


class TestLLMOperations:
    async def test_add(self) -> None:
        fake = FakeLLM([_tool_resp([
            {"operation": "ADD", "rationale": "novel fact"}
        ])])
        d = await _promoter(fake).decide_operation(
            _fact(), [_si("loosely related", 0.70)]
        )
        assert d.op == "ADD"
        assert d.target_id is None
        assert d.short_circuited is False
        assert d.model == "gpt-4o-mini"
        assert (d.tokens_in, d.tokens_out) == (120, 30)
        # tool schema actually passed to the model
        kw = fake.kwargs[0]
        assert kw["tools"][0]["function"]["name"] == "memory_operation"
        assert kw["tool_choice"] == "required"

    async def test_update(self) -> None:
        tid = uuid.uuid4()
        fake = FakeLLM([_tool_resp([{
            "operation": "UPDATE", "target_id": str(tid),
            "rationale": "augments", "merged_text": "Alice works at Acme Corp (NYC)",
        }])])
        d = await _promoter(fake).decide_operation(_fact(), [_si("x", 0.7)])
        assert d.op == "UPDATE"
        assert d.target_id == tid
        assert d.merged_text == "Alice works at Acme Corp (NYC)"

    async def test_delete(self) -> None:
        tid = uuid.uuid4()
        fake = FakeLLM([_tool_resp([{
            "operation": "DELETE", "target_id": str(tid),
            "rationale": "contradicts prior employer",
        }])])
        d = await _promoter(fake).decide_operation(_fact(), [_si("x", 0.7)])
        assert d.op == "DELETE"
        assert d.target_id == tid
        assert d.merged_text is None

    async def test_noop(self) -> None:
        fake = FakeLLM([_tool_resp([
            {"operation": "NOOP", "rationale": "already known"}
        ])])
        d = await _promoter(fake).decide_operation(_fact(), [_si("x", 0.7)])
        assert d.op == "NOOP" and d.target_id is None

    async def test_invalid_op_coerced_to_noop(self) -> None:
        fake = FakeLLM([_tool_resp([
            {"operation": "FRObNICATE", "rationale": "?"}
        ])])
        d = await _promoter(fake).decide_operation(_fact(), [_si("x", 0.7)])
        assert d.op == "NOOP"

    async def test_target_forced_null_for_add(self) -> None:
        fake = FakeLLM([_tool_resp([{
            "operation": "ADD", "target_id": str(uuid.uuid4()),
            "rationale": "new",
        }])])
        d = await _promoter(fake).decide_operation(_fact(), [_si("x", 0.7)])
        assert d.op == "ADD" and d.target_id is None   # schema rule enforced


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class TestBatch:
    async def test_batch_one_call_for_many(self) -> None:
        fake = FakeLLM([_tool_resp([
            {"operation": "ADD", "rationale": "a"},
            {"operation": "NOOP", "rationale": "b"},
            {"operation": "UPDATE", "target_id": str(N_ID),
             "rationale": "c", "merged_text": "m"},
        ])])
        sink = RecordingSink()
        p = _promoter(fake, sink)
        items = [
            (_fact("f0"), [_si("n", 0.7)]),
            (_fact("f1"), [_si("n", 0.7)]),
            (_fact("f2"), [_si("n", 0.7)]),
        ]
        out = await p.decide_operations_batch(items)
        assert [d.op for d in out] == ["ADD", "NOOP", "UPDATE"]
        assert fake.calls == 1                       # ONE call for three
        assert len(sink.records) == 3

    async def test_short_circuit_excluded_from_call(self) -> None:
        fake = FakeLLM([_tool_resp([{"operation": "ADD", "rationale": "x"}])])
        p = _promoter(fake, RecordingSink())
        items = [
            (_fact("new"), [_si("n", 0.20)]),          # short-circuit ADD
            (_fact("dup"), [_si("n", 0.99)]),          # short-circuit NOOP
            (_fact("ask"), [_si("n", 0.70)]),          # → LLM
        ]
        out = await p.decide_operations_batch(items)
        assert [d.op for d in out] == ["ADD", "NOOP", "ADD"]
        assert [d.short_circuited for d in out] == [True, True, False]
        assert fake.calls == 1                          # only the 3rd hit LLM

    async def test_batch_size_chunks_calls(self) -> None:
        fake = FakeLLM([
            _tool_resp([{"operation": "ADD", "rationale": "a"},
                        {"operation": "ADD", "rationale": "b"}]),
            _tool_resp([{"operation": "ADD", "rationale": "c"}]),
        ])
        p = _promoter(fake, batch_size=2)
        items = [(_fact(f"f{i}"), [_si("n", 0.7)]) for i in range(3)]
        out = await p.decide_operations_batch(items)
        assert len(out) == 3
        assert fake.calls == 2                          # 3 items / batch 2


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestAudit:
    async def test_every_decision_audited(self) -> None:
        fake = FakeLLM([_tool_resp([{"operation": "ADD", "rationale": "r"}])])
        sink = RecordingSink()
        p = _promoter(fake, sink)
        await p.decide_operations_batch([
            (_fact("sc"), [_si("n", 0.1)]),    # short-circuit
            (_fact("llm"), [_si("n", 0.7)]),   # llm
        ])
        assert len(sink.records) == 2
        keys = {"op", "candidate_text", "target_id", "llm_model",
                "llm_rationale", "tokens_in", "tokens_out"}
        assert all(keys == set(r) for r in sink.records)
        sc, llm = sink.records
        assert sc["llm_model"] is None and sc["op"] == "ADD"
        assert llm["llm_model"] == "gpt-4o-mini" and llm["tokens_in"] == 120

    async def test_audit_failure_does_not_break(self) -> None:
        class BadSink:
            async def __call__(self, _r: list[dict]) -> None:
                raise RuntimeError("db down")

        fake = FakeLLM([])
        p = _promoter(fake, BadSink())
        d = await p.decide_operation(_fact(), [])   # short-circuit ADD
        assert d.op == "ADD"                          # decision still returned


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGraceful:
    async def test_llm_failure_degrades_chunk_to_noop(self) -> None:
        fake = FakeLLM([RuntimeError("boom")] * 3)
        sink = RecordingSink()
        p = _promoter(fake, sink)
        items = [(_fact("a"), [_si("n", 0.7)]),
                 (_fact("b"), [_si("n", 0.7)])]
        out = await p.decide_operations_batch(items)
        assert [d.op for d in out] == ["NOOP", "NOOP"]   # safe default
        assert all("unavailable" in d.rationale for d in out)
        assert fake.calls == 3                            # retried
        assert len(sink.records) == 2                     # still audited


# ---------------------------------------------------------------------------
# Postgres audit sink
# ---------------------------------------------------------------------------


class TestPostgresAuditSink:
    async def test_requires_dsn_or_factory(self) -> None:
        with pytest.raises(ValueError, match="dsn or conn_factory"):
            make_postgres_audit_sink()

    async def test_inserts_rows(self) -> None:
        executed: list[tuple[str, Any]] = []

        class MockConn:
            async def execute(self, sql: str, params: Any = None) -> None:
                executed.append((sql, params))

        @asynccontextmanager
        async def factory() -> AsyncIterator[MockConn]:
            yield MockConn()

        sink = make_postgres_audit_sink(conn_factory=factory)
        await sink([
            Decision(op="ADD", target_id=None, rationale="r",
                     candidate_text="c", model="gpt-4o-mini",
                     tokens_in=5, tokens_out=1).audit_record()
        ])
        assert len(executed) == 1
        sql, params = executed[0]
        assert "INSERT INTO memory_promotions" in sql
        assert params["op"] == "ADD"
        assert params["candidate_text"] == "c"
        assert params["llm_model"] == "gpt-4o-mini"
