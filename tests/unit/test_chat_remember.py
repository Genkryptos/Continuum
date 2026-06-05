"""
tests/unit/test_chat_remember.py
===============================
The decider-driven LTM write in the chat REPL (``_remember`` + helpers).
Fakes for the LTM store, embedder, and decider — no Postgres, no network, no
model. Asserts the op→LTM-operation mapping, especially that a retraction
DELETE retires-only while a contradiction DELETE supersedes (retire + store).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.chat import _cosine, _neighbor_text, _remember, _split_clauses
from continuum.core.types import MemoryItem, MemoryTier
from continuum.promotion.mem0_promoter import Decision

pytestmark = pytest.mark.unit


def test_split_clauses_separates_assertion_and_retraction() -> None:
    # The combined turn must split so "Bhilai" can be stored AND Bangalore retracted.
    parts = _split_clauses("I live in Bhilai actually I never went to Bangalore")
    assert any("Bhilai" in p for p in parts)
    assert any("never went to Bangalore" in p for p in parts)
    assert len(parts) == 2


def test_split_clauses_single_sentence_is_unchanged() -> None:
    assert _split_clauses("My name is Mayank") == ["My name is Mayank"]


def test_split_clauses_drops_tiny_fragments() -> None:
    parts = _split_clauses("I live in Pune. Ok.")
    assert parts == ["I live in Pune"]  # "Ok" (1 word) dropped


def test_cosine_basic() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([], [1.0]) == 0.0
    assert _cosine(None, [1.0]) == 0.0


def _si(content: str, nid: str) -> Any:
    return SimpleNamespace(item=MemoryItem(id=nid, content=content, tier=MemoryTier.LTM))


def test_neighbor_text_lookup() -> None:
    ns = [_si("user in Bangalore", "n1"), _si("user likes chess", "n2")]
    assert _neighbor_text(ns, "n1") == "user in Bangalore"
    assert _neighbor_text(ns, "missing") == "missing"


# ── _remember op routing ────────────────────────────────────────────────────


class _FakeLTM:
    def __init__(self) -> None:
        self.upserted: list[str] = []
        self.invalidated: list[str] = []
        self.updated: list[tuple[str, dict]] = []
        self._neighbors: list[Any] = []

    def set_neighbors(self, ns: list[Any]) -> None:
        self._neighbors = ns

    async def search_hybrid(self, q: Any, k: int) -> list[Any]:
        return self._neighbors

    async def upsert(self, item: MemoryItem) -> None:
        self.upserted.append(item.content)

    async def invalidate(self, tid: Any, at: Any = None) -> None:
        self.invalidated.append(str(tid))

    async def update(self, tid: Any, patch: dict) -> None:
        self.updated.append((str(tid), patch))


class _FakeDecider:
    def __init__(self, decision: Decision) -> None:
        self._decision = decision

    async def decide_operation(self, fact: Any, neighbors: list[Any]) -> Decision:
        return self._decision


def _info(ltm: _FakeLTM, decision: Decision, *, decider_llm: bool = True) -> dict[str, Any]:
    # embedder=None so _remember skips cosine-rescore (decision is injected anyway).
    return {
        "ltm": ltm,
        "embedder": None,
        "decider": _FakeDecider(decision),
        "decider_llm": decider_llm,
    }


async def test_add_upserts() -> None:
    ltm = _FakeLTM()
    note = await _remember(_info(ltm, Decision(op="ADD", target_id=None, rationale="")), "s", "I live in Pune")
    assert "ADD" in note
    assert ltm.upserted == ["I live in Pune"]
    assert ltm.invalidated == []


async def test_retraction_delete_retires_only() -> None:
    ltm = _FakeLTM()
    ltm.set_neighbors([_si("user moved to Bangalore", "bng")])
    dec = Decision(op="DELETE", target_id="bng", rationale="", metadata={"retraction": True})
    note = await _remember(_info(ltm, dec), "s", "I was never in Bangalore")
    assert "retraction" in note
    assert ltm.invalidated == ["bng"]
    assert ltm.upserted == []  # negation is NOT stored


async def test_contradiction_delete_supersedes() -> None:
    ltm = _FakeLTM()
    ltm.set_neighbors([_si("I live in Hyderabad", "hyd")])
    dec = Decision(op="DELETE", target_id="hyd", rationale="", metadata={})  # no retraction flag
    note = await _remember(_info(ltm, dec), "s", "I moved to Bangalore")
    assert "SUPERSEDE" in note
    assert ltm.invalidated == ["hyd"]          # old retired
    assert ltm.upserted == ["I moved to Bangalore"]  # new stored


async def test_update_rewrites_target() -> None:
    ltm = _FakeLTM()
    dec = Decision(op="UPDATE", target_id="t1", rationale="", merged_text="merged")
    note = await _remember(_info(ltm, dec), "s", "new info")
    assert "UPDATE" in note
    assert ltm.updated and ltm.updated[0][0] == "t1"


async def test_noop_without_llm_falls_back_to_add() -> None:
    ltm = _FakeLTM()
    # LLM-band NOOP (short_circuited=False) + no decider LLM → store anyway.
    dec = Decision(op="NOOP", target_id=None, rationale="", short_circuited=False)
    note = await _remember(_info(ltm, dec, decider_llm=False), "s", "I like hiking")
    assert "ADD" in note
    assert ltm.upserted == ["I like hiking"]


async def test_genuine_noop_with_llm_is_respected() -> None:
    ltm = _FakeLTM()
    dec = Decision(op="NOOP", target_id=None, rationale="", short_circuited=True)
    note = await _remember(_info(ltm, dec, decider_llm=True), "s", "dup")
    assert "NOOP" in note
    assert ltm.upserted == []
