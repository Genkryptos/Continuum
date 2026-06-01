"""
tests/unit/extraction/test_candidate_extractor.py
==================================================
Unit coverage for :class:`MemoryCandidateExtractor` — the regex/heuristic
v1 bridge from MTM ``SummaryBlock`` text to ``MemoryCandidate`` objects.

Pure + deterministic (no model dependency). A fake fact-extractor is
injected to exercise the FactExtractor integration + secret-overlap
suppression without pulling in the real extractor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from continuum.core.types import SummaryBlock
from continuum.extraction.candidate_extractor import MemoryCandidateExtractor, _extract
from continuum.extraction.entity_extractor import Entity
from continuum.policies.models import MemoryCandidateType, Sensitivity

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────


def _block(text: str, *, role: str = "user", session_id: Any = None) -> SummaryBlock:
    return SummaryBlock(
        text=text,
        id=str(uuid4()),
        created_at=datetime(2026, 5, 31, tzinfo=UTC),
        metadata={"role": role},
        session_id=session_id,
    )


def _entity(text: str = "Postgres") -> Entity:
    return Entity(text=text, type="TECH", start=0, end=len(text), confidence=0.9)


def _types(cands: list[Any]) -> set[MemoryCandidateType]:
    return {c.candidate_type for c in cands}


# ── secrets (detected first, suppress overlaps) ──────────────────────────────


async def test_openai_key_detected_as_sensitive() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("my key is sk-ABCDEFGHIJKLMNOPQRSTUV"), [])
    secrets = [c for c in out if c.candidate_type == MemoryCandidateType.SENSITIVE_DATA]
    assert secrets
    assert secrets[0].sensitivity == Sensitivity.RESTRICTED
    assert any(lbl.startswith("secret:openai_key") for lbl in secrets[0].labels)


async def test_github_pat_and_kv_secret_detected() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(
        _block("token=ghp_ABCDEFGHIJKLMNOPQRSTUV and password: hunter2xyz"), []
    )
    labels = {lbl for c in out for lbl in c.labels}
    assert any("github_pat" in lbl for lbl in labels)
    assert any("kv_secret" in lbl for lbl in labels)


# ── behavioural regex passes ─────────────────────────────────────────────────


async def test_preference_detected() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("I prefer dark roast coffee"), [_entity()])
    assert MemoryCandidateType.USER_PREFERENCE in _types(out)


async def test_task_detected() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("remind me to email Alice tomorrow"), [])
    assert MemoryCandidateType.TASK in _types(out)


async def test_decision_detected() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("we decided to use Postgres for storage"), [_entity()])
    assert MemoryCandidateType.DECISION in _types(out)


async def test_correction_detected() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("actually it's Tuesday not Monday"), [])
    assert MemoryCandidateType.CORRECTION in _types(out)


async def test_procedure_detected() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("whenever the build fails restart the runner"), [])
    assert MemoryCandidateType.PROCEDURE in _types(out)


async def test_meeting_whole_block_signal() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("had a standup with the platform crew today"), [])
    assert MemoryCandidateType.MEETING_EPISODE in _types(out)


async def test_meeting_suppressed_when_decision_present() -> None:
    # A block that is both a "meeting" and a "decision" should NOT emit a
    # separate MEETING_EPISODE (decision already covers it).
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(
        _block("in the sync we decided to ship on Friday"), [_entity()]
    )
    types = _types(out)
    assert MemoryCandidateType.DECISION in types
    assert MemoryCandidateType.MEETING_EPISODE not in types


async def test_plain_text_yields_no_typed_candidates() -> None:
    ex = MemoryCandidateExtractor()
    out = await ex.extract_candidates(_block("the weather is mild"), [])
    assert out == []


# ── fact-extractor integration ───────────────────────────────────────────────


async def test_fact_extractor_facts_become_fact_candidates() -> None:
    fact = SimpleNamespace(
        text="The sky is blue",
        category="nature",
        confidence=0.8,
        entities_mentioned=["sky"],
        source_block_id=uuid4(),
    )

    class _FE:
        async def extract_facts(self, block: Any, entities: list[Any]) -> list[Any]:
            return [fact]

    ex = MemoryCandidateExtractor(fact_extractor=_FE())
    out = await ex.extract_candidates(_block("The sky is blue"), [])
    facts = [c for c in out if c.candidate_type == MemoryCandidateType.FACT]
    assert facts
    assert facts[0].text == "The sky is blue"
    assert facts[0].metadata["source_fact_id"] == str(fact.source_block_id)


async def test_fact_overlapping_secret_is_suppressed() -> None:
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUV"
    fact = SimpleNamespace(
        text=f"the api key is {secret}",
        category=None,
        confidence=0.9,
        entities_mentioned=[],
        source_block_id=uuid4(),
    )

    class _FE:
        async def extract_facts(self, block: Any, entities: list[Any]) -> list[Any]:
            return [fact]

    ex = MemoryCandidateExtractor(fact_extractor=_FE())
    out = await ex.extract_candidates(_block(f"the api key is {secret}"), [])
    # The secret candidate is emitted; the overlapping FACT is dropped.
    assert any(c.candidate_type == MemoryCandidateType.SENSITIVE_DATA for c in out)
    assert not any(c.candidate_type == MemoryCandidateType.FACT for c in out)


async def test_fact_extractor_exception_is_swallowed() -> None:
    class _BoomFE:
        async def extract_facts(self, block: Any, entities: list[Any]) -> list[Any]:
            raise RuntimeError("model down")

    ex = MemoryCandidateExtractor(fact_extractor=_BoomFE())
    # Regex passes still run; the exception doesn't propagate.
    out = await ex.extract_candidates(_block("I prefer tea"), [])
    assert MemoryCandidateType.USER_PREFERENCE in _types(out)


# ── session-uuid + _extract helpers ──────────────────────────────────────────


async def test_session_uuid_valid_and_invalid() -> None:
    ex = MemoryCandidateExtractor()
    valid = uuid4()
    out = await ex.extract_candidates(_block("I prefer tea", session_id=str(valid)), [])
    pref = next(c for c in out if c.candidate_type == MemoryCandidateType.USER_PREFERENCE)
    assert pref.session_id == valid

    out2 = await ex.extract_candidates(_block("I prefer tea", session_id="not-a-uuid"), [])
    pref2 = next(c for c in out2 if c.candidate_type == MemoryCandidateType.USER_PREFERENCE)
    assert pref2.session_id is None


def test_extract_dedupes_and_skips_secret_overlap() -> None:
    import re

    pat = re.compile(r"prefer[^.?!]+", re.IGNORECASE)
    text = "prefer tea. prefer tea. prefer the secret sk-XXXXXXXXXXXXXXXXXXXX"
    # No secret spans → both unique matches kept (case-folded dedupe).
    res = _extract(text, pat, secret_spans=[])
    assert len(res) == 2
    # With the secret span supplied, the overlapping match is dropped.
    res2 = _extract(text, pat, secret_spans=["sk-XXXXXXXXXXXXXXXXXXXX"])
    assert all("sk-" not in r for r in res2)
