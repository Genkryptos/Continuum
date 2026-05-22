"""
continuum/extraction/candidate_extractor.py
============================================
``MemoryCandidateExtractor`` — bridges raw MTM blocks → ``MemoryCandidate``
objects ready for the Policy Engine.

v1 implementation: **regex + heuristics**, intentionally simple and
deterministic so the policy layer can be developed without a model
dependency. The same surface is preserved for later LLM-driven candidate
extraction — swap the body without touching callers.

Pipeline
--------
1. Reuse :class:`continuum.extraction.fact_extractor.FactExtractor` when
   provided → emits ``FACT`` candidates with provenance.
2. Run lightweight regex patterns on the block text to surface
   preferences, tasks/deadlines, decisions, procedures, corrections,
   meetings, and secrets.
3. **Secret detection is performed FIRST**: if a secret is found the
   secret-candidate is emitted *instead of* fact candidates that overlap
   that span — the policy engine can then redact / defer safely.

Each candidate carries:
* a stable ``id`` (UUID)
* ``source_ref`` linking it back to the source block id
* ``source_span`` quoting the substring that triggered the match
* a sensible default for ``urgency`` / ``volatility`` / ``sensitivity`` /
  ``source_authority`` — policies may override.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from continuum.core.types import SummaryBlock
from continuum.extraction.entity_extractor import Entity
from continuum.policies.models import (
    MemoryCandidate,
    MemoryCandidateType,
    Sensitivity,
    SourceAuthority,
    Urgency,
    Volatility,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — deliberately conservative; precision > recall for v1.
# ---------------------------------------------------------------------------

_PREFERENCE_RE = re.compile(
    r"\b(?:i|user|my|we)\b[^.?!]{0,40}\bprefer(?:s|red)?\b[^.?!]+",
    re.IGNORECASE,
)
_TASK_RE = re.compile(
    r"\b(?:remind\s+me|todo|to-do|task|deadline|due)\b[^.?!]+",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:we|team|i)\b\s+(?:decided|chose|selected|agreed)\b[^.?!]+",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"\b(?:whenever|always|every\s+time|procedure|workflow|"
    r"the\s+way\s+to)\b[^.?!]+",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(?:actually|correction|i\s+meant|to\s+be\s+clear)\b[^.?!]+",
    re.IGNORECASE,
)
_MEETING_RE = re.compile(
    r"\b(?:meeting|standup|sync|call|1[:-]1)\b[^.?!]+",
    re.IGNORECASE,
)
# Each (label, pattern). Order matters: more-specific first.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    (
        "kv_secret",
        re.compile(
            r"\b(?:api[_-]?key|token|password|secret)\b\s*[:=]\s*[\w./+\-]{6,}",
            re.IGNORECASE,
        ),
    ),
)


class MemoryCandidateExtractor:
    """
    Turn one :class:`SummaryBlock` into a list of :class:`MemoryCandidate`.

    Parameters
    ----------
    fact_extractor:
        Optional :class:`FactExtractor`-shaped collaborator. When provided
        its facts seed the candidate list with proper provenance; when
        absent, the regex passes still run.
    """

    def __init__(self, fact_extractor: Any | None = None) -> None:
        self._fact_extractor = fact_extractor

    # ── public API ──────────────────────────────────────────────────────────

    async def extract_candidates(
        self,
        block: SummaryBlock,
        entities: Sequence[Entity],
    ) -> list[MemoryCandidate]:
        out: list[MemoryCandidate] = []
        seen_spans: set[str] = set()

        # 1. Secrets FIRST — any matching span dominates and suppresses
        #    overlapping fact candidates.
        secret_spans: list[str] = []
        for kind, pattern in _SECRET_PATTERNS:
            for m in pattern.finditer(block.text):
                span = m.group(0)
                secret_spans.append(span)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                out.append(self._make(
                    block=block,
                    span=span,
                    candidate_type=MemoryCandidateType.SENSITIVE_DATA,
                    labels=[f"secret:{kind}"],
                    sensitivity=Sensitivity.RESTRICTED,
                    source_authority=SourceAuthority.INFERRED,
                    volatility=Volatility.STABLE,
                    importance=0.9,
                    confidence=0.95,
                ))

        # 2. Facts from the existing extractor (rich provenance + entities).
        if self._fact_extractor is not None:
            try:
                facts = await self._fact_extractor.extract_facts(
                    block, list(entities)
                )
            except Exception:
                log.exception(
                    "fact extractor failed in candidate pipeline — "
                    "continuing with regex passes only",
                )
                facts = []
            for fact in facts:
                # Skip facts that overlap a detected secret to avoid leaking
                # the raw payload into LTM.
                if any(s.lower() in fact.text.lower() for s in secret_spans):
                    continue
                out.append(self._make(
                    block=block,
                    span=fact.text,
                    candidate_type=MemoryCandidateType.FACT,
                    labels=[fact.category] if fact.category else [],
                    sensitivity=Sensitivity.PRIVATE,
                    source_authority=SourceAuthority.INFERRED,
                    importance=0.6,
                    confidence=float(fact.confidence),
                    entities=list(fact.entities_mentioned),
                    metadata={"source_fact_id": str(fact.source_block_id)},
                ))

        # 3. Behavioural / typed candidates via regex.
        for span in _extract(block.text, _PREFERENCE_RE, secret_spans):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            out.append(self._make(
                block=block, span=span,
                candidate_type=MemoryCandidateType.USER_PREFERENCE,
                sensitivity=Sensitivity.PRIVATE,
                source_authority=SourceAuthority.USER_EXPLICIT,
                volatility=Volatility.STABLE,
                importance=0.8, confidence=0.85,
                entities=[e.text for e in entities],
            ))

        for span in _extract(block.text, _TASK_RE, secret_spans):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            out.append(self._make(
                block=block, span=span,
                candidate_type=MemoryCandidateType.TASK,
                urgency=Urgency.SOON,
                volatility=Volatility.TEMPORARY,
                source_authority=SourceAuthority.USER_EXPLICIT,
                importance=0.7, confidence=0.8,
                entities=[e.text for e in entities],
            ))

        for span in _extract(block.text, _DECISION_RE, secret_spans):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            out.append(self._make(
                block=block, span=span,
                candidate_type=MemoryCandidateType.DECISION,
                source_authority=SourceAuthority.USER_EXPLICIT,
                volatility=Volatility.HISTORICAL,
                importance=0.85, confidence=0.85,
                entities=[e.text for e in entities],
            ))

        for span in _extract(block.text, _CORRECTION_RE, secret_spans):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            out.append(self._make(
                block=block, span=span,
                candidate_type=MemoryCandidateType.CORRECTION,
                source_authority=SourceAuthority.USER_EXPLICIT,
                importance=0.75, confidence=0.8,
            ))

        for span in _extract(block.text, _PROCEDURE_RE, secret_spans):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            out.append(self._make(
                block=block, span=span,
                candidate_type=MemoryCandidateType.PROCEDURE,
                source_authority=SourceAuthority.USER_EXPLICIT,
                volatility=Volatility.STABLE,
                importance=0.7, confidence=0.75,
                entities=[e.text for e in entities],
            ))

        # Meetings are a *whole-block* signal: tag the block once if any
        # meeting keyword appears, but only if no other typed candidate
        # already covers it (avoid double-counting).
        if _MEETING_RE.search(block.text) and not any(
            c.candidate_type
            in (
                MemoryCandidateType.DECISION,
                MemoryCandidateType.TASK,
                MemoryCandidateType.SENSITIVE_DATA,
            )
            for c in out
        ):
            out.append(self._make(
                block=block, span=block.text[:200],
                candidate_type=MemoryCandidateType.MEETING_EPISODE,
                source_authority=SourceAuthority.CALENDAR,
                importance=0.55, confidence=0.7,
                entities=[e.text for e in entities],
            ))

        return out

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _block_session_uuid(block: SummaryBlock) -> UUID | None:
        if block.session_id is None:
            return None
        try:
            return UUID(str(block.session_id))
        except (ValueError, AttributeError):
            return None

    def _make(
        self,
        *,
        block: SummaryBlock,
        span: str,
        candidate_type: MemoryCandidateType,
        labels: list[str] | None = None,
        sensitivity: Sensitivity = Sensitivity.PRIVATE,
        source_authority: SourceAuthority = SourceAuthority.INFERRED,
        urgency: Urgency = Urgency.NORMAL,
        volatility: Volatility = Volatility.STABLE,
        importance: float = 0.5,
        confidence: float = 0.7,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            id=uuid4(),
            text=span.strip(),
            candidate_type=candidate_type,
            labels=labels or [],
            source_ref=str(block.id),
            source_span=span.strip(),
            session_id=self._block_session_uuid(block),
            speaker=block.metadata.get("role"),
            entities=entities or [],
            confidence=max(0.0, min(1.0, confidence)),
            importance=max(0.0, min(1.0, importance)),
            urgency=urgency,
            volatility=volatility,
            sensitivity=sensitivity,
            source_authority=source_authority,
            occurred_at=block.created_at,
            metadata=metadata or {},
        )


def _extract(
    text: str, pattern: re.Pattern[str], secret_spans: list[str]
) -> list[str]:
    """Return de-duped match strings, skipping any that overlap a secret."""
    out: list[str] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        s = m.group(0).strip(" .,;:'\"")
        if not s or s.lower() in seen:
            continue
        if any(sec.lower() in s.lower() for sec in secret_spans):
            continue
        seen.add(s.lower())
        out.append(s)
    return out


__all__ = ["MemoryCandidateExtractor"]
