"""
evals/longmemeval/evidence_packet.py
====================================
Minimal evidence packet for the final answerer.

Why this exists
---------------
Through the previous passes the pipeline gained an extractor (typed
:class:`evals.longmemeval.candidates.Candidate`), a verifier
(:mod:`evals.longmemeval.claim_verifier`), a span selector
(:mod:`evals.longmemeval.evidence_spans`), and a session narrower
(:mod:`evals.longmemeval.session_narrowing`). What still leaks to the
final synthesis prompt is the **raw evidence text** of every
sub-answer's bundle — a wide window that contains lexically-near
distractors (ShareGPT, UltraChat, Apple finance facts when the
question is about a Spotify playlist).

The minimal evidence packet is the answer-time replacement for that
window. It carries:

* The original question.
* Only the **verified** answer-bearing claims (the verifier's PASS set).
* Each claim's exact source span + source session id + confidence.
* The number of candidates **excluded** at this stage, so the trace
  shows the noise that was dropped.
* A strict instruction: answer ONLY from the packet; abstain otherwise.

Token cost
----------
A 4-item wiki window is typically 800–1 200 tokens. A packet of 3
claims with their spans is typically 80–180 tokens. The compression
is what makes accuracy + cost both move in the same direction.

Telemetry
---------
:class:`EvidencePacket` exposes ``selected_evidence_count`` and
``excluded_noise_count`` so the JSONL trace records exactly how
narrow the answer prompt was.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from evals.longmemeval.candidates import Candidate

# ─── Data shapes ───────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PacketClaim:
    """One claim entry inside the evidence packet.

    Carries everything the answerer needs to use the claim correctly
    — and nothing else. No raw paragraph text, no neighbouring turns.
    """

    claim: str                     # short factual statement (the verified claim text)
    source_span: str               # the exact substring that anchored the claim
    source_session_id: str         # session id for attribution / dedup
    confidence: float              # verifier's confidence in this claim
    subject: str = ""              # what the claim is about (for the answerer's prompt)
    relation: str = ""             # canonical relation (is_worth, located_at, ...)
    object_: str = ""              # the answer-side of the relation
    answer_type: str = ""          # finer-grained than candidate_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "source_span": self.source_span,
            "source_session_id": self.source_session_id,
            "confidence": round(self.confidence, 3),
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object_,
            "answer_type": self.answer_type,
        }


@dataclasses.dataclass(frozen=True)
class EvidencePacket:
    """A minimal, answer-time evidence container for the final synthesiser."""

    question: str
    claims: list[PacketClaim]
    excluded_noise_count: int

    @property
    def selected_evidence_count(self) -> int:
        return len(self.claims)

    @property
    def is_empty(self) -> bool:
        return not self.claims

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "claims": [c.to_dict() for c in self.claims],
            "selected_evidence_count": self.selected_evidence_count,
            "excluded_noise_count": self.excluded_noise_count,
        }


# ─── Builder ───────────────────────────────────────────────────────────────


def build_evidence_packet(
    question: str,
    verified_candidates: Iterable[Candidate],
    *,
    all_candidates: Iterable[Candidate] | None = None,
    max_claims: int = 6,
) -> EvidencePacket:
    """
    Construct an :class:`EvidencePacket` from the verifier's output.

    Parameters
    ----------
    question:
        The original user question. Passed verbatim into the packet so
        the answerer prompt can echo it back to the model.
    verified_candidates:
        The :class:`Candidate` objects that **passed** the claim
        verifier. These become the packet's ``claims`` — no other
        candidate enters the final answer context.
    all_candidates:
        Optional total candidate pool (verified + failed). When
        provided, the difference against ``verified_candidates``
        becomes ``excluded_noise_count``. When omitted, the excluded
        count defaults to 0 (no audit-trail).
    max_claims:
        Hard cap on packet size. With > 6 claims the answerer's
        prompt starts to bloat past the compression target. Claims
        are ranked by confidence before truncation.

    Returns
    -------
    EvidencePacket
        A minimal packet. ``is_empty`` is True when the verifier kept
        nothing — the caller MUST treat this as an abstain signal,
        not "let me guess from the wider context".
    """
    verified_list = list(verified_candidates)
    # Sort by confidence descending; preserve insertion order on ties
    # so the trace is reproducible.
    verified_list.sort(key=lambda c: -c.confidence)
    kept = verified_list[:max_claims]

    if all_candidates is None:
        excluded = 0
    else:
        # ``all_candidates`` may overlap with ``verified_candidates``.
        # Count by identity-set so a candidate that the caller passed
        # in both lists isn't double-counted as "excluded".
        all_ids = {id(c) for c in all_candidates}
        kept_ids = {id(c) for c in verified_list}
        excluded = max(0, len(all_ids - kept_ids))

    claims = [
        PacketClaim(
            claim=c.claim or c.source_text[:200] or c.source_span,
            source_span=c.source_span or c.claim,
            source_session_id=c.source_session_id,
            confidence=c.confidence,
            subject=c.subject,
            relation=c.relation,
            object_=c.object_ or c.value,
            answer_type=c.answer_type or c.candidate_type,
        )
        for c in kept
    ]
    return EvidencePacket(
        question=question,
        claims=claims,
        excluded_noise_count=excluded,
    )


# ─── Prompt formatter ─────────────────────────────────────────────────────


def render_evidence_packet_prompt(packet: EvidencePacket) -> str:
    """
    Render the packet as the answerer's prompt.

    Two distinct shapes:

    * ``packet.is_empty`` — no verified evidence. The prompt forces
      ``"I don't know"`` and explicitly forbids guessing from absent
      data. Token cost: ~30 tokens.
    * Non-empty — header + numbered claim list + answer instructions.
      Token cost scales linearly with claim count; capped via
      ``max_claims`` at packet construction time.

    The prompt is intentionally short. Every word is there for the
    answerer; no compression rules, no aggregation rules, no
    duplicate-of rules — those live in the synthesis prompts that
    consume the packet, not in the packet itself.
    """
    if packet.is_empty:
        return (
            "Original question: " + packet.question + "\n\n"
            "Verified evidence: (none — the upstream verifier rejected "
            "every candidate)\n\n"
            "Instruction: You MUST answer \"I don't know\" — do not "
            "infer from prior context, do not guess.\n\n"
            "Answer:"
        )

    lines: list[str] = []
    for idx, c in enumerate(packet.claims, start=1):
        # One numbered claim per line. Keep each line compact but
        # carry every field the answerer might need:
        #   - claim text (the factual statement)
        #   - source_span (the exact substring that supports it)
        #   - source_session_id (for attribution)
        #   - confidence (so a low-confidence claim can be deprioritised)
        confidence_str = f"{c.confidence:.2f}"
        line = (
            f"[{idx}] {c.claim}"
            f"\n    span: \"{c.source_span}\""
            f"\n    session: {c.source_session_id or '?'}"
            f"  confidence: {confidence_str}"
        )
        if c.subject or c.relation or c.object_:
            line += (
                f"\n    subject={c.subject!r}"
                f"  relation={c.relation!r}"
                f"  object={c.object_!r}"
            )
        lines.append(line)

    return (
        "Answer the original question USING ONLY the verified evidence "
        "below. Each entry carries the exact source span, the session "
        "it came from, and the verifier's confidence. Do NOT invent "
        "facts, do NOT use any information outside this list, and do "
        "NOT use lower-confidence claims when a higher-confidence one "
        "directly answers the question. If none of the evidence answers "
        "the question, you MUST reply \"I don't know\".\n\n"
        f"Original question: {packet.question}\n\n"
        f"Verified evidence ({packet.selected_evidence_count} item"
        + ("s" if packet.selected_evidence_count != 1 else "")
        + f"; {packet.excluded_noise_count} excluded as noise):\n"
        + "\n".join(lines)
        + "\n\nAnswer:"
    )


__all__ = [
    "EvidencePacket",
    "PacketClaim",
    "build_evidence_packet",
    "render_evidence_packet_prompt",
]
