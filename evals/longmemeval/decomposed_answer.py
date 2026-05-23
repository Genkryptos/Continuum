"""
Structured sub-question answering helpers for LongMemEval.

The retrieval-only decomposer improves context coverage slightly but still
asks the final model to compose all reasoning in one pass. These helpers
support the stronger path: answer each atomic sub-question first, then
synthesize the final answer from those intermediate answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from continuum.core.types import ContextBundle


@dataclass(frozen=True)
class SubAnswer:
    subquestion: str
    answer: str
    evidence_session_ids: list[str]
    evidence_text: str
    hit_count: int


def extract_session_ids(ctx: ContextBundle | None) -> list[str]:
    if ctx is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in getattr(ctx, "items", []) or []:
        metadata = getattr(item, "metadata", {}) or {}
        raw_ids: list[object] = []
        sid = metadata.get("session_id")
        if sid:
            raw_ids.append(sid)
        session_ids = metadata.get("session_ids")
        if isinstance(session_ids, list):
            raw_ids.extend(session_ids)
        for raw_id in raw_ids:
            sid_s = str(raw_id)
            if sid_s in seen:
                continue
            seen.add(sid_s)
            out.append(sid_s)
    return out


def _context_text(ctx: ContextBundle | None) -> str:
    if ctx is None or not getattr(ctx, "items", None):
        return ""
    chunks: list[str] = []
    for idx, item in enumerate(ctx.items, start=1):
        role = str((item.metadata or {}).get("role", "user"))
        sid = str((item.metadata or {}).get("session_id", ""))
        prefix = f"[{idx}]"
        if sid:
            prefix += f" session={sid}"
        chunks.append(f"{prefix} role={role}\n{item.content}")
    return "\n\n".join(chunks)


def build_subanswer_prompt(subquestion: str, ctx: ContextBundle | None) -> str:
    evidence = _context_text(ctx)
    return (
        "Answer the sub-question using only the evidence. "
        "If the evidence does not contain the answer, say \"I don't know\".\n\n"
        f"Evidence:\n{evidence or '[no evidence retrieved]'}\n\n"
        f"Sub-question: {subquestion}\n"
        "Sub-answer:"
    )


def build_final_synthesis_prompt(
    question: str,
    subanswers: list[SubAnswer],
) -> str:
    blocks: list[str] = []
    for idx, sub in enumerate(subanswers, start=1):
        sessions = ", ".join(sub.evidence_session_ids) or "none"
        blocks.append(
            f"Sub-question {idx}: {sub.subquestion}\n"
            f"Sub-answer {idx}: {sub.answer}\n"
            f"Sessions {idx}: {sessions}\n"
            f"Evidence {idx}: {sub.evidence_text or '[no evidence]'}"
        )
    return (
        "Use the evidence blocks directly to answer the original question. "
        "Sub-answers are intermediate notes and may be incomplete. "
        "Resolve comparisons, arithmetic, before/after ordering, and updates "
        "from the evidence and sub-answers. If neither the evidence nor the "
        "sub-answers contain the answer, say \"I don't know\". Return only "
        "the final answer.\n\n"
        f"Original question: {question}\n\n"
        + "\n\n".join(blocks)
        + "\n\nFinal answer:"
    )


__all__ = [
    "SubAnswer",
    "extract_session_ids",
    "build_subanswer_prompt",
    "build_final_synthesis_prompt",
]
