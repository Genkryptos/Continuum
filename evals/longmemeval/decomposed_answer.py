"""
Structured sub-question answering helpers for LongMemEval.

The retrieval-only decomposer improves context coverage slightly but still
asks the final model to compose all reasoning in one pass. These helpers
support the stronger path: answer each atomic sub-question first, then
synthesize the final answer from those intermediate answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from continuum.core.types import ContextBundle
from evals.longmemeval.evidence_packet import (
    EvidencePacket,
    render_evidence_packet_prompt,
)
from evals.longmemeval.evidence_spans import (
    EvidenceSpan,
    render_spans_for_prompt,
)

# ---------------------------------------------------------------------------
# Question-shape detection — shared between the adapter and the synthesis
# prompt so they agree on what counts as "aggregation".
# ---------------------------------------------------------------------------


_AGGREGATE_RE = re.compile(
    r"\b(?:how many|how much|total|count|list|all|both|either|or)\b",
    re.IGNORECASE,
)


def is_aggregate_question(question: str) -> bool:
    """True for count / list / how-many / aggregation questions."""
    return bool(_AGGREGATE_RE.search(question))


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


def build_subanswer_prompt(
    subquestion: str,
    ctx: ContextBundle | None,
    *,
    spans: list[EvidenceSpan] | None = None,
) -> str:
    """
    Build the per-sub-question prompt.

    When ``spans`` is provided, the prompt renders the compact answer-
    bearing spans instead of the full evidence-window text. Two
    distinguished cases:

    * ``spans is None`` — legacy behaviour: full window text via
      :func:`_context_text`. Used when span selection is disabled.
    * ``spans == []`` — the span selector ran but found nothing
      answer-bearing. The prompt then carries no evidence AND an
      explicit abstain instruction so the model returns
      ``"I don't know"`` instead of hallucinating from the wide window.
    * ``spans`` non-empty — render the compact spans only.

    See :mod:`evals.longmemeval.evidence_spans` for the selector.
    """
    use_spans = spans is not None
    if use_spans:
        if spans:
            evidence = render_spans_for_prompt(spans)
        else:
            evidence = ""  # forces the abstain branch below
    else:
        evidence = _context_text(ctx)

    # Type-aware hints. The wiki retriever's evidence windows pull
    # surrounding turns, so the answer to a "where" question often
    # lives in a neighbouring turn rather than the matched turn — the
    # model needs an explicit shape-of-answer constraint to avoid
    # echoing the matched-turn's most salient noun. Same for "what"
    # and "when" — the model defaults to the literal subject of the
    # matched turn when the answer is actually elsewhere in scope.
    q_lower = subquestion.lower()
    type_hints: list[str] = []
    if re.search(r"\bwhere\b", q_lower):
        type_hints.append(
            "This is a WHERE question — the answer must be a specific "
            "location (store name, city, address, app/platform name, "
            "venue, country). It is NOT the object or event involved. "
            "Scan ALL evidence turns, not just the one most directly "
            "about the event; the location is often mentioned in a "
            "neighbouring turn."
        )
    if re.search(r"\bwhen\b", q_lower):
        type_hints.append(
            "This is a WHEN question — the answer must be a specific "
            "time reference (date, day, month, year, relative phrase "
            "like 'last Sunday' or 'two weeks ago')."
        )
    if re.search(r"\bwho\b", q_lower):
        type_hints.append(
            "This is a WHO question — the answer must be a person, "
            "organisation, or named role."
        )
    hint_block = (
        "\n\nAdditional constraints:\n  - " + "\n  - ".join(type_hints) + "\n"
        if type_hints else ""
    )

    # When the span selector produced zero spans we hard-anchor abstain:
    # the wider window is intentionally NOT included even as fallback,
    # because the whole point of span selection is to refuse to guess
    # when nothing in the bundle directly supports the question.
    no_span_block = ""
    if use_spans and not spans:
        no_span_block = (
            "\nNote: the evidence selector did NOT find any answer-bearing "
            "span for this sub-question. You MUST answer \"I don't know\" "
            "and not infer from absent evidence.\n"
        )

    return (
        "Answer the sub-question using only the evidence. "
        "For count, total, or list questions, first extract every matching item "
        "or event from the evidence, deduplicate repeated mentions, then provide "
        "the count or concise list requested. Use this format for those "
        "questions: Matching facts: <bullet list>; Sub-answer: <concise final>. "
        "If the evidence does not contain the answer, say \"I don't know\"."
        f"{hint_block}"
        f"{no_span_block}\n"
        f"Evidence:\n{evidence or '[no evidence retrieved]'}\n\n"
        f"Sub-question: {subquestion}\n"
        "Sub-answer:"
    )


def build_final_synthesis_prompt(
    question: str,
    subanswers: list[SubAnswer],
    *,
    evidence_packet: EvidencePacket | None = None,
) -> str:
    """
    Build the final-answer synthesis prompt.

    When ``evidence_packet`` is provided, the prompt switches to the
    minimal-packet shape — the model receives only the verified
    claims and their exact source spans, NOT the raw evidence blocks.
    This is the "noisy memory window → minimal evidence packet"
    change from the spec. Sub-answers are still included as
    intermediate notes (they're cheap and sometimes carry
    decomposition reasoning the model needs), but the heavyweight
    evidence text is replaced.

    When ``evidence_packet`` is None, the legacy shape is preserved
    (full ``evidence_text`` per sub-answer) so callers that haven't
    opted in keep their current behaviour.
    """
    if evidence_packet is not None:
        # Packet-driven path. The packet renderer carries the strict
        # "use only this evidence" instructions and the abstain
        # contract — we just append a compact summary of the
        # sub-answers as intermediate notes so the model can see the
        # decomposition's intermediate conclusions.
        packet_block = render_evidence_packet_prompt(evidence_packet)
        if subanswers:
            sub_notes = "\n".join(
                f"  - sub-Q {i}: {sub.subquestion} → {sub.answer}"
                for i, sub in enumerate(subanswers, start=1)
            )
            return (
                packet_block.rstrip("Answer:").rstrip()
                + "\n\nSub-answer notes (intermediate, may be incomplete; "
                "the verified evidence above is authoritative):\n"
                + sub_notes
                + "\n\nFinal answer:"
            )
        return packet_block

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
        "from the evidence and sub-answers. For count or list questions, "
        "before computing the total: enumerate each candidate item across "
        "all sub-answers, and collapse items that refer to the same "
        "underlying entity under different wording (e.g. 'data analysis "
        "team' / 'the analytics initiative' = ONE project; the same project "
        "mentioned in two sessions counts ONCE). Preserve decimals exactly "
        "(2 weeks + 1.5 weeks = 3.5 weeks, not 3). Preserve the unit and "
        "any descriptive qualifier from the question (e.g. answer '5 model "
        "kits', not just '5'). If neither the evidence nor the sub-answers "
        "contain the answer, say \"I don't know\". Return only the final "
        "answer.\n\n"
        f"Original question: {question}\n\n"
        + "\n\n".join(blocks)
        + "\n\nFinal answer:"
    )


# ---------------------------------------------------------------------------
# Option 2 — structured JSON aggregation for count / list / how-many
# questions. The synthesis prompt asks the model to emit a candidate
# list with explicit duplicate links, then a final answer string.
# Parsing is graceful: malformed JSON falls back to the raw text so the
# caller never crashes.
# ---------------------------------------------------------------------------


def build_aggregation_synthesis_prompt(
    question: str, subanswers: list[SubAnswer],
) -> str:
    """Demand a structured JSON response so dedup happens at the schema layer."""
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
        "You are answering a COUNT / LIST / HOW-MANY question. Use only "
        "the evidence in the sub-answers below. Output a single JSON "
        "object — no surrounding prose, no markdown fences — with EXACTLY "
        "this schema:\n\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "item": "<short noun phrase naming the candidate>",\n'
        '      "source_sub_q": <int — which sub-question mentioned it>,\n'
        '      "duplicate_of": "<the item string of the first occurrence, or null>"\n'
        "    }\n"
        "  ],\n"
        '  "unique_count": <int — count of candidates whose duplicate_of is null>,\n'
        '  "final_answer": "<prose answer to the ORIGINAL question (not the sub-questions)>"\n'
        "}\n\n"
        "Rules:\n"
        "  1. List every candidate item mentioned in ANY sub-answer. The "
        "same item mentioned by multiple sub-answers gets one entry per "
        "mention; for each mention after the first, set duplicate_of to "
        "the item string of the first occurrence.\n"
        "  2. Two items are duplicates ONLY when they share the SAME "
        "DEFINING ATTRIBUTES — same specific instance (same name, same "
        "date/event, same store, same purpose). Items that share a "
        "common noun but differ on any defining attribute are DISTINCT. "
        "Examples:\n"
        "       • 'data analysis team' and 'the analytics initiative' "
        "naming the same project at the same employer → DUPLICATE.\n"
        "       • 'team of five engineers' mentioned in session A and "
        "again in session B → DUPLICATE (same team).\n"
        "       • 'new pair of boots to pick up' and 'boots to return to "
        "Zara' → DISTINCT (different events: pickup vs return).\n"
        "       • Two camping trips to different parks → DISTINCT.\n"
        "       • When uncertain, mark DISTINCT rather than merging.\n"
        "  3. Do NOT invent items not stated in the evidence — no "
        "inference from related-sounding facts.\n"
        "  4. unique_count MUST equal the number of candidates whose "
        "duplicate_of is null.\n"
        "  5. final_answer must preserve the question's unit AND any "
        "descriptive qualifier — '5 model kits' not just '5'; '3.5 weeks' "
        "not '3 weeks'; '8 days' not '8'. For sum/arithmetic questions "
        "(durations, costs), compute the sum across DISTINCT items.\n"
        "  6. If the evidence does not contain the answer, emit "
        '{"candidates":[],"unique_count":0,"final_answer":"I don\'t know"}.\n\n'
        "Worked example for a 'how many pets do I have' question with "
        "evidence mentioning a dog 'Rex' and a cat 'Whiskers' (twice):\n"
        "{\n"
        '  "candidates": [\n'
        '    {"item": "Rex (dog)", "source_sub_q": 1, "duplicate_of": null},\n'
        '    {"item": "Whiskers (cat)", "source_sub_q": 1, "duplicate_of": null},\n'
        '    {"item": "Whiskers (cat)", "source_sub_q": 2, "duplicate_of": "Whiskers (cat)"}\n'
        "  ],\n"
        '  "unique_count": 2,\n'
        '  "final_answer": "2 pets"\n'
        "}\n\n"
        f"Original question: {question}\n\n"
        + "\n\n".join(blocks)
        + "\n\nJSON:"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_aggregation_response(text: str) -> tuple[str, dict[str, Any]]:
    """
    Parse a JSON aggregation response.

    Returns ``(final_answer, metadata)``. ``metadata`` always carries
    a ``parse_status`` field — ``"ok"`` when the schema parsed, or a
    short reason string on fallback. On ANY parse failure the raw
    text is returned as the final answer so the caller never crashes;
    the metadata documents what went wrong for downstream logging.

    Also computes ``unique_count_consistent`` so production telemetry
    can monitor the model's dedup correctness independently of the
    final answer.
    """
    raw = (text or "").strip()
    if not raw:
        return "", {"parse_status": "empty"}

    # Strip ```json fences if the model wrapped despite our instructions.
    fenced = re.match(
        r"^```(?:json)?\s*(?P<body>.*?)\s*```$",
        raw, re.DOTALL | re.IGNORECASE,
    )
    candidate_text = fenced.group("body") if fenced else raw

    match = _JSON_OBJECT_RE.search(candidate_text)
    if match is None:
        return raw, {"parse_status": "no_json_object"}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return raw, {"parse_status": f"json_decode_error: {exc.msg}"}
    if not isinstance(data, dict):
        return raw, {"parse_status": "not_object"}

    final = data.get("final_answer")
    if not isinstance(final, str) or not final.strip():
        return raw, {"parse_status": "missing_final_answer", "data": data}

    candidates = data.get("candidates") or []
    declared = data.get("unique_count")
    computed = sum(
        1 for c in candidates
        if isinstance(c, dict)
        and c.get("duplicate_of") in (None, "", "null")
    )
    return final.strip(), {
        "parse_status": "ok",
        "candidates": candidates,
        "unique_count_declared": declared,
        "unique_count_computed": computed,
        "unique_count_consistent": declared == computed,
    }


_SCAFFOLD_PREFIX_RE = re.compile(
    r"^\s*matching\s+facts?\s*:.*?(?=sub[-\s]?answer\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_SUBANSWER_TAIL_RE = re.compile(
    r"sub[-\s]?answer\s*:\s*(?P<body>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_subanswer_scaffold(text: str) -> str:
    """
    Strip the extractive scaffold (``Matching facts: …; Sub-answer: <X>``)
    from an LLM reply, returning just ``<X>``.

    Used by the atomic-extractive code path so atomic-question answers
    come out clean even though we use the sub-answer prompt to force
    evidence quoting. Falls back to returning the input untouched when
    no recognisable scaffold is present.
    """
    if not text:
        return ""
    stripped = text.strip()
    # Prefer the "Sub-answer: …" tail when present — that's the
    # canonical final answer the prompt asks for.
    tail = _SUBANSWER_TAIL_RE.search(stripped)
    if tail:
        body = tail.group("body").strip()
        if body:
            return body
    # No tail: drop a leading "Matching facts: …" block if present.
    return _SCAFFOLD_PREFIX_RE.sub("", stripped).strip() or stripped


__all__ = [
    "SubAnswer",
    "_strip_subanswer_scaffold",
    "build_aggregation_synthesis_prompt",
    "build_final_synthesis_prompt",
    "build_subanswer_prompt",
    "extract_session_ids",
    "is_aggregate_question",
    "parse_aggregation_response",
]
