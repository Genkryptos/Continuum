"""
evals/longmemeval/evidence_spans.py
===================================
Compact, answer-bearing span selection for retrieved evidence windows.

Why this exists
---------------
The WikiMemoryRetriever's evidence window (see
``bootstrap_ollama.WikiMemoryRetriever._evidence_window``) returns a
``MemoryItem`` whose ``content`` is a 5-turn Markdown block —
the matched turn plus ±2 neighbours. That's great for *recall* (the
answer almost certainly lives somewhere in those 5 turns) but
terrible for *precision*: the model is now asked to pick the right
number / location / name from a window that often contains multiple
candidates ("the bookshelf took 4 hours, the table took 2 hours" /
"my Bachelor's is from UCLA, I'm looking at Master's programs at
USC and Berkeley").

The 500-row LongMemEval analysis pinned this as the dominant failure
mode on multi-session + temporal-reasoning rows: retrieval was right,
but the answerer picked the wrong span inside the window.

Contract
--------
``select_spans(question, items, *, max_spans=3, min_overlap=0.2)``
returns 0-3 :class:`EvidenceSpan` objects. *Zero* is a valid answer —
when nothing in the bundle scores above ``min_overlap`` the caller
MUST treat it as "no evidence" and force abstain rather than guess.

Each :class:`EvidenceSpan` preserves the exact source text, the
``source_session_id`` (so trace logs can attribute back), the
``source_turn_id`` (when the bundle item carries one), the role
(``"user"`` / ``"assistant"`` / ``""``), and a short ``reason``
string explaining which signal won — handy for the debug-trace
JSONL.

What this is NOT
----------------
* Not a re-ranker — operates on already-retrieved context, doesn't
  hit the embedder or any LLM.
* Not a paraphraser — span text is the exact substring of the
  source item's content.
* Not a storage schema change — purely an answer-time transform.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Sequence
from typing import Any

from continuum.core.types import MemoryItem
from evals.longmemeval.question_type import QuestionType

# ─── Public dataclass ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class EvidenceSpan:
    """One compact, answer-bearing snippet pulled from a wider window."""

    text: str                       # exact substring of the source item
    source_session_id: str          # session_id stamped on the source item
    source_turn_id: str             # MemoryItem.id of the source, or "" when unknown
    role: str                       # "user" / "assistant" / ""
    score: float                    # composite score for trace/debug
    reason: str                     # which signal(s) won — debug aid

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ─── Tokenisation + stopwords ──────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,}|\d+(?:\.\d+)?")

#: Closed-class stopwords. Kept small on purpose — the question's
#: content words ("bookshelf", "UCLA", "commute") are what drive the
#: overlap score. Removing too many words washes out the signal.
#:
#: Note: quantifier words ("many", "much", "few", "several") are
#: stopworded because they always pair with the question signal
#: ("how many shirts") and would otherwise inflate overlap on
#: irrelevant lines that happen to use the same quantifier.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in",
    "on", "at", "by", "with", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "me", "my", "you", "your", "we", "us", "our", "they", "them",
    "this", "that", "these", "those", "it", "its", "if", "then",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "all", "any", "some", "no", "not", "so", "too", "very",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "than", "about", "into", "over", "under", "between", "during",
    "many", "much", "few", "several", "long", "lot", "lots", "more",
    "less", "most", "least", "another", "each", "every",
})

_QUESTION_VERBS = frozenset({
    # Verbs that often appear in questions but aren't the answer's anchor.
    "tell", "say", "mention", "describe", "explain", "give", "show",
})


def _tokens(text: str) -> list[str]:
    """Lowercase content tokens; numerics preserved as their literal."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _content_words(text: str) -> set[str]:
    """Set of useful tokens for overlap scoring (stopwords stripped)."""
    return {
        t for t in _tokens(text)
        if t not in _STOPWORDS and t not in _QUESTION_VERBS and len(t) > 1
    }


# ─── Splitting items into candidate spans ──────────────────────────────────


_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_BULLET_RE = re.compile(r"^\s*-\s+(?:\((?P<role>[a-z]+)\):\s*)?(?P<body>.+)$")
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"  # split on sentence terminators
)


def _split_into_spans(item: MemoryItem) -> list[tuple[str, str]]:
    """
    Split one bundle item into candidate ``(text, role)`` spans.

    Markdown headings are dropped. Bullet lines split into one span
    each — these are the wiki evidence window's primary structure.
    Bare paragraphs are sentence-split as a fallback.
    """
    out: list[tuple[str, str]] = []
    item_role = str((item.metadata or {}).get("role", "") or "")
    for raw_line in (item.content or "").splitlines():
        line = raw_line.strip()
        if not line or _HEADING_RE.match(line):
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            body = bullet.group("body").strip()
            role = (bullet.group("role") or item_role or "").lower()
            if len(body) >= 5:
                out.append((body, role))
            continue
        # Bare line — sentence-split so a paragraph of 4 sentences
        # becomes 4 candidate spans rather than one fat one.
        for sent in _SENTENCE_RE.split(line):
            sent = sent.strip()
            if len(sent) >= 5:
                out.append((sent, item_role.lower()))
    return out


# ─── Question-type shape bonuses + distractor penalties ────────────────────


_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_TIME_UNIT_RE = re.compile(
    r"\b(?:second|minute|hour|day|week|month|year|min|hr|sec|yr)s?\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\b"
    r"|\b(?:Mon|Tue|Tues|Wed|Thu|Thurs|Fri|Sat|Sun)(?:day)?\b"
    r"|\b(?:yesterday|today|tomorrow|tonight)\b",
    re.IGNORECASE,
)
_LOCATION_HINT_RE = re.compile(
    r"\b(?:store|shop|venue|park|library|office|school|university|college|"
    r"restaurant|cafe|clinic|hospital|airport|hotel|building|street|"
    r"avenue|road|state|country|city|gym|studio|warehouse|app|website)\b",
    re.IGNORECASE,
)
_PERSON_HINT_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Professor)\.?\s+[A-Z][a-zA-Z]+"
    r"|\b[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\b",
)

#: Lines describing *hypothetical / alternative* content are the
#: classic distractor for "bachelor's at UCLA / master's at USC|Berkeley".
#: We knock these down so the answer-bearing factual line wins.
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:would|could|might|maybe|perhaps|considering|thinking about|"
    r"looking at|options?|alternative|either|or|vs\.?|versus|instead)\b",
    re.IGNORECASE,
)
#: Lines that are themselves questions ("Did you …?", "Have you …?")
#: rarely carry the answer.
_TRAILING_QUESTION_RE = re.compile(r"\?\s*$")


def _shape_bonus(span_text: str, qtype: QuestionType | None) -> tuple[float, str]:
    """Return ``(bonus, reason)`` for the question-type shape match."""
    if qtype is None:
        return 0.0, ""
    if qtype == QuestionType.COUNT:
        if _NUMERIC_RE.search(span_text):
            return 0.6, "has_number"
        return 0.0, ""
    if qtype == QuestionType.DURATION:
        if _NUMERIC_RE.search(span_text) and _TIME_UNIT_RE.search(span_text):
            return 0.8, "has_duration"
        if _TIME_UNIT_RE.search(span_text):
            return 0.3, "has_time_unit"
        return 0.0, ""
    if qtype == QuestionType.DATE_TIME:
        if _DATE_RE.search(span_text):
            return 0.6, "has_date_phrase"
        if _NUMERIC_RE.search(span_text):
            return 0.2, "has_number"
        return 0.0, ""
    if qtype == QuestionType.LOCATION:
        if _LOCATION_HINT_RE.search(span_text):
            return 0.5, "has_location_keyword"
        if re.search(r"\b[A-Z][a-zA-Z]+\b", span_text):
            return 0.3, "has_capitalised_token"
        return 0.0, ""
    if qtype == QuestionType.PERSON:
        if _PERSON_HINT_RE.search(span_text):
            return 0.6, "has_person_name"
        return 0.0, ""
    return 0.0, ""


def _distractor_penalty(span_text: str) -> tuple[float, str]:
    """Penalty for hypothetical / question-shaped / option-list spans."""
    penalty = 0.0
    reasons: list[str] = []
    if _HYPOTHETICAL_RE.search(span_text):
        penalty += 0.4
        reasons.append("hypothetical")
    if _TRAILING_QUESTION_RE.search(span_text):
        penalty += 0.3
        reasons.append("question_shape")
    return penalty, ",".join(reasons)


# ─── Public selector ───────────────────────────────────────────────────────


def select_spans(
    question: str,
    items: Sequence[MemoryItem],
    *,
    max_spans: int = 3,
    min_overlap: float = 0.2,
    question_type: QuestionType | None = None,
) -> list[EvidenceSpan]:
    """
    Pick up to ``max_spans`` answer-bearing spans from ``items``.

    Parameters
    ----------
    question:
        The natural-language sub-question (after decomposition).
    items:
        Retrieved bundle items — usually the
        ``WikiMemoryRetriever`` evidence windows, each containing
        5 turns of Markdown.
    max_spans:
        Hard cap on returned spans. Defaults to 3 — empirically the
        answerer can handle 1-3 anchored spans well but starts to
        regress to the broad-window failure mode beyond that.
    min_overlap:
        Minimum content-word overlap (as a fraction of the question's
        content words) before a span is eligible. Defaults to 0.2 —
        the wiki window's neighbour turns often share zero overlap
        with the question, and filtering them out is the whole point.
    question_type:
        Optional :class:`QuestionType` for shape-aware scoring. When
        present, spans matching the shape (digits for COUNT,
        dates for DATE_TIME, etc.) get a bonus that breaks ties.

    Returns
    -------
    list[EvidenceSpan]
        0-``max_spans`` spans, ordered by descending score. **An empty
        list is a valid result** and means "no span supports the
        question" — the caller must convert this to an abstain
        signal rather than handing the broad window through.
    """
    q_words = _content_words(question)
    if not q_words or not items:
        return []

    # Generate (overlap, is_distractor, span) tuples. We carry `overlap`
    # (absolute count) so the anchor-overlap rule below can filter
    # secondaries that share fewer question words than the top span
    # (the "IKEA bookshelf vs IKEA table" distractor pattern). We carry
    # `is_distractor` so that — if EVERY candidate is a hypothetical /
    # question-shape line — we return empty and force abstain instead
    # of surfacing the least-bad option.
    candidates: list[tuple[int, bool, EvidenceSpan]] = []
    for item in items:
        sid = str((item.metadata or {}).get("session_id") or "")
        turn_id = str(getattr(item, "id", "") or "")
        for span_text, role in _split_into_spans(item):
            span_words = _content_words(span_text)
            if not span_words:
                continue
            overlap = len(q_words & span_words)
            overlap_frac = overlap / max(1, len(q_words))
            if overlap_frac < min_overlap:
                continue
            shape_bonus, shape_reason = _shape_bonus(span_text, question_type)
            penalty, distract_reason = _distractor_penalty(span_text)
            score = overlap_frac + shape_bonus - penalty
            if score <= 0.0:
                continue
            reason_parts = [f"overlap={overlap}/{len(q_words)}"]
            if shape_reason:
                reason_parts.append(shape_reason)
            if distract_reason:
                reason_parts.append(f"penalty:{distract_reason}")
            candidates.append((overlap, bool(distract_reason), EvidenceSpan(
                text=span_text,
                source_session_id=sid,
                source_turn_id=turn_id,
                role=role,
                score=round(score, 4),
                reason=" ".join(reason_parts),
            )))

    if not candidates:
        return []

    # Abstain when nothing factually supports the question: if EVERY
    # candidate is a hypothetical / question-shape distractor, the
    # bundle doesn't contain a directly-answering span. Returning
    # empty here is the spec's "force abstain" path.
    if all(is_distractor for _ov, is_distractor, _sp in candidates):
        return []

    # Sort descending by score; tie-break by longer span (more context).
    candidates.sort(key=lambda t: (-t[2].score, -len(t[2].text)))

    # Anchor-overlap rule: the top span defines the answer's topic via
    # its content-word overlap with the question. Any secondary span
    # with strictly fewer question-words shared is on a *different*
    # subtopic (e.g. "IKEA bookshelf 4 hours" anchors on {ikea,bookshelf};
    # "IKEA table 2 hours" only shares {ikea} — drop it). Without this
    # rule the selector regresses to the original broad-window failure
    # mode where the distractor sits next to the correct span.
    top_overlap = candidates[0][0]
    candidates = [t for t in candidates if t[0] >= top_overlap]

    # Dedup: drop spans whose text is a substring of a higher-scoring
    # span — the wiki window often repeats the matched turn inside
    # the "Nearby Context" block.
    deduped: list[EvidenceSpan] = []
    for _overlap, _is_distractor, cand in candidates:
        if any(
            cand.text in keeper.text or keeper.text in cand.text
            for keeper in deduped
        ):
            continue
        deduped.append(cand)
        if len(deduped) >= max_spans:
            break
    return deduped


# ─── Prompt-friendly rendering ─────────────────────────────────────────────


def render_spans_for_prompt(spans: Iterable[EvidenceSpan]) -> str:
    """
    Format spans as a compact evidence block for the answerer prompt.

    Each span is rendered as ``- [session=…] (role): exact text``,
    preserving the source attribution the answer-shape validator can
    later cross-reference.
    """
    lines: list[str] = []
    for sp in spans:
        sid = sp.source_session_id or "?"
        role = sp.role or "?"
        lines.append(f"- [session={sid}] ({role}): {sp.text}")
    return "\n".join(lines)


__all__ = [
    "EvidenceSpan",
    "render_spans_for_prompt",
    "select_spans",
]
