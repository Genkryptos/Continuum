"""
evals/longmemeval/claims.py
===========================
Claim-first evidence layer.

What a claim is
---------------
Coarser than a :class:`evals.longmemeval.candidates.Candidate` — a
:class:`Claim` is *one source-backed sentence* lifted from the
retrieved evidence window, with the role/session/turn/date metadata
preserved so downstream reasoning heads can rank and pick from
*sentences*, not from regex fragments.

The previous failure mode (rows e47becba / 58ef2f1c / 5d3d2817 / etc)
was that the regex extractors filtered the candidate pool *before* the
question-type router ran. A wrong question_type guess removed the
correct evidence entirely, leaving the answerer with nothing. Working
at the claim level means the question-type assignment becomes a *soft*
ranking signal, not a hard filter that can throw away the answer.

Pipeline shape
--------------
::

    retrieved bundle
       │
       ▼  extract_claims_from_context
    list[Claim]   ← scaffold-filtered, role-tagged
       │
       ▼  rank_claims(question, mode)
    ordered list[Claim]
       │
       ▼  one of six reasoning heads
    final answer string

This module owns the first two steps. The reasoning heads live in
:mod:`evals.longmemeval.reasoning_heads`; the router that picks which
head to call lives in :mod:`evals.longmemeval.task_router`.

Filtering
---------
``extract_claims_from_context`` drops every line that looks like
prompt scaffold (``## Matched Turn``, ``Sub-answer:``) AND every line
that looks like transcript-timestamp junk (``02 share``,
``04 principles``, ``0:02``, ``00:00:12``) — small numeric prefixes
that the wiki retriever's evidence window picks up from speaker
metadata. Without this filter, "02 share" lands in the candidate pool
as a count of "share" objects.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Iterable
from typing import Any

from continuum.core.types import ContextBundle, MemoryItem
from evals.longmemeval.scaffold_filter import (
    is_label_metadata_line,
    is_scaffold_line,
    is_scaffold_text,
    is_user_request_sentence,
)

log = logging.getLogger(__name__)


# ─── Claim dataclass ───────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Claim:
    """
    One sentence-level evidence claim with full provenance.

    ``answer_source_span`` is the substring of ``text`` that the
    downstream reasoning head selected as the answer; it stays empty
    at extraction time and is populated when a head fills it in.
    """

    text: str                       # the source sentence (full)
    source_session_id: str          # session_id from the bundle metadata
    source_role: str                # "user" / "assistant" / "" (unknown)
    turn_index: int | None = None   # index within the session, when available
    source_date: str = ""           # ISO date / timestamp on the source item
    confidence: float = 0.6         # extractor's initial confidence
    answer_source_span: str = ""    # populated by the reasoning head

    @property
    def is_user(self) -> bool:
        return self.source_role.lower() == "user"

    @property
    def is_assistant(self) -> bool:
        return self.source_role.lower() == "assistant"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_session_id": self.source_session_id,
            "source_role": self.source_role,
            "turn_index": self.turn_index,
            "source_date": self.source_date,
            "confidence": round(self.confidence, 3),
            "answer_source_span": self.answer_source_span,
        }


# ─── Transcript-timestamp noise filter ─────────────────────────────────────


#: Lines that look like timestamp metadata leaking from the source
#: transcript. Examples observed: "02 share", "04 principles",
#: "0:02 something", "00:00:12 here". The pattern requires a numeric
#: prefix (with or without a colon) followed by a short word.
_TIMESTAMP_NOISE_RE = re.compile(
    r"^\s*\d{1,3}(?::\d{1,2}){0,2}\s+[a-z][a-z\-]{1,15}\s*$",
    re.IGNORECASE,
)
#: Even narrower: just "MM:SS" / "HH:MM:SS" on its own line.
_BARE_TIMESTAMP_RE = re.compile(r"^\s*\d{1,3}(?::\d{1,2}){1,2}\s*$")


def is_transcript_timestamp_noise(line: str) -> bool:
    """True when the line is transcript timestamp metadata, not content."""
    if not line or not line.strip():
        return False
    s = line.strip()
    if _BARE_TIMESTAMP_RE.match(s):
        return True
    if _TIMESTAMP_NOISE_RE.match(s):
        return True
    return False


# ─── Claim extraction ──────────────────────────────────────────────────────


#: Matches the wiki-bundle's per-turn role prefix in any of three
#: live shapes (mirror of :data:`structured_claims._BULLET_RE`):
#:   * ``- (user): text``            — Matched Turn line
#:   * ``- turn 1 (assistant): text`` — Nearby Context line
#:   * ``assistant: text``           — inline role prefix embedded mid-content
#: The ``role`` group is the speaker; ``body`` is the content. Without
#: the ``turn N`` alternative the legacy extractor swallowed the whole
#: ``turn 1 (user): Give me 100 prompt parameters...`` string as the
#: claim body — and the assistant-memory head returned it verbatim as
#: the answer (the 8752c811 regression). Stripping the prefix at the
#: source lets every downstream head operate on real content.
_BULLET_RE = re.compile(
    r"^\s*-\s+"
    r"(?:turn\s+\d+\s+)?"
    r"(?:\((?P<role>user|assistant|system)\):\s*)?"
    r"(?P<body>.+)$",
    re.IGNORECASE,
)
_INLINE_ROLE_RE = re.compile(
    r"^\s*(?P<role>user|assistant|system)\s*:\s*(?P<body>.+)$",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_MIN_CLAIM_LEN = 4


def _split_into_sentences(body: str) -> list[str]:
    """Sentence-split, dropping fragments under ``_MIN_CLAIM_LEN`` chars."""
    body = body.strip()
    if not body:
        return []
    out: list[str] = []
    for s in _SENTENCE_SPLIT_RE.split(body):
        s = s.strip()
        if len(s) >= _MIN_CLAIM_LEN:
            out.append(s)
    return out


def _claims_from_item(
    item: MemoryItem, *, item_index: int,
) -> list[Claim]:
    """Extract claims from one bundle item, with role / session metadata."""
    metadata = item.metadata or {}
    session_id = str(metadata.get("session_id") or item.session_id or "")
    item_role = str(metadata.get("role") or "")
    item_date = (
        str(metadata.get("date") or metadata.get("timestamp") or "")
        if metadata else ""
    )

    out: list[Claim] = []
    for line_no, raw_line in enumerate((item.content or "").splitlines()):
        line = raw_line.rstrip()
        # 1) scaffold / wrapper lines → skip
        if is_scaffold_line(line):
            continue
        # 2) transcript-timestamp noise → skip
        if is_transcript_timestamp_noise(line):
            continue
        # 3) label-only metadata lines (``Occasion: Sister's birthday``)
        #    → skip. The label tells us the topic but the body is not
        #    a memory claim; if the body has any content the per-
        #    sentence loop below will pick it up from the wider bundle.
        if is_label_metadata_line(line):
            continue

        # 3) Parse the per-line role prefix in any of three shapes:
        #    "- (user): text", "- turn N (assistant): text", or the
        #    inline "assistant: text" form (used when the bundler
        #    stuffed an assistant reply into a user turn's content).
        #    Falls back to the outer item role when no prefix matches.
        bullet = _BULLET_RE.match(line)
        if bullet:
            role = (bullet.group("role") or item_role or "").lower()
            body = bullet.group("body").strip()
        else:
            inline = _INLINE_ROLE_RE.match(line)
            if inline:
                role = inline.group("role").lower()
                body = inline.group("body").strip()
            else:
                role = item_role.lower()
                body = line.strip()

        # 4) split body into sentences and filter each one
        for sent_no, sentence in enumerate(_split_into_sentences(body)):
            if is_scaffold_text(sentence):
                continue
            if is_transcript_timestamp_noise(sentence):
                continue
            out.append(Claim(
                text=sentence,
                source_session_id=session_id,
                source_role=role,
                turn_index=item_index * 1000 + line_no * 10 + sent_no,
                source_date=item_date,
                confidence=0.6,
            ))
    return out


def extract_claims_from_context(
    ctx: ContextBundle | None,
) -> list[Claim]:
    """
    Lift every non-scaffold sentence from the retrieved bundle into a
    :class:`Claim`. ``turn_index`` is monotonically increasing across
    items so downstream heads can use it as a "later" / "earlier"
    signal for knowledge-update questions.
    """
    if ctx is None or not getattr(ctx, "items", None):
        return []
    out: list[Claim] = []
    for idx, item in enumerate(ctx.items):
        out.extend(_claims_from_item(item, item_index=idx))
    return out


# ─── Claim ranking ─────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d+")
_CLAIM_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on",
    "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "i", "me", "my", "you", "your", "we", "us", "our", "they",
    "them", "this", "that", "it", "its", "if", "then",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "do", "does", "did", "have", "has", "had", "can", "could",
    "should", "would", "will", "shall",
    "many", "much", "few", "several", "long", "lot", "more", "less",
})

#: Personal questions ("I"/"my"/"me") → boost user-turn claims.
_PERSONAL_RE = re.compile(r"\b(?:i|my|me|mine)\b", re.IGNORECASE)
#: Assistant-memory cues — questions asking what *the assistant said*.
_ASSISTANT_CUE_RE = re.compile(
    r"\b(?:you\s+(?:told|said|recommended|suggested|mentioned|"
    r"asked|advised)|did\s+you\s+(?:recommend|suggest|mention|tell))\b",
    re.IGNORECASE,
)


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower() for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _CLAIM_STOPWORDS and len(t) > 1
    }


def rank_claims(
    question: str,
    claims: Iterable[Claim],
    *,
    prefer_role: str = "",
) -> list[Claim]:
    """
    Rank claims by question-overlap, then role bias.

    Parameters
    ----------
    question:
        The original user question. Drives the content-word overlap
        scoring; question stopwords are filtered.
    claims:
        Claims emitted by :func:`extract_claims_from_context`.
    prefer_role:
        ``"user"`` → boost direct autobiographical claims;
        ``"assistant"`` → boost assistant-memory claims;
        ``""`` → autodetect from question wording (``"I"/"my"/"me"`` →
        user-preference; ``"did you tell"`` → assistant-preference;
        else neutral).

    Returns
    -------
    list[Claim]
        Sorted by descending score; claims with overlap=0 are kept at
        the tail (never dropped — answer-type validation is *soft*).
    """
    q_tokens = _content_tokens(question)
    if not q_tokens:
        return list(claims)

    # Auto-detect role preference if caller didn't specify.
    role_pref = prefer_role.lower().strip()
    if not role_pref:
        if _ASSISTANT_CUE_RE.search(question):
            role_pref = "assistant"
        elif _PERSONAL_RE.search(question):
            role_pref = "user"

    # When the benchmark question is a remembered-fact question (not
    # ``"did you tell me..."``), user-request sentences are almost
    # never the answer — they're the user asking the assistant for
    # help. We penalise them hard so they fall below real claims.
    asks_about_assistant = role_pref == "assistant"

    def score(c: Claim) -> float:
        ctokens = _content_tokens(c.text)
        overlap = len(q_tokens & ctokens)
        s = float(overlap) + 0.2 * c.confidence
        # Boost the favoured role; penalise the other softly so it
        # isn't fully discarded ("never delete all evidence" rule).
        if role_pref == "user":
            if c.is_user:
                s += 0.5
            elif c.is_assistant:
                s -= 0.2
        elif role_pref == "assistant":
            if c.is_assistant:
                s += 0.5
            elif c.is_user:
                s -= 0.1

        # Penalise user-request sentences for fact questions. A claim
        # like "Can you help me organize my sales data from previous
        # markets?" must NOT outrank "I used to be a marketing
        # specialist at a small startup". Skip the penalty when the
        # caller explicitly wants assistant-memory content (those
        # questions sometimes legitimately echo a user request).
        if not asks_about_assistant and is_user_request_sentence(c.text):
            s -= 1.5
        # Penalise heading-only / scaffold-shaped claims that survived
        # extraction. Same intent as the scaffold filter — defence in
        # depth at the ranking layer.
        if is_scaffold_text(c.text):
            s -= 2.0
        return s

    scored = [(score(c), c) for c in claims]
    scored.sort(key=lambda pair: -pair[0])
    return [c for _s, c in scored]


__all__ = [
    "Claim",
    "extract_claims_from_context",
    "is_transcript_timestamp_noise",
    "rank_claims",
]
