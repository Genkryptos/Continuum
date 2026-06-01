"""
evals/longmemeval/session_narrowing.py
======================================
Top-session narrowing for LongMemEval-style single-session questions.

The wiki retriever returns evidence windows from the top-K *items*,
which on LongMemEval often spans several distinct *sessions* — the
correct one plus 2-4 ShareGPT / UltraChat distractor sessions that
were lexically close. Empirically on the diagnostic_50 sample, the
correct session is rank 1 with ``partial_recall = 1.0`` on the
majority of single-session-* rows, and the model still hallucinates
because the answer prompt contains evidence from unrelated sessions.

This module implements a deterministic narrowing pass:

* When the **top-ranked session** carries a verified answer-bearing
  candidate, retain only that session's items.
* When the question is **explicitly multi-session** (classifier hint
  or category metadata), narrowing is bypassed and every session is
  passed through.
* When the **top session has no verified evidence** OR every
  verified candidate is below ``min_verified_confidence``, all
  sessions are kept — narrowing should never silently drop evidence
  when there's no positive signal that the top session is right.

Telemetry
---------
:class:`NarrowResult` records the kept session id, the dropped session
ids, the verifier confidence that drove the decision, and a short
human-readable reason. The adapter stamps this onto
``last_decomposition_stats`` so the JSONL trace shows every narrowing
decision.

Wiring contract
---------------
The caller passes:

* ``items`` — the retrieved bundle items, in rank order (item[0] is
  the top-ranked item).
* ``verified_candidates`` — the candidates that already passed the
  claim verifier (see :mod:`evals.longmemeval.claim_verifier`). The
  narrowing key is the verified candidate's ``source_session_id``.
* ``multi_session_hint`` — ``True`` when the question is known to
  span sessions (e.g. ``QuestionType.MULTI_SESSION`` or the
  ``is_multi_session_hint`` regex fired). Bypasses narrowing.

Why deterministic
-----------------
The signal is concrete: ranking already happened, candidates already
went through the verifier, and the question type is already
classified. There is nothing for an LLM to add here — the decision
is "is the top session a verified hit?" and we already know the
answer.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from continuum.core.types import MemoryItem
from evals.longmemeval.candidates import Candidate

log = logging.getLogger(__name__)


# ─── Result dataclass ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class NarrowResult:
    """Outcome of one narrowing decision."""

    items: list[MemoryItem]
    chosen_session_id: str
    dropped_session_ids: list[str]
    narrowed: bool
    reason: str
    verified_confidence: float
    n_input_sessions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_session_id": self.chosen_session_id,
            "dropped_session_ids": list(self.dropped_session_ids),
            "narrowed": self.narrowed,
            "reason": self.reason,
            "verified_confidence": round(self.verified_confidence, 3),
            "n_input_sessions": self.n_input_sessions,
            "n_output_items": len(self.items),
        }


# ─── Helpers ───────────────────────────────────────────────────────────────


def _session_id_of(item: MemoryItem) -> str:
    """Stringified session id from the item or its metadata."""
    metadata = item.metadata or {}
    sid = metadata.get("session_id") or item.session_id or ""
    return str(sid)


def _ordered_session_ids(items: Sequence[MemoryItem]) -> list[str]:
    """Distinct session ids in first-seen rank order. Items without an id contribute the empty string."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        sid = _session_id_of(item)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _items_from_session(
    items: Sequence[MemoryItem], session_id: str,
) -> list[MemoryItem]:
    """Items whose session_id matches, preserving rank order."""
    return [item for item in items if _session_id_of(item) == session_id]


# ─── Public entry point ───────────────────────────────────────────────────


def narrow_to_top_session(
    items: Sequence[MemoryItem],
    verified_candidates: Iterable[Candidate],
    *,
    multi_session_hint: bool = False,
    min_verified_confidence: float = 0.5,
) -> NarrowResult:
    """
    Drop evidence from sessions other than the top-ranked one **only**
    when the top session carries a verified answer-bearing candidate.

    Parameters
    ----------
    items:
        Retrieved bundle items in rank order. Item[0] is the top-ranked
        item; its ``session_id`` defines "the top session". Items from
        the same session are grouped wherever they sit in the rank list.
    verified_candidates:
        Candidates that already passed the claim verifier. The
        narrowing keys on ``Candidate.source_session_id``.
    multi_session_hint:
        ``True`` when the question explicitly spans sessions
        (``QuestionType.MULTI_SESSION`` / ``is_multi_session_hint``
        regex / dataset-tag). Bypasses narrowing — returns every item
        with ``narrowed=False``, ``reason="multi_session_hint"``.
    min_verified_confidence:
        Floor on the verified candidate's confidence. Below this we
        refuse to narrow — the verifier passed it, but not with enough
        conviction to throw away other sessions.

    Returns
    -------
    NarrowResult
        ``items`` is either the original list (no narrowing happened)
        or only the top-session items. ``reason`` explains the choice
        so the JSONL trace can be read without re-running the verifier.
    """
    items_list = list(items)
    session_order = _ordered_session_ids(items_list)
    n_sessions = len(session_order)

    # Empty input — nothing to narrow.
    if not items_list or not session_order:
        return NarrowResult(
            items=items_list,
            chosen_session_id="",
            dropped_session_ids=[],
            narrowed=False,
            reason="no_items",
            verified_confidence=0.0,
            n_input_sessions=0,
        )

    # Explicit multi-session question — bypass.
    if multi_session_hint:
        return NarrowResult(
            items=items_list,
            chosen_session_id="",
            dropped_session_ids=[],
            narrowed=False,
            reason="multi_session_hint",
            verified_confidence=0.0,
            n_input_sessions=n_sessions,
        )

    # Only one session is present — nothing to narrow.
    if n_sessions == 1:
        return NarrowResult(
            items=items_list,
            chosen_session_id=session_order[0],
            dropped_session_ids=[],
            narrowed=False,
            reason="single_session_only",
            verified_confidence=0.0,
            n_input_sessions=1,
        )

    top_session = session_order[0]
    # Find the highest-confidence verified candidate sourced from the
    # top session. The verifier may have produced multiple passes for
    # the same session — pick the strongest one to drive the decision.
    top_verified = [
        c for c in verified_candidates
        if c.source_session_id == top_session
    ]
    if not top_verified:
        return NarrowResult(
            items=items_list,
            chosen_session_id=top_session,
            dropped_session_ids=[],
            narrowed=False,
            reason="top_session_unverified",
            verified_confidence=0.0,
            n_input_sessions=n_sessions,
        )

    best_conf = max(c.confidence for c in top_verified)
    if best_conf < min_verified_confidence:
        return NarrowResult(
            items=items_list,
            chosen_session_id=top_session,
            dropped_session_ids=[],
            narrowed=False,
            reason=(
                f"top_session_confidence_below_threshold:"
                f"{best_conf:.2f}<{min_verified_confidence:.2f}"
            ),
            verified_confidence=best_conf,
            n_input_sessions=n_sessions,
        )

    # Narrow: keep only items belonging to the top session.
    narrowed_items = _items_from_session(items_list, top_session)
    dropped_sessions = [sid for sid in session_order if sid != top_session]
    log.info(
        "session_narrow: kept=%s dropped=%s (confidence=%.2f)",
        top_session, dropped_sessions, best_conf,
    )
    return NarrowResult(
        items=narrowed_items,
        chosen_session_id=top_session,
        dropped_session_ids=dropped_sessions,
        narrowed=True,
        reason="narrowed_to_top_session",
        verified_confidence=best_conf,
        n_input_sessions=n_sessions,
    )


__all__ = [
    "NarrowResult",
    "narrow_to_top_session",
]
