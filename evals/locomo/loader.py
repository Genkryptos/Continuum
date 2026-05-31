"""
evals/locomo/loader.py
======================
Parse ``data/locomo10.json`` into normalised shapes the LOCOMO runner
and both adapters consume.

LOCOMO sample shape (from snap-research/locomo)
-----------------------------------------------
Each of the 10 samples is::

    {
      "sample_id": "...",
      "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob",
        "session_1_date_time": "1:00 pm on 8 May, 2023",
        "session_1": [
          {"speaker": "Alice", "dia_id": "D1:1", "text": "..."},
          ...
        ],
        "session_2_date_time": "...", "session_2": [...],
        ...
      },
      "qa": [
        {"question": "...", "answer": "...",
         "evidence": ["D1:3", "D2:7"], "category": 4},
        ...
      ]
    }

We flatten ``conversation`` into an ordered list of turns (each tagged
with its session id + date + speaker + dia_id) and lift the ``qa`` list
into :class:`LocomoQuestion` rows. ``evidence`` (dialog ids) is the
recall ground truth; ``category`` is the question type.

Category integers (per the LOCOMO paper; we map for reporting but never
hard-depend on the exact set):

    1 multi-hop · 2 temporal · 3 open-domain · 4 single-hop · 5 adversarial
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: LOCOMO category int → human label. ``adversarial`` questions have no
#: answer in the conversation (the correct response is a refusal /
#: "not mentioned"); the runner can treat them specially if desired.
_CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def category_name(category: Any) -> str:
    """Map a LOCOMO category value to a label; unknowns pass through as str."""
    try:
        return _CATEGORY_NAMES.get(int(category), f"category-{category}")
    except (TypeError, ValueError):
        return f"category-{category}"


@dataclass
class LocomoTurn:
    """One dialogue turn, tagged with where it sits in the conversation."""

    speaker: str
    text: str
    dia_id: str
    session_id: str  # e.g. "session_3"
    session_date: str  # raw LOCOMO date-time string ("" if absent)


@dataclass
class LocomoConversation:
    """One LOCOMO sample's full multi-session dialogue."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[LocomoTurn] = field(default_factory=list)


@dataclass
class LocomoQuestion:
    """One QA row from a sample's ``qa`` list."""

    sample_id: str
    question: str
    answer: str
    evidence: list[str]  # dialog ids (dia_id) containing the answer
    category: int | None
    category_label: str


# ── parsing ──────────────────────────────────────────────────────────────


_SESSION_RE = re.compile(r"^session_(\d+)$")


def _parse_conversation(sample: dict[str, Any]) -> LocomoConversation:
    conv = sample.get("conversation", {}) or {}
    sample_id = str(sample.get("sample_id", ""))
    speaker_a = str(conv.get("speaker_a", "speaker_a"))
    speaker_b = str(conv.get("speaker_b", "speaker_b"))

    # Collect session keys in numeric order so turns are chronological.
    session_keys: list[tuple[int, str]] = []
    for key in conv:
        m = _SESSION_RE.match(key)
        if m:
            session_keys.append((int(m.group(1)), key))
    session_keys.sort()

    turns: list[LocomoTurn] = []
    for _num, skey in session_keys:
        date = str(conv.get(f"{skey}_date_time", "") or "")
        session = conv.get(skey) or []
        if not isinstance(session, list):
            continue
        for turn in session:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text", "") or "").strip()
            if not text:
                continue
            turns.append(
                LocomoTurn(
                    speaker=str(turn.get("speaker", "") or ""),
                    text=text,
                    dia_id=str(turn.get("dia_id", "") or ""),
                    session_id=skey,
                    session_date=date,
                )
            )
    return LocomoConversation(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        turns=turns,
    )


def _parse_questions(sample: dict[str, Any]) -> list[LocomoQuestion]:
    sample_id = str(sample.get("sample_id", ""))
    out: list[LocomoQuestion] = []
    for qa in sample.get("qa", []) or []:
        if not isinstance(qa, dict):
            continue
        q = str(qa.get("question", "") or "").strip()
        if not q:
            continue
        cat = qa.get("category")
        ev = qa.get("evidence") or []
        if not isinstance(ev, list):
            ev = [ev]
        out.append(
            LocomoQuestion(
                sample_id=sample_id,
                question=q,
                # answers can be numeric/bool in LOCOMO — coerce to str.
                answer=str(qa.get("answer", "")),
                evidence=[str(e) for e in ev],
                category=int(cat) if isinstance(cat, int) else None,
                category_label=category_name(cat),
            )
        )
    return out


def load_locomo(
    path: Path | str,
) -> list[tuple[LocomoConversation, list[LocomoQuestion]]]:
    """
    Load ``locomo10.json`` → list of ``(conversation, questions)`` per
    sample. Raises ``FileNotFoundError`` with a download hint when the
    dataset isn't present.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"LOCOMO dataset not found at {p}. Download it with:\n"
            "  mkdir -p evals/locomo/data && curl -L -o evals/locomo/data/locomo10.json \\\n"
            "    https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        )
    raw = json.loads(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of samples in {p}, got {type(raw)}")
    out: list[tuple[LocomoConversation, list[LocomoQuestion]]] = []
    for sample in raw:
        out.append((_parse_conversation(sample), _parse_questions(sample)))
    return out


__all__ = [
    "LocomoConversation",
    "LocomoQuestion",
    "LocomoTurn",
    "category_name",
    "load_locomo",
]
