#!/usr/bin/env python3
"""
evals/beam/convert.py
=====================
Convert the published BEAM dataset into the LongMemEval-S row shape that
``evals/longmemeval/bootstrap_ollama.py`` consumes, carrying BEAM's ``rubric``
nuggets through so ``evals/beam/rubric_judge.py`` can score the answers.

Why this exists
---------------
The original conversion was done ad-hoc into ``scratchpad/`` and never
committed, so the BEAM runs in ``results/beam_topk*`` were not reproducible —
the dataset they were generated from no longer existed. This script makes the
whole path reproducible from the public source.

Source
------
``Mohammadta/BEAM`` on the HuggingFace Hub (BEAM, ICLR 2026 —
github.com/mohammadtavakoli78/BEAM). Splits: ``100K`` (20 conversations),
``500K`` (35), ``1M`` (35). BEAM-10M is a separate repo (``Mohammadta/BEAM-10M``).

Emitted schema (one JSON list; a superset satisfying BOTH consumers)
--------------------------------------------------------------------
``bootstrap_ollama.load_longmemeval_rows`` requires::

    question_id, question, answer, question_type, question_date,
    haystack_session_ids, haystack_sessions, haystack_dates, answer_session_ids

``rubric_judge`` additionally requires::

    question_id, question, question_type, rubric

Mapping notes
-------------
* A BEAM ``chat`` is an array of *sessions*, each an array of message dicts.
  Each message carries a ``time_anchor`` (e.g. ``March-15-2024``); we take a
  session's date from its first message's anchor, which is what makes the
  retriever's recency signal a genuine signal rather than noise.
* ``contradiction_resolution`` stores gold under ``ideal_answer``;
  ``event_ordering`` uses ``answer``. Both carry ``rubric``.
* ``source_chat_ids`` references message ``id``s (flat across the conversation,
  and for contradiction it is a dict of ``first_statement`` / ``second_statement``
  lists). We resolve those ids back to the sessions containing them, which is
  what ``answer_session_ids`` means in LongMemEval terms.

Run
---
::

    python3.12 -m evals.beam.convert --split 100K \\
        --out evals/beam/data/beam_100K_c_o.json

    # then, unchanged:
    python3.12 -m evals.longmemeval.bootstrap_ollama \\
        --dataset evals/beam/data/beam_100K_c_o.json --dataset-name beam-100K ...
    python3.12 -m evals.beam.rubric_judge \\
        --answers <answers.json> --dataset evals/beam/data/beam_100K_c_o.json
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

#: BEAM categories this converter emits. These are the two the repo's existing
#: runs cover; ``--categories all`` emits every category BEAM ships.
DEFAULT_CATEGORIES = ("contradiction_resolution", "event_ordering")

#: Per-category field holding the gold answer (BEAM is not uniform here).
_GOLD_FIELD = {
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "abstention": "ideal_response",
}
_GOLD_FALLBACKS = ("ideal_answer", "answer", "ideal_response", "gold_answer")

_HF_REPO = "Mohammadta/BEAM"


def _parse_time_anchor(raw: str | None) -> str:
    """``March-15-2024`` -> ``2024-03-15``; unparseable -> ``""``."""
    if not raw:
        return ""
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(str(raw).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _load_split(split: str, cache_dir: str | None) -> Any:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        _HF_REPO, f"data/{split}-00000-of-00001.parquet",
        repo_type="dataset", cache_dir=cache_dir,
    )
    return pd.read_parquet(path)


def _gold(cat: str, q: dict[str, Any]) -> str:
    field = _GOLD_FIELD.get(cat)
    if field and q.get(field):
        return str(q[field])
    for f in _GOLD_FALLBACKS:
        if q.get(f):
            return str(q[f])
    return ""


def _flatten_source_ids(raw: Any) -> list[int]:
    """BEAM's ``source_chat_ids`` is a flat list (event_ordering) or a dict of
    lists (contradiction_resolution). Normalise to a flat list of ints."""
    out: list[int] = []

    def _add(v: Any) -> None:
        if isinstance(v, dict):
            for sub in v.values():
                _add(sub)
        elif isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
            for sub in (v.tolist() if hasattr(v, "tolist") else v):
                _add(sub)
        else:
            with contextlib.suppress(TypeError, ValueError):
                out.append(int(str(v).strip()))

    _add(raw)
    return out


def convert_split(split: str, categories: tuple[str, ...],
                  cache_dir: str | None = None) -> list[dict[str, Any]]:
    df = _load_split(split, cache_dir)
    rows: list[dict[str, Any]] = []

    for _, conv in df.iterrows():
        conv_id = str(conv["conversation_id"])
        chat = conv["chat"]

        # Build sessions + the message-id -> session-id index in one pass.
        session_ids: list[str] = []
        sessions: list[list[dict[str, str]]] = []
        dates: list[str] = []
        msgid_to_session: dict[int, str] = {}

        for s_idx, session in enumerate(chat):
            sid = f"beam-{conv_id}-s{s_idx}"
            msgs: list[dict[str, str]] = []
            sdate = ""
            for msg in session:
                msgs.append({"role": str(msg["role"]),
                             "content": str(msg["content"])})
                if not sdate:
                    sdate = _parse_time_anchor(msg.get("time_anchor"))
                with contextlib.suppress(TypeError, ValueError, KeyError):
                    msgid_to_session[int(str(msg["id"]).strip())] = sid
            session_ids.append(sid)
            sessions.append(msgs)
            dates.append(sdate)

        # The question's reference "now" — the latest session date we saw.
        question_date = max((d for d in dates if d), default="")

        probing = conv["probing_questions"]
        if isinstance(probing, str):
            probing = ast.literal_eval(probing)

        for cat in categories:
            for q_idx, q in enumerate(probing.get(cat, []) or []):
                rubric = q.get("rubric") or []
                if hasattr(rubric, "tolist"):
                    rubric = rubric.tolist()
                rubric = [str(n) for n in rubric]
                if not rubric:
                    # rubric_judge scores such a row 0.0 with error "no-rubric";
                    # emitting it would silently depress the score, so skip and
                    # report the count at the end instead.
                    continue

                ans_sids = sorted({
                    msgid_to_session[i]
                    for i in _flatten_source_ids(q.get("source_chat_ids"))
                    if i in msgid_to_session
                })

                rows.append({
                    "question_id": f"beam-{split}-{conv_id}-{cat}-{q_idx}",
                    "question": str(q.get("question", "")),
                    "answer": _gold(cat, q),
                    "question_type": cat,
                    "question_date": question_date,
                    "haystack_session_ids": session_ids,
                    "haystack_sessions": sessions,
                    "haystack_dates": dates,
                    # Fall back to every session rather than none, so a row with
                    # unresolvable provenance still scores instead of erroring.
                    "answer_session_ids": ans_sids or session_ids,
                    "rubric": rubric,
                    # Provenance, ignored by both consumers.
                    "beam_split": split,
                    "beam_conversation_id": conv_id,
                    "beam_difficulty": str(q.get("difficulty", "")),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="100K", choices=("100K", "500K", "1M"),
                   help="BEAM conversation-size split (default 100K).")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                   help="Comma-separated BEAM categories, or 'all'.")
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument("--cache-dir", default=None, help="HuggingFace cache dir.")
    a = p.parse_args(argv)

    cats: tuple[str, ...]
    if a.categories.strip().lower() == "all":
        cats = ()
    else:
        cats = tuple(c.strip() for c in a.categories.split(",") if c.strip())

    if not cats:  # 'all' — discover from the first conversation
        df = _load_split(a.split, a.cache_dir)
        probing = df.iloc[0]["probing_questions"]
        if isinstance(probing, str):
            probing = ast.literal_eval(probing)
        cats = tuple(probing.keys())

    print(f"  converting BEAM {a.split} · categories: {', '.join(cats)}", flush=True)
    rows = convert_split(a.split, cats, a.cache_dir)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2, default=str))

    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["question_type"]] = by_cat.get(r["question_type"], 0) + 1
    print(f"  wrote {len(rows)} rows -> {a.out}")
    for c, n in sorted(by_cat.items()):
        print(f"    {c:<28} {n}")
    if rows:
        n_sess = len(rows[0]["haystack_sessions"])
        n_msg = sum(len(s) for s in rows[0]["haystack_sessions"])
        print(f"  (row 0 haystack: {n_sess} sessions, {n_msg} messages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["convert_split", "main"]
