"""
evals.locomo
============
LOCOMO benchmark harness for Continuum, with a Mem0 head-to-head.

LOCOMO (snap-research/locomo, ``data/locomo10.json``) is 10 long
multi-session ``speaker_a``/``speaker_b`` conversations, each annotated
with a ``qa`` list of ``{question, answer, evidence, category}``. We
evaluate two memory systems on the same questions, same answerer model,
same LLM judge:

* :mod:`evals.locomo.continuum_adapter` — v1's *actual* winning stack:
  direct retrieval (hybrid + session-aware) over a FlatHaystackStore,
  one answerer call. (NOT the IterativeReasoner — this session's evals
  showed it net-negative; comparing against a known-loser component
  would be dishonest.)
* :mod:`evals.locomo.mem0_adapter` — the ``mem0ai`` package, standard
  config, not handicapped.
"""

from __future__ import annotations

from evals.locomo.loader import (
    LocomoConversation,
    LocomoQuestion,
    LocomoTurn,
    category_name,
    load_locomo,
)

__all__ = [
    "LocomoConversation",
    "LocomoQuestion",
    "LocomoTurn",
    "category_name",
    "load_locomo",
]
