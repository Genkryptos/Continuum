"""
Reproduce the coverage numbers in ``continuum.promotion.scalar_supersession``.

    python3.12 -m evals.audit.scalar_supersession_coverage

Zero LLM calls, zero network, no reader — it reads the LongMemEval-S file and
prints the table. Positives are the real gold supersession pairs; negatives
are the *hardest* available, i.e. the highest-overlap non-gold user-turn pairs
from the same haystacks, because random pairs would flatter the rule.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from continuum.promotion.attribute_extract import extract_attribute
from continuum.promotion.scalar_supersession import (
    DEFAULT_MIN_OVERLAP,
    frame_overlap,
    supersedes,
)
from evals.longmemeval.continuum_ltm_store import _UPDATE_MARKERS
from evals.longmemeval.evidence import gold_turns_from_raw

DATA = Path("evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json")
HARD_NEG_HAYSTACKS = 35   # haystacks to mine negatives from
HARD_NEG_PER_HAYSTACK = 12


def main() -> int:
    rows = json.loads(DATA.read_text())
    ku = [q for q in rows if q["question_type"] == "knowledge-update"]
    pairs = [(q, g) for q in ku if len(g := gold_turns_from_raw(q)) >= 2]

    hard: list[tuple[str, str]] = []
    for q, gold in pairs[:HARD_NEG_HAYSTACKS]:
        seen = {t.content for t in gold}
        turns = [
            t["content"]
            for s in q["haystack_sessions"] for t in s
            if t.get("role") == "user" and t["content"] not in seen
        ][:60]
        scored = sorted(
            (
                (ov, a, b)
                for a, b in itertools.combinations(turns, 2)
                if (ov := frame_overlap(a, b)) >= DEFAULT_MIN_OVERLAP
            ),
            reverse=True, key=lambda x: x[0],
        )
        hard.extend([(a, b) for _, a, b in scored[:HARD_NEG_PER_HAYSTACK]])

    marker = sum(1 for _, g in pairs if _UPDATE_MARKERS.search(g[-1].content))
    attr = sum(
        1 for _, g in pairs
        if (x := extract_attribute(g[0].content)) is not None
        and x == extract_attribute(g[-1].content)
    )
    tp = sum(1 for _, g in pairs if supersedes(g[-1].content, g[0].content))
    fp = sum(1 for a, b in hard if supersedes(a, b))

    n = len(pairs)
    print(f"knowledge-update questions with a gold pair: {n}")
    print(f"hard negatives (top-overlap non-gold pairs): {len(hard)}\n")
    print(f"{'detector':<42}{'recall':>8}{'hard-neg FP':>14}")
    print("-" * 64)
    print(f"{'_UPDATE_MARKERS regex':<42}{marker / n:>7.0%}{'—':>14}")
    print(f"{'attribute_extract key collision':<42}{attr / n:>7.0%}{'—':>14}")
    print(f"{'scalar_supersession (this module)':<42}{tp / n:>7.0%}"
          f"{fp / len(hard) if hard else 0:>13.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
