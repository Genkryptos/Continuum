"""
scripts/build_diagnostic_sample.py
==================================
Build a deterministic, category-stratified 50-row diagnostic sample
of LongMemEval-S question IDs for cheap regression testing.

Composition (per the spec):

    single-session-user        10
    multi-session              10
    temporal-reasoning         10
    knowledge-update           10
    single-session-preference   5
    abstain                     5

When a category is short on rows, the script logs a warning and pads
from the closest-related category (e.g. abstain shortage → pull from
``*_abs`` question_ids; preference shortage → fall back to
single-session-user with a note).

The output is committed under ``samples/diagnostic_50_ids.json`` so
re-running the eval pipeline against the same sample is one flag:

    --question-ids-file samples/diagnostic_50_ids.json

Usage::

    python scripts/build_diagnostic_sample.py \\
        --input evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json \\
        --output samples/diagnostic_50_ids.json \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

log = logging.getLogger("build_diagnostic_sample")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO / "evals" / "longmemeval" / "LongMemEval" / "data"
    / "longmemeval_s_cleaned.json"
)
DEFAULT_OUTPUT = REPO / "samples" / "diagnostic_50_ids.json"

# Spec composition. Keys must match LongMemEval's ``question_type``
# values; "abstain" is detected separately (see _is_abstain_row).
COMPOSITION: dict[str, int] = {
    "single-session-user":        10,
    "multi-session":              10,
    "temporal-reasoning":         10,
    "knowledge-update":           10,
    "single-session-preference":   5,
    "abstain":                     5,
}


_ABSTAIN_ANSWER_SIGNALS = (
    "not mentioned", "not enough information", "no information",
    "do not know", "don't know", "cannot determine",
    "unable to determine", "no record", "not available",
)


def _is_abstain_row(row: dict[str, Any]) -> bool:
    """True when a row is an abstain-style question.

    LongMemEval marks abstention with ``_abs`` suffix on ``question_id``;
    we also fall back to a phrase match on the expected answer because
    not every abstain row carries the suffix in the cleaned set.
    """
    qid = str(row.get("question_id") or "")
    if qid.endswith("_abs"):
        return True
    expected = str(row.get("answer") or "").lower()
    return any(s in expected for s in _ABSTAIN_ANSWER_SIGNALS)


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return ``{category: [question_id, ...]}`` from the dataset."""
    out: dict[str, list[str]] = {k: [] for k in COMPOSITION}
    for row in rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        qtype = str(row.get("question_type") or "")
        # Abstain bucket — checked first so abstain examples aren't
        # double-counted in their category-of-origin bucket.
        if _is_abstain_row(row):
            out["abstain"].append(qid)
            continue
        if qtype in out:
            out[qtype].append(qid)
    return out


def build_sample(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> tuple[list[str], dict[str, int]]:
    """
    Return ``(question_ids, breakdown)`` from the dataset *rows*.

    The breakdown reflects what was ACTUALLY sampled per bucket;
    when a bucket fell short, the deficit is reported via warnings.
    """
    buckets = _bucket_rows(rows)
    rng = random.Random(seed)
    picked: list[str] = []
    breakdown: dict[str, int] = {}

    for category, want in COMPOSITION.items():
        pool = list(buckets[category])
        pool.sort()  # deterministic before shuffle
        rng.shuffle(pool)
        take = pool[:want]
        if len(take) < want:
            log.warning(
                "bucket %r short: wanted %d, got %d",
                category, want, len(take),
            )
        picked.extend(take)
        breakdown[category] = len(take)

    # Final dedup — abstain may overlap with another category if the
    # dataset is small enough; pulling unique ids preserves the count
    # of distinct questions actually graded.
    deduped: list[str] = []
    seen: set[str] = set()
    for qid in picked:
        if qid in seen:
            continue
        seen.add(qid)
        deduped.append(qid)
    return deduped, breakdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Source dataset JSON.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Where to write the sample JSON (overwrites).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed; same seed + same dataset → same sample.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-category warnings.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"dataset not found: {args.input}")

    rows = json.loads(args.input.read_text())
    if not isinstance(rows, list):
        raise SystemExit(f"unexpected dataset shape in {args.input}")

    qids, breakdown = build_sample(rows, seed=args.seed)

    payload = {
        "name": "diagnostic_50",
        "seed": args.seed,
        "created_from": str(args.input),
        "question_ids": qids,
        "breakdown": breakdown,
        "total": len(qids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    log.info("wrote %d ids to %s", len(qids), args.output)
    log.info("breakdown: %s", breakdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
