"""
scripts/build_budget_sample.py
==============================
Build the frozen, category-proportional subset the cost/accuracy budget
ablation runs on (``docs/LOW_BUDGET_PLAN.md``, Phase 1).

Why a subset, and why proportional
----------------------------------
The ablation is five budget points, and every later phase (router,
facts-not-turns, rerank tail) re-runs against the same rows. At ~10s per
question that is hours of wall clock, and wall clock — not money — is the
binding constraint: five full-500 runs is ~7h serial, five 250-runs is
~3.5h.

Proportional, not the fixed composition of ``build_diagnostic_sample.py``.
The diagnostic sample deliberately over-weights small categories to catch
regressions. This one must preserve LongMemEval-S's real category mix so
the subset's aggregate accuracy is directly comparable to the published
full-500 number, and so the per-category cells stay large enough to read
(n>=15 at the default size).

The sample is **frozen and committed**. Every budget point must run on
byte-identical rows or the points are not comparable — that is the whole
premise of the curve. Do not regenerate it mid-ablation.

Usage::

    python scripts/build_budget_sample.py
    python scripts/build_budget_sample.py --size 250 --seed 7

Then::

    --question-ids-file samples/budget_strat_250.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger("build_budget_sample")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO / "evals" / "longmemeval" / "LongMemEval" / "data"
    / "longmemeval_s_cleaned.json"
)
DEFAULT_SIZE = 250
DEFAULT_SEED = 0


def _allocate(counts: dict[str, int], total: int, size: int) -> dict[str, int]:
    """
    Split ``size`` across categories in proportion to ``counts``.

    Largest-remainder apportionment: floor every share, then hand the
    leftover seats to the largest fractional remainders. Guarantees the
    allocation sums to exactly ``size`` — a naive round() does not, and a
    subset of 249 or 251 would quietly break the "half the dataset"
    framing this sample is chosen for.
    """
    exact = {k: size * v / total for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    leftover = size - sum(alloc.values())
    # Ties broken by category name so the result is fully deterministic.
    order = sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in order[:leftover]:
        alloc[k] += 1
    return alloc


def build_sample(
    rows: list[dict[str, Any]],
    *,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], dict[str, int]]:
    """Return ``(question_ids, breakdown)`` — proportional and deterministic."""
    buckets: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        buckets.setdefault(str(row.get("question_type") or "unknown"), []).append(qid)

    counts = {k: len(v) for k, v in buckets.items()}
    total = sum(counts.values())
    if size > total:
        raise SystemExit(f"--size {size} exceeds the dataset's {total} rows")
    alloc = _allocate(counts, total, size)

    rng = random.Random(seed)
    picked: list[str] = []
    breakdown: dict[str, int] = {}
    for category in sorted(buckets):
        pool = sorted(buckets[category])  # deterministic before shuffle
        rng.shuffle(pool)
        take = pool[: alloc[category]]
        if len(take) < alloc[category]:
            log.warning(
                "bucket %r short: wanted %d, got %d",
                category, alloc[category], len(take),
            )
        picked.extend(take)
        breakdown[category] = len(take)

    # Preserve the dataset's source order — trace logs stay comparable
    # across runs and diffs against other samples stay readable.
    order = {
        str(r.get("question_id")): i
        for i, r in enumerate(rows)
    }
    picked.sort(key=lambda q: order.get(q, 1 << 30))
    return picked, breakdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Defaults to samples/budget_strat_<size>.json",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Same seed + same dataset -> same sample.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not args.input.exists():
        raise SystemExit(f"dataset not found: {args.input}")
    rows = json.loads(args.input.read_text())
    if not isinstance(rows, list):
        raise SystemExit(f"unexpected dataset shape in {args.input}")

    out_path = args.output or (
        REPO / "samples" / f"budget_strat_{args.size}.json"
    )
    if out_path.exists():
        log.warning(
            "%s already exists — overwriting. If an ablation is in flight, "
            "its earlier budget points ran on the OLD sample and are no "
            "longer comparable to anything you run after this.",
            out_path,
        )

    qids, breakdown = build_sample(rows, size=args.size, seed=args.seed)
    source = Counter(str(r.get("question_type") or "unknown") for r in rows)

    payload = {
        "name": f"budget_strat_{args.size}",
        "purpose": "Frozen subset for the cost/accuracy budget ablation "
                   "(docs/LOW_BUDGET_PLAN.md, Phase 1).",
        "seed": args.seed,
        "created_from": str(args.input),
        "question_ids": qids,
        "breakdown": breakdown,
        "source_distribution": dict(sorted(source.items())),
        "total": len(qids),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    log.info("wrote %d ids to %s", len(qids), out_path)
    n = len(qids)
    for category in sorted(breakdown):
        got, src = breakdown[category], source[category]
        log.info(
            "  %-27s %3d  (%4.1f%% vs %4.1f%% in the full set)",
            category, got, 100 * got / n, 100 * src / len(rows),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
