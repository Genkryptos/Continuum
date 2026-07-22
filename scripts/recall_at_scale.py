#!/usr/bin/env python3
"""
scripts/recall_at_scale.py
==========================
Does the **right** memory still come back once the store is large?

`scale_test.py` answers a different question — it loads random vectors and
measures latency and index behaviour. It cannot tell you whether retrieval is
still *correct*, and it says so. This does: build a store of real embedded
sentences, hide twenty needles with known answers among them, and ask.

    make recall-at-scale ARGS="--rows 20000"
    python3 scripts/recall_at_scale.py --rows 3000 --shape formulaic

Two corpus shapes, because the difference mattered:

* ``realistic`` (default) — many topical families, varied phrasing, few repeats.
  What a person's store actually looks like.
* ``formulaic`` — a handful of templates repeated with small substitutions, so
  the embedding space collapses into a few tight clusters. Close to a worst case
  for approximate search. A 50k formulaic store showed HNSW losing 8 of 20
  needles that an exact scan found; the same measurement on realistic data showed
  **no index loss at all**. Quoting the first number as if it were general would
  have been wrong, which is why both shapes ship.

Needs a **throwaway** migrated database — it writes tens of thousands of rows.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time

# ── the needles ───────────────────────────────────────────────────────────────
# Each answer is a full sentence and is matched EXACTLY. An earlier version of
# this harness matched a short substring, and '%tin%' happily matched "meeting"
# and "getting" — inflating recall and, on a second pass, deflating it. Never
# score retrieval by substring.
NEEDLES: list[tuple[str, str]] = [
    ("My grandmother's maiden name was Quintanilha.", "who was my grandmother"),
    ("I am deathly allergic to shellfish.", "can I eat prawns"),
    ("My blood type is AB negative.", "what is my blood type"),
    ("I broke my left wrist skiing in 2011.", "have I ever broken a bone"),
    ("My first car was a 1998 Fiat Punto.", "what car did I start with"),
    ("I am terrified of deep water.", "what am I scared of"),
    ("My childhood nickname was Pikas.", "what did people call me as a kid"),
    ("I play the bandoneon badly.", "do I play an instrument"),
    ("My passport expires in March 2031.", "when does my passport run out"),
    ("I am lactose intolerant.", "can I drink milk"),
    ("I once lived on a boat for two years.", "have I lived anywhere unusual"),
    ("My thesis was on photonic crystals.", "what did I write my thesis about"),
    ("I collect vintage fountain pens.", "do I have any collections"),
    ("My wedding anniversary is 14 September.", "when is my anniversary"),
    ("I am colourblind in the red-green range.", "do I have any vision issues"),
    ("I ran the Berlin marathon in 2019.", "have I run a marathon"),
    ("My mother's side of the family is from Goa.", "where is my family from"),
    ("I am allergic to penicillin as well.", "any drug allergies"),
    ("I keep my emergency cash in a tin above the fridge.", "where is my emergency cash"),
    ("My accountant is called Filipa Rego.", "who does my taxes"),
]

PLACES = [
    "Lisbon",
    "Porto",
    "Coimbra",
    "Braga",
    "Faro",
    "Berlin",
    "Munich",
    "Hamburg",
    "Madrid",
    "Seville",
    "Bilbao",
    "Paris",
    "Lyon",
    "Nantes",
    "Rome",
    "Milan",
    "Naples",
    "Vienna",
    "Graz",
    "Prague",
    "Brno",
    "Dublin",
    "Cork",
    "Oslo",
    "Bergen",
    "Helsinki",
    "Warsaw",
    "Krakow",
    "Athens",
    "Zurich",
]
HOBBIES = [
    "sourdough",
    "bouldering",
    "birdwatching",
    "woodworking",
    "sea swimming",
    "chess",
    "pottery",
    "cycling",
    "gardening",
    "photography",
    "sketching",
    "running",
    "knitting",
    "beekeeping",
    "kayaking",
    "astronomy",
]
TOPICS = [
    "the tax deadline",
    "the new lease",
    "the recycling schedule",
    "the boiler service",
    "the bike repair",
    "the dentist referral",
    "the insurance renewal",
    "the parking permit",
    "the water bill",
    "the roof inspection",
    "the phone contract",
    "the gym membership",
]
FOODS = ["bacalhau", "pastel de nata", "ramen", "pierogi", "goulash", "risotto", "tagine", "pho"]
PEOPLE = ["a colleague", "my sister", "my neighbour", "an old friend", "my cousin", "a client"]
FAMILY = ["my mother", "my father", "my brother", "my aunt", "my grandfather", "my niece"]
LANGS = ["Python", "Rust", "Go", "TypeScript", "Java", "Ruby", "Elixir", "Scala", "Kotlin"]


def distractors(n: int, shape: str = "realistic", seed: int = 11) -> list[str]:
    """*n* distinct sentences that are never a needle's answer.

    Raises if the template space cannot produce *n*. An earlier version looped
    forever when asked for more strings than its vocabulary allowed — it spun at
    100% CPU for ten minutes looking like a slow embedder. A generator that
    cannot deliver must say so, not hang.
    """
    rng = random.Random(seed)
    if shape == "formulaic":
        families = [
            lambda: f"I lived in {rng.choice(PLACES)} for {rng.randint(1, 9)} years.",
            lambda: f"I write {rng.choice(LANGS)} at work.",
            lambda: f"I cooked {rng.choice(FOODS)} on the weekend.",
            lambda: f"My dentist is near the {rng.choice(PLACES)} station.",
        ]
        suffix = lambda: f" That was around {rng.choice(range(1998, 2026))}."  # noqa: E731
    else:
        families = [
            lambda: f"I spent a weekend in {rng.choice(PLACES)} back in {rng.randint(2005, 2025)}.",
            lambda: (
                f"{rng.choice(PEOPLE).capitalize()} recommended a place for {rng.choice(FOODS)}."
            ),
            lambda: f"{rng.choice(TOPICS).capitalize()} turned out to be more work than expected.",
            lambda: f"I tried {rng.choice(HOBBIES)} for about {rng.randint(2, 30)} weeks.",
            lambda: f"{rng.choice(FAMILY).capitalize()} called about {rng.choice(TOPICS)}.",
            lambda: f"I paid {rng.randint(12, 900)} euros towards {rng.choice(TOPICS)}.",
            lambda: f"The {rng.choice(HOBBIES)} group meets every {rng.randint(1, 4)} weeks.",
            lambda: (
                f"I spent {rng.randint(1, 14)} hours on {rng.choice(HOBBIES)} in {rng.choice(PLACES)}."
            ),
            lambda: (
                f"The {rng.choice(FOODS)} place near {rng.choice(PLACES)} closes at {rng.randint(17, 23)}."
            ),
            lambda: (
                f"{rng.choice(PEOPLE).capitalize()} is moving to {rng.choice(PLACES)} next year."
            ),
        ]
        suffix = lambda: f" That was around {rng.choice(range(2005, 2026))}."  # noqa: E731

    out: list[str] = []
    seen: set[str] = set()
    stalled = 0
    while len(out) < n:
        s = rng.choice(families)() + suffix()
        if s in seen:
            stalled += 1
            if stalled > 100_000:
                raise ValueError(
                    f"shape {shape!r} cannot produce {n} distinct sentences "
                    f"(reached {len(out)}). Widen the templates or ask for fewer."
                )
            continue
        stalled = 0
        seen.add(s)
        out.append(s)
    return out


async def load(dsn: str, rows: int, shape: str, namespace: str) -> float:
    import psycopg

    from continuum.core.config import ContinuumConfig
    from continuum.db.pgvector_upgrade import to_halfvec_literal
    from continuum.embeddings import EmbeddingService

    texts = distractors(rows, shape) + [fact for fact, _q in NEEDLES]
    emb = EmbeddingService(ContinuumConfig.load().embedding)
    started = time.perf_counter()
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for i in range(0, len(texts), 256):
            chunk = texts[i : i + 256]
            vectors = await emb.embed(chunk)
            cur.executemany(
                'INSERT INTO memory_nodes (layer, "text", embedding, namespace, kind,'
                " importance, confidence) VALUES ('LTM', %s, %s::halfvec, %s, 'fact', 0.5, 1.0)",
                [
                    (t, to_halfvec_literal(v), namespace)
                    for t, v in zip(chunk, vectors, strict=True)
                ],
            )
    return time.perf_counter() - started


async def score(dsn: str, namespace: str, k: int) -> tuple[int, float]:
    """Recall through the real product path, matching the needle sentence exactly."""
    from continuum.memory import Memory

    mem = Memory.from_postgres(dsn, embeddings=True, namespace=namespace)
    await mem.start()
    try:
        await mem.recall("warm up", k=1)
        hits = 0
        latencies: list[float] = []
        misses: list[str] = []
        for fact, query in NEEDLES:
            t = time.perf_counter()
            found = await mem.recall(query, k=k)
            latencies.append((time.perf_counter() - t) * 1000)
            if fact.strip() in [(h.content or "").strip() for h in found]:
                hits += 1
            else:
                misses.append(query)
        latencies.sort()
        if misses:
            print(f"  missed: {misses}")
        return hits, latencies[len(latencies) // 2]
    finally:
        await mem.aclose()


def _row_count(dsn: str, namespace: str) -> int:
    try:
        import psycopg

        with psycopg.connect(dsn) as c, c.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memory_nodes WHERE namespace = %s AND invalidated_at IS NULL",
                (namespace,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else -1
    except Exception:
        return -1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--shape", choices=("realistic", "formulaic"), default="realistic")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--namespace", default="scale")
    p.add_argument("--skip-load", action="store_true", help="corpus already loaded")
    args = p.parse_args(argv)

    dsn = os.environ.get("CONTINUUM_DB_DSN")
    if not dsn:
        print("[recall-at-scale] set CONTINUUM_DB_DSN to a THROWAWAY migrated db.", file=sys.stderr)
        return 1

    if not args.skip_load:
        took = asyncio.run(load(dsn, args.rows, args.shape, args.namespace))
        print(f"  loaded {args.rows + len(NEEDLES)} {args.shape} memories in {took:.0f}s")

    # Report the store's ACTUAL size, never the requested one: with --skip-load
    # they differ, and a label that guesses is how a measurement quietly starts
    # describing something other than what it measured.
    rows = _row_count(dsn, args.namespace)
    hits, p50 = asyncio.run(score(dsn, args.namespace, args.k))
    total = len(NEEDLES)
    shape = args.shape if not args.skip_load else "existing store"
    print(
        f"\n  recall@{args.k} = {hits}/{total} = {hits / total:.0%}"
        f"   p50 {p50:.0f}ms   ({rows} memories, {shape})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
