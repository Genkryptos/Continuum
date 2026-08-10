"""
bench/scale_ingest_retrieval.py
===============================
Scale benchmark for the **real** PostgresLTM write + hybrid-read path.

Why this exists
---------------
An earlier ad-hoc audit reported ">1,000 QPS ingestion" and "hybrid search
surfaced the needle" against a database in which **every row had a NULL
embedding**. With no vectors stored, ``search_hybrid``'s dense CTE
(``WHERE ... AND embedding IS NOT NULL``) matches nothing, so that run
measured trigram-only retrieval and embedding-free writes — neither of
which is the production path. It also reported "PostgreSQL 16" while
running against a 14.18 instance.

This harness is built so those specific mistakes cannot recur silently:

* **Embedding coverage is asserted, not assumed.** The run aborts if any
  stored row lacks a vector.
* **Server version and index parameters are captured** into the result
  JSON, so a claim about "PG16, m=32" is checked rather than typed.
* **Embed and write are timed separately.** Embedding dominates real
  ingestion; folding it out inflates throughput by an order of magnitude.
* **Needles are retrieved by paraphrase**, never by a literal rare token.
  Searching for ``CVE-2026-9999`` when that exact string is in the row
  tests ``pg_trgm``, not the memory layer.
* **Retrieval is repeated** so latency and recall carry a spread rather
  than a single flattering draw.

Usage
-----
    python3.12 -m bench.scale_ingest_retrieval --n 5000
    python3.12 -m bench.scale_ingest_retrieval --n 50000 --concurrency 16

Writes ``bench/results/scale_<n>_<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from continuum.core.types import MemoryItem, MemoryTier, Query
from continuum.stores.postgres.ltm import PostgresLTM

DEFAULT_DSN = "postgresql://localhost:5433/continuum_bench"
RESULTS = Path(__file__).parent / "results"
EMBED_DIM = 1024

# ── synthetic corpus ──────────────────────────────────────────────────────
# Distractors are deliberately *topically close* to the needles. A corpus of
# unrelated noise makes any retriever look good; the interesting regime is
# when the corpus contains many plausible near-misses.

_SUBJECTS = ["the billing service", "the auth gateway", "the search cluster",
             "the ingest worker", "the report scheduler", "the webhook relay",
             "the media transcoder", "the notification fanout"]
_REGIONS = ["us-east-1", "us-west-2", "eu-central-1", "ap-south-1",
            "sa-east-1", "eu-west-2", "ap-northeast-1"]
_ACTIONS = ["was migrated to", "was rolled back from", "now reads replicas in",
            "failed over to", "was pinned to", "stopped serving traffic from"]
_TOPICS = ["retry budget", "connection pool size", "cache TTL", "batch window",
           "circuit breaker threshold", "backoff ceiling", "shard count"]


def _distractor(rng: random.Random, i: int) -> str:
    """
    One distractor. The trailing reference id keeps every row textually
    distinct.

    Without it the template space is only ~17k combinations, so migration
    007's unique-live-text constraint silently collapses duplicates — a
    50,000-record run would store far fewer than 50,000 rows while the
    harness reported the requested figure. A natural-looking ticket
    reference is the cheapest way to guarantee the corpus is the size the
    result claims.
    """
    ref = f"(ref INC-{i:07d})"
    kind = rng.random()
    if kind < 0.45:
        return (f"{rng.choice(_SUBJECTS).capitalize()} {rng.choice(_ACTIONS)} "
                f"{rng.choice(_REGIONS)} during the week {i % 52} maintenance "
                f"window. {ref}")
    if kind < 0.8:
        return (f"The {rng.choice(_TOPICS)} for {rng.choice(_SUBJECTS)} was set to "
                f"{rng.randint(2, 900)} after the incident review in quarter "
                f"{i % 4 + 1}. {ref}")
    return (f"On-call for {rng.choice(_SUBJECTS)} rotated to team "
            f"{chr(65 + i % 26)}{i % 10} covering {rng.choice(_REGIONS)}. {ref}")


@dataclass(frozen=True)
class Needle:
    """A planted fact plus the query that must retrieve it."""
    text: str
    query: str
    #: "paraphrase" — semantic restatement, no shared rare tokens. The dense
    #: channel has to earn these.
    #: "identifier" — the query contains a rare literal (error code, SKU,
    #: version, surname) that also appears in the fact. This is the ONLY
    #: regime where the lexical channel has a mechanism argument, and the
    #: original 20 needles contained none of them — which is why "hybrid adds
    #: nothing" could not be concluded from them.
    kind: str = "paraphrase"


#: 20 needles. Each query is a *paraphrase* — it shares little or no rare
#: vocabulary with its fact, so trigram alone should struggle and the dense
#: channel has to earn the hit.
NEEDLES: list[Needle] = [
    Needle("The primary datastore for customer invoices lives in Frankfurt and is never replicated outside the EU.",
           "Where is invoice data kept, and does it leave Europe?"),
    Needle("Rolling restarts of the auth gateway must happen before 04:00 UTC or sessions are dropped mid-flight.",
           "What is the safe window to bounce authentication nodes?"),
    Needle("The transcoder rejects any upload larger than 512 megabytes with a silent failure rather than an error.",
           "Why would a big video upload vanish without a message?"),
    Needle("Two engineers hold the production database break-glass credentials: the platform lead and the on-call SRE.",
           "Who can access emergency database credentials?"),
    Needle("Search relevance degrades sharply once a tenant exceeds roughly four hundred thousand indexed documents.",
           "At what corpus size does tenant search quality fall off?"),
    Needle("Webhook deliveries are retried nine times over a span of six hours before being moved to the dead letter queue.",
           "How many delivery attempts before a callback is abandoned?"),
    Needle("The nightly reconciliation job cannot run concurrently with the ledger export or both produce duplicates.",
           "Which two batch jobs must never overlap?"),
    Needle("Customer-facing latency budgets are measured at the ninety-ninth percentile, not the median.",
           "Which percentile governs our published response time targets?"),
    Needle("The mobile client caches feature flags for twelve hours, so flag changes take up to half a day to appear.",
           "Why do toggles take so long to reach phones?"),
    Needle("Refunds above two thousand dollars require a second approver drawn from the finance rotation.",
           "When does a money-back request need a second sign-off?"),
    Needle("The staging environment shares a Redis instance with CI, which is why cache tests are flaky on busy afternoons.",
           "What explains intermittent caching test failures before a release?"),
    Needle("Log retention was cut from ninety days to thirty after the storage bill tripled in the spring.",
           "How long do we keep logs now, and why did it change?"),
    Needle("Any schema migration touching the orders table must be shipped behind a dual-write for one full release.",
           "What is the rule for changing the orders schema safely?"),
    Needle("The recommendation model is retrained weekly on Sunday and takes about eleven hours end to end.",
           "How often is the recommender refreshed and how long does it take?"),
    Needle("Support tickets from enterprise accounts bypass the triage queue and page the duty manager directly.",
           "How are premium customer issues escalated?"),
    Needle("Our payment provider throttles at three hundred requests per second per merchant identifier.",
           "What rate limit does the payments vendor enforce?"),
    Needle("The data warehouse copy lags production by roughly forty minutes during business hours.",
           "How stale is analytics data mid-day?"),
    Needle("Deleting a workspace is soft for thirty days, after which the purge job removes it irrecoverably.",
           "What happens after a team removes their workspace?"),
    Needle("The image CDN strips EXIF metadata on upload, which is why photo timestamps disappear.",
           "Why do picture dates go missing after upload?"),
    Needle("Only the platform team can approve new outbound network egress rules in production.",
           "Who signs off on letting a service call the internet?"),
]

#: Identifier-style needles. Rare literal tokens shared between fact and
#: query — exactly what dense embeddings smooth away and trigram matches
#: exactly. If the lexical channel earns its 2.1s anywhere, it is here.
NEEDLES += [
    Needle("Deployment ERR_4021 fires when the sidecar cannot reach the metrics collector within the startup probe window.",
           "What causes ERR_4021?", "identifier"),
    Needle("The affected batch is SKU-77device-Q3B, recalled after the humidity sensor tolerance was found out of spec.",
           "Why was SKU-77device-Q3B recalled?", "identifier"),
    Needle("Kafka consumer group orders-reconciler-v7 was pinned to broker set B after the rebalance storm.",
           "Which broker set is orders-reconciler-v7 pinned to?", "identifier"),
    Needle("Dr. Yevgenia Kowalczyk-Brandt signed off on the revised anticoagulation protocol in the cardiology unit.",
           "Who signed off on the anticoagulation protocol? Kowalczyk-Brandt?", "identifier"),
    Needle("Rolling back to image tag v2.14.3-hotfix.2 resolved the memory leak in the notification fanout.",
           "What does image tag v2.14.3-hotfix.2 fix?", "identifier"),
    Needle("Incident INC-0042317 was the root cause of the duplicate invoice run in the billing service.",
           "What happened in INC-0042317?", "identifier"),
    Needle("The compliance exception is tracked under policy clause 14.7(b)(iii) of the data residency agreement.",
           "What does clause 14.7(b)(iii) cover?", "identifier"),
    Needle("Feature flag enable_async_ledger_writes remains off in production pending the dual-write bake.",
           "Is enable_async_ledger_writes on in production?", "identifier"),
    Needle("CVE-2026-31337 affects the PDF rendering path and is mitigated by disabling embedded JavaScript.",
           "How do we mitigate CVE-2026-31337?", "identifier"),
    Needle("Customer account ACT-9931-KX was migrated to the dedicated tenant pool after the noisy-neighbour report.",
           "Why was ACT-9931-KX moved to a dedicated pool?", "identifier"),
]


@dataclass
class Phase:
    name: str
    seconds: float = 0.0
    n: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def qps(self) -> float:
        return self.n / self.seconds if self.seconds > 0 else 0.0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


# ── environment capture ───────────────────────────────────────────────────

async def capture_env(ltm: PostgresLTM) -> dict[str, Any]:
    """Record what we are ACTUALLY running against, not what we assume."""
    async with ltm._connect() as conn:
        async def one(sql: str) -> Any:
            # The store's pool uses a dict row factory, so index by position
            # through .values() rather than assuming tuples.
            cur = await conn.execute(sql)
            row = await cur.fetchone()
            if not row:
                return None
            return next(iter(row.values())) if isinstance(row, dict) else row[0]

        return {
            "server_version": await one("select version()"),
            "pgvector_version": await one(
                "select extversion from pg_extension where extname='vector'"),
            "migrations": await one(
                "select string_agg(version, ',' order by version) from schema_migrations"),
            "hnsw_index": await one(
                "select indexdef from pg_indexes where indexname like '%hnsw%'"),
            # The server default is meaningless here: the store issues
            # `SET hnsw.ef_search` per query (ltm.py:759), so record what the
            # store is CONFIGURED to use, not what a fresh session reports.
            "hnsw_ef_search_server_default": await one("show hnsw.ef_search"),
            "hnsw_ef_search_used_by_store": ltm._hnsw_ef_search,
        }


async def embedding_coverage(ltm: PostgresLTM) -> tuple[int, int]:
    async with ltm._connect() as conn:
        cur = await conn.execute(
            "select count(*) filter (where embedding is not null) as with_vec, "
            "count(*) as total from memory_nodes")
        row = await cur.fetchone()
        if isinstance(row, dict):
            return int(row["with_vec"]), int(row["total"])
        return int(row[0]), int(row[1])


# ── phases ────────────────────────────────────────────────────────────────

def _real_distractors(n: int, seed: int) -> list[str]:
    """
    Distractors drawn from the LongMemEval haystack — real conversational
    turns rather than templates.

    The synthetic generator produces one enormous tight cluster in embedding
    space (distractor-to-distractor cosine 0.600 mean, 0.998 max), which is
    close to the worst case for a proximity graph and almost certainly
    overstates the ANN recall gap. Real turns are the control for that.
    """
    import json as _json

    path = (Path(__file__).resolve().parents[1] / "evals" / "longmemeval"
            / "LongMemEval" / "data" / "longmemeval_s_cleaned.json")
    if not path.exists():
        raise SystemExit(f"dataset not found for --corpus real: {path}")
    rows = _json.loads(path.read_text())
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        for session in row.get("haystack_sessions") or []:
            for turn in session:
                text = " ".join(str(turn.get("content", "")).split())
                # Skip fragments and anything already stored: migration 007's
                # unique-live-text constraint would collapse duplicates and the
                # stored corpus would be smaller than the reported size.
                if len(text) < 60 or text in seen:
                    continue
                seen.add(text)
                out.append(text[:2000])
                if len(out) >= n:
                    return out
    raise SystemExit(f"only {len(out)} distinct turns available, need {n}")


def build_corpus(
    n: int, seed: int, source: str = "synthetic",
) -> tuple[list[str], list[int]]:
    """Distractors with the needles planted at deterministic positions."""
    rng = random.Random(seed)
    if source == "real":
        texts = _real_distractors(n, seed)
    else:
        texts = [_distractor(rng, i) for i in range(n)]
    step = max(1, n // (len(NEEDLES) + 1))
    positions = []
    for i, needle in enumerate(NEEDLES):
        pos = min(n - 1, (i + 1) * step)
        texts[pos] = needle.text
        positions.append(pos)
    return texts, positions


#: BGE-M3 accepts 8192 tokens by default. Real conversational turns are long
#: enough that a 64-wide batch at that limit exhausts MPS memory outright, and
#: retrieval gains little from the tail — 512 is the usual retrieval setting.
#: Held constant across corpora so the synthetic/real comparison stays fair.
MAX_SEQ_LENGTH = 512


def _load_encoder() -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed_all(texts: list[str], batch_size: int) -> tuple[list[list[float]], float]:
    """Local BGE-M3 — the production embedder. Timed on its own."""
    model = _load_encoder()
    t0 = time.perf_counter()
    vecs = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    assert vecs.shape[1] == EMBED_DIM, f"expected {EMBED_DIM}-dim, got {vecs.shape[1]}"
    return [v.tolist() for v in vecs], elapsed


async def write_all(
    ltm: PostgresLTM, texts: list[str], vecs: list[list[float]],
    concurrency: int,
) -> tuple[Phase, list[float]]:
    """Concurrent upserts through the real store. Per-item latency recorded."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    lock = asyncio.Lock()
    now = datetime.now(UTC)

    async def one(text: str, vec: list[float], idx: int) -> None:
        item = MemoryItem(
            id=str(uuid.uuid4()), content=text, tier=MemoryTier.LTM,
            embedding=vec, created_at=now,
            metadata={"kind": "fact", "source": "bench", "seq": idx},
        )
        async with sem:
            t0 = time.perf_counter()
            await ltm.upsert(item)
            dt = (time.perf_counter() - t0) * 1000
        async with lock:
            latencies.append(dt)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(t, v, i) for i, (t, v) in enumerate(zip(texts, vecs, strict=True))))
    elapsed = time.perf_counter() - t0
    return Phase("write", elapsed, len(texts)), latencies


async def retrieve(
    ltm: PostgresLTM, k: int, repeats: int, batch_size: int,
) -> dict[str, Any]:
    """Paraphrase queries against the planted needles. Recall@k + latency."""
    model = _load_encoder()
    queries = [n.query for n in NEEDLES]
    qvecs = model.encode(queries, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True)

    per_run_recall: list[float] = []
    latencies: list[float] = []
    misses: list[str] = []
    # Paraphrase and identifier needles exercise opposite channels. Averaging
    # them hides exactly the effect this benchmark exists to measure, so keep
    # per-kind tallies alongside the aggregate.
    by_kind_hits: dict[str, int] = {}
    by_kind_n: dict[str, int] = {}
    by_kind_lat: dict[str, list[float]] = {}

    for run in range(repeats):
        hits = 0
        for needle, qvec in zip(NEEDLES, qvecs, strict=True):
            q = Query(text=needle.query)
            q.embedding = [float(x) for x in qvec]
            t0 = time.perf_counter()
            results = await ltm.search_hybrid(q, k=k)
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            by_kind_lat.setdefault(needle.kind, []).append(dt)
            found = any(
                (getattr(r.item, "content", "") or "").strip() == needle.text
                for r in results
            )
            hits += found
            by_kind_hits[needle.kind] = by_kind_hits.get(needle.kind, 0) + int(found)
            by_kind_n[needle.kind] = by_kind_n.get(needle.kind, 0) + 1
            if not found and run == 0:
                misses.append(f"[{needle.kind}] {needle.query}")
        per_run_recall.append(hits / len(NEEDLES))

    by_kind = {
        kind: {
            "recall": by_kind_hits[kind] / by_kind_n[kind],
            "n_needles": by_kind_n[kind] // max(1, repeats),
            "latency_p50_ms": _pct(by_kind_lat[kind], 0.50),
        }
        for kind in sorted(by_kind_n)
    }
    return {
        "k": k, "repeats": repeats, "n_needles": len(NEEDLES),
        "by_kind": by_kind,
        "recall_per_run": per_run_recall,
        "recall_mean": statistics.fmean(per_run_recall),
        "recall_min": min(per_run_recall), "recall_max": max(per_run_recall),
        "latency_p50_ms": _pct(latencies, 0.50),
        "latency_p95_ms": _pct(latencies, 0.95),
        "latency_p99_ms": _pct(latencies, 0.99),
        "misses_first_run": misses,
    }


# ── main ──────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> int:
    ltm = PostgresLTM(dsn=args.dsn, pool_max_size=max(4, args.concurrency),
                      lexical_channel=not args.no_lexical)
    try:
        env = await capture_env(ltm)
        print(f"server   : {str(env['server_version'])[:38]}")
        print(f"pgvector : {env['pgvector_version']}  migrations {env['migrations']}")
        print(f"hnsw     : {str(env['hnsw_index']).split('WITH')[-1].strip()}  "
              f"ef_search={env['hnsw_ef_search_used_by_store']} (store-set per query)")

        covered, total = await embedding_coverage(ltm)
        if total and covered != total:
            print(f"ABORT: pre-existing rows lack embeddings ({covered}/{total}). "
                  "Use a clean database.")
            return 2
        if total:
            print(f"note     : database already holds {total} embedded rows")

        print(f"\ncorpus   : building {args.n} {args.corpus} records "
              f"({len(NEEDLES)} needles planted)")
        texts, _ = build_corpus(args.n, args.seed, args.corpus)

        print("embed    : BGE-M3 (local, the production embedder)...")
        vecs, embed_s = embed_all(texts, args.batch_size)
        print(f"           {embed_s:.2f}s  →  {args.n / embed_s:,.0f} texts/s")

        print(f"write    : {args.n} upserts, concurrency {args.concurrency}...")
        wphase, wlat = await write_all(ltm, texts, vecs, args.concurrency)
        print(f"           {wphase.seconds:.2f}s  →  {wphase.qps:,.0f} writes/s  "
              f"(p50 {_pct(wlat,0.5):.2f}ms  p99 {_pct(wlat,0.99):.2f}ms)")

        covered, total = await embedding_coverage(ltm)
        if covered != total:
            print(f"ABORT: {total - covered} of {total} rows stored WITHOUT an embedding.")
            return 2
        print(f"verify   : {covered}/{total} rows carry a vector ✓")
        if total < args.n:
            print(f"WARNING  : {args.n - total} of {args.n} records collapsed on the "
                  "unique-live-text constraint — the stored corpus is smaller "
                  "than the requested size. Throughput is still valid; corpus "
                  "size claims must use the stored count.")

        end_to_end = embed_s + wphase.seconds
        print(f"\nend-to-end ingest: {end_to_end:.2f}s  →  "
              f"{args.n / end_to_end:,.0f} records/s "
              f"(embedding is {100 * embed_s / end_to_end:.0f}% of it)")

        print(f"\nretrieve : {len(NEEDLES)} paraphrase queries × {args.repeats} runs, k={args.k}...")
        r = await retrieve(ltm, args.k, args.repeats, args.batch_size)
        print(f"           recall@{args.k} {r['recall_mean']:.1%} "
              f"(min {r['recall_min']:.0%} max {r['recall_max']:.0%})  "
              f"p50 {r['latency_p50_ms']:.1f}ms  p99 {r['latency_p99_ms']:.1f}ms")
        for kind, cell in r["by_kind"].items():
            print(f"           {kind:<11} recall {cell['recall']:.0%} "
                  f"(n={cell['n_needles']})  p50 {cell['latency_p50_ms']:.0f}ms")
        if r["misses_first_run"]:
            print(f"           missed: {len(r['misses_first_run'])} — "
                  f"e.g. {r['misses_first_run'][0][:60]!r}")

        payload = {
            "n_records": args.n, "concurrency": args.concurrency, "seed": args.seed,
            "corpus": args.corpus,
            "lexical_channel": not args.no_lexical,
            "environment": env,
            "embed": {"seconds": embed_s, "texts_per_s": args.n / embed_s,
                      "model": "BAAI/bge-m3", "dim": EMBED_DIM,
                      "max_seq_length": MAX_SEQ_LENGTH,
                      "batch_size": args.batch_size},
            "write": {"seconds": wphase.seconds, "writes_per_s": wphase.qps,
                      "latency_p50_ms": _pct(wlat, 0.50),
                      "latency_p95_ms": _pct(wlat, 0.95),
                      "latency_p99_ms": _pct(wlat, 0.99)},
            "end_to_end": {"seconds": end_to_end, "records_per_s": args.n / end_to_end,
                           "embed_share_pct": 100 * embed_s / end_to_end},
            "embedding_coverage": {"with_vector": covered, "total": total},
            "retrieval": r,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / (f"scale_{args.corpus}_{args.n}_"
                         f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nresults → {out}")
        return 0
    finally:
        await ltm.aclose()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--no-lexical", action="store_true",
        help="Disable the pg_trgm sparse channel (dense-only retrieval).")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--corpus", choices=("synthetic", "real"), default="synthetic",
        help=("'real' draws distractors from the LongMemEval haystack. The "
              "synthetic generator forms one tight embedding cluster that is "
              "adversarial for HNSW; 'real' is the control for that."))
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
