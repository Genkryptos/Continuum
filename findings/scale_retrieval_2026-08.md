# Scale at 5k and 50k — storage holds, the ANN index destroys recall

*Status: measured · 2026-08-09 · supersedes an ad-hoc "FAANG scale audit" that was invalid (§1)*

**Headline: at 50,000 rows the HNSW index costs 50 percentage points of end-to-end recall.**
`search_hybrid` returns **40%** recall@10 with the index and **90%** with it dropped, on
identical rows and identical queries. Ingestion scales fine; retrieval is where the system
breaks.

The secondary result is methodological and applies to anyone repeating this: **four separate
mistakes in my own harness and first draft produced confident, plausible, wrong numbers**
before the measurement was trustworthy (§5).

---

## 1. Why the earlier audit had to be discarded

An ad-hoc audit reported ">1,000 QPS ingestion", "hybrid search surfaced the exact CVE", and
"PostgreSQL 16 + pgvector scales brilliantly". Verified against the database it actually ran on
(`:5432/continuum_db`):

| claim | verified reality |
|---|---|
| PostgreSQL 16 | **PostgreSQL 14.18.** The PG16 instance is on `:5433` and had 0 rows. |
| "hybrid search" found the needle | **0 of 10,591 rows had an embedding.** `search_hybrid`'s dense CTE filters on `embedding IS NOT NULL` (`ltm.py:703`), so the dense channel matched nothing. Every result came from `pg_trgm` alone. |
| 1,028 writes/sec | Writes **with no embedding computed**. Embedding is 73–89% of real ingestion time (§2). |
| needle recall ✓ | n=1, and the needle was the literal string `CVE-2026-9999` — the best case for trigram and the weakest possible test of a vector store. |
| "switch to a code embedding model" | Would change nothing. No embedding model was being invoked. |

The `m=32` HNSW index on that database indexes zero non-null vectors, so it was irrelevant to
every number reported.

This is the same class of error as `findings/budget_curve_2026-08.md` §1 (a budget flag that
never bound) and the telemetry that reported `$0` cost: **an instrument not measuring what it
claims, producing confident output.**

---

## 2. Ingestion — this part is genuinely fine

Clean PostgreSQL **16.10**, migrations 001–007, HNSW `m=32, ef_construction=200`, BGE-M3
(1024-dim, local — the production embedder), 16-way write concurrency.

| n | embed | write | end-to-end | write p50 / p99 | vector coverage |
|---:|---|---|---|---|---|
| 5,000 | 18.7s (267/s) | 2.35s (**2,125/s**) | 21.1s → **238 rec/s** | 5.98 / 9.78 ms | 5000/5000 ✓ |
| 50,000 | 179.4s (279/s) | 65.4s (**765/s**) | 244.7s → **204 rec/s** | 14.12 / 60.26 ms | 50000/50000 ✓ |

* **Embedding is 73–89% of ingestion.** Any throughput figure that excludes it overstates real
  ingest by roughly an order of magnitude.
* **Write throughput fell 2,125 → 765/s at 10× scale** — HNSW graph maintenance growing with
  the table, not a Postgres limit.
* Embedding coverage is asserted by the harness; the run aborts if any row lands without a
  vector. That check is the direct fix for §1.

---

## 3. Retrieval — where it breaks

20 needles, retrieved by **paraphrase** (no shared rare tokens), 3 repeats, identical results
across repeats.

| corpus | `search_hybrid` recall@10 | p50 | p99 |
|---:|---:|---:|---:|
| 5,000 | **90%** | 53.8 ms | 93.3 ms |
| 50,000 | **40%** | 52.8 ms | 325.6 ms |

### 3.1 The index is the entire loss

Same 50k rows, same queries, only the index removed:

| `search_hybrid` at 50k | recall@10 | p50 |
|---|---:|---:|
| with HNSW index | **40%** | 52.8 ms |
| **HNSW index dropped** | **90%** | 79.0 ms |

**RRF fusion is not the culprit.** All 50pp is attributable to ANN approximation.

### 3.2 Dense-only, `ef_search` swept

| method | recall@10 | p50 |
|---|---:|---:|
| exact scan (no index) | **90%** | 66–69 ms |
| HNSW `ef_search=40` (server default) | **5–15%** | 2–36 ms |
| HNSW `ef_search=100` | 20% | 2.1 ms |
| HNSW `ef_search=400` | 25% | 2.7 ms |
| HNSW `ef_search=1000` (pgvector's cap, what the store sets) | **30–40%** | 4.5–11.6 ms |

Within a single index build, recall is monotone in `ef` — as it must be — and **saturates
50–60pp below exact at pgvector's cap.** No `ef_search` setting recovers it.

**The ranges are across index *builds*, not runs.** Three consecutive runs against one build
are identical to the point (5% / 30% / 90%); rebuilding the index moves `ef=40` between 5% and
15% and `ef=1000` between 30% and 40%. HNSW construction assigns node layers randomly, so
**recall is a property of the build, not of the configuration** — the same observation as
commit `f1e3703` ("HNSW recall is a distribution, not a number"), reproduced here at 50k.

The practical consequence is worse than the headline: **you cannot tune `m`/`ef_construction`
by comparing single builds.** Two builds of identical parameters differ by more than the
`ef_search` sweep does. Any such comparison needs several builds per configuration and a
reported spread.

Exact search, by contrast, returned **90% on every run of every build** — its only variance is
latency.

### 3.3 What did *not* fix it

* **REINDEX on the populated table.** The index is created empty by migration 001 and built
  incrementally during load, so a rebuild on 50k rows was the obvious hypothesis. It did not
  help — and a later DROP + CREATE on the populated table produced a *worse* graph (5% at
  `ef=40` against the original 15%). Not an incremental-build artifact.
* **Larger k.** `ef_search` **caps the number of rows an HNSW scan can return** — a `LIMIT 500`
  at `ef_search=40` silently returns 40. Over-fetch-then-rescore cannot rescue this, because
  the graph never reaches the needles at any `ef`.

### 3.4 A planner hazard worth knowing

pgvector's HNSW cost estimate is **non-monotonic in `ef_search`**. On this table the planner
refuses the index at `ef_search=200` (falls back to a sort over all rows) but uses it at 400
and 1000. Latency and recall can therefore flip on a cost estimate you do not control — an
"ANN" query silently becoming an exact scan, or the reverse.

---

## 4. The ceiling, and the honest caveat

### 4.1 90% is an embedding ceiling, not a retrieval one

Exact search tops out at 90%, not 100%. Measured in embedding space: the margin between a
query's true needle and its best distractor is **+0.122 mean, but negative for 2 of 20
queries**. Those two needles are genuinely outranked by a distractor under exact cosine. No
index, no `k`, and no reranker can recover them — that requires better embeddings or chunk
enrichment.

### 4.2 This corpus is unusually hostile to HNSW

**The 50pp gap should not be quoted as a production number.** Measured structure of the
synthetic corpus:

| statistic | value |
|---|---:|
| distractor ↔ distractor cosine (mean) | **0.600** |
| distractor ↔ distractor cosine (p95 / max) | 0.798 / **0.998** |
| needle ↔ distractor cosine (mean) | 0.428 |

50,000 templated near-duplicates form **one enormous dense cluster**, with the needles as
isolated outliers. That is close to the worst case for a proximity graph: greedy descent enters
the dominant cluster and sparsely-connected outliers are unreachable. Real corpora are more
diverse, so the production gap is likely **smaller than 50pp**. Re-run on a sample of real data
before treating this figure as ours.

Also: 20 needles means **1 needle = 5pp**. The 15% vs 20% steps are noise; the 40% vs 90% gap
is 10 needles and is not.

---

## 5. Four mistakes, each of which produced believable numbers

Recorded because the failure mode generalises, and because the numbers above are only
trustworthy given that these were caught.

1. **`SET LOCAL` outside a transaction is a silent no-op.** Settings never applied; the run
   reported ef-search values it was not using.
2. **Session settings leak across pooled connections.** `enable_indexscan=off` from the exact
   probe survived release back to the pool and was inherited by a later "ANN" probe — which
   then ran an exact scan and scored **90% under an ANN label.**
3. **`"Index Scan" in plan` also matches a btree scan.** With `enable_seqscan=off` the planner
   used an unrelated index plus a Sort — exact results at seq-scan latency, again labelled ANN.

**Every one was caught by the same tell: physical impossibility.** A wider beam scoring *worse*
than a narrow one; a wider beam running 17× *faster*; recall flat from k=10 to k=500. None of
those can happen, so the harness — not the system — was wrong.

A fourth was caught only after this document's first draft: the `ef` sweep in §3.2 was written
from a single index build and reported as though the figures were exact. They are build-
dependent (§3.2). The draft was corrected rather than left standing.

The harness now (a) reads settings back after writing them and asserts, (b) issues `RESET ALL`
on entry and exit of every probe, and (c) requires the plan to name
`memory_nodes_embedding_hnsw_idx` specifically.

---

## 6. What to do

Ranked by measured effect at 50k:

1. **Drop the ANN index. +50pp for ~26 ms** (40% → 90%, p50 52.8 → 79.0 ms). At 50k a
   sequential scan over halfvec is ~70 ms, negligible beside a generation call. This is the
   whole answer at current scale.
2. **For >90%, work on embeddings, not retrieval.** The ceiling is set by the 2/20 negative
   margins.
3. **If ANN becomes unavoidable (>10⁶ rows): raise `m`, not `ef_search`** — connectivity is the
   failing property and `m` is baked in at build time. But **measure several builds per
   setting** (§3.2): build-to-build spread here exceeded the entire `ef_search` sweep, so a
   single-build A/B would have been noise. Verify recall against exact after every rebuild;
   never assume a rebuild improved anything.
4. **For per-tenant workloads, the question is moot.** Scoped to one user's ~1,000 memories,
   exact search is ~1 ms and perfectly accurate. The ANN index is pure downside at that scale.
   The open question is whether any query path is un-scoped.

---

## 7. Reproducing

```bash
createdb -p 5433 continuum_bench && for m in migrations/00*.sql; do psql -p 5433 -d continuum_bench -f "$m"; done
python3.12 -m bench.scale_ingest_retrieval --n 50000 --concurrency 16
python3.12 -m bench.ann_vs_exact --ef-search 40 100 400 1000
```

Raw outputs in `bench/results/scale_*.json` and `bench/results/ann_vs_exact_*.json`, each
recording server version, migrations, index parameters, and embedding coverage — so a claim
about the environment is checked rather than typed.
