# Scale at 50k — the ANN index is fine; the lexical channel costs 2.1 seconds

*Status: measured · 2026-08-10 · **corrects this document's own first version**, which
reported the opposite conclusion from a corpus that turned out to be the artifact*

**Headline: on realistic data the HNSW index costs 0pp of recall and is 25× faster than
exact search.** Both return 85% recall@10 at 50k rows; HNSW does it in 3.0 ms against 77.3 ms.

**The real cost is elsewhere: `search_hybrid` takes 2,317 ms per query, and 2,112 ms of that
is the `pg_trgm` lexical channel** — a channel a previous experiment (commit `03715dd`) already
found delivered no measurable recall benefit.

---

## 0. What this document got wrong, and how

The first version of this file — committed in `7557455` — concluded:

> *"the ANN index costs 50 percentage points of end-to-end recall … Drop the ANN index. +50pp
> for ~26ms. This is the whole answer at current scale."*

**That was an artifact of the synthetic corpus, and the recommendation was wrong.** The
document did carry the caveat (§4.2, "this corpus is unusually hostile to HNSW … the production
gap is likely smaller than 50pp"), but it hedged toward *smaller* when the honest answer is
*zero*, and it put the disputed number in the title.

Re-running with distractors drawn from the real LongMemEval haystack, changing nothing else:

| dense-only, 50k rows | synthetic corpus | **real corpus** |
|---|---:|---:|
| exact scan | 90% | **85%** |
| HNSW `ef_search=40` | 15% | **85%** |
| **gap attributable to ANN** | **−75pp** | **0pp** |

The lesson generalises past this benchmark: **a synthetic corpus can invert a conclusion, not
merely soften it.** 50,000 templated near-duplicates form one dense embedding cluster with the
needles as isolated outliers — close to the worst case for a proximity graph. Nothing about
that resembles production data, and it produced a confident, wrong recommendation.

---

## 1. The earlier ad-hoc audit (still invalid, for different reasons)

An ad-hoc audit reported ">1,000 QPS ingestion", "hybrid search surfaced the exact CVE", and
"PostgreSQL 16 + pgvector scales brilliantly". Verified against the database it actually ran on
(`:5432/continuum_db`):

| claim | verified reality |
|---|---|
| PostgreSQL 16 | **PostgreSQL 14.18.** The PG16 instance is on `:5433` and had 0 rows. |
| "hybrid search" found the needle | **0 of 10,591 rows had an embedding.** `search_hybrid`'s dense CTE filters on `embedding IS NOT NULL` (`ltm.py:703`), so the dense channel matched nothing. Every result came from `pg_trgm` alone. |
| 1,028 writes/sec | Writes **with no embedding computed**. Embedding is 94% of real ingestion on this corpus (§2). |
| needle recall ✓ | n=1, and the needle was the literal string `CVE-2026-9999` — the best case for trigram and the weakest possible test of a vector store. |

---

## 2. Ingestion

Clean PostgreSQL **16.10**, migrations 001–007, HNSW `m=32, ef_construction=200`, BGE-M3
(1024-dim, local, `max_seq_length=512`), 16-way write concurrency.

| corpus | n stored | embed | write | end-to-end | write p50 / p99 |
|---|---:|---|---|---|---|
| synthetic | 50,000 | 179.4 s (279/s) | 65.4 s (2,125/s… see note) | 204 rec/s | 5.98 / 9.78 ms |
| **real** | **49,989** | 1,729.8 s (**29/s**) | 119.0 s (**420/s**) | **27 rec/s** | 18.88 / 128.74 ms |

* **Embedding is 94% of ingestion on real text**, against 73% on synthetic. Real turns are
  roughly 10× longer, so any throughput figure that excludes embedding is off by more than an
  order of magnitude — which is exactly how the ad-hoc audit reached 1,028 QPS.
* Write throughput falls with row width as well as row count (2,125 → 420/s).
* 11 of 50,000 real turns collapsed on migration 007's unique-live-text constraint; the harness
  warns and all figures use the **stored** count.
* Embedding coverage is asserted, not assumed — the run aborts if any row lands without a
  vector.

---

## 3. Retrieval

20 needles retrieved by **paraphrase** (no shared rare tokens), 3 repeats, identical across
repeats.

### 3.1 The ANN index is not the problem

Dense-only, real corpus, 49,989 rows:

| method | recall@10 | p50 |
|---|---:|---:|
| exact scan (no index) | 85% | 77.3 ms |
| **HNSW `ef_search=40`** (server default) | **85%** | **3.0 ms** |
| HNSW `ef_search=100` | 85% | 7.5 ms |
| HNSW `ef_search=1000` | — | planner refuses the index |

**Zero recall cost, 25× faster.** Keep the index.

At `ef_search=1000` the planner declines the HNSW scan and falls back to a sort, because real
rows are ~733 bytes wide against 131 synthetic. Since recall is already saturated at `ef=40`,
the store's `DEFAULT_HNSW_EF_SEARCH = 1000` buys nothing here and pushes the planner toward the
expensive plan — **worth reconsidering as a default.**

### 3.2 The lexical channel is the real cost

| path | p50 |
|---|---:|
| dense only, HNSW `ef=40` | **3.0 ms** |
| dense only, exact | 77.3 ms |
| `pg_trgm` sparse channel alone | **2,112 ms** |
| `search_hybrid` (both, RRF) | **2,317 ms** |

**The sparse channel is 91% of hybrid query latency** — trigram similarity over 50k long
documents, where the synthetic corpus's short strings made it look cheap.

Set beside commit `03715dd`, which found hybrid-vs-cosine **statistically indistinguishable**
on the needle set, the position is now: *the lexical channel has no measured recall benefit and
costs 2.1 seconds per query.* That is the strongest case yet for gating it — by query type, by
corpus size, or off by default with an opt-in for identifier-style queries.

### 3.3 The 85% ceiling is the embedder

Exact search tops out at 85%, so 3 of 20 needles are unreachable by dense retrieval at any `k`
or `ef`. On the synthetic corpus the equivalent figure was 90%, and the measured cause there
was margin: 2 of 20 queries scored their true needle *below* the best distractor under exact
cosine. Raising this requires better embeddings or chunk enrichment, not retrieval work.

---

## 4. Caveats

* **20 needles: one needle is 5pp.** The 85%-vs-85% tie is meaningful (identical hit sets); the
  85-vs-90 difference between corpora is one needle and should not be read as real.
* **`max_seq_length` was capped at 512** for both corpora after real turns exhausted MPS memory
  at BGE-M3's 8192 default. Standard for retrieval, but the synthetic figures in §3.1 predate
  the cap — they never approached it (≈30-token rows), so the comparison holds.
* **HNSW recall varies by index *build*.** On the synthetic corpus, rebuilding the same
  configuration moved `ef=40` recall between 5% and 15% — more than the entire `ef_search`
  sweep. Same observation as commit `f1e3703` ("HNSW recall is a distribution, not a number").
  On the real corpus recall is saturated, so build variance is invisible; it would reappear
  under a harder workload. **Never A/B `m` or `ef_construction` from single builds.**
* One corpus, one embedder, one machine.

---

## 5. Four mistakes, each of which produced believable numbers

Recorded because the failure mode generalises.

1. **`SET LOCAL` outside a transaction is a silent no-op.** Settings never applied.
2. **Session settings leak across pooled connections.** `enable_indexscan=off` from the exact
   probe survived release back to the pool and was inherited by an "ANN" probe — which then ran
   an exact scan and scored **90% under an ANN label.**
3. **`"Index Scan" in plan` also matches a btree scan.** With `enable_seqscan=off` the planner
   used an unrelated index plus a Sort — exact results at seq-scan latency, again labelled ANN.
4. **The corpus itself.** The three above were caught within minutes by physical impossibility
   — a wider beam scoring worse, a wider beam running 17× faster, recall flat from k=10 to
   k=500. **The fourth produced entirely plausible numbers and was only caught by changing the
   data.** It is the one that reached a committed conclusion.

The harness now reads settings back and asserts, issues `RESET ALL` around every probe, and
requires the plan to name `memory_nodes_embedding_hnsw_idx`. Nothing in it could have caught
mistake 4 — only running against real data could.

---

## 6. What to do

1. **Keep the HNSW index.** 0pp recall cost, 25× faster. The previous "drop the index"
   recommendation is withdrawn.
2. **Gate or drop the `pg_trgm` channel.** 2,112 ms per query for no measured recall benefit
   (`03715dd`). Biggest single latency win available.
3. **Reconsider `DEFAULT_HNSW_EF_SEARCH = 1000`.** Recall saturates at `ef=40` here, and 1000
   pushes the planner off the index entirely on wide rows.
4. **For >85%, work on embeddings, not retrieval.** 3 of 20 needles are unreachable by exact
   dense search.
5. **Re-measure before generalising.** This document's first version had the opposite headline
   from the same harness on different data.

---

## 7. Reproducing

```bash
createdb -p 5433 continuum_bench && for m in migrations/00*.sql; do psql -p 5433 -d continuum_bench -f "$m"; done
python3.12 -m bench.scale_ingest_retrieval --n 50000 --concurrency 16 --corpus real
python3.12 -m bench.ann_vs_exact --ef-search 40 100
```

`--corpus synthetic` reproduces the adversarial control. Raw outputs in
`bench/results/scale_*.json` and `ann_vs_exact_*.json`, each recording server version,
migrations, index parameters, embedding coverage, and corpus source.
