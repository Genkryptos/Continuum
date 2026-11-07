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

30 needles — 20 retrieved by **paraphrase** (no shared rare tokens) and 10 by **identifier**
(a rare literal shared with the fact) — 3 repeats, identical across repeats. §3.1 and §3.2
predate the identifier needles and use the original 20; §3.2b uses all 30.

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

### 3.1b `DEFAULT_HNSW_EF_SEARCH = 1000` pushes the planner off the index

The store sets `hnsw.ef_search = 1000` per query (`ltm.py:759`), chosen in commit `0e31ae4`
when recall was degrading at scale. On this corpus that default is not merely unnecessary — it
is **counterproductive**. Measured across the 30-needle set, real corpus, 49,989 rows:

| `ef_search` | planner picks | recall@10 | p50 |
|---:|---|---:|---:|
| 40 | HNSW index | 87% | 14.9 ms |
| **100** | **HNSW index** | **90%** | **5.5 ms** |
| 200 | HNSW index | — | — |
| 400 | **Seq Scan + Sort** | — | — |
| **1000** *(the store default)* | **Seq Scan + Sort** | — | ~76 ms |

pgvector's cost estimate for an HNSW scan grows with `ef_search`; past ~200 on these
733-byte-wide rows it exceeds a full sort, so the planner abandons the index. **At the shipped
default the dense channel is doing a sequential scan of the whole table** — which is why
recall matched exact search exactly, and why it cost ~76 ms instead of 5.5 ms.

`ef_search = 100` matches exact recall (90%, i.e. 27/30 — the same three the embedder cannot
reach) at **14× lower latency than the default actually delivers**. `ef=40` loses exactly one
needle.

**Recommendation: lower `DEFAULT_HNSW_EF_SEARCH` to ~100.** Caveat: the crossover depends on
row width, so a corpus of short rows may keep the index at higher `ef`. The check is one
`EXPLAIN` — if the plan does not name `memory_nodes_embedding_hnsw_idx`, the setting is
buying a sequential scan.

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

### 3.2b The lexical channel contributes nothing — tested in both regimes

§3.2 established the sparse channel's cost. The obvious objection was that the
20 original needles are all *paraphrase* style — semantic restatements sharing
no rare vocabulary — which is precisely the regime where lexical matching
should not help. Concluding "hybrid adds nothing" from them was not valid.

So the needle set was extended with **10 identifier-style needles**, where the
query shares a rare literal with the fact (`ERR_4021`, `SKU-77device-Q3B`,
`v2.14.3-hotfix.2`, `CVE-2026-31337`, `clause 14.7(b)(iii)`, a hyphenated
surname). That is the one regime with a mechanism argument for trigram matching.

Same fixture, same queries, same 3 repeats — only `lexical_channel` gated:

| | hybrid (lexical on) | dense-only | delta |
|---|---:|---:|---:|
| **identifier** recall@10 (n=10) | 100% | **100%** | 0pp |
| **paraphrase** recall@10 (n=20) | 85% | **85%** | 0pp |
| aggregate | 90% | 90% | 0pp |
| p50 latency | 2,319 ms | **43 ms** | **54× faster** |

**The miss sets are byte-identical.** Zero documents were retrieved by the
lexical channel that the dense channel had not already found — in either
regime. This is not a coincidental tie at the aggregate; it is the same three
needles missed by both, and they are missed by exact dense search too (§3.3).

The mechanism argument therefore fails empirically for this embedder: BGE-M3's
subword tokenisation preserves rare literals well enough that identifier
queries are already at 100% without any lexical help. **The channel costs
2,276 ms per query and buys nothing measurable.**

Recommendation: **flip `lexical_channel` to default off**, with the caveat in
§4 about embedder dependence. The flag exists now
(`PostgresLTM(lexical_channel=...)`) and still defaults to on, so nothing has
changed for existing callers.

### 3.3 The 85% ceiling is the embedder

Exact search tops out at 85% on the paraphrase needles, so 3 of 20 are unreachable by dense
retrieval at any `k` or `ef` — and unreachable by the lexical channel too (§3.2b). All 10
identifier needles are found. On the synthetic corpus the equivalent figure was 90%, and the measured cause there
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
* **The identifier result depends on the embedder, and that is the main limit
  on generalising it.** BGE-M3 is a strong multilingual model whose subword
  tokenisation handles rare literals well. A smaller or more aggressively
  pooled embedder may not, and the lexical channel could earn its place there.
  Anyone adopting §3.2b's recommendation should re-run it on their own
  embedder before flipping the default.
* **n=10 identifier needles: one needle is 10pp.** 100% vs 100% is a tie with
  no difference to detect, but a regression smaller than 10pp would be
  invisible. The identical miss sets are the stronger evidence — they show the
  channel returned nothing new, not merely that the totals matched.
* Measured at `k=10`. A larger `k` gives the lexical channel more room to
  contribute, and was not tested.
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
2. **Gate the `pg_trgm` channel off.** Now tested in both regimes (§3.2b):
   identical recall, identical miss sets, 54× faster. `PostgresLTM` takes a
   `lexical_channel` flag; it still defaults to on pending a decision on the
   default and a check against other embedders.
3. **Lower `DEFAULT_HNSW_EF_SEARCH` to ~100** (§3.1b). At 1000 the planner abandons the index
   and sequentially scans; 100 matches exact recall at 5.5 ms against the ~76 ms the default
   actually delivers.
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
