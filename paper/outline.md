# Continuum — Research Paper Outline & Abstract (working draft)

**Status:** draft skeleton · **Target:** arXiv preprint → optional agent-memory workshop
(NeurIPS/ICLR/EMNLP) · **Artifacts:** open-source repo, `continuum-mcp` on PyPI, released benchmarks.

> Every number below is one Continuum already produces — see `README.md` and `findings/`.
> Cells marked **[NEEDS]** are the measurement gaps to close before submission (they map to
> the retrieval-metrics / baselines / supersession-benchmark / frontier-answerer tasks).

---

## Title (candidates)

1. **Continuum: Bi-Temporal, Supersession-Aware Memory for AI Agents**
2. When Facts Go Stale: Bi-Temporal Memory for Correct Agent Knowledge-Update
3. Retrieval Isn't the Ceiling: An Honest Decomposition of Long-Term Agent Memory

Lead with (1); (3) is the honest-analysis angle if the venue prefers empirical findings.

## Thesis (the one claim the paper defends)

Agents fail at long-term memory in two distinct ways that the field conflates:
**(a) knowledge-update** — old facts stay retrievable after they're superseded, and
**(b) as-of / temporal** — "what was true at time *T*" can't be answered at all.
Vector stores and flat/recency memory get both wrong. Continuum's **bi-temporal,
supersession-aware** design gets them **deterministically correct** (100% vs 38%/75%
baselines). And a controlled decomposition shows the residual LongMemEval ceiling is
**multi-hop reasoning in the answerer, not retrieval** — so throwing a bigger retriever
at it does nothing, but a stronger answerer over clean retrieval lifts it.

---

## Abstract (draft)

> AI agents built on LLMs have no native persistence: across sessions they forget, and
> within a growing memory they surface *stale* facts as if current. We present **Continuum**,
> a tiered (short/mid/long-term) memory system whose long-term store is **bi-temporal** —
> every fact carries both valid-time and transaction-time — and **supersession-aware**:
> a newer fact retires the one it contradicts *in place* (soft-invalidation), so retrieval
> stays current while history stays intact. On scripted knowledge-update and "as-of"
> benchmarks, Continuum answers **100%** correctly versus **38%** and **75%** for a
> vector-store baseline. On LongMemEval-S it reaches **~74%** judged accuracy. We further
> run a controlled decomposition that separates retrieval quality from answering, and find
> the accuracy ceiling on this benchmark is set by **multi-hop reasoning in the answerer,
> not by retrieval recall** — a reasoning loop we built and measured was *net-negative* and
> removed. Continuum is open-source, hybrid (dense bge-m3 + BM25, fused with RRF over an
> HNSW-indexed pgvector store), and reproducible; we release the code, the memory server,
> and the supersession/bi-temporal benchmarks. **[NEEDS: add retrieval-only Recall@k/NDCG
> and the Mem0/pgvector baseline numbers before finalizing.]**

---

## Section outline

### 1. Introduction
- The two failure modes (forgetting vs staleness); why they're different; why benchmarks
  that report a single end-to-end score hide (b).
- Contributions: (i) a bi-temporal supersession-aware memory design; (ii) released
  benchmarks for knowledge-update and as-of queries; (iii) an honest retrieval-vs-reasoning
  decomposition on LongMemEval; (iv) a production, MCP-exposed open-source system.

### 2. Related work
Mem0 · MemGPT/Letta · A-Mem · vector-RAG · temporal knowledge graphs / bi-temporal DBs ·
LongMemEval & LOCOMO (benchmarks). Position: most agent-memory work optimizes recall on a
static store; Continuum's contribution is *correctness under contradiction and time*.

### 3. System
- Tiered architecture: STM (session) → MTM (consolidation) → LTM (durable).
- **Bi-temporal model:** valid_from/valid_to vs created_at/invalidated_at; soft-invalidation
  (`invalidated_at`), nothing deleted; `current()` and `timeline()` semantics.
- **Hybrid retrieval:** dense (bge-m3, 1024-d) + BM25, fused with Reciprocal Rank Fusion;
  HNSW (halfvec_cosine_ops), `hnsw.ef_search` tuning.
- MCP surface: `remember` / `recall` / `current` / `timeline` — memory as tool calls.

### 4. Evaluation setup
- Datasets: LongMemEval-S (500 Q, judged); the scripted **supersession** set (50 updates);
  the scripted **bi-temporal "as-of"** set (20 timelines).
- Metrics: judged accuracy (answerer + non-reasoning judge); **[NEEDS]** retrieval-only
  Recall@k / NDCG / MRR; supersession correctness; as-of correctness.
- Baselines: **[NEEDS]** plain pgvector cosine; flat/recency store; **Mem0** (clean run).

### 5. Results
| Result | Continuum | Baseline | Source |
|---|---|---|---|
| Supersession correctness (50 scripted updates) | **100%** | 38% | README |
| Bi-temporal "as of date Y" (20 scripted timelines) | **100%** | 75% | README |
| LongMemEval-S (500 Q, judged, gpt-oss-120b) | **~74%** (73.6–75.6%) | 60.8% v1.0 · 34.4% May ceiling | README |
| Retrieval recall vs store size | ~100% @ tens · 95% @ 3k · 75% @ 47k | — | embedder_bakeoff |
| **Retrieval-only @ 3k** (real hybrid pipeline) | **R@1 .900 · R@5 .950 · R@10 1.00 · R@20 1.00 · MRR .916 · NDCG@20 .935** | **[NEEDS pgvector/Mem0 baselines]** | `scripts/retrieval_metrics.py` |
| Retrieval-only sweep @ 25k / 47k | **[NEEDS run]** | — | `scripts/retrieval_metrics.py --sizes 25000 47000` |
| Mem0 head-to-head (LOCOMO) | **[NEEDS clean run]** | — | README (preliminary) |

- **Ablations** (from existing harnesses): dense-only vs sparse-only vs RRF; `ef_search`
  sweep; embedder bake-off; HNSW vs exact-scan crossover; reranker (built, measured
  net-negative — report it).

### 6. The retrieval-vs-reasoning decomposition (the honest finding)
- Seven sweeps × four model families × six retrievers found a hard **32–34% substring
  ceiling even at 100% recall** → bottleneck is multi-hop reasoning.
- What actually moved the number: stronger answerer + clean direct retrieval + honest
  scoring (v1.0 → 60.8%, then ~74%). The reasoning loop we predicted we'd need was built,
  A/B-tested, and **cut as net-negative** (findings/reasoning_loop_2026-06.md).
- **[NEEDS]** frontier-answerer pass (task d) — the clean experiment that isolates this.

### 7. Limitations (be blunt — this is a strength)
- Judged ~74% is not SOTA on LongMemEval; the win is the deterministic correctness axes.
- Recall decays with store size (75% at ~47k) — an open problem; document honestly.
- Single-writer supersession; no multi-writer conflict resolution yet.
- Scripted benchmarks are small (50/20) — release them so others can extend.

### 8. Conclusion & released artifacts
Open-source repo, `pip install continuum-mcp`, 1800+ tests, the harnesses, and the
supersession/bi-temporal benchmark sets.

---

## Gaps to close before submission (the real work; ordered)
1. **[b] Retrieval-only metrics** — Recall@k / NDCG / MRR harness (decouple from answerer). *Highest leverage.*
2. **[c] Formalize + release the supersession & as-of benchmarks** — described sets + baselines. *Most novel/citable.*
3. **Baselines** — pgvector-cosine, flat/recency, and a **clean Mem0** run on the same eval.
4. **[d] Frontier-answerer pass** — isolates retrieval-vs-reasoning for §6.
5. Ablation tables from existing harnesses (`embedder_bakeoff`, `index_crossover`, ef_search, reranker).

## Reproducibility checklist
- [ ] Dataset provenance + fetch scripts (LongMemEval not committed — document source)
- [ ] Seeds + exact configs for every reported number
- [ ] One-command repro per table (`make bench-*`)
- [ ] Released benchmark sets (supersession, as-of) with a scoring script
- [ ] Judge model + prompt released (currently gpt-4o-mini)
