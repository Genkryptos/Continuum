# Continuum: Bi-Temporal, Supersession-Aware Memory for AI Agents

**Draft preprint — working version.** Target: arXiv → agent-memory workshop (NeurIPS/ICLR/EMNLP).
Artifacts: open-source repository, `continuum-mcp` on PyPI, released benchmark sets.

> Numbers marked **[NEEDS]** are measurement gaps still open before submission. Every other
> number is one the repository already produces; sources are named inline.

---

## Abstract

AI agents built on large language models have no native persistence: across sessions they
forget, and within a growing memory they surface *stale* facts as if they were current. We
present **Continuum**, a tiered (short-/mid-/long-term) memory system whose long-term store is
**bi-temporal** — every fact carries both a *valid-time* (when it was true in the world) and a
*transaction-time* (when the system learned it) — and **supersession-aware**: a newer fact
retires the one it contradicts *in place* via soft-invalidation, so retrieval stays current
while history stays intact and auditable. On scripted knowledge-update and point-in-time
("as-of") benchmarks, Continuum answers **100%** correctly, versus **38%** for a naive
append-only store and **20%/75%** for single-axis (recency / chronological) baselines. On
LongMemEval-S it reaches **~74%** judged accuracy. We further run a controlled decomposition
that separates retrieval quality from answering, and find that the accuracy ceiling on this
benchmark is set by **multi-hop reasoning in the answerer, not by retrieval recall** — a
reasoning loop we built and measured was net-negative and removed. Finally, we report an
honest negative result on the BEAM benchmark and a clean **presentation-order** finding:
displaying retrieved memory in chronological rather than relevance order lifts contradiction
resolution by **+25 percentage points** under BEAM's own rubric judge. Continuum is
open-source, hybrid (dense bge-m3 + BM25 fused with RRF over an HNSW-indexed pgvector store),
MCP-exposed, and reproducible; we release the code, the memory server, and the
supersession/bi-temporal benchmark sets.

---

## 1. Introduction

Long-term memory for LLM agents is usually framed as a *retrieval* problem: store past
interactions, embed them, and fetch the top-k most similar passages at query time. This framing
conflates two failure modes that are, in fact, distinct:

- **(a) Knowledge-update.** A fact changes ("I moved from Boston to Seattle"). The old fact
  remains in the store and remains retrievable, so the agent answers with stale information.
  Recall is not the problem — the stale fact is retrieved *correctly*; the problem is that
  nothing marks it as retired.
- **(b) As-of / point-in-time.** "Where did I live in 2021?" requires reconstructing what was
  true *at a past time*, not what is true now. A store with a single time axis cannot answer
  this at all, and a store with no time axis cannot even represent the question.

Vector stores and flat/recency memories get both wrong, and single-end-to-end benchmark scores
hide (b) entirely. Continuum's contribution is to treat memory as a **correctness-under-time**
problem rather than a similarity problem. Concretely:

1. A **bi-temporal, supersession-aware** long-term store that makes knowledge-update and as-of
   queries *deterministically* correct — no LLM in the critical path.
2. Released **benchmark sets** for knowledge-update (supersession) and as-of queries, with
   defined baselines and a scoring script.
3. An **honest retrieval-vs-reasoning decomposition** on LongMemEval showing where the ceiling
   actually is, including a reasoning enhancement we removed because it hurt.
4. A production, **MCP-exposed** open-source system: memory as `remember` / `recall` /
   `current` / `timeline` tool calls.

**Why correctness, not recall.** In a long-lived, correctness-sensitive assistant — a customer
agent that must not quote a stale address, an assistant asked "what was true at the time" for an
audit — a memory error is not a lost benchmark point but a wrong answer with consequences, and a
*confidently* wrong one, since the stale fact is retrieved perfectly. In these settings a
guarantee that the current value is returned *without a model guessing* is worth more than a few
points of end-to-end accuracy. This is the regime Continuum targets: its deterministic
`current()`/`timeline()` operators keep the correctness path free of the LLM, and the
append-only, soft-invalidation store makes history auditable rather than overwritten. We claim
the *design* is aimed at this regime, not that the current single-writer, self-hosted
implementation is a drop-in for a specific regulated deployment; scale, multi-tenancy, and
compliance hardening are engineering work beyond this paper's scope.

The through-line is honesty about what is and isn't a win. Where our advantage is large and
deterministic (supersession, as-of), we show it against defined baselines. Where a comparison is
null (hybrid vs. cosine) or negative (BEAM, the reasoning loop), we report it plainly, because
that is what keeps the strong claims credible.

## 2. Related work

**Agent-memory systems.** Mem0, MemGPT/Letta, and A-Mem optimize what to store and what to
retrieve from a largely static store. Mem0 in particular performs LLM-based fact extraction
(add/update/delete) and multi-signal retrieval. Continuum differs in *where* it puts the
guarantee: correctness under contradiction and time is enforced by the store's bi-temporal
schema and deterministic `current()` / `timeline()` operators, not by an extraction LLM.

**Vector RAG and temporal KGs.** Dense/sparse retrieval and reranking (the RAG stack) address
recall on a fixed corpus; bi-temporal databases and temporal knowledge graphs address
valid-vs-transaction time in structured data. Continuum sits at the intersection: a bi-temporal
store with a hybrid retrieval front-end, exposed to an agent as tools.

**Benchmarks.** LongMemEval and LOCOMO measure end-to-end long-term QA; BEAM adds rubric-scored
categories (contradiction resolution, event ordering, etc.). We use LongMemEval-S as our primary
end-to-end benchmark, add scripted supersession/as-of sets to isolate the two failure modes
above, and report BEAM as a measured-honestly comparison point (§7).

## 3. System

### 3.1 Tiered architecture

Continuum organizes memory in three tiers: **STM** (session-scoped working memory), **MTM**
(consolidated mid-term), and **LTM** (durable long-term). Consolidation promotes salient content
upward; the LTM tier is where the bi-temporal guarantees live.

### 3.2 Bi-temporal model

Every LTM fact carries two independent time axes:

- **valid-time** (`valid_from` / `valid_to`) — the interval during which the fact holds in the
  world;
- **transaction-time** (`created_at` / `invalidated_at`) — the interval during which the system
  believed it.

Supersession is **soft**: when a newer fact contradicts an older one on the same attribute, the
older row's `invalidated_at` is set — nothing is deleted. This yields two deterministic
operators:

- **`current(subject, attribute)`** — an exact-tag + valid-time lookup returning the live value.
  No LLM, no embedding, no ranking.
- **`timeline(subject, attribute)`** — the full, ordered history including retired values.

Splitting the two axes is what makes *retroactive* corrections representable: a fact learned
*late* about the *past* (a correction) is invisible to a single-axis (chronological) store but is
recovered exactly by the valid-vs-transaction split (§5).

### 3.2.1 Formal model

A **fact** is a tuple

  f = ⟨ s, a, v, [v⁻, v⁺), [t⁻, t⁻⁺) ⟩

where *s* is the subject, *a* the attribute, *v* the value, [v⁻, v⁺) the **valid-time** interval
(when *v* holds in the world; v⁺ = ∞ denotes "still true"), and [t⁻, t⁺) the **transaction-time**
interval (when the system believed *f*; t⁺ = ∞ denotes "not yet invalidated"). The store *S* is a
set of such facts. A fact is **live** at transaction time τ_t iff τ_t ∈ [t⁻, t⁺).

**Point-in-time query.** For valid time τ_v and transaction time τ_t (both default to *now*):

  current(s, a ; τ_v, τ_t) = v(f*),  where
  f* = argmax_{f ∈ S}  v⁻(f)  subject to  s(f)=s, a(f)=a, τ_v ∈ [v⁻,v⁺), τ_t ∈ [t⁻,t⁺).

i.e. among facts for (s, a) whose valid interval contains τ_v and that were believed at τ_t,
return the one with the latest valid-from. `timeline(s, a)` returns all such facts (including
those with v⁺ ≠ ∞) ordered by v⁻. Both are pure index lookups — **no embedding, ranking, or LLM.**

**Deterministic supersession on write (Algorithm 1).** Adding
f_new = ⟨s, a, v_new, [v⁻_new, ∞), [now, ∞)⟩:

```
for f_old in S with s(f_old)=s, a(f_old)=a, v⁺(f_old)=∞:      # open prior facts, same key
    if v⁻(f_old) < v⁻_new:  v⁺(f_old) ← v⁻_new                # close its valid interval
    else:                   t⁺(f_new) ← ... ; keep both       # older fact wins valid-time; retain history
insert f_new
```

The `(s, a)` key makes this **decidable without an LLM**. Generalizing it from tagged writes to
open conversation text — deriving `(s, a)` for untagged facts — is the widening described in the
engineering plan; the *guarantee* is unchanged, only its coverage grows.

**Why one axis is insufficient (retroactive corrections).**

*Proposition.* A correction may carry a valid-from *earlier* than the fact it supersedes. Any
store that orders facts by a single clock ranks such a correction as stale.

*Instance (from the shipped store, memory.py:344).* At τ₁ the user asserts "I live in Lisbon"
(v⁻ = τ₁). At τ₂ > τ₁ the user says "I moved to Porto on July 1" with July 1 < τ₁ (v⁻ = July 1,
learned at τ₂). Ordering by valid-time alone makes *Lisbon* (v⁻ = τ₁) look newer than the *Porto*
correction (v⁻ = July 1), so a single valid-axis store answers "Lisbon" — wrong. Transaction time
disambiguates: Porto was **believed later** (t⁻ = τ₂), so it supersedes. This is exactly the
`naive_chronological` failure measured in §5 (point-in-time 100%, retroactive **0/5**); the
bi-temporal split recovers all five.

### 3.3 Hybrid retrieval

The `recall` path is hybrid: dense retrieval (bge-m3, 1024-d) and BM25, fused with Reciprocal
Rank Fusion, over an HNSW-indexed (`halfvec_cosine_ops`) pgvector store with tunable
`hnsw.ef_search`. Retrieval is relevance-ranked and is the path that must *filter* superseded
facts (whereas `current`/`timeline` are exact and always correct).

### 3.4 MCP surface

Continuum exposes memory as four MCP tools — `remember`, `recall`, `current`, `timeline` —
shipped as `continuum-mcp` (PyPI). This makes the store usable by any MCP-capable agent as
ordinary tool calls, and makes the deterministic operators (`current`/`timeline`) directly
available to the model rather than hidden behind a similarity search.

## 4. Evaluation setup

**Datasets.** (i) LongMemEval-S — 500 questions, LLM-judged. (ii) A scripted **supersession**
set — 50 knowledge-update scenarios across 6 attribute types (location, employer, pet, marital,
vehicle, hobby). (iii) A scripted **bi-temporal as-of** set — 20 timelines (15 point-in-time +
5 retroactive corrections).

**Metrics.** Judged accuracy (answerer + a non-reasoning judge); retrieval-only Recall@k / MRR /
NDCG; supersession correctness (`current()` and `recall` top-1); as-of correctness.

**Baselines.** `naive_append` (append-only, no supersession); `naive_latest` (single recency
axis); `naive_chronological` (single valid-time axis); plain pgvector cosine; **[NEEDS: a clean
Mem0 run on the supersession/as-of sets — see §8].**

## 5. Results

| Result | Continuum | Baseline(s) | Source |
|---|---|---|---|
| **Supersession — in-memory schema sim** (50) | **100%** (50/50, 0 stale) | `naive_append` **38%** (19/50) → **+62pp** | `bench/supersession_correctness.py` |
| **Supersession — end-to-end through Postgres** (50, no LLM) | `current()` **100%** (50/50, 0 stale) · `recall` top-1 **20%** | — | `bench/supersession_e2e.py` |
| **Bi-temporal "as-of"** (20 = 15 PIT + 5 retroactive) | **100%** (20/20) | `naive_latest` **20%** (PIT 0/15) · `naive_chronological` **75%** (PIT 100%, **retroactive 0/5**) | `bench/bi_temporal.py` |
| LongMemEval-S (500, judged, gpt-oss-120b) | **~74%** (73.6–75.6%) | 60.8% v1.0 · 34.4% May ceiling | README |
| **Retrieval-only decay** (real hybrid, 20 needles, depth 20) | R@10/MRR: **3k 1.00/.916 · 25k .90/.90 · 47k .85/.80**; NDCG@20 .935→.900→.813 | pgvector-cosine (below) | `scripts/retrieval_metrics.py` |
| **Hybrid vs pgvector-cosine** (MRR) | 3k **.916** · 25k **.900** · 47k **.800** | 3k .916 · 25k .903 · 47k .750 | `scripts/retrieval_metrics.py` |
| ↳ verdict | within HNSW build noise (±1–2 needles); **hybrid ≥ cosine, no clean win** | — | — |

**The two-axis argument (why bi-temporal, not just chronological).** The bi-temporal result is
the paper's cleanest figure. `naive_chronological` (a single valid-time axis) gets point-in-time
queries **100%** right yet **0/5** on retroactive corrections: a fact learned late about the past
is invisible to a one-axis store. Splitting valid-time from transaction-time is exactly what
recovers those 5, taking Continuum to 20/20. This is a mechanism result, not just a headline
number.

**End-to-end validation, and a sharper architecture claim.** Replaying the same 50 supersession
scenarios through the *shipped* `Memory.from_postgres` path — `add(attribute=…)` then
`current("user", attribute)` — gives `current()` **100%** (50/50, 0 stale) with **no LLM and no
API key**: the deterministic bi-temporal exact-tag lookup. Through the same path, `recall` top-1
is only **20%**. This *locates the LLM precisely*: the supersession decider is needed only to
invalidate rows so the relevance-ranked `recall` path stops surfacing stale facts — it is **not**
needed for `current`/`timeline` correctness. So the robust, no-dependency claim is the
deterministic axis; recall-level staleness filtering is an explicitly optional, LLM-gated
enhancement.

**Hybrid vs. pure cosine — an honest, null-ish ablation.** Across 3k/25k/47k, the shipped hybrid
and a plain pgvector cosine scan are statistically indistinguishable: MRR ties at 3k (.916),
cosine edges it at 25k (.903 vs .900), hybrid edges it at 47k (.800 vs .750). Every gap is a
single needle out of 20 — within the ±1–2-needle HNSW build noise the repository's own harnesses
document. **We do not claim hybrid beats cosine** on this set; the honest claim is *hybrid ≥
cosine, no worse*. The lexical channel's value must be shown on a purpose-built exact-match query
set (future work), not this semantic one. Reporting the null here is deliberate — it is what
keeps the (large, deterministic) supersession and bi-temporal claims credible.

**Retrieval degrades gracefully, and it bounds the ceiling claim.** Through the real hybrid
pipeline, R@10 falls 1.00 → .90 → .85 as the store grows 3k → 25k → 47k (16×); misses are
near-bimodal (a needle is rank-1 or absent). So "retrieval isn't the ceiling" holds *strongly* at
small/mid stores (R@10 = 100% at 3k while judged accuracy is ~74%) and remains the honest read at
47k (retrieval 85% vs answerer ~74%) — but the curve shows retrieval *does* start to matter at
very large stores, which we state rather than claiming retrieval is solved unconditionally.

## 6. The retrieval-vs-reasoning decomposition

Seven sweeps across four model families and six retrievers found a hard **~32–34% substring
ceiling even at 100% recall**, which localizes the bottleneck to multi-hop reasoning in the
answerer rather than to retrieval. What actually moved the end-to-end number was a stronger
answerer over clean direct retrieval plus honest (judged) scoring: v1.0 60.8% → ~74%. Critically,
a **reasoning loop** we predicted we would need was built, A/B-tested, and **cut as net-negative**
(`findings/reasoning_loop_2026-06.md`). We report this because it is the empirical core of the
"retrieval isn't the ceiling" claim: given clean retrieval, the lever is answerer reasoning, and
naive attempts to add reasoning on top can hurt. **[NEEDS: a controlled frontier-answerer pass
that isolates this axis.]**

## 7. Presentation order, and an honest BEAM result

BEAM scores categories with per-question rubric nuggets (each judged 0/0.5/1.0; a question passes
at mean ≥ 0.5). Two findings, one methodological and one negative:

**Presentation order is a real lever.** On BEAM-100K (gpt-oss-120b answerer, rubric-nugget judge),
presenting the top-k retrieved turns in **chronological** order rather than **relevance** order
lifts `contradiction_resolution` from **7.5% → 32.5%** (+25pp) and nudges `event_ordering`
27.5% → 32.5%. Inspection of passing rows shows the mechanism: time-ordering surfaces *both sides*
of a contradiction adjacently, so the answerer mentions both rather than one. This is consistent
with the bi-temporal thesis — order carries information that relevance ranking destroys.

**The honest negative.** Continuum does **not** out-score Mem0 on BEAM. Mem0's published BEAM
numbers (48.6% contradiction / 60.0% event ordering) use a **gpt-5** answerer *and* judge on the
1M split; our numbers use an open answerer and a lighter judge on the 100K split. The setups are
not comparable, and where they are closest, Mem0's stronger answerer leads. We also observe that
the rubric rewards *verbose, both-sides* answers, so a stronger model answering *tersely and
correctly* can score *lower* than a weaker model that enumerates both sides — i.e., answer style,
not just capability, drives the rubric. We therefore treat BEAM as a measured comparison point,
not a headline, and locate Continuum's advantage on the deterministic axes of §5 instead.

## 8. Limitations

- Judged ~74% on LongMemEval-S is not state-of-the-art; the win is the deterministic correctness
  axes, which are answerer-independent.
- Retrieval recall decays with store size (85% R@10 at ~47k) — an open problem, documented
  honestly rather than hidden.
- Single-writer supersession; no multi-writer conflict resolution yet.
- The scripted benchmarks are small (50 / 20); we release them so others can extend.
- The 20-needle retrieval set cannot resolve sub-5pp differences (1 needle = 5pp), which is why
  the hybrid-vs-cosine comparison is inconclusive. Fix before submission: a larger needle set +
  bootstrap CIs, and a separate exact-match query set to isolate the BM25 channel.
- **[NEEDS: a clean Mem0 baseline on the supersession/as-of sets.]** This is the single
  highest-value remaining experiment — showing Mem0 fails the knowledge-update / as-of scenarios
  where Continuum is 100% would convert "we beat toy baselines" into "we beat the SOTA memory
  system where correctness-under-time matters."

## 9. Conclusion & released artifacts

Continuum reframes agent memory from similarity to **correctness under contradiction and time**.
Its bi-temporal, supersession-aware store answers knowledge-update and as-of queries
deterministically (100% vs 38% / 20% / 75% baselines), while an honest decomposition shows the
residual end-to-end ceiling is answerer reasoning, not retrieval. We release the open-source
repository, `pip install continuum-mcp`, the test suite, the retrieval/decomposition harnesses,
and the supersession/bi-temporal benchmark sets with scoring scripts.

---

## Reproducibility checklist

- [ ] Dataset provenance + fetch scripts (LongMemEval not committed — document source).
- [ ] Seeds + exact configs for every reported number.
- [ ] One-command repro per table (`make bench-*`).
- [ ] Released benchmark sets (supersession, as-of) with scoring script.
- [ ] Judge model + prompt released (currently gpt-4o-mini).
- [ ] BEAM converter + rubric judge released (`evals/beam/`).
