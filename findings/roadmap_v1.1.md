# Continuum v1.1 roadmap — targeting accuracy & recall

*Status: proposed · 2026-05-31 · supersedes nothing (forward-looking)*

This roadmap is grounded in the v1 results in
[`reasoning_loop_2026-06.md`](reasoning_loop_2026-06.md). It is **not** a
wish-list — every workstream targets a specific, measured v1 weakness and
carries an expected pp lift, an effort estimate, and a risk.

---

## 1 · Where v1 actually loses

LongMemEval-S, n=500, judged accuracy + recall (from the v1 report):

| category | n | judged | recall | bottleneck |
|---|---:|---:|---:|---|
| single-session-assistant | 56 | 98.2% | ~98% | ✅ solved |
| single-session-user | 70 | 94.3% | ~94% | ✅ solved |
| **temporal-reasoning** | **133** | **41.4%** | high | date arithmetic / counting — *not* recall |
| **multi-session** | **133** | **54.9%** | 85% | aggregation/counting at recall=1.0 |
| knowledge-update | 78 | 51.3% | 98.7% | picking the *current* value when old+new co-retrieved |
| single-session-preference | 30 | 50.0% | 93% | residual after the judge-artifact fix |
| **overall** | **500** | **60.8%** | **91.6%** | |

### The two facts that set the strategy

1. **Recall is largely solved (91.6%).** The dominant headroom is *turning
   retrieved context into correct answers*, not retrieving more.
2. **The headroom is concentrated.** temporal (133) + multi-session (133) are
   **53% of the dataset** and sit at 41% / 55%. Fixing them is the whole game.

The v1 report classifies both as "genuine model-reasoning limits, not memory
or retrieval gaps." The trap — which v1 already fell into and **cut** — is
answering that with a *general* reasoning loop (decompose → verify → compose;
it lost to direct retrieval everywhere). 

> **v1.1 thesis: narrow, scoped tooling beats general scaffolding.**
> Give the model a date calculator and an enumeration step where it
> demonstrably can't do the work itself — do **not** rebuild the reasoner.

---

## 2 · Workstreams (ranked by expected overall lift)

### WS-1 · Temporal — normalize dates, compute deltas in code ⭐
- **Failure**: "how many days between X and Y", "what happened first" — the
  model miscounts from raw prose; the judge can't rescue genuine arithmetic.
- **Approach**:
  1. Normalize each memory item's date to ISO at ingest (populate
     `occurred_at` / `valid_from`).
  2. A **temporal-question classifier** routes temporal Qs to a path that
     surfaces normalized timestamps and performs date arithmetic in a
     **deterministic tool / code**, not the LLM's mental math.
  3. No decompose/verify loop — dates in → delta out.
- **Expected**: 41% → 60-70% on 133 Q ≈ **+5-8pp overall**.
- **Effort**: M · **Risk**: classifier precision (misrouting can hurt
  non-temporal Qs — gate on the full set).

### WS-2 · Multi-session aggregation — count-then-answer
- **Failure**: recall=1.0 but "3 weddings" → "1"; the model doesn't enumerate
  before counting.
- **Approach**: upgrade the existing aggregation branch so count/list
  questions force *"enumerate every distinct instance across sessions →
  dedupe → then count/answer."*
- **Expected**: 55% → 68-72% on 133 Q ≈ **+3-5pp overall**.
- **Effort**: S · **Risk**: low (prompt-level; trivial to A/B).

### WS-3 · Knowledge-update — surface recency/supersession in the prompt
- **Failure**: when an old fact and its current supersessor are both
  retrieved, the model sometimes answers stale.
- **Approach**: tag retrieved items with `invalidated_at` / recency and
  instruct "prefer the most recent non-invalidated value." Leverages the
  supersession data Continuum already produces.
- **Expected**: 51% → ~62% on 78 Q ≈ **+1.5pp**.
- **Effort**: S · **Risk**: low.

### WS-4 · Recall residual — reranker + embedder upgrade *(the "recall" half)*
- **Where recall still misses**: 8.4% overall; multi-session ~85%.
- **Approach**: add a **cross-encoder / Cohere reranker** over the hybrid
  (cosine ⊕ BM25 ⊕ RRF) candidate set; A/B a stronger embedder
  (MiniLM → bge-large or text-embedding-3-small).
- **Expected**: recall 91.6% → ~95%, plus precision gains that lift every
  category ≈ **+2-4pp overall**.
- **Effort**: M · **Risk**: added latency/cost (rerank = +1 call); embedder
  swap re-indexes everything.

### WS-5 · Measurement honesty *(prerequisite, do first)*
- **Per-category ablation harness** — every lever reports its pp delta on the
  **full 500**, never just the failure slice. (This is the *exact* mistake the
  v1 raw-context regression made: a fix that helped failures regressed the
  success set −6pp because only failures were checked.)
- **Cost accounting** — wire OpenRouter pricing; kill the `$0.00` artifact so
  we track accuracy-per-dollar.
- **LOCOMO clean run** — un-handicap Mem0 (verify near-zero JSON parse errors),
  run full, publish a fair second-benchmark head-to-head.

---

## 3 · Target & guardrails

- **Stacked target: 60.8% → ~70-72% judged** from WS-1+2+3+4, *without*
  resurrecting the general reasoner.
- **Gate**: every change is A/B'd on the full 500 with the LLM judge **before
  it lands**. Any regression on the solved categories (single-session
  94-98%) blocks the change.
- **Anti-goal**: no decompose → verify → compose loop. Scoped tools only
  (date math, enumeration), and only where the model provably can't.
- **Honesty**: if a lever doesn't move the full-set number, it doesn't ship —
  we report it as a tested negative result (as we did for the IterativeReasoner).

---

## 4 · Suggested sequence

1. **WS-5 harness** — so everything after is measured honestly.
2. **WS-2 aggregation** — cheapest win, validates the harness.
3. **WS-1 temporal** — biggest prize.
4. **WS-3 knowledge-update** — cheap, leverages existing supersession data.
5. **WS-4 reranker/embedder** — the recall half; most infra.

Each lands as its own branch + PR with a before/after full-500 judged table in
the description, so the pp contribution of every lever is on the record.
