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
- **Approach**: **wire the existing `continuum/retrieval/reranker.py`** into
  the eval-winning path — it's already built but *not used* in the v1 direct
  config. Then A/B a stronger embedder (MiniLM → bge-large or
  text-embedding-3-small).
- **Expected**: recall 91.6% → ~95%, plus precision gains that lift every
  category ≈ **+2-4pp overall**.
- **Effort**: S-M (reranker exists) · **Risk**: added latency/cost
  (rerank = +1 call); embedder swap re-indexes everything.

### WS-6 · Wire the dormant bi-temporal knowledge graph ⭐⭐ *(the buried moat)*
- **What's already built** (a capability audit surprise): the entire graph
  stack exists and is **bi-temporal** — `memory_edges` (edges carry
  `invalidated_at`), a production recursive-CTE `neighbors()` traversal
  (~2 ms, cycle-guarded), and `_graph_expand()` already wired into the
  retriever (gated by `graph_expand_n`). The extractors even produce
  `(entities, relations)`.
- **The single missing step**: nothing **writes** edges. The promoter's
  `_neighbors()` is semantic-similarity search, not graph edges; the
  extracted `relations` are discarded. No `INSERT INTO memory_edges` exists.
- **Approach**: add a relation-writer — persist extracted
  `(subject, PREDICATE, object)` triples to `memory_edges` at promotion time,
  carrying bi-temporal `invalidated_at` so superseded relations drop out of
  traversal automatically. Turn on `graph_expand_n` in the eval config.
- **Why it matters**: this *is* Zep's winning architecture (a temporal
  knowledge graph with validity windows). Continuum already architected it;
  populating edges unlocks **multi-hop** ("X's manager's project") — exactly
  where flat retrieval dies and graphs win — on top of the bi-temporal
  invalidation we already have.
- **Expected**: multi-hop / cross-entity questions are the residual in
  multi-session (55%) and part of temporal; realistic **+3-6pp overall**,
  and a genuine architectural differentiator.
- **Effort**: M · **Risk**: relation-extraction precision (bad edges add
  noise — gate edge writes on confidence; measure with `graph_expand_n` on
  vs off on the full set).

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

- **Stacked target: 60.8% → ~72-75% judged** from WS-1+2+3+4+6, *without*
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
3. **WS-4 reranker** — the reranker already exists; wiring it is near-free recall.
4. **WS-1 temporal** — biggest single-category prize.
5. **WS-6 graph** — the buried moat; unlocks multi-hop + the differentiator story.
6. **WS-3 knowledge-update** — cheap, leverages existing supersession data.

> **Capability-audit note:** WS-4 (reranker) and WS-6 (bi-temporal graph) are
> both *already built* in the codebase and merely dormant. Before adding new
> machinery, the cheapest wins are turning on what's already there.

Each lands as its own branch + PR with a before/after full-500 judged table in
the description, so the pp contribution of every lever is on the record.

---

## 5 · Outcomes log (what actually happened)

### WS-5 — DONE
`findings/charts/ablate.py` (per-category A/B + regression guard) + OpenRouter
cost wiring shipped. The harness immediately earned its keep — see below.

### Measurement reality (discovered via WS-5) — **the gating constraint**
- **Answerer nondeterminism**: `openai/gpt-oss-120b` on OpenRouter varies
  ~44% of answers run-to-run even at temperature 0 (multi-provider routing +
  MoE). Mitigated with `--openrouter-provider <pin> --seed 0` (added to
  `OpenRouterLLM`): cuts cross-provider swing, but **~5% correctness flips
  remain** (seed not honoured at the token level on the pinned backend).
- **Noise floor**: ~±5pp per category at n=20; ~±2-5pp overall at n=120.
  **A lever is only cleanly measurable if its expected lift exceeds ~5pp.**
- **Judge nondeterminism**: the LLM judge is *also* unstable on soft-rubric
  categories (and isn't provider-pinned). On single-session-preference the
  judge verdict count swung 13–18/30 across identical runs.

### WS-4 reranker — wired, but a NO-OP in the v1 config
The reranker now fires in `_DirectAnswerAdapter` (keeps `rerank_to`), but the
session-aware retriever returns only ~4-8 candidates (`max_items` cap), so
there's nothing to rerank. **Needs a retriever over-fetch fix before it can
help.** First A/B was pure nondeterminism (reranker never engaged).

### WS-7 preference conditioning — built, tested, **SHELVED (unmeasurable)**
Added `--pref-conditioning` (gated to preference questions; provably can't
touch other categories). It fires and changes answers (rescues abstentions:
"I don't know" → tailored pick). But on the n=30 preference category the
**A/A noise floor is ±16.7pp** — two *identical* runs disagreed 60.0% vs
43.3%. Both v1 (+4/−4) and v2 ("concrete pick, not a menu") A/B signals are
far inside that noise. Preference compounds answer-nondeterminism × soft-rubric
judge-nondeterminism → worst-case to measure, on 6% of the dataset. Flag stays
(off by default, documented); not shipped as a win. *Lesson: this category is
not where measurable wins live — go where the metric is crisp.*

### WS-1 temporal conditioning — **WIN (the first measurable lever)**
Root cause was NOT arithmetic difficulty (the original assumption): the direct
adapter built `[role] content` and **dropped the turn dates entirely**, so the
model answered *"the conversation doesn't include any dates" / "0"*. WS-1
(`--temporal-conditioning`, gated): prefix each retrieved turn with its
`[YYYY-MM-DD]` date, inject the reference "now" (`question_date`), prompt for an
explicit scoped date calculation.

**Result (n=133, judged, DeepInfra-pinned):**

| | judged |
|---|---:|
| baseline (dates dropped) | 39.8% |
| **+ temporal conditioning** | **72.9%** |
| **Δ** | **+33.1pp** |
| A/A noise floor (2 identical baselines) | 0.8pp |

Signal beats noise ~40×. Gated to temporal questions → other categories
provably untouched → **~+8.8pp overall (60.8% → ~69.6%, projected)**. Confirm
with one full-500 judged run for the published headline. Recommend making
`--temporal-conditioning` **default-on** for v1.1 (gated, no downside).

### Next levers (by expected value)
- **WS-2 aggregation** (multi-session, n=133, ~55%) — cheap prompt branch, next.
- **WS-6 graph** — the dormant bi-temporal KG (differentiator).
- WS-4 reranker — needs the retriever over-fetch fix; likely sub-noise, low priority.
