# Continuum — Engineering Improvement Plan

**Status:** proposed · **Frame:** strengthen the *actual* differentiator — deterministic
correctness under contradiction and time — and remove the LLM from the correctness path.
**Measure everything against the existing harnesses** (`bench/`, `scripts/retrieval_metrics.py`,
`evals/`) so each change has a before/after number, not a vibe.

## Guiding principle

The moat is `current()`/`timeline()` being **deterministically correct with no LLM in the
critical path**. Today that guarantee is *scoped to attribute-tagged facts*; untagged
real-conversation text falls back to relevance retrieval or an optional gpt-4o-mini decider.
Every phase below either (a) widens the deterministic guarantee to open text, or (b) fixes a
measured weakness, and none of them should re-introduce an LLM into `current`/`timeline`.

---

## Phase 1 — Deterministic supersession on open text (highest leverage)

**Problem (grounded).** `continuum/memory.py::add` supersedes deterministically only when an
`attribute=` tag is supplied (the `touch(..., attribute=...)` path, memory.py:329). Untagged
facts get `valid_to=None`, so `_is_superseded` (memory.py:703, `valid_to is not None`) never
fires, and `_prefer_current_versions` — which only *reorders the already-retrieved set* — can't
demote a stale fact that outranks its correction or, worse, when the correction isn't in top-k
at all. This is why supersession `recall` top-1 = **20%** while `current()` = **100%**
(`bench/supersession_e2e.py`).

**1a. Open-domain attribute keying on write.**
Reuse the existing extractors (`continuum/extraction/fact_extractor.py`,
`continuum/promotion/attribute_extract.py`) to derive a `(subject, predicate)` key for untagged
`add()` calls — deterministically where possible (regex/entity), LLM only as an *optional*
enrichment, never on the read path. Store it as the same `attribute` metadata the tagged path
already uses, so everything downstream (`_current_by_tag`, `_prefer_current_versions`) just works.

**1b. Deterministic supersession-on-write.**
When a new fact's `(subject, predicate)` key matches an open prior fact, **close the prior's
`valid_to`** at write time — the same operation the tagged `touch` path performs — with **no
LLM**. Guard with the existing `continuum/policies/conflict_policy.py` for the "is this actually
a contradiction vs. an addition" decision, kept rule-based.

**1c. Validity-aware `recall`.**
- Over-fetch at the store layer (`continuum/core/session.py::search`, line 406): fetch `k·3`,
  then let `_prefer_current_versions` promote — so a correction that ranks below its stale
  original still surfaces in the returned `k`.
- Add an explicit **validity/recency term** to the ranking so `valid_to IS NULL` (open) facts
  outrank closed ones deterministically, and honour `as_of` at the SQL `WHERE` when provided.

**Acceptance:** on `bench/supersession_e2e.py`, `recall` top-1 **20% → ≥ 80%** *with the LLM
decider OFF*. `current()` stays 100%. No regression on LongMemEval-S (`~74%`).
**Size:** M. **Risk:** false-positive supersession (an *addition* mistaken for a *replacement*)
— mitigate with the conflict policy + a "both retained, newest ranked first" fallback (never
delete).

---

## Phase 2 — `timeline()`/`current()`-backed retrieval mode

**Why.** Today's finding: presenting retrieved memory in **chronological, deduped,
current-flagged** order lifted BEAM contradiction **+25pp** (`--chrono-sort`). That was an eval
hack; it belongs in the product.

**Build.** A retrieval/context mode that, for a query, assembles context from `timeline()` +
`current()` output (ordered, superseded facts explicitly labelled "was X, now Y") instead of raw
cosine hits. Expose it via the MCP `recall` path as an option and via the eval adapter so the
same code is measured.

**Acceptance:** reproduce the chrono-sort contradiction gain **through the product path** (not
the eval-only flag) on the BEAM converter set; no LongMemEval regression.
**Size:** M. **Risk:** context bloat — cap with `current()`-first, `timeline()` on demand.

---

## Phase 3 — Consolidation quality (parity with Mem0's core strength)

**Why.** Mem0's edge is its extract/merge/dedupe pipeline. Continuum has the scaffolding
(`continuum/promotion/promoter.py`, `mem0_promoter.py`, `synthesis.py`) — the question is how
good the merge/dedup is.

**Build.** Audit the promotion path; add/strengthen **near-duplicate detection** (embedding
threshold), **entity linking** (co-reference "my dog" ↔ "Rex"), and **fact merging** (fold
compatible facts, supersede contradictory ones via Phase 1). Emit dedup/merge telemetry.

**Acceptance:** measurable drop in redundant LTM rows on a fixed ingest; LongMemEval
knowledge-update subset up; no loss on retrieval recall.
**Size:** M–L. **Risk:** over-merging distinct facts — keep a provenance link, never destroy.

---

## Phase 4 — Retrieval at scale (fix the decay)

**Why.** R@10 falls **1.00 → .90 → .85** across 3k → 25k → 47k, misses are bimodal (rank-1 or
absent) → an *index/recall* problem, not ranking (ef_search sweeps didn't move it).

**Build.** **Two-stage hierarchical retrieval**: coarse recall over session/entity summaries (or
the entity graph in `continuum/extraction/entity_extractor.py`), then fine retrieval within the
selected regions. Optionally fold in graph-expansion (the `graph_expand` idea already prototyped
in the eval harness).

**Acceptance:** R@10 at 47k **.85 → ≥ .95** on `scripts/retrieval_metrics.py`; latency p95 within
budget.
**Size:** L. **Risk:** two-stage latency — measure, cache summaries.

---

## Phase 5 — Hygiene (cheap, do in passing)

- **Resolve the BM25 null.** Hybrid == cosine on the semantic set. Validate BM25 on a purpose-built
  exact-match query set; fix RRF weighting or **drop the channel** (latency/simplicity win) if it
  genuinely adds nothing. Don't ship a no-op.
- **Harden the eval harness.** Auto-detect reasoning models and raise `--answer-max-tokens`
  (default 256 starves them → empty answers); warn when `substring`/`recall` are structurally
  meaningless for a dataset (e.g. BEAM); fix the `--help` crash (`unsupported format character
  ';'` — a raw `%` in a help string).
- **Index determinism.** The ±1–2-needle HNSW build noise is non-determinism — pin build params /
  seed so results are reproducible run-to-run.
- **Incremental embedding.** Confirm production doesn't re-embed the whole store per `recall`
  (the eval retriever rebuilds each call).

---

## Sequencing & the validation gate

1. **Phase 1 first** — it generalizes the moat to open text and removes the LLM from the
   correctness path. Biggest rating mover.
2. **Phase 2** — turns the chrono-sort finding into a shipped feature.
3. Then the **validation gate**: run **Mem0 through `bench/supersession_*` and `bench/bi_temporal`**
   (the cheap, decisive experiment). If Continuum still wins post-Phase-1 while Mem0 does not, the
   moat is proven on open text — that's the paper's headline and the go-signal for Phases 3–4.
4. **Phases 3–4** are parity/scale work, valuable but not differentiating; do after the gate.
5. **Phase 5** anytime — low cost, low risk.

Each phase is independently shippable and independently measured. Nothing here depends on beating
BEAM; the target is deterministic correctness under time, general to open conversation.
