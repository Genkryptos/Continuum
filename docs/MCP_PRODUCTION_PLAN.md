# Continuum MCP — Production-Readiness Plan

**Status:** draft · **Owner:** Mayank · **Context:** the MCP server works well in
tests but is not yet something a stranger can rely on across sessions and
contexts. This plan is what stands between the two, ordered by what actually
gates production — not by what is easy.

Everything here is grounded in bugs/limits found this cycle by *using* the
server, not by benchmarks. The engine is now good; the gaps are in **product
shape** (single-tenant, passive) and **operability**.

---

## Definition of done

A stranger can: `pip install continuum-memory[mcp]` → point it at their Postgres
→ and across separate Claude sessions it (a) remembers what they told it,
(b) never surfaces another context's memories, (c) fires without being asked, and
(d) never silently loses a write or returns a stale fact as current. Reproducible
by someone who is not us.

---

## Already done this cycle (baseline — do not redo)

- Retrieval is relevance-ranked (was query-independent: every query returned the same rows).
- Valid time is plumbed end-to-end; `current` is exact via `attribute` tags + bi-temporal `as_of`.
- `remember` reports **durability** — no more "stored" for a write that vanishes.
- Backend resilience: no crash when the DB is down, opt-in autostart, DB-level (not port-level) readiness probe.
- Compound statements split before embedding; supersession available via the Mem0 decider (opt-in, needs an LLM).
- Adversarial fixes: negative/zero `k`, `as_of` naive/aware crash, SQL-injection inert.
- **End-to-end test suite** driving the real binary against Postgres, wired into CI.

---

## Phase 0 — Correctness & safety blockers (nothing ships public without these)

### 0.1 Tenant scoping (the #1 blocker) — ✅ DONE (migration 005, namespace on every LTM query + STM bound to it; e2e isolation test)
**Problem (proven):** LTM has no session/user/agent column and no filter anywhere.
Alice's `recall("lives in")` returned Bob's fact. `recall` and `current` leak
across every session — and across your own projects, since one database is shared.
**Fix:** add a scope key (start with `namespace`/`agent_id`) to `memory_nodes`;
thread it through `upsert`, `search_hybrid`, `by_tags`, and the MCP server (from a
`CONTINUUM_MCP_NAMESPACE` env or the MCP client id). Migration + backfill default.
**Acceptance:** an e2e test where two namespaces write overlapping facts and
neither can recall or `current` the other's. **Effort:** M (schema + every query).
**Decision needed:** scope granularity — per-user? per-project? per-agent? Default
to a single `namespace` string and let the operator choose what it means.

### 0.2 No silent data loss under the decider
**Problem:** with supersession on but the LLM unavailable mid-run, a NOOP on the
ambiguous band would drop the new fact. Currently guarded by falling back to ADD,
but there is no test proving it under a *flaky* (not absent) LLM.
**Fix:** a test that injects an erroring/timing-out completion_fn and asserts the
fact still lands. **Acceptance:** e2e proves no write is ever lost. **Effort:** S.

---

## Phase 1 — Make it *memory*, not *a tool Claude sometimes calls*

### 1.1 Automatic recall — ✅ DONE (UserPromptSubmit hook: continuum.mcp.recall_hook, sparse, never blocks)
**Problem (proven):** everything good this cycle happened because the tools were
called explicitly. In a real session Claude must *choose* to `recall`, so memory
fires only when it thinks to look.
**Fix:** a `UserPromptSubmit` hook that recalls against the actual prompt and
injects the top hits as context. At ~85ms/call it is cheap enough per turn.
**Acceptance:** a scripted multi-session run where a fact stated in session 1 is
used in session 2 *without any explicit recall*. **Effort:** M. Ship as a
documented, optional hook first; do not force it on.

### 1.2 Automatic capture (optional, bigger)
**Problem:** the user (or Claude) must explicitly `remember`. Real memory should
also accrue from conversation.
**Fix:** the promotion pipeline already exists (`continuum/promotion/`). Wire a
`Stop`/`PostToolUse` hook or an ingest path that extracts durable facts from
turns. **Risk:** noise/PII capture — must be opt-in and reviewable. **Effort:** L.
Defer until 0.1 and 1.1 land.

---

## Phase 2 — Reliability at scale

### 2.1 Scale test (unknown territory)
**Problem:** tested at 62 memories; real use is thousands. Index behaviour,
`recall` precision, and latency at 10k+ rows are unmeasured.
**Fix:** a harness that loads 10k–100k memories and measures recall@k, p95, and
index size; tune HNSW `ef`/`m` if needed. **Acceptance:** documented curves; p95
< 200ms at 10k. **Effort:** M.

### 2.2 Forgetting / pruning
**Problem:** memory only grows. No eviction, no decay, no summarisation of stale
rows. **Fix:** use the optimizer strategies already in the tree (currently
untested); TTL or importance-based pruning. **Effort:** M.

### 2.3 Deduplication — ✅ DONE (assembler dedups LTM+STM by content, LTM copy wins)
**Problem (observed):** `recall` returns LTM+STM duplicates of the same fact
("Alice lives in Paris" twice). **Fix:** collapse by content/id in the assembler.
**Effort:** S.

---

## Phase 3 — Retrieval quality

### 3.1 Changed-facts reliability
**Problem (measured):** without `attribute` tags, changed facts rank the stale
version first. The decider catches ~half (gpt-4o-mini is a noisy judge:
Infosys→Nimbus retired, Nimbus→Stripe and Air→Pro missed at identical cosine).
**Fix (ranked):** (a) auto-extract `attribute` on write so the deterministic
`current` path applies without the user naming it — this is the reliable answer;
(b) optionally a stronger decider model. **Effort:** M. This is your v2.1 "typed
memory" roadmap item.

### 3.2 Paraphrase precision
**Problem (measured):** recall@1 ~70% on paraphrased queries (100% @3). Livable
because Claude reads k=8, but rank-1 is not reliable. **Fix:** the cross-encoder
reranker exists but is off by default; enable + measure. **Effort:** S–M.

---

## Phase 4 — Operability & packaging

### 4.1 Input hygiene — ✅ DONE (empty→no-op, text capped at MAX_FACT_CHARS, k clamped to MAX_RECALL_K)
Length cap on `text` (100k chars is embedded today), reject empty strings, cap
`k`. All found adversarially. **Effort:** S.

### 4.2 Observability
Structured logs / counters for: what was recalled, hit/miss, latency, backend
fallbacks, decider decisions. Today failures are silent by design — the exact
property that hid this cycle's worst bugs. **Effort:** M.

### 4.3 Install & secrets story
`pip install continuum-memory[mcp]` clean-venv path; the MCP registration
one-liner; document that `continuum.yaml` is git-tracked and must not hold a
machine DSN (nearly leaked a username into the public repo this cycle).
**Effort:** S.

---

## Suggested order

1. **0.1 tenant scoping** — correctness/safety; blocks everything public.
2. **1.1 automatic recall** — turns "tool" into "memory"; highest felt value.
3. **2.3 dedup + 4.1 input hygiene + 2.2 pruning** — cheap reliability.
4. **3.1 auto-attribute extraction** — the real fix for changing facts.
5. **2.1 scale test + 4.2 observability** — prove it holds and make it debuggable.
6. **1.2 automatic capture** — the ambitious one; last, because it is the riskiest.

Phases 0–1 are the line between "impressive demo" and "production." Everything
after is what keeps it production once people rely on it.
