# Continuum positioning & strategy

*2026-05-31 · how Continuum wins, grounded in the market + a capability audit*

This is the "why we win" narrative. It pairs the 2026 agent-memory market
with an honest audit of what Continuum has, what's dormant, and what's
missing — and proposes a wedge plus three bets.

---

## 1 · The market (2026)

No single "best" — the market splits by use case, and **temporal reasoning is
the dividing line** on accuracy:

| System | Philosophy | Wins on | LongMemEval* |
|---|---|---|---|
| **Mem0** | auto-extract facts → vector+graph+KV | personalization, easiest adoption (~47K★) | ~49% |
| **Zep** | **bi-temporal knowledge graph** (fact validity windows) | **temporal reasoning** | **~63.8%** |
| **Letta** (ex-MemGPT) | literal "memory OS": context=RAM, archival=disk, self-managed | long-running agents | — |
| **Cognee** | KG + ETL pipelines | structured/graph domains | — |
| **Continuum** | tiered STM/MTM/LTM + supersession + bi-temporal + policy engine | *(see wedge)* | **60.8%** (gpt-oss-120b)** |

\* Vendor/blog numbers on GPT-4o; directional, not apples-to-apples.
\*\* Continuum's 60.8% is a different model + an LLM judge — not directly
comparable to Zep's 63.8%-on-GPT-4o. Treated as "same arena," not "we win."

**Two lessons from the market:**
1. **Accuracy ≠ adoption.** Mem0 leads the market while scoring *lowest* —
   on DX, community, and a free hosted tier. Capability alone doesn't win.
2. **The accuracy leader (Zep) wins on time.** Its edge is a *bi-temporal
   knowledge graph* — storing when facts are valid, not just snapshots.

---

## 2 · The capability audit (the surprise)

Zep's moat is a **bi-temporal knowledge graph**. Here's the honest state of
that exact architecture in Continuum:

| Layer | Status | Evidence |
|---|---|---|
| Bi-temporal node schema | ✅ **built & used** | `valid_from`/`valid_to` + `created_at`/`invalidated_at`; supersession = `invalidated_at IS NULL` |
| Bi-temporal **edge** schema | ✅ **built**, dormant | `memory_edges` with `predicate`, `weight`, `invalidated_at` |
| Graph traversal | ✅ **built**, dormant | `neighbors()` recursive-CTE walk, cycle-guarded, ~2 ms, bi-temporal |
| Graph-expansion retrieval | ✅ **built**, dormant | `_graph_expand()` wired into the retriever (gated by `graph_expand_n`) |
| Relation extraction | ✅ **produces relations** | extractors return `(entities, relations)` |
| **Edge population** | ❌ **the one missing step** | nothing writes `memory_edges`; extracted relations are discarded |
| Reranker | ✅ **built**, dormant | `continuum/retrieval/reranker.py` exists, not in the v1 path |
| Policy / privacy engine | ✅ **built & tested** | sensitivity → redact/encrypt/ask-user; 8 policies |
| Token-budget optimizer | ✅ **built** | 5 compression strategies incl. LLMLingua |

**Headline:** Continuum *already architected Zep's winning design* — a
bi-temporal knowledge graph — and it sits **dormant behind a single missing
step** (writing edges). This is the highest-leverage fact in the whole
strategy. (Tracked as roadmap **WS-6**.)

---

## 3 · The wedge

> **Continuum: memory with first-class _time_ and _relationships_ —
> privacy-aware, cost-efficient, and honestly benchmarked.**

A lane each competitor only half-occupies:
- **Zep** owns *time* but markets no privacy/cost story.
- **Mem0** owns *adoption* but discards the timeline (we measured it collapse
  to 4.2% on temporal vs Continuum's 33%).
- **Letta** owns the *OS metaphor* but not benchmark accuracy.
- **Nobody** markets **privacy-aware memory** (don't leak secrets into LTM) or
  **cost-per-correct-answer** — both of which Continuum already implements.

---

## 4 · The three bets

1. **Cash in time + graph (capability → beat Zep on its own turf).**
   Wire the dormant bi-temporal graph (WS-6) + exploit the temporal schema at
   answer time (WS-1) + turn on the existing reranker (WS-4). Goal: a *fair*
   LongMemEval/LOCOMO run that matches or beats Zep — using architecture we
   already have.

2. **One framework adapter + a 3-line SDK (adoption → the real gap).**
   Continuum ships **zero** framework adapters (no LangChain/LangGraph/
   AutoGen/LlamaIndex) and is library-only. This is almost certainly why a
   60.8% system has less traction than a 49% one. A `ContinuumMemory`
   drop-in for LangGraph + a `add()/search()` SDK is the adoption unlock.

3. **Make the measurement the marketing (credibility).**
   Every competitor publishes self-serving numbers. Continuum already
   documents what it *cut* (the reasoner) and what's *preliminary* (LOCOMO).
   Publish a reproducible LongMemEval + a *fair* LOCOMO head-to-head + a
   **tokens-per-correct-answer** cost number (the optimizer chain nobody else
   benchmarks). "We measured honestly, here's the repro" is itself a moat.

---

## 5 · Honest weaknesses (what we're NOT pretending)

- **No framework adapters / no hosted option** — the biggest adoption blocker.
- **Single benchmark** at scale (LongMemEval); LOCOMO is preliminary.
- **Temporal underperforms (41%) despite owning the schema** — storage built,
  answer-time exploitation not.
- **Possible over-engineering**: the 8-policy engine is a real privacy wedge,
  but shouldn't expand before adapters exist. Keep it lean.
- **Graph precision is unproven** — populating edges adds value *only* if
  relation extraction is precise; bad edges add noise.

---

## 6 · What we will NOT do

- **Rebuild the general reasoner.** The IterativeReasoner (decompose → verify →
  compose) was built, A/B'd, and **cut** as net-negative. Scoped tools only.
- **Chase a leaderboard with vanity tuning.** Numbers stay reproducible and
  honestly scoped, or they don't ship.

---

*Companion: the execution plan lives in
[`findings/roadmap_v1.1.md`](../findings/roadmap_v1.1.md).*
