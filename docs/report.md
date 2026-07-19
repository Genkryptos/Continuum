# Continuum: an honest technical report

*Memory for AI agents is a state problem, not a retrieval problem.*

This report describes what Continuum is, how it performs on LongMemEval-S, and —
unusually — **what we tried that did not work**. The negative results are the
point: they mark where the ceiling actually is on this class of reader, and they
save the next person the same spend.

---

## 1. The problem

Most "memory" for LLM agents is one of three things, each with a failure mode:

- a **vector store** — great recall, no notion of *current* vs *stale* facts;
- an **append-only history** — no structure, grows expensive;
- a **curated profile object** — works until the user changes their mind.

The hard part isn't retrieval — it's **state**: when a user says "I moved to
Boston" and later "actually, I'm in NYC now", the system must know NYC is current
*and* still be able to answer "where did I live as of March?".

## 2. Continuum's approach

- **Tiered storage** (STM / MTM / LTM) so recent turns, mid-term summaries, and
  long-term facts are addressable separately.
- **Supersession** — a superseded LTM fact is stamped `invalidated_at` (never
  deleted); every "current" read filters `invalidated_at IS NULL`, so the system
  always knows which version is live.
- **Bi-temporal columns** — `valid_from`/`valid_to` (world time) and
  `created_at`/`invalidated_at` (system time) — enabling "as of date Y" queries
  including retroactive corrections.
- **Hybrid retrieval** — dense + BM25 fused by Reciprocal Rank Fusion.

The public API is five verbs: `add`, `recall`, `current`, `timeline`,
`remember`. An MCP server exposes the same as tools for any agent client.

## 3. Where it wins (scripted benchmarks, deterministic)

| benchmark | Continuum | baseline |
|---|---:|---:|
| Supersession correctness (50 scripted updates) | **100%** | 38% |
| Bi-temporal "as of date Y" (20 scripted timelines) | **100%** | 75% |

These are the differentiator: if your agent forgets or uses stale facts, this is
the fix.

## 4. LongMemEval-S — measured honestly

LongMemEval-S is 500 questions over ~50-session histories, LLM-judged. We report
with `gpt-oss-120b` as the reader (open, cheap) and `llama-3.3-70b` as the judge.

### The key methodological result

**Measuring gains only on a set of known failures structurally overstates them.**
A failure can improve or stay failed — it can *never regress* — so a
failure-only view shows every win and none of the damage. Early failure-only
runs looked like +20–30% recovery; a full-500 run with a **same-setup control**
showed the true effect was ~neutral. Always carry a control.

### The numbers (full-500, same-setup, llama-3.3 judge)

| config | judged |
|---|---:|
| same-setup baseline (no additions) | 73.8% |
| + bounded reflect (preference/KU) + vote-3, run A | 75.6% |
| + bounded reflect (preference/KU) + vote-3, run B (fresh) | 73.6% |

**The LongMemEval-S number is ~74% (73.6–75.6% across runs).** `gpt-oss-120b` has
~±3–5pp per-category run-to-run variance (MoE routing + batch nondeterminism), so
a single run is not a stable estimate — the 76.4% quoted in earlier notes was a
favorable draw. Critically, the read-side additions (reflect + vote-3) land
**within that noise of the baseline** — approximately accuracy-neutral overall.
Even the most promising per-category signal, single-session-preference, swung
53 / 60 / 70 / 80% across runs on just 30 rubric-judged questions: we first read
one 80% draw as a "+20pp win", but it did **not** reproduce (a fresh run put it
back at 60%, tied with the no-reflect control). The reproducible wins are the
deterministic benches in §3 — not the LongMemEval per-category deltas.

## 5. What did NOT work (the honest core)

We tested four ways to have the *memory layer* fix the reader's mistakes. All
were **net-negative** on the full benchmark:

| lever | mechanism | why it failed |
|---|---|---|
| synthesis / aggregation | count members in code, inject the count | reader ignores/over-counts; scoping errors |
| deterministic router | return the code-computed count, skip the reader | fired on duration/recall questions → 82% wrong |
| evidence distillation | filter to the relevant turns before answering | **undercounts** — counting needs completeness; the filter drops members |
| temporal codemath | model emits a date-spec, code computes | model emits bad specs ~43% of the time → confident wrong answers |

**The pattern:** you cannot bolt deterministic machinery onto `gpt-oss`'s
intermediate outputs — its counts, extractions, and date-specs are too
unreliable, so the code faithfully computes the wrong thing. Both *more* context
(distractors) and *less* context (dropped members) lowered accuracy.

**What did NOT hurt:** *prompting* (the bounded reflect pass) and *sampling*
(vote-of-3) are approximately accuracy-neutral (within run-to-run noise) — vote-3
mainly *stabilises* the number rather than raising it. That's the honest ceiling:
the deterministic levers hurt, the prompt/sample levers are neutral, and the
residual error is **reader-bound** (counting via coreference, inference). A
stronger reader moves it; the memory layer does not.

## 6. Positioning

| system | philosophy | LongMemEval-S* |
|---|---|---:|
| Mem0 | auto-extract → vector+graph+KV | ~49% |
| Zep | bi-temporal knowledge graph | ~63.8% |
| **Continuum** | tiered + supersession + bi-temporal, honest measurement | **~74%** |

\*Numbers are not perfectly comparable across setups/readers; treat as
directional. Continuum's edge is temporal/supersession correctness, at
~5k tokens/query (comparable to Mem0's token efficiency).

## 7. Reproducibility

- Scripted benches (no infra, no key): `make bench-all` (~60s).
- LongMemEval-S: `make repro-everything` (needs `OPENROUTER_API_KEY`; the dataset
  is fetched on demand, raw outputs gitignored).
- The full-500 verification, the disproven levers, and the control methodology:
  `findings/roadmap_v3.md` §9.

## 8. Takeaway

Continuum is the state layer agents lack: supersession and bi-temporal recall,
measured honestly, at low token cost. It is *not* a reasoning engine — final
answer quality is bounded by the reader above it. We shipped what the benchmarks
support and documented, in full, the ideas that didn't survive contact with a
same-setup control.
