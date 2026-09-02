# Continuum v4 roadmap — the instrument, the representation, and the moat

*Status: research complete, build proposed · 2026-09-02 · supersedes the
"accuracy is reader-bound" conclusion in `docs/limitations.md` and `docs/report.md`*

Constraint for this cycle, stated up front: **the reader is fixed at
`gpt-oss-120b`.** No flagship model is available for evaluation. Everything
below is designed to hold under that constraint — and the central argument of
this document is that the constraint is **not** what is limiting us.

---

## 0. The one finding that reframes the whole project

`docs/limitations.md` opens with:

> *"Accuracy is reader-bound, not memory-bound. […] that's a reader problem a
> bigger/stronger model addresses, not a memory layer."*

**That claim is falsified.** Hindsight ([arXiv 2512.12818](https://arxiv.org/abs/2512.12818))
reports **89.0% on LongMemEval with `gpt-oss-120b` as the answer generator** —
the exact reader we are pinned to. We are at **~74%**.

A **15pp gap at an identical reader** cannot be a reader bound. It is an
architecture bound. `findings/roadmap_v3.md` §2 actually spotted this a quarter
ago and wrote the right sentence —

> *"their reader reasons over structured, synthesised memory, not raw turns.
> Ours reasons over raw turns […] or, at best, atomic facts."*

— and then §5 built a **numeric aggregator** (counts and sums) and, when it went
net-negative, we generalised the failure of *one* mechanism into the death of
*the whole category*. Re-read §3 of that document with fresh eyes: it explicitly
records that Hindsight's observations are **narrative summaries, not
precomputed counts**, and we built the counter anyway.

**We disproved numeric aggregation. We never tested write-time narrative
synthesis.** That is the open lever, and it is the largest one on the board.

---

## 1. Where the field actually is (and why our positioning table is now a liability)

`docs/report.md` §6 publishes this:

| system | LongMemEval-S |
|---|---:|
| Mem0 | ~49% |
| Zep | ~63.8% |
| **Continuum** | **~74%** |

Those Mem0/Zep figures are from the 2025 literature. As of September 2026 the
landscape is:

| system | reported LongMemEval | reader used | source class |
|---|---:|---|---|
| Mastra Observational Memory | **94.87%** | gpt-5-mini | vendor research page |
| | 84.23% | gpt-4o | vendor research page |
| Mem0 (2026 token-efficient algo) | **93.4%** | unstated | vendor blog |
| Hindsight | **91.4%** | Gemini-3 Pro | peer-reviewed preprint |
| **Hindsight** | **89.0%** | **gpt-oss-120b** | **peer-reviewed preprint** |
| Zep | 63.8% | 2025-era | 2025 literature |
| **Continuum (us)** | **~74%** | **gpt-oss-120b** | our own full-500 control |

Two consequences, both urgent:

1. **Publishing the stale table is now the single biggest credibility risk we
   carry.** Honest measurement is our entire brand. A comparison table that
   flatters us by two-year-old competitor numbers destroys exactly the thing it
   is trying to establish. Fix it in this cycle regardless of what else ships.
2. **The only row that is a fair comparison is the Hindsight `gpt-oss-120b`
   row.** Every other number varies the reader, and per
   [MemDelta (arXiv 2606.29914)](https://arxiv.org/abs/2606.29914) — a
   controlled-evaluation study on LongMemEval-S — swapping *only the embedding
   model* in an otherwise identical pipeline moves accuracy **+6.2pp
   (n=500, p=0.004)**, and swapping the reader flips system rankings outright
   (Gemini gains +14pp from full context; Sonnet gains +31pp from RAG, partly
   because it refuses 63% of full-context queries). Cross-reader leaderboard
   numbers are close to meaningless. **89.0% at our reader is the only target
   that matters.**

### The architectural split the field has settled into

- **Retrieval-per-turn** (Mem0, Zep, MemMachine, **Continuum**): search on every
  query. Unbounded memory size, but the prompt changes every turn — which
  **invalidates prompt caching** and adds retrieval latency.
- **Stable-context observation log** (Mastra OM, and in spirit Hindsight): keep
  a compressed, dated observation log permanently in context. Prompt-cacheable,
  ~10x cheaper, currently top of the benchmark — but **capped by the context
  window**, and supersession is decided by an LLM "Reflector".

We are in the losing camp on accuracy *and* on cost. But the winning camp has a
real, structural weakness, and it is precisely our strength — see §3.

---

## 2. Before any of that: our measuring instrument does not work

This is the prerequisite. Every experiment below is unrunnable until it is fixed.

### 2.1 We pin the provider and the seed and still get ±3–5pp

`make repro-everything` already passes `--openrouter-provider DeepInfra --seed 0`
at `temperature=0`. We still observe ±3–5pp per-category run-to-run variance.
We attributed this to "MoE routing + batch nondeterminism" and treated it as a
law of nature. It is not.

The mechanism is now well characterised: temperature-0 nondeterminism is
dominated by the **batch-size dependence of reduction kernels**, not by sampling.
A request dispatched under a different dynamic batch size takes a different
reduction tree, producing different logits and eventually a different token.
**A seed cannot fix this**, because the seed was never the source. Thinking
Machines Lab demonstrated bit-identical outputs across 1,000 repeated runs by
substituting batch-invariant kernels (RMSNorm, matmul, attention) under vLLM's
FlexAttention backend, at ~61.5% throughput cost; SGLang's later integration
with CUDA graphs brings that to ~34%.

**We cannot buy this from OpenRouter at any price.** It requires owning the
serving stack.

### 2.2 We have no statistical machinery whatsoever

A grep across the entire repository for `mcnemar`, `bootstrap_ci`,
`confidence_interval`, `binomtest`, `scipy.stats`, `p_value`, `wilson` returns
**zero hits**. We compare 73.6% against 75.6% by eye and call it noise. That
call happens to be correct, but we have no instrument that could have told us
otherwise, and no way to detect a real 2pp improvement if we built one.

At n=500 with unpaired runs and ±3–5pp per-category noise, our **minimum
detectable effect is roughly the size of every improvement we would plausibly
ship.** We have been running experiments that were statistically incapable of
answering their own question. That, and not any individual lever, is why v3
produced a quarter of net-neutral results.

### 2.3 What this costs us

`findings/roadmap_v3.md` §9 records the methodology lesson — *"always carry a
same-setup control"* — and it is the most valuable thing in the document. But it
was learned once, by hand, and never encoded. Nothing in CI enforces it. The
next experiment will re-learn it.

---

## 3. The constraint is leverage: `gpt-oss-120b` is open-weight

Being pinned to `gpt-oss-120b` is treated in our docs as a ceiling. It is
actually the most exploitable position we could be in, because **we can own the
weights.** `gpt-oss-120b` is a 116.8B-total / 5.1B-active MoE that serves at
TP=1 with mxfp4 on a single H100. Self-hosting unlocks five capabilities that no
API path provides:

| capability | what it unlocks | currently |
|---|---|---|
| **Batch-invariant kernels** | bit-identical outputs → A/A noise floor of **0.0pp** → 1pp effects become measurable | impossible via OpenRouter |
| **Logprobs** | calibrated abstention (a whole LongMemEval category), confidence-weighted selection instead of majority vote, confidence-gated escalation | unavailable |
| **Prefix caching (APC)** | a stable context prefix becomes near-free → makes the observation-log architecture cheap | defeated by per-query retrieval |
| **`reasoning_effort` low/med/high** (harmony) | a 3-run sweep on a knob we have never touched | not plumbed |
| **Guided / constrained decoding** | schema-valid extraction that *cannot* emit a malformed spec — the exact failure mode that killed temporal codemath at 43% bad specs | not plumbed |

We have never used any of them. The eval harness supports eight providers
(`ollama, groq, nvidia, gemini, openai, openrouter, bedrock, lmstudio`) and not
one of them exposes logprobs, reasoning effort, or a pinned local endpoint for
the answerer. Note that `LMStudioLLM` (`bootstrap_ollama.py:1375`) already
speaks OpenAI-compatible `/v1/chat/completions` against a configurable
`base_url` — **a vLLM adapter is a small diff, not a project.**

Cost sanity check: our retrieved context is ~5–6k tokens/question, so a full-500
run is ~3M prefill tokens. On one rented H100 (~$2–3/hr) that is well under an
hour and a few dollars per run, even after the batch-invariance throughput
penalty. **Self-hosting is cheaper than what we currently spend on OpenRouter,
and it is the only path to a working instrument.**

---

## 4. The plan

Six workstreams. **WS-0 is a hard gate: no number produced before it lands is
admissible as evidence for anything else.**

### WS-0 — Build the instrument (2 weeks, blocking)

| # | deliverable | acceptance gate |
|---|---|---|
| 0.1 | `VLLMLLM` adapter — base-url flag, `logprobs`, `reasoning_effort`, `seed`, guided-JSON passthrough. Extend `LMStudioLLM`. | answers 500 questions end-to-end |
| 0.2 | Self-hosted `gpt-oss-120b` on vLLM, mxfp4, TP=1, batch-invariant kernels (`thinking-machines-lab/batch-invariant-ops` or SGLang equivalent) | **SHA-256 of the output token ids is identical across 1,000 repeats of a fixed prompt at 3 different concurrency levels** |
| 0.3 | `evals/stats.py` — paired McNemar exact test, bootstrap CI on aggregate accuracy, Wilson interval per category, Cohen's κ for judge agreement | unit-tested against known fixtures |
| 0.4 | **Power calculator, printed before every run**: "at n=500 with this design, minimum detectable effect = X pp" | refuses to launch an underpowered A/B without `--i-know` |
| 0.5 | **Paired A/B runner** — identical question set, identical retrieved context ids, treatment is the only delta; **judge only the discordant pairs** | 5–10x judging cost reduction, verified |
| 0.6 | `run.lock.json` manifest — model sha, kernel set, seed, retrieval config, dataset hash, harness git sha | every result file carries one |
| 0.7 | Judge reliability: 100-item human-labelled gold subset; report judge-vs-gold κ and judge self-consistency; dual-judge disagreement rate | judge noise reported **separately** from reader noise |
| 0.8 | **A/A gate in CI** | A/A must return **exactly 0.0pp**. Anything else means the instrument is broken and the run is void. |

Our current A/A noise floor is 0.8pp (`findings/roadmap_v1.1.md`). **The target
is 0.0pp — bit-identical.** That is the difference between guessing and
measuring.

> **This workstream is itself a competitive moat.** MemDelta's entire thesis is
> that the field cannot tell what it is measuring, and MemTrace
> ([arXiv 2606.17328](https://arxiv.org/abs/2606.17328)) argues pooled accuracy
> hides the behaviours that matter. **No memory system currently ships a
> bit-reproducible evaluation harness.** Being first is a stronger and far more
> durable claim than any single benchmark number — and unlike a benchmark
> number, nobody can beat it next month with a bigger model.

### WS-1 — The representation switch: Observation View over a bi-temporal store

The architecture bet, and the one experiment that can plausibly move 74% → mid-80s.

**Build:** a deterministic projection from the LTM store into a **stable, dated,
entity-scoped narrative observation log** — and put *that* in the reader's
context instead of top-k raw turns.

```
current LTM facts (invalidated_at IS NULL)  ─┐
MTM session summaries                       ─┼─► ObservationView ─► stable prompt prefix
entity index                                ─┘    (dated, sorted,
                                                    narrative, scoped)
```

**Why ours is strictly better than Mastra's, not merely equivalent:**

| | Mastra OM Reflector | Continuum ObservationView |
|---|---|---|
| what is superseded | an **LLM decides**, and rewrites the log | **schema decides** — `invalidated_at IS NULL` |
| auditability | the old text is gone | every retired row is retained, bi-temporally |
| recoverability | a bad reflection is unrecoverable | supersession is reversible by construction |
| determinism | LLM-nondeterministic | pure projection — same store, same bytes |
| scale ceiling | **capped by context window** | store is source of truth; the view is a **scoped, paged projection** |

That last row is the answer to the stable-context camp's one structural
weakness. They must fit all memory in context. We do not — the bi-temporal store
holds everything and the view is a deterministic window over it. **We can adopt
the representation that is winning without inheriting its ceiling.**

**Key property to engineer for: `view_etag`.** The view changes only when memory
changes, not per query. Expose the hash so callers can hold a prompt cache
across turns. This converts our worst cost characteristic (cache-hostile
per-query retrieval) into our best.

**Experiments, in order, all on the WS-0 instrument, reader fixed:**
1. ObservationView **vs** current top-k retrieval, paired, full-500, McNemar.
2. Ablate the *narrative* property: narrative observations vs the same facts as
   atomic bullets. This isolates the variable v3 never tested.
3. Ablate scoping: entity-scoped vs flat chronological.
4. No-regression check on the categories we have already solved (single-session
   user/assistant at 91–98%).

**Honest risk:** narrative synthesis is LLM-generated write-time content, so it
can hallucinate. Mitigation is native to our schema and is itself a
differentiator: **every observation carries provenance to its source fact ids**,
and a synthesis whose claims do not resolve to live facts is rejected at write
time. Nobody else in the stable-context camp can do this, because they have no
underlying fact store to check against.

### WS-2 — Reader-side levers that only exist once we own the weights

All cheap, all previously impossible, all measurable on the WS-0 instrument.

1. **`reasoning_effort` sweep** (low / medium / high) — three runs. We have
   never touched this knob. Possibly the highest value-per-hour item in the
   entire plan.
2. **Logprob-calibrated abstention.** LongMemEval scores abstention as its own
   ability. Threshold on answer logprob rather than prompting the model to
   decide. Deterministic, tunable, and sweepable on a held-out split.
3. **Confidence-gated escalation.** Re-ask only the bottom-N% by confidence at
   higher reasoning effort. Bounded extra compute aimed exactly at the error
   mass — as opposed to vote-of-3, which spends 3x on every question and, per
   `docs/report.md`, mainly *stabilises* the number rather than raising it.
   **Replace vote-of-3 with logprob-weighted selection**: same sample budget,
   strictly more information used.
4. **Guided decoding for extraction.** Temporal codemath died because the model
   emitted bad specs ~43% of the time. A JSON-schema-constrained decode cannot
   emit a structurally invalid spec. This does not resurrect codemath on its own
   — semantic errors survive constrained decoding — but it retires the format
   half of that failure, and the honest test is now cheap to run.

### WS-3 — Own the benchmark the field is missing: `MemState-Bench`

Our deterministic benches are **50 supersession scenarios and 20 bi-temporal
timelines, both saturated at 100%.** A saturated benchmark has no signal and
cannot be lost, which means it also cannot be won.

Build a generated, adversarial, **zero-LLM, fully deterministic** suite over
memory *state* semantics — the axis where our schema is genuinely unmatched and
where no reader variance can contaminate the result:

- out-of-order arrival; retroactive correction; re-assertion after retraction
- conflicting simultaneous sources; partial retraction; equal-timestamp tie-breaks
- entity merge and split; coreference-driven identity change
- cross-namespace isolation (a leak is a **failure**, not a lower score)
- idempotent replay; crash recovery mid-promotion; clock skew; DST/timezone edges
- erasure-then-query (see WS-4.3)

Property-based generation (Hypothesis) to 1,000+ cases. Adopt MemTrace's
**"knowledge point"** unit: probe the *same* fact repeatedly while varying memory
age, question type, and evidence condition, instead of scoring isolated rows.

Then publish scores for **Continuum, Mem0, Zep, Letta, and raw pgvector**.

Why this is the moat: it costs nothing to run (no LLM, so it lives in CI
alongside `bench-all`), it is where our architecture is strongest, and
**append-only stores and vector databases cannot score well on it without
re-implementing bi-temporal supersession.** That is a defensible category, and
unlike a LongMemEval number it does not decay when someone ships a better model.

Secondary target: **BEAM**, which runs at 1M and 10M token scales and whose ten
categories include *contradiction resolution*, *knowledge update*, *event
ordering*, *temporal reasoning* and *abstention* — five categories that are
literally what supersession and bi-temporal columns were built for. Current SOTA
is 0.79 @ 100K and 0.67 @ 10M. **BEAM is a better fit for our architecture than
LongMemEval has ever been**, and the 10M scale is where the stable-context camp's
context-window ceiling becomes fatal and our store-of-record design wins.

### WS-4 — Production engineering (the credibility layer)

Concrete gaps, each verified against the current tree:

| # | gap | evidence | fix |
|---|---|---|---|
| 4.1 | **Zero observability.** No OpenTelemetry, no Prometheus, no structured logging — 38 files on stdlib `logging`. | grep: 0 hits for `opentelemetry`/`prometheus`/`structlog` in `continuum/` | OTel spans per tier (retrieve / score / rerank / assemble / promote), RED metrics, exemplar traces |
| 4.2 | **No row-level security.** Namespace scoping (migration 005) is an application-level `WHERE`. One missing clause is a cross-tenant leak. | `migrations/005_namespace_scoping.sql`; no `ROW LEVEL SECURITY` anywhere | Postgres RLS on `memory_nodes` + a test asserting a deliberately unfiltered query **still** cannot cross namespaces |
| 4.3 | **No hard erasure — GDPR Art. 17 / CCPA cannot be satisfied.** `forget` sets `invalidated_at`; the README states plainly that *nothing is deleted*. The embedding row survives. | `continuum/memory.py:491`; no `DELETE FROM memory_nodes` in the tree | `purge(subject)`: real `DELETE`, audit tombstone, index maintenance, and a test asserting the vector is **gone** |
| 4.4 | **No auth on the MCP HTTP transport.** `continuum-mcp --http` binds a port with no bearer, no mTLS, no rate limit. | `continuum/mcp/server.py` — no auth path | bearer/mTLS + per-namespace rate limiting before the HTTP transport is recommended anywhere near production |
| 4.5 | **Integration testing is thin.** 87 unit test files vs **6 integration and 2 acceptance**. No load, soak, or chaos test in the repo (the 6h soak was run by hand and never encoded). | `find tests -name 'test_*.py'` | k6/Locust ingest+recall profile in CI; kill-Postgres-mid-write recovery test; backpressure under queue saturation |
| 4.6 | **Vector index is untuned and unpublished.** `m=16, ef_construction=64` — conservative defaults. `halfvec` is commented out. We already found `ef_search 400→1000` was the real recall fix, which proves this surface matters and is unexplored. | `migrations/001_ltm_schema.sql:366`, `:504` | a published **recall@k vs p99-latency curve** over (m, ef_construction, ef_search) × {vector, halfvec} at 1e5 / 1e6 / 1e7 rows |
| 4.7 | **No SLOs.** | — | publish p50/p99 recall latency, ingest throughput, and cost per 1k turns; regression-gate them in CI exactly as `check_bench_regressions.py` already gates correctness |

4.1, 4.2, 4.3 and 4.4 are the four that block an enterprise buyer outright.
4.3 in particular is not a missing feature but a **design decision that
currently makes compliance impossible** — soft-retire is the correct default,
but it must not be the only option.

### WS-5 — Pay down the eval harness

`evals/longmemeval/bootstrap_ollama.py` is **6,477 lines**. The `evals/` tree is
**19,842 LOC against a 22,628 LOC product.** The research harness is nearly the
size of the thing it measures, and it is where all of our reproducibility risk
is concentrated: a reviewer cannot verify an experiment they cannot read.

Decompose into: dataset loader / context builder / reader adapter / judge /
stats / runner. Target under 500 lines per module. This is not cosmetic — it is
a precondition for anyone outside the project trusting a number we publish, and
we are asking them to trust our numbers as the core of our positioning.

---

## 5. Predicted outcomes — honestly bounded

Built lever by lever from the **73.8% same-setup control**, reader fixed at
`gpt-oss-120b`:

| stack | predicted | confidence |
|---|---:|---|
| control (today) | 73.8% | measured |
| + WS-0 instrument | 73.8% | **no accuracy change — it makes everything below measurable** |
| + `reasoning_effort=high` | 74–78% | low confidence, cheap to find out |
| + ObservationView (narrative, entity-scoped) | **80–86%** | **the bet; the widest error bar here** |
| + logprob abstention + confidence-gated escalation | +1–3pp | medium |
| Hindsight parity at our reader | 89.0% | reachable only if WS-1 lands near its top end |

**Call: 82–87% is the realistic stacked target; 89% requires WS-1 to work about
as well as it does for Hindsight.** State the error bar this way in public and
we keep the one asset that distinguishes us.

And the honest counterweight: **the accuracy number is the least defensible
thing we can chase.** Anyone can beat it next quarter with a better model. WS-3
and WS-4 cannot be beaten that way. If we ship exactly one workstream, it should
not be the one that chases the leaderboard.

---

## 6. Sequencing

```
Weeks 1–2   WS-0   instrument            [BLOCKING — gate: A/A == 0.0pp]
Weeks 2–3   WS-4.1–4.4 + WS-2.1          observability, RLS, purge, MCP auth,
                                          reasoning_effort sweep (runs in parallel;
                                          none of it depends on WS-1)
Weeks 3–6   WS-1   ObservationView       the architecture bet
Weeks 4–8   WS-3   MemState-Bench        zero-LLM, runs in CI, no reader dependency
Weeks 6–8   WS-2.2–2.4                   abstention, escalation, guided decoding
Ongoing     WS-5   harness decomposition
Week 1      positioning table correction  ← do this immediately, independent of everything
```

**Cost:** one rented H100 for the WS-0/WS-1/WS-2 experiment window. At ~$2–3/hr
and well under an hour per full-500 run, this is materially **cheaper** than the
OpenRouter spend it replaces — and it is the only way to get a working
instrument at any price.

---

## 7. What to stop doing

1. **Stop treating ±3–5pp as a law of nature.** It is a serving-stack property
   we have the option to remove, and it has silently invalidated a quarter of
   experiments.
2. **Stop running unpaired A/Bs.** Paired + McNemar + judge-only-the-discordant
   is strictly better on cost *and* power. There is no reason to run the old way.
3. **Stop publishing the 2025 competitor table.** It reads as flattery-by-stale-data
   and it undermines the only thing that makes this project distinctive.
4. **Stop generalising from one disproven mechanism to a whole category.**
   Numeric aggregation failed. Write-time narrative synthesis is a different
   mechanism, is what the current SOTA actually does, and remains untested here.
5. **Stop optimising a saturated benchmark.** 100% on 50 scripted scenarios is
   not a result. Build a benchmark that can be lost.

---

## 8. The one-paragraph version

We concluded we were reader-bound. Hindsight scores 89.0% on the same
`gpt-oss-120b` reader we are pinned to, so we are not — we are bound by feeding
the reader retrieved raw turns while the state of the art feeds it synthesised
narrative observations. We cannot currently detect an improvement smaller than
our own noise floor, because we buy inference from a router that cannot give us
batch-invariant kernels, and because the repository contains no statistical
machinery at all. So: own the weights, make the A/A floor exactly zero, then
make the one architectural change we talked ourselves out of a quarter ago —
and build the deterministic state-correctness benchmark and the production
hardening that no competitor can beat by shipping a bigger model. Being pinned
to an open-weight reader is not our constraint. It is the reason all of this is
available to us.
