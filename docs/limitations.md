# Known limitations (read this)

Continuum's pitch is honest measurement, so here is what it does *not* do well,
and what we tried that **didn't work**. Negative results are part of the product.

## Accuracy is reader-bound, not memory-bound

On LongMemEval-S, retrieval recall is ~98% — the right context is almost always
retrieved. The residual error is the *reader* (the LLM answering over that
context) failing, not the memory layer. Concretely, with a `gpt-oss-120b` reader:

- **Counting / aggregation is the floor (~63% multi-session).** "How many tanks
  do I have?" requires enumerating a set with coreference ("community tank" =
  "main tank" = one tank) and excluding hypotheticals. The reader can't reliably
  do this, and — importantly — **no memory-side trick fixed it:**
  - a write-time synthesis/aggregation layer (count in code): net-negative;
  - a deterministic answer *router* (return the code-computed count): 82% wrong
    when it fired — it answered duration/recall questions with garbage counts;
  - *evidence distillation* (filter to the relevant turns before answering):
    net-negative — it **undercounts**, because counting needs completeness and
    the filter drops members.
  Both *more* context (hurts: distractors) and *less* context (hurts: dropped
  members) lower accuracy. Counting is genuinely reader-bound.

- **Temporal arithmetic/ordering.** Computing date deltas or ordering events in
  code (the model emits a spec, code computes) was also **net-negative**: the
  model emits bad specs (wrong dates) ~43% of the time, so the code faithfully
  computes a confident wrong answer.

**The pattern:** you cannot bolt deterministic machinery onto `gpt-oss`'s
intermediate outputs — its counts, extractions, and date-specs are too
unreliable. The levers that *did* help were **better prompting** (a bounded
preference-application pass: +20pp on preference questions) and **sampling**
(vote-of-3 self-consistency: ~+2pp overall, and it tames run-to-run noise).

## The benchmark number has real variance

`gpt-oss-120b` (via OpenRouter, MoE routing + batch nondeterminism) has ~±3–5pp
per-category run-to-run variance. A single run is not a stable estimate. On a
**same-setup control** (no reflect, no synthesis), the full LongMemEval-S judged
number is ~73.8%; the best mandate-clean config (bounded reflect + vote-3) is
~75.6%. A higher single-run figure elsewhere is a favorable draw, not a
reproducible headline — always compare against a same-setup control and average
noise before claiming a delta.

**Methodology note (the lesson we paid for):** measuring "recovery" only on a
set of known failures *structurally overstates* gains — a failure can improve or
stay failed, but it can never regress, so you see every win and none of the
damage. Only a full same-setup run reveals the true (often ~neutral) effect.
Carry a control.

## Retrieval

- **Recall@4 on the synthetic 200-session corpus is tied with naive cosine** —
  the recency signal doesn't help there. Continuum's retrieval edge shows up on
  real multi-session workloads, not this synthetic bench.
- More retrieved context is not always better: past a point, distractors lower
  answer accuracy. Precision matters more than raw recall.

## Product surface

- **`Memory.in_memory()` is for demos/tests.** Full supersession (the
  ADD/UPDATE/DELETE decider) and valid-time bi-temporal history live on the
  **Postgres** path. In-memory `current`/`timeline` are best-effort over the
  session's stores.
- **Not a reasoning engine.** Continuum is the memory layer; the quality of
  final answers depends on the reader you put above it.

## What this means for you

If your bottleneck is *"the agent forgets / uses stale facts"* — supersession
and bi-temporal recall are exactly the wins (100% on the scripted supersession
and as-of-date benches). If your bottleneck is *"the model can't count or do
date math over what it retrieved"* — that's a reader problem a bigger/stronger
model addresses, not a memory layer.

*(Sources: `findings/roadmap_v3.md` §9 for the full-500 verification, the
disproven levers, and the control methodology.)*
