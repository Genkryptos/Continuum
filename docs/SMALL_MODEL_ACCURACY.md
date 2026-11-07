# Making small models accurate: the mechanism

*Research note, 2026-08-11. Companion to `docs/ACCURACY_AND_BENCHMARK_PLAN.md`.*

**The claim this note defends:** Continuum's reason to exist is that a *small*
model plus a temporally-correct store should match a large model plus naive
retrieval. That is a stronger and more useful claim than "we have bi-temporal
metadata", and the repo already contains partial evidence for it.

The mechanism is not better retrieval and not more reasoning scaffolding. It is:

> **Convert reasoning problems into query problems.** Every question a
> deterministic store can *compute* is a question the answerer does not have to
> *reason* about — and reasoning is exactly the capacity a small model lacks.

---

## 1. What the repo already proves

Three measured results, all internal, that point the same way:

1. **The failure is reasoning, not retrieval.** `wrong_retrieval` outnumbers
   `missing_fact` by ~10× across every run — including the long-context run,
   which has 100% recall by construction and *still* logs 334 `wrong_retrieval`
   failures (`findings/longmemeval_2026-05.md`). When the model fails, it fails
   with the right context in front of it.

2. **Six retrieval levers were flat.** Optimizer chain, cross-encoder, scorer
   weights, long-context, decomposition, WikiAdapter: none moved more than ~+3pp.
   `mode=long` scored *identically* to top-k=4 (32.2% = 32.2%).

3. **Where the store computes, Continuum already wins.** LOCOMO hard slice:
   temporal **33.3% vs Mem0's 4.2%** (`findings/reasoning_loop_2026-06.md` §6),
   because Mem0 distills to facts and discards the timeline while Continuum keeps
   sequence. And `--use-ltm` supersession took knowledge-update to **98.7%
   recall**. Those are the two categories where a *store operation* replaced a
   *reasoning operation*.

That third point is the whole thesis in miniature. It has never been generalized.

---

## 2. The correction: scaffolding hurts small models *more*, not less

`findings/reasoning_loop_2026-06.md` §4 cut the IterativeReasoner after it lost
badly (88% direct vs ~50% iterative single-hop; 16% vs 6% multi-hop), and left
this note:

> *"The class stays in the tree … as a possible lever for **weaker** answerers."*

**The literature says that is backwards.** ["More Is Not Always Better:
Cross-Component Interference in LLM Agent Scaffolding"](https://arxiv.org/html/2605.05716)
measures a *capability gradient* — the gap between the best component subset and
the fully-scaffolded agent, by model size:

| model | gap between optimal subset and full scaffold |
|---|---:|
| 8B | **32%** |
| 70B | 19% |
| Claude Haiku | ~0% |

Small models are hurt *most* by added components. The mechanisms named are
bandwidth competition, signal dilution, and context-window crowding — all of
which get worse as capacity shrinks. Their practitioner guidance: start minimal,
don't add greedily (56% of combinations violate submodularity, so incremental
addition is unreliable), and note that **tool use alone captures ~70% of all
scaffold value**.

So the plan for small models is *not* to reinstate the reasoning loop. It is to
delete reasoning work, and replace it with a tool call. Which is convenient,
because that is the one component class the paper says pays.

---

## 3. The mechanism: typed temporal queries instead of temporal reasoning

The external work on temporal QA converges on the same answer. Query-based KGQA
generates an *executable query* rather than an answer, and is reported to improve
"multi-hop and temporal QA with small language models" specifically; the surveys
note LLMs "perform poorly in symbolic temporal reasoning, especially in
multi-step tasks" ([2507.11954](https://arxiv.org/abs/2507.11954),
[TKGQA survey](https://arxiv.org/pdf/2406.14191)). The small-model memory work
independently reaches the write-time version of the same idea: push structured
extraction, compression, and indexing to ingest so query time is lookup, and
"structured memory organization compensates for small models' reduced reasoning
capacity" ([2604.07798](https://arxiv.org/pdf/2604.07798)).

**Continuum is unusually well-positioned here because the temporal algebra already
exists.** `current()`, `as_of`, `timeline()`, `valid_from`/`valid_to` are built,
tested, and 100%-correct on the bench suites. What is missing is that the
answerer cannot *reach* them — it gets dated turns and is asked to reason.

### The shape to build

```
question → [small model, constrained decode] → typed query
                                                  ↓
                                        deterministic execution
                                        (as_of / timeline / current / count)
                                                  ↓
                              result rows → [small model] → verbalize
```

The model's two jobs become **parameter extraction** and **verbalization**. Both
are small-model-tractable. Neither is multi-hop reasoning. The correctness comes
from the store.

This also fits the rule already load-bearing in the harness
(`reasoning_loop_2026-06.md` §5): reasoning models for answering, fast
non-reasoning models for structured/classification sub-tasks. Query translation
is a structured sub-task — schema-bounded, constrained-decodable, and
*deterministically validatable* (a malformed query is caught and retried; a wrong
free-form inference is not).

### Which categories this converts

| category | today | what the store could compute instead |
|---|---:|---|
| temporal-reasoning | 41.4% | `as_of` / `timeline` — "what was true at T" is a query, not an inference |
| knowledge-update | 51.3% | already partly converted via `current()` — 98.7% recall |
| multi-session aggregation | 54.9% | `COUNT` / `GROUP BY` over episodes — "how many times did I…" is arithmetic |
| single-session-* | 94–98% | leave alone, already solved |
| multi-hop composition | — | **not convertible**; genuinely reasoning-bound |

The honest boundary: multi-hop composition stays the answerer's problem. The
claim is not "small models get everything", it is "small models get everything
the store can compute, which is more than the field currently assumes."

---

## 4. What to build, in order

**S1. Typed query interface over the temporal algebra.** Expose `current`,
`as_of`, `timeline`, and a bounded aggregate as a callable schema. No new storage
— this is a facade over what `PostgresLTM` already does.

**S2. NL → typed query translator.** Fast non-reasoning model, constrained
decoding, schema-validated with deterministic retry. Falls back to today's
direct-retrieval path when translation fails — so the floor is the current
number, by construction.

**S3. Answer-from-rows verbalizer.** Small model, rows in, sentence out.

**S4. The model-size ladder — this is the actual paper result.** Run the same
pipeline across a size ladder (e.g. 8B / 20B / 120B) in two arms: direct
retrieval vs typed-query. The claim is that the **typed-query arm flattens the
curve** — that small models lose much less accuracy than they do under direct
retrieval. A flat curve is the proof that memory substitutes for model scale.

Note S4 is the deliverable. S1–S3 exist to make S4 measurable.

---

## 5. How this could fail (state before running)

- **Translation is the new bottleneck.** If the small model cannot reliably emit a
  valid typed query, accuracy moves from "reasoning failure" to "translation
  failure" and nothing is gained. *Mitigation:* constrained decoding + schema
  validation + fallback to direct retrieval; measure translation validity as its
  own metric, separately from end accuracy.
- **Question coverage is narrower than hoped.** If only a small slice of
  temporal-reasoning questions map onto typed queries, the ceiling barely moves.
  *Mitigation:* hand-classify a 50-question sample for convertibility **before**
  building S2. If under ~40% convert, the lever is weaker than this note assumes
  and should be re-scoped.
- **CCI applies to us too.** The typed-query path is itself a component. If it is
  added *on top of* the existing pipeline rather than replacing a branch of it,
  the same interference the IterativeReasoner suffered may recur. *Mitigation:*
  route, don't stack — a question goes down exactly one path.
- **The ladder shows a flat curve for the wrong reason.** If accuracy is flat
  because the pipeline is bottlenecked somewhere shared, that is not the claim.
  *Mitigation:* the direct-retrieval control arm must show the expected *steep*
  degradation with model size. If both arms are flat, the experiment is void.

---

## 5a. Design consequence: run the head-to-head at *small* answerer size

If Continuum's accuracy comes from the store and Mem0's comes from the answerer
reasoning over retrieved facts, then **the size of the shared answerer decides how
big Continuum's edge looks.** At gpt-5 the answerer can reason its way past a
missing timeline, and the architectures converge. At 8B it cannot, and the
difference between "compute the answer" and "infer the answer" is maximally
exposed.

This inverts the usual instinct to benchmark at the strongest available model:

| shared answerer | expected Continuum − Mem0 gap |
|---|---|
| frontier (gpt-5) | smallest — model compensates for the missing timeline |
| mid (gpt-oss-120b) | moderate — today's measurements live here |
| small (8B–20B) | **largest — the architecture is the only thing left** |

So Phase 2 of the benchmark plan should run the head-to-head **across the ladder**,
not at one size, and report the gap *as a function of answerer size*. That single
curve is a stronger claim than any single-cell win: it says the advantage is
structural, and it predicts where it matters most (cheap local models — the actual
deployment case).

It also resolves the apparent tension between "beat Mem0" and "prove the
small-model claim". They are the same experiment.

## 6. Relationship to the benchmark plan

This note changes `docs/ACCURACY_AND_BENCHMARK_PLAN.md` Phase 1:

- **A5** (wire supersession into LongMemEval) remains first — it is the cheapest
  test and a prerequisite for measuring anything else honestly.
- **S1–S4 become the main Phase-1 line**, ahead of A2/A3/A4, because they attack
  the category ceiling rather than trimming points off retrieval.
- **A1** (annotate valid_from/valid_to into context) is the *fallback* form of the
  same idea: if the model must reason, at least give it dates and the rule. Keep
  it as the control arm for S1–S3.
- The multi-hop categories stay written off. That is unchanged and correct.

---

## Sources

- [More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding](https://arxiv.org/html/2605.05716)
- [Lightweight LLM Agent Memory with Small Language Models](https://arxiv.org/pdf/2604.07798)
- [The benefits of query-based KGQA systems for complex and temporal questions in LLM era](https://arxiv.org/abs/2507.11954)
- [Temporal Knowledge Graph Question Answering: A Survey](https://arxiv.org/pdf/2406.14191)
- [State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
