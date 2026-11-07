# Architecture — what the failure data says to build

*2026-09-07 · branch `experiment/lexical-channel-gating` · reader pinned
`openai/gpt-oss-120b` · control run `results/v3_baseline_control/judged.json`*

Companion to `findings/roadmap_v4.md`, which asks what to build next. This
asks a narrower question — *what does the error data permit us to build?* —
and answers it from the 500 rows of the same-setup control rather than from
a competitor's headline number.

Every figure below is reproducible from files in this repository. Where a
claim rests on inspection of individual failures rather than on a metric, it
says so.

---

## 1. The measurement that decides the architecture

The control's failure taxonomy is a single bucket:

| bucket | n |
|---|---:|
| `wrong_span` | 130 |
| `missing_fact` | 1 |
| everything else | 0 |

**114 of 131 errors occur with *every* answer-bearing session retrieved**;
130 with at least one (which is all `wrong_span` asserts — for a
multi-session question needing four gold sessions, three of four still lands
in that bucket). So 17 errors carry *some* session-level retrieval deficit:
an upper bound of 3.4pp, and the per-category recall column locates it in
multi-session and temporal-reasoning. Per category:

| category | n | correct | wrong | pp if perfect | session recall |
|---|---:|---:|---:|---:|---:|
| multi-session | 133 | 83 | 50 | 10.0 | 0.968 |
| temporal-reasoning | 133 | 100 | 33 | 6.6 | 0.976 |
| knowledge-update | 78 | 50 | 28 | 5.6 | **1.000** |
| single-session-preference | 30 | 18 | 12 | 2.4 | 0.967 |
| single-session-user | 70 | 64 | 6 | 1.2 | 1.000 |
| single-session-assistant | 56 | 54 | 2 | 0.4 | 1.000 |
| **total** | **500** | **369** | **131** | **26.2** | **0.981** |

Knowledge-update — the category this store was designed for — retrieves
**every** gold session and gets 28 wrong. No retrieval change can move it.

This is not a new result, it is a confirmed one.
`findings/longmemeval_2026-05.md` §6 recorded six retrieval levers and their
outcomes: optimizer chain 0pp, cross-encoder reranker 0pp, scorer-weight
tuning −2pp, **full long-context identical to top-k=4 (32.2% = 32.2%)**,
decomposition +3pp, wiki adapter regressed. It also drew the conclusion this
document restates four months later:

> "The one experimental signal that *might* break the ceiling — applying
> Continuum's real promotion pipeline … — was never wired into this eval."

---

## 2. The architecture is not the thing under measurement

Four findings, each verified against the tree.

**Supersession has never fired in a LongMemEval run.** From the in-flight
work in `evals/longmemeval/continuum_ltm_store.py`: the store upserted before
deciding, so every candidate found its own just-inserted row among its
neighbours and returned NOOP instead of comparing against the older
contradicting fact — *"why every `--use-ltm` run reported supersessions=0
with ~all ADDs."*

**The headline config stores whole turns, not facts.** `make
repro-everything` passes `--no-llm-promoter`, which routes to
`_heuristic_facts()`, which appends the entire user turn body as a `Fact`.
`[CURRENT FACT]` in the reader's context therefore marks a raw turn.

**`current()` and `timeline()` are never called by the eval.** Zero call
sites in `evals/longmemeval/`. The resolution API exists
(`continuum/memory.py:567`) and ships as an MCP tool; the benchmark does not
touch it.

**The eval and the product are different retrieval code.** Product:
`Retriever` → `_graph_expand` → `Scorer(rel/imp/rec/conf)` → policy filter.
Eval: `SessionAwareSemanticRetriever` / `BM25HaystackRetriever` /
`HybridRetriever`. They share `BM25Index` and `rrf` and nothing else. Graph
expansion has **0 references** in anything the eval executes.

So there are three disjoint evidence bases:

| evidence base | measures | on | result |
|---|---|---|---|
| `bench/` | the product architecture | scripted scenarios | saturated, 100% |
| LongMemEval | a hybrid retriever over raw turns | real data | 73.8% |
| — | the product architecture | real data | **does not exist** |

---

## 3. The schema gap that makes deterministic resolution impossible

`memory_nodes` is `(id, layer, text, embedding, kind, confidence, importance,
valid_from, valid_to, invalidated_at, source_ids, tags)`. There is **no
subject and no attribute column**; attribute lives in `tags->>'attribute'`
(`continuum/stores/postgres/ltm.py:516`).

`docs/architecture.md:96` draws the table as
`subject | attribute | value | superseded_by`. The document describes an
architecture the store does not have.

The consequence is recorded in `continuum/promotion/attribute_extract.py`:

> "`current(subject, attribute)` is only exact for facts stored with an
> attribute (else it falls back to fuzzy retrieval, **which ranks a stale
> value above its correction**)."

Without an attribute key, "does this contradict that?" cannot be a key
collision. It is a regex over `no longer|used to|moved to`, or an LLM call
per fact. Both are what exists, and both are why the mechanism could sit
inert for months without anyone noticing.

**A unique partial index on `(namespace, subject, attribute) WHERE valid_to
IS NULL` makes supersession a database invariant rather than an inference.**
That is the highest-leverage schema change available.

---

## 4. What the 131 failures actually are

Classified by inspecting answer/expected pairs, then partly automated (§6).

**Knowledge-update (28), all at recall 1.000**
- **~14 stale or direction-inverted.** `Chicago` vs `the suburbs`; `Hawaii`
  vs `Paris`; `$350,000` vs `$400,000`; `8:30 am` vs `7:30 am`. Four are
  *inversions* — the model holds both values and assigns them to the wrong
  times ("you switched to **more** water" when the answer is less; "you
  **decreased** your coffee limit" when it increased; a previously/now pair
  swapped whole). Not a retrieval failure and not a reasoning failure: a
  **labelling** failure.
- ~12 counting or duration.
- 2 abstention — confabulated where the expected answer is "not enough
  information" (one answered about *footballs* where the user collects
  autographed *baseballs*).

**Temporal-reasoning (33)**
- ~8 ordering / enumeration (one answer missed JetBlue from a four-airline
  list — a completeness miss).
- ~8 raw date arithmetic (`26` vs `21` days; `0` vs `21` days).
- **~5 false abstentions** — "The conversation doesn't specify who gave you
  the jewellery" when the answer (`my aunt`) was in the retrieved context.

**Multi-session (50)**
- **40 of 50 failed with `partial_recall == 1.0`.**
- Among numeric answers: **20 undercount vs 7 overcount** — systematically
  low (2 vs 3, 1 vs 2, 3 vs 4, 1 vs 4).

**Abstention as a cross-cutting ability**: 30 of the 500 questions are `_abs`
rows. They score **63.3% against 74.5% on everything else** — an 11-question
block worth 2.2pp, spread over four categories.

---

## 5. The reframing: two populations, one label

A completeness hypothesis for the undercount — that the `--turns-per-session
6` and `--session-top-k 12` caps were cutting set members — was tested
against the dataset and **refuted**:

| | max gold turns in one session | questions needing >12 gold sessions |
|---|---:|---:|
| all 500 rows | **3** | **0** |

The caps are not structurally cutting gold evidence. But inspecting the gold
turns to establish that surfaced something better. The two categories filed
under "counting" are not the same problem:

**Knowledge-update "counting" is not counting.** The gold evidence for *How
many Korean restaurants have I tried?* is two turns:

```
[2023/08/11] "I've tried three different ones recently"
[2023/09/30] "I've tried four different ones so far"      ← answer: four
```

Nothing needs enumerating. Both turns state a complete count; the answer is
whichever is **later**. The model returned `three`. Same shape for the H&M
tops (`three` → five) and the MCU films (`4` → 5). These are supersession
failures wearing a number.

**Multi-session counting is genuinely enumerative.** Its gold turns each
contribute *one member* — "I'm having issues with my nasal spray prescription
from Dr. Patel" … "I just got diagnosed by an ENT specialist, Dr. Patel" —
with coreference to resolve (one doctor, mentioned twice) and duplicates to
drop. Five gold turns, expected answer three.

`docs/limitations.md` files both under *"counting / aggregation is the
floor … counting is genuinely reader-bound."* That verdict looks sound for
multi-session and wrong for the knowledge-update subset. It also explains a
standing negative result better than "counting is hard" does: **v3's numeric
aggregator was built to enumerate members, which is the right tool for
multi-session and the wrong tool for the knowledge-update rows it was tested
on.** The 58-row counting-failure set mixed both populations.

---

## 6. The instrument (landed with this document)

`evals/longmemeval/evidence.py` — two metrics at the granularity the failures
live at, wired into `RowResult`, `by_question_type`, and the failure
taxonomy.

**`gold_turn_recall`** — LongMemEval flags the individual answer-bearing
turns (`has_answer`). Session recall scores a session as recalled when *any*
of its turns survives; this counts the turns. New bucket
`missing_gold_turn` separates "the evidence turn never arrived" from "the
evidence was in front of the reader", which is the half of `wrong_span` that
is still a retrieval failure.

**`supersession_verdict`** — `"stale"` when the answer carries a value only
the *superseded* gold turn holds and none of the value that replaced it.
Conservative by construction: it fires only when the expected answer confirms
the later turn, and abstains otherwise.

Measured on the control:

| | n |
|---|---:|
| stale verdicts on **wrong** answers | **15** (14 knowledge-update, 1 temporal) |
| stale verdicts on **correct** answers (false positives) | **0** |
| fresh verdicts | 61 |
| questions receiving any verdict | 76 / 500 |

15 questions — 3.0pp of the benchmark — are now *provably* answers that took
the retired value with the current one also in context. Treat it as a **lower
bound**: the detector is numeric-first with a guarded lexical fallback, so
non-numeric supersessions it cannot decide are abstained on, not counted.

`stale_fact` previously keyed off marker words in the matched claim and read
**0 on every run ever recorded**. That path is removed: it tested the
retrieved span's phrasing rather than the answer.

---

## 7. The architecture

**Change the memory→reader interface from `List[Item]` to `ResolvedState`.**

Everything above says one thing: retrieval finds the evidence and then hands
the reader an unresolved, unordered, unlabelled pile. The 14 stale answers,
the inversions, the false abstentions are the reader doing resolution work
the store should already have done and can do deterministically.

**L1 — Turn the mechanism on.** The in-flight `continuum_ltm_store.py` fix.
Every number below is meaningless while supersessions = 0.

**L2 — The EAV spine.** Promote `subject` and `attribute` out of `tags`
into columns; add the unique partial index on live
`(namespace, subject, attribute)`. Supersession becomes a write-time key
collision the database enforces.

**L3 — The projection.** A pure SQL function from store to dated, resolved,
entity-scoped view:

```
## Current state (as of 2023-05-20)
- residence   = Boston   since 2023-04-02  (was: NYC, 2021-06-01 → 2023-04-02)
- coffee.limit = 2 cups  since 2023-05-11  (was: 1 cup)

## Timeline
2023-04-02  moved NYC → Boston
2023-05-11  raised coffee limit 1 → 2

## Supporting turns
[2023-04-02] [user] I just moved to Boston...
```

Three properties carry the design. It **shows the transition rather than
hiding it** — which is what the inverted-direction failures need, and what a
plain `WHERE superseded_by IS NULL` filter would destroy. It is a
**deterministic projection**: same store, same bytes, no LLM, reproducible
under any serving stack. And it is **stable per subject across queries**, so
it is prefix-cacheable.

**L4 — Shape routing keyed on the data, not the question.** v3's router
classified the *question* by regex and returned a code-computed count; it was
82% wrong. Invert it. If `current(subject, attribute)` returns one live value
with a supersession chain, the view is authoritative. If it returns a set of
*n* members with no supersession, hand over the **complete set,
unsummarised** — the evidence-distillation result says lossy projection
undercounts. The store knows which shape it holds; the question classifier
never did.

**Why this is defensible.** Mastra's reflector has an LLM decide supersession
and rewrite the log; Hindsight synthesises narrative observations. Both are
nondeterministic, unauditable, irreversible, and capped by the context
window. A SQL projection over a bi-temporal store is reproducible
byte-for-byte, retains every retired row, answers "what did you believe on
March 3rd", and pages over an unbounded store.

---

## 8. The honest ceiling

Perfect knowledge-update (+28) and perfect temporal-reasoning (+33) is +61
questions: **73.8% → 86.0%**. Hindsight parity at 89.0% needs +76, so it
requires **at least 15 of the 50 multi-session enumeration errors** as well.
There is no path to 89% that routes around enumeration.

From the failure classification rather than a stack of assumptions:
~14 stale/inverted + ~8 ordering + ~5 false abstention + a few preference
≈ 30 questions ≈ **+6pp → ~80%**. If enumeration yields to complete-set
delivery, add ~20 → **~84%**. If it does not, ~80% is the architecture
ceiling here and the remainder belongs to the reader.

---

## 9. Ablation ladder

Each rung deterministic, one run each, paired against the same control,
reported per-category, with `gold_turn_recall` and `stale_answers` beside
accuracy.

| rung | change | predict |
|---|---|---|
| A0 | control | 73.8% |
| A1 | chronological order + dedup (zero LLM) | temporal ordering moves |
| A2 | real supersession + `was X → now Y` annotation (needs L1) | **`stale_answers` → 0**; KU moves |
| A3 | typed EAV resolved-state header (needs L2) | KU, preference |
| A4 | as-of scoping to `question_date` | temporal |
| A5 | complete-set delivery on set-shaped questions (needs L4) | multi-session |

A2 has a sharper acceptance test than accuracy: **the 15 measured stale
answers should go to zero.** If they do not, supersession is not reaching the
reader and no accuracy number from that arm means anything. If they do and
accuracy does not move, the architecture thesis is wrong — and that is worth
knowing for one run rather than one quarter.

---

## 10. What not to build

**Retrieval.** Six independent results say the same thing: reranker 0pp,
optimizer 0pp, scorer weights −2pp, full long-context identical to top-k=4,
decomposition +3pp, and the ef_search work that ended with the planner
abandoning the index (`82a7802`). With session recall at 98% and 130 of 131
errors downstream of retrieval, there is at most ~1pp there.

**A benchmark that cannot be lost.** `bench/` supersession and bi-temporal
both pass at their acceptance bar on scripted scenarios. They demonstrate the
mechanism; they cannot discriminate designs.

---

## 11. Open, and cheap

1. **Re-derive the reader-bound verdict per population.** `docs/limitations.md`
   currently generalises one conclusion across two different problems (§5).
2. **`docs/limitations.md` and `docs/report.md` §6 contradict each other**
   — one says reader-bound, the other architectural. Both are shipped.
3. **Extend the stale detector to non-numeric supersessions** without losing
   the 0-false-positive property. Today's 15 is a floor, not a count.

---

## 12. P0 result — measured 2026-09-12

The A5 arms ran on all 78 knowledge-update questions. **The gate failed, and
the reason is more useful than a pass would have been.**

| condition | acc (substring) | stale | promoter path | LTM facts | items to reader |
|---|---:|---:|---|---:|---:|
| control | 56.4% | 14 | heuristic (whole turns) | 271 | 73 |
| arm A `invalidate` | 50.0% | 13 | llm (atomic facts) | 933 | 20 |
| arm B `bitemporal` | 52.6% | 15 | llm (atomic facts) | 930 | 20 |

**Control vs arms is confounded — do not cite it.** The arms also switched the
promoter from heuristic to LLM, which replaced 271 whole turns with ~930
atomic facts and cut evidence reaching the reader by 3.6×. The accuracy drop
is that context collapse, not supersession. `docs/ARCHITECTURE_PLAN.md` §P0
framed control→armA as measuring "does supersession working at all help"; that
framing was wrong and is retracted.

**A vs B is valid** — identical but for the supersession mode. Bi-temporal beat
destructive by net **+2 questions of 78** (fixed 6, broke 4), i.e. the A/A
noise floor. Directionally consistent with the thesis, not evidence for it.

### The finding: detection is the bottleneck, not retention

**`supersessions = 4`** across 78 questions — against ~70 that carry a gold
supersession pair. The promoter caught roughly **6%**.

Every detector now measured on this data:

| detector | recall |
|---|---:|
| `attribute_extract` key collision | 0% |
| `_UPDATE_MARKERS` regex | 3% |
| **gpt-4o-mini `ContinuumPromoter`** | **~6%** |
| `scalar_supersession` | **51%** |

The A/B therefore compared what happens to four facts out of ~930, and no
retention policy could have moved a 78-question benchmark from there. The gate
(`stale_answers` → 0) failed because the mechanism never fired on the stale
questions — exactly the "diagnose the wiring, do not interpret the score" case
the plan reserved for this.

**Consequence for sequencing.** P1 (presentation) assumes supersession is
being detected; it mostly is not. Detection moves ahead of it:
`scalar_supersession` is built, tested and measured, and wiring it into
`ContinuumPromoter` as a third deterministic short-circuit is now the first
accuracy-relevant step rather than a P2 refinement.

**Cost:** arm A $3.78 (78 rows, ~$0.05/row — 2.5× the 5-row calibration, whose
haystacks were smaller than average). Arm B was effectively free: its ingest
replayed from arm A's cache.
