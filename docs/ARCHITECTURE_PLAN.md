# Architecture Plan — getting to the best available memory architecture

*2026-09-08 · evidence: `findings/architecture_v1.md` · reader pinned
`openai/gpt-oss-120b` · control `results/v3_baseline_control/judged.json`*

**This plan proposes almost no new work.** Most of what the error data calls
for is already specified in `docs/ACCURACY_AND_BENCHMARK_PLAN.md` §A1–A5 and
`docs/ENGINEERING_PLAN.md` Phases 1–2. What has been missing is an ordering
derived from where the errors actually are, a gate for each step that the
harness can measure, and three items nothing covers.

So this document does four things: it **credits** the existing plans, it
**re-sequences** them against the error budget, it **adds** the three missing
pieces, and it **gates** every phase on a number rather than a judgement.

---

## 0. The budget everything is spent against

131 errors. **114 of them occur with *every* answer-bearing session already
retrieved** (130 with at least one — all the `wrong_span` bucket asserts).
The residual 17 carry some session-level retrieval deficit: an upper bound of
3.4pp, concentrated in multi-session and temporal-reasoning.

| category | n | wrong | pp if perfect | session recall |
|---|---:|---:|---:|---:|
| multi-session | 133 | 50 | 10.0 | 0.968 |
| temporal-reasoning | 133 | 33 | 6.6 | 0.976 |
| knowledge-update | 78 | 28 | 5.6 | **1.000** |
| single-session-preference | 30 | 12 | 2.4 | 0.967 |
| single-session-user | 70 | 6 | 1.2 | 1.000 |
| single-session-assistant | 56 | 2 | 0.4 | 1.000 |

Two consequences fix the sequencing:

1. **Retrieval is not the main lever.** Six measured results agree (reranker
   0pp, optimizer 0pp, scorer weights −2pp, full long-context identical to
   top-k=4, decomposition +3pp, wiki regressed). The 17 errors with a
   session-level deficit cap it at 3.4pp, and none of them are
   knowledge-update — that category retrieves everything and still loses 28.
2. **Knowledge-update is the cleanest target.** Recall 1.000, 28 wrong, and
   `evals/longmemeval/evidence.py` now shows **15 of the control's answers
   provably took the superseded value** (14 knowledge-update, 1 temporal;
   0 false positives). That is a defect with a name, a count, and a test.

---

## 1. What is already written down

| need (from the evidence) | already specified in | status |
|---|---|---|
| Bi-temporal supersession actually exercised in the eval | ACCURACY §A5 | **built, unrun** — `--supersession-mode bitemporal` |
| Dates + validity rule carried in-band to the reader | ACCURACY §A1 | specified |
| Transition written into the fact text ("was X, now Y") | ACCURACY §A2 | specified |
| Observation date vs ingest date | ACCURACY §A3 | specified |
| Deterministic supersession on *open* text | ENGINEERING §1 | specified |
| `timeline()`/`current()`-backed retrieval mode | ENGINEERING §2 | specified |
| Consolidation / dedup / entity linking | ENGINEERING §3 | specified |

**Nothing below replaces those. This plan orders them and adds the three
items in §2.**

One correction to the existing plans, made in passing: both treat
LongMemEval as a *regression guard* ("no regression on ~74%"). Given recall
1.000 and 15 measured stale answers on knowledge-update, A5 + A1 + A2 should
**move** that category, and failing to move it is now an informative negative
rather than a neutral pass.

---

## 2. What nothing covers

**2a. The supersession invariant lives in application code, not the schema.**
`memory_nodes` has no `subject` and no `attribute` column — attribute is
`tags->>'attribute'` (`continuum/stores/postgres/ltm.py:516`), while
`docs/architecture.md:96` draws the table as
`subject | attribute | value | superseded_by`. ENGINEERING §1a proposes
deriving the key and *"storing it as the same `attribute` metadata"*. That
keeps contradiction detection an inference. Promoting the key to columns with
a **unique partial index on `(namespace, subject, attribute) WHERE valid_to IS
NULL`** makes it a write-time key collision the database enforces. This is
the difference between "we soft-invalidate" and a guarantee, and it is what
no LLM-synthesis competitor can copy without rebuilding their store.

**2b. Routing must key on the data, not the question.** v3's router
classified the *question* by regex and returned a code-computed count; it was
82% wrong and shipped off. Invert it: if `current(subject, attribute)`
returns one live value with a supersession chain, the resolved view is
authoritative; if it returns a set of *n* members with no supersession, hand
over the **complete set, unsummarised**. The store knows which shape it
holds; the question classifier never did.

**2c. "Counting" is two different problems.** The gold evidence for *How many
Korean restaurants have I tried?* is two turns that each state a complete
count (`three` on Aug 11, `four` on Sep 30) — a supersession wearing a
number. Multi-session counting is genuinely enumerative: one member per gold
turn, coreference to resolve, duplicates to drop. `docs/limitations.md` files
both under "counting is genuinely reader-bound". That verdict looks sound for
the second and wrong for the first — and it explains v3's aggregator failing
better than "counting is hard" does: **it was built to enumerate members and
tested on rows that do not pose that problem.**

---

## 3. Phases

Each phase: one change, one gate, one predicted yield. Yields are from
failure inspection, not extrapolation. A phase that misses its gate is
recorded as a negative result and the next phase does not inherit its
assumption.

### P0 — Run the experiment that is already built *(blocking, ~1 run)*

`--supersession-mode bitemporal` is wired, and `--ingest-cache` makes the
second arm cost no LLM calls. Arm A (`invalidate`, Mem0 0.1.67's destructive
model) vs arm B (`bitemporal`), everything else held.

**Gate:** in arm B, `stale_answers` (new metric) goes **15 → 0**. That is a
sharper test than accuracy: if the count does not fall, supersession is not
reaching the reader and no accuracy number from that arm means anything —
diagnose the wiring, do not interpret the score. Report per-category with
`gold_turn_recall` beside accuracy.

**Predicted:** knowledge-update +2 to +4pp (≈ +0.5 to +1pp overall). The
15 stale answers are the floor of what is addressable, not the ceiling — the
detector abstains on non-numeric supersessions.

**Why first:** it is built and it tests the paper's central claim on a public
benchmark. Every phase below assumes supersession reaches the reader; P0 is
what establishes that.

> **Pre-flight, measured 2026-09-08 — P0 must run with the LLM promoter.**
> The heuristic path (`--no-llm-promoter`) detects supersession by regex over
> update markers (`no longer|used to|moved to|switched to|…`). Across the 15
> measured stale answers, **0** have a marker in the later gold turn. Across
> all 70 knowledge-update questions that have a gold pair, **2 (3%)** do.
>
> These restatements — "I've tried three" → "I've tried four", `$350,000` →
> `$400,000`, `8:30 am` → `7:30 am` — are supersessions by **(subject,
> attribute) key collision**, with no linguistic marker to key on. Running P0
> in the headline `--no-llm-promoter` config would therefore return arm B ≈
> arm A and `stale_answers` unchanged at 15, for a reason that has nothing to
> do with the bi-temporal claim.
>
> **This raises P2's priority.** `ENGINEERING_PLAN`'s stated moat is
> "deterministic correctness with **no LLM in the critical path**". On this
> data the deterministic path currently sees 3% of the supersessions. Until
> the EAV spine exists, any arm-B win is an *LLM-promoter* win — a materially
> weaker claim for the paper than the one it wants to make. Consider running
> P2's detector offline (§6) before spending on P0.

### P1 — Presentation: dates, validity rule, transition text *(ACCURACY §A1 + §A2)*

The 14 stale and 4 direction-inverted knowledge-update answers are labelling
failures: the model holds both values and assigns them to the wrong times
("you switched to **more** water" when the answer is less). Suppression alone
cannot fix an inversion — the transition has to be *shown*, not hidden.

**Gate:** A1's own invariant test (`current()` and as-of results byte-identical
— if they move, the change leaked into the decider, revert), plus
`stale_answers` at 0 and knowledge-update up. Run A1's specified ablation
(`off` / `chrono-sort only` / `dates only` / `dates + rule`) so the
contribution is attributable.

**Predicted:** +2 to +3pp overall.

### P2 — The EAV spine *(new, §2a)*

Promote `subject` and `attribute` to columns; add the unique partial index on
live `(namespace, subject, attribute)`; migrate `tags->>'attribute'` forward.
Supersession becomes a database invariant.

**Gate:** `ENGINEERING_PLAN` Phase 1's existing acceptance —
`bench/supersession_e2e.py` recall top-1 **20% → ≥80% with the LLM decider
OFF** — now met by the schema rather than by application code. Plus: P0 and
P1's gains must survive with `--no-llm-promoter`, i.e. on open text rather
than on tagged facts.

**Why here and not first:** P0/P1 are measurable without it. P2 is what makes
them *hold* outside the eval's tagged path, which is the difference between a
benchmark result and a product guarantee.

> **Revised by measurement, 2026-09-08 — the spine cannot be fed by the
> extractor we have.** Run
> `python3.12 -m evals.audit.scalar_supersession_coverage`:
>
> | detector | recall on the 70 KU gold pairs | hard-negative FP |
> |---|---:|---:|
> | `_UPDATE_MARKERS` regex | 3% | — |
> | `attribute_extract` key collision | **0%** | — |
> | `scalar_supersession` (new) | **51%** | 3.1% |
>
> `extract_attribute` returns `None` on **every** gold turn in this category.
> It keys on canonical forms — "I live in X", "I work at X", "my X is Y" — and
> conversational turns are not shaped that way. A unique index on
> `(namespace, subject, attribute)` is still the right invariant for facts
> written *with* an attribute (the MCP `remember` path tags them), but on open
> conversation there would be nothing to key on.
>
> `continuum/promotion/scalar_supersession.py` is the deterministic path that
> does work on this data: **same topic frame, restated, different scalar.** It
> is a fast path, not a replacement — it removes the LLM from about half the
> supersession decisions and defers the rest, exactly as `Mem0Promoter`'s
> existing cosine short-circuits do. Wiring it in as a third short-circuit is
> the concrete P2 deliverable; the EAV columns remain valuable but serve the
> tagged path, not this benchmark.

### P3 — ResolvedState through the product path *(ENGINEERING §2)*

The `timeline()`/`current()`-backed retrieval mode, assembling context as a
dated, resolved, entity-scoped view instead of ranked raw hits. This is the
already-planned home for the `--chrono-sort` finding (+25pp on BEAM
contradiction), promoted from eval flag to product feature.

Three properties are load-bearing and should be stated as requirements, not
side effects: it **shows the transition** rather than filtering it away; it is
a **deterministic projection** (same store, same bytes, no LLM — so it needs
no reproducible-serving work to be trustworthy); and it is **stable per
subject across queries**, hence prefix-cacheable.

**Gate:** the existing one (chrono-sort gain reproduced through the product
path on BEAM, no LongMemEval regression) **plus** temporal-reasoning ordering
failures down — ~8 of the 33 are ordering/enumeration.

**Predicted:** +1 to +2pp overall.

### P4 — Complete-set delivery and data-keyed routing *(new, §2b/§2c)*

Multi-session is the largest block (50 errors, 10.0pp) and is systematically
*under*-counting (20 undercount vs 7 overcount). The evidence-distillation
result already showed lossy projection makes this worse. So the route for
set-shaped data is the opposite of the route for state-shaped data: deliver
the complete member set, unsummarised, and let the store decide which is
which.

**Gate:** the undercount ratio moves, measured on the multi-session numeric
subset. A wrong-direction result here reverts cleanly because the routing is
keyed on store shape, not on a question classifier.

**Predicted:** +3 to +4pp if enumeration yields; 0 if it is genuinely
reader-bound. This is the phase most likely to return a clean negative, and
the one whose negative is most worth having — it would finally *earn* the
reader-bound verdict for the population it actually applies to.

### P5 — Abstention *(small, cross-cutting)*

30 of the 500 questions are `_abs` rows scoring **63.3% against 74.5%**
elsewhere — 11 questions, 2.2pp. Failures run both ways: confabulation
("you lead four engineers" where the expected answer is *not enough
information*), and false abstention ("the conversation doesn't specify who
gave you the jewellery" when `my aunt` was in the retrieved context).

The second kind is a presentation failure and may fall out of P1/P3 for free
— measure it there before building anything. If a gap remains, an explicit
"absent from the resolved view" signal is a store-side answer that does not
require logprobs.

**Predicted:** +1pp.

---

## 4. Deprioritised, with reasons

**`ENGINEERING_PLAN` Phase 4 (retrieval at scale).** Not because scale does
not matter for the product, but because (a) it is not where the benchmark
errors are, and (b) its evidence base has been withdrawn twice — the
50k-scale conclusion was reversed in `findings/scale_retrieval_2026-08.md`
("corrects this document's own first version"), and `HARDENING_PLAN` §2.1
carries a further correction plus the newer finding that `ef_search=1000`
makes the planner abandon the index entirely (`82a7802`). Settle the
measurement before planning against it.

**More retrieval channels.** Six negative results; the pg_trgm channel was
just gated for contributing nothing at 2.3s.

**Optimising the deterministic benches.** `bench/` supersession and
bi-temporal pass at their acceptance bar on scripted scenarios. They
demonstrate the mechanism; they cannot discriminate designs.

---

## 5. The honest ceiling

Perfect knowledge-update (+28) and perfect temporal-reasoning (+33) is +61
questions: **73.8% → 86.0%**. Hindsight parity at 89.0% needs +76, so it
requires at least 15 of the 50 multi-session enumeration errors as well.
**There is no path to 89% that routes around enumeration.**

Stacking the predictions above: **~80%**, or **~84%** if P4's enumeration bet
lands. State it with that error bar publicly.

And the counterweight worth keeping from `roadmap_v4` §06: the accuracy
number is the least defensible thing here. What P2 buys — a supersession
invariant the database enforces, reversible, auditable, answerable as-of, on
an unbounded store — is not beaten by shipping a bigger model.

---

## 6. Sequencing

**Re-sequenced 2026-09-12, after P0 ran.** P0's verdict was that the promoter
detects ~6% of the supersessions in this category (`supersessions = 4` across
78 questions, against ~70 gold pairs), so the arms compared what happens to
four facts and nothing downstream could move. Every later phase assumed
detection works. It does not, so detection goes first.

| | phase | depends on | cost |
|---|---|---|---|
| ~~done~~ | ~~**P0** run A5 arms~~ | — | **$3.78, complete — see §12 of `findings/architecture_v1.md`** |
| **now** | **P2a** wire `scalar_supersession` into `Mem0Promoter` as a third deterministic short-circuit | built + tested | S |
| then | **P0′** re-run the A/B with detection working | P2a | ~$0.30 (ingest cached) |
| then | **P1** dates + rule + transition text | P0′ verdict | S–M |
| then | **P3** ResolvedState in the product | P1 | M |
| then | **P4** complete-set + data-keyed routing | P3 | M |
| any time | **P5** abstention | measure during P1/P3 | S |

P2a is now the gate. Until detection clears ~6%, no retention policy,
annotation scheme or resolved view can be measured — they all operate on
facts the promoter never flagged. `scalar_supersession` measures 51% on this
data, and re-running the A/B after wiring it costs almost nothing because the
ingest cache is already warm.

**A caution carried forward from P0.** Switching the promoter path from
heuristic to LLM cut the evidence reaching the reader from 73 items to 20 and
cost more accuracy than supersession could plausibly return. Any change to
the promoter must report `retrieved_items` and `gold_turn_recall` alongside
accuracy, or a context collapse will be misread as a mechanism result.

---

## 7. Open items this plan does not resolve

1. **`docs/limitations.md` and `docs/report.md` §6 contradict each other** —
   one says the ceiling is reader-bound, the other architectural. Both ship.
   §2c says the honest answer is per-population.
2. **The stale detector is numeric-first** and abstains on non-numeric
   supersessions. 15 is a floor. Extending it without losing the
   0-false-positive property would sharpen every gate above.
3. **The product and eval retrieval paths are different code**
   (`Retriever` + graph expansion vs `SessionAwareSemanticRetriever`). P3
   narrows this; it does not close it.
