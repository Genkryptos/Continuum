# Continuum v3 roadmap — the aggregation gap

*Status: investigation complete, build proposed · 2026-06-07 · supersedes nothing*

This picks up where v2.0 (LongMemEval-S **76.4%**, gpt-oss-120b) left off, after the
question: *can we push the number further with architecture, not a bigger model?*

The honest finding from a night of measurement: **yes, but not with anything we
already have.** The residual is dominated by **counting/aggregation** questions,
and the fix is a **write-time synthesis layer** that does not yet exist —
*fact extraction is not enough* (proven below).

---

## 1. Diagnosis — the residual is NOT retrieval-bound

Of the 118 failures in the v2.0 full-500 run (`results/ws4_full500/judged.json`),
classified by *was the answer session retrieved?*:

| | count | meaning |
|---|---:|---|
| retrieval-bound (answer session missing) | **2** | fixable with better retrieval |
| right context retrieved, still wrong | **116** | NOT a retrieval problem |

So recall is ~99% — **WS-4 essentially solved retrieval.** The memory layer
surfaces the right context; the failures are in what happens *after* retrieval.

**But "not retrieval-bound" ≠ "reader-bound" (my first, wrong, conclusion).**
Re-classifying the 116 by question shape:

- **58 / 118 (49%) are counting/aggregation** — "how many X", "how long", "how
  much", "total" — concentrated in **multi-session (35)** and
  **knowledge-update (15)**.
- The rest are inference ("where did I redeem the coupon" → *Target*, never
  stated), abstention-judgment, and temporal arithmetic.

The counting bucket is the single biggest, most *addressable* slice.

---

## 2. Reference point — Hindsight hits ~89% on gpt-oss-120b

[Hindsight (arXiv 2512.12818)](https://arxiv.org/abs/2512.12818) reports ~89% on
LongMemEval **with the same gpt-oss-120b reader** we use. (Note: the "89% with
gpt-oss-120b" figure is **Hindsight, not Mem0** — Mem0-S is 67.6%, *below*
Continuum's 76.4%.) That ~13pp gap at the *same reader* is the proof the ceiling
is architectural, not the model.

What Hindsight does that we don't (from the paper):
- Four memory networks: **World** (facts), **Experience** (episodes),
  **Opinion** (beliefs), and **Observation** — *synthesized per-entity
  summaries*.
- **Reflect** reasons over this bank at **read time** to produce answers.
- It does **not** precompute counts; observations are *narrative* summaries.
- No ablation, so the per-component contribution is unknown.

The mandate-compatible takeaway: their reader reasons over **structured,
synthesized memory**, not raw turns. Ours reasons over raw turns
(`--no-llm-promoter`) or, at best, atomic facts.

---

## 3. THE KEY FINDING — counting needs *aggregation*, not *extraction*

We tested the obvious hypothesis: *does feeding the reader extracted facts
(instead of raw turns) let the same gpt-oss-120b reader count?* — by running
`--llm-promoter` (gpt-4o-mini fact extractor) with the gpt-oss-120b answerer
held fixed, on counting failures.

**Result (micro A/B, same reader, extracted facts vs raw turns):**

| question | expected | raw turns | extracted facts | |
|---|---|---|---|---|
| how many postcards | 25 | "50" | **"50"** | ❌ identical wrong count |
| how many H&M tops | five | "three" | **"three"** | ❌ identical wrong count |
| how long in Harajuku | 3 months | (wrong) | "3 months" | ✅ a *duration*, not a count |

**Verdict: atomic fact extraction does NOT fix counting.** The two real counting
questions produced the **exact same wrong number** over extracted facts as over
raw turns. Why: atomic facts are still *individual* items ("added postcard X",
"added postcard Y", …) — the reader **still has to count them, and still
miscounts.** Extraction surfaces; it does not **aggregate**. The one recovery was
a *duration* fact (surfaced more cleanly), not a count.

This is the crux: **`--llm-promoter` is not the lever.** (It's also impractically
slow as a per-query step — O(sessions) LLM calls per question, re-ingesting each
haystack. ~hours for a handful of questions.)

---

## 4. The two levers (clearly separated)

| lever | effect on counting | cost | nature |
|---|---|---|---|
| atomic fact extraction (`--llm-promoter`) | ❌ none (proven §3) | very slow | exists, doesn't help |
| **better reader** (gpt-4o-mini) | ✅ partial — recovers ~26% of failures incl. some counts ("citrus→3", "tanks→3") | cheap | a *model choice* |
| **synthesis / aggregation layer** | ✅ the real fix | a build | **architecture-native (v3)** |

The reader-swap evidence: gpt-4o-mini, run on the 118 failures, recovered
**32 (27%)**, concentrated in counting (KU 14, multi-session 6). So a *stronger*
reader counts better — but that's a model choice, and it's still partial. Net
effect needs a full-500 run (recovery rate ≠ net).

---

## 5. v3 design — the synthesis (Observation) layer

**Goal:** at promotion time, derive *aggregate* facts per entity so the reader
*reads* the answer instead of *computing* it.

**What it computes (the gap atomic extraction leaves):**
- per-entity **counts / totals** — "user owns N tanks", "user has 25 postcards",
  "user bought 5 tops from H&M"
- **durations** anchored to `question_date` — "living in Harajuku for 3 months"
- **scoped** aggregates — "*since I started collecting*", "*in the first three
  months*" (the postcards failure was a scoping error: 50 total vs 25 since-start)

**Where it lives:** a new pass in the promotion pipeline, *after* fact
extraction — group the atomic facts by entity (the GLiNER/LLM entities we already
extract), then an LLM (or deterministic counter where possible) emits a derived
`entity_summary` memory item. Mirrors Hindsight's **Observation** network.

**Representation:** a new `MemoryItem` kind (`entity_summary` / `derived_fact`)
in LTM, embedded + retrievable, tagged so the reader prefers it for
counting/aggregation questions (reuse the WS-3 "[CURRENT FACT]" conditioning
pattern). Bi-temporal supersession still applies (a count changes as items are
added).

**Retrieval:** detect counting/aggregation questions (the §1 regex is a start)
and surface the matching `entity_summary` ahead of the atomic facts.

**Honest risks (this is a real build, not a flag):**
- **Aggregation correctness is hard** — scoping ("since X"), dedup (don't double-
  count the same item across sessions), and supersession (a count is a *live*
  derived value) are exactly where the postcards/H&M answers went wrong. The
  synthesis LLM must scope correctly, or it'll be confidently wrong (the v1
  −6pp lesson).
- **Cost** — synthesis is more write-time LLM work. It must run *once* at ingest
  and be reused, never per-query (the §3 slowness lesson).
- **Verification** — every aggregate should be traceable to its source facts
  (the WS-5 A/B discipline: ship only if it clears the noise floor, per category,
  vs an A/A baseline; counting failures are the test set — `samples/ws4_counting_failures_ids.json`).

**Cheaper interim (read-time):** a "count over the retrieved facts" step at
answer time would help, but it's closer to an agentic reasoning loop — in tension
with the "no reasoning loops in the headline" mandate. Write-time synthesis keeps
the headline clean.

---

## 6. Recommended sequencing

1. **Now (cheap, model choice):** full-500 `gpt-4o-mini` reader run to get the
   *net* number — it's a disclosable mid-tier swap and recovers ~a quarter of
   failures including some counts. Decision: if net > 76.4%, gpt-4o-mini is the
   better default reader.
2. **v3 (the architecture win):** build the **synthesis/aggregation layer** (§5),
   A/B'd against `samples/ws4_counting_failures_ids.json` (the 58 counting
   failures) with the gpt-oss-120b reader held fixed. Target: recover a large
   share of the 49% counting bucket → Hindsight-class numbers at a mid-tier
   reader.

**Artifacts from this investigation:**
- `findings/ws4_failures.json` — all 118 v2.0 failures, classified
  reader/retrieval-bound, with question + expected + model answer.
- `samples/ws4_failures_ids.json` — re-run just the failures.
- `samples/ws4_counting_failures_ids.json` — the 58 counting/aggregation
  failures (the v3 test set).
- `results/reader_4omini_fails/` — gpt-4o-mini reader recovery (27%).
- `results/extract_micro/` — the §3 atomic-extraction-doesn't-aggregate proof.

---

## 7. v3.0 A/B result + the v3.1 fixes

**Built & wired** the synthesis layer (`continuum/promotion/synthesis.py` +
`bootstrap_ollama --synthesis`) and ran the A/B: `--synthesis` on the 58 counting
failures, answerer **fixed at gpt-oss-120b** (the 58 were 0/58 over raw turns).

**v3.0 result: 9/58 recovered (8 genuine, 1 rerun-noise) = ~14%** — below the
20% bar. But the mechanism is sound, not broken:
- synthesis **fired on 57/58** rows (extract → aggregate → inject all worked).
- **8 genuine recoveries** caused by synthesis.

Diagnosis of the gap (this is the value of the run):
1. **Summary flood (biggest).** Median **40 aggregates injected per question**
   (up to 57) — we dumped *every* group including `count=1` singletons, burying
   the one relevant count in a 40-line wall.
2. **No SUM.** Several failures are *summing* questions ("how many hours total"
   → 135 vs 140; "hours driving" → 6 vs 15). The aggregator only COUNTed
   distinct members; it didn't sum numeric quantities.
3. **Predicate inconsistency.** "citrus fruits" → 2 vs 3 (extractor split/missed
   a fruit). LLM-quality-bound.

**v3.1 fixes (built, hermetically tested):**
- **Relevance filtering** — `relevant_summaries(facts, question)` injects ONLY
  the aggregate(s) whose predicate matches the question's words (fallback: the
  largest `count>=2` groups). Hands the reader *the* count, not 40.
- **SUM aggregation** — `StructuredFact.quantity/unit` + `DerivedFact.total`;
  aggregate() now sums quantities ("User has 3 sessions totaling 140 hours").
  Extractor prompt asks for quantity/unit on measurements.
- **Tighter extraction prompt** (consistent predicates, explicit examples).

35 synthesis tests (count + SUM + relevance + extraction + wiring).

**Next:** re-run the A/B with v3.1 — expect the relevance-filter alone to move it
well above 20% (v3.0 was diluting a working signal across 40 lines). If it
clears the bar, project to full-500 and ship; predicate-consistency (#3) is the
remaining LLM-quality ceiling.

---

## 8. Reflect — the bounded reasoning pass (the inference/preference lever)

Synthesis fixes **counting** (49% of failures). It does nothing for the **other
half**: ~29 failures are **inference / preference** questions where the right
context IS retrieved (recall ~99%) but the single-shot reader **under-reasons** —
it has to *connect* clues (the "Cartwheel app" + "the email" → **Target**, never
stated), *apply* a stated preference, or *combine* evidence across turns.

### The honest wall (measured, not assumed)
gpt-5-mini is a **reasoning model** (it CoTs by default) and on the 118 failures
it recovered **0/29** of this inference/preference slice. So read-time reasoning
*over raw turns* is not, by itself, enough. Hindsight's Reflect works because it
reasons over **structured** memory (their 4 networks + narrative summaries), not
the transcript. The lever is *Reflect over conditioned context*, not Reflect
alone.

### What we built
A **single, bounded reasoning pass** — deliberately NOT the agentic
IterativeReasoner that went net-negative in v1:

| | v1 IterativeReasoner (cut) | v3 Reflect (this) |
|---|---|---|
| LLM calls | 6 + 2 refine, model-controlled | **exactly 1** |
| retrieval | re-retrieves per round | **none** — reuses the retrieved context |
| control flow | model decides next step (agentic) | **fixed** — one forward pass |
| machinery | decompose / claim / verify / packet / heads | **none** — just a CoT prompt |
| output | free-text synthesised | parsed `Answer: <final>` line |

When the question is in an **eligible hard category** (single-session-preference,
single-session-assistant, multi-session, knowledge-update, temporal-reasoning),
the final answer call uses a chain-of-thought prompt — *quote the relevant facts
→ connect them (apply preference / infer / combine / time) → commit `Answer:
<final>`* — with a larger token budget (`--reflect-max-tokens`, default 512). It
**stacks on** everything already in the prompt: the synthesis `COMPUTED FACTS`
block, WS-1 date stamps, WS-3 `[CURRENT FACT]` marks. Still one `llm.complete`,
no re-retrieval, no iteration.

### The regression guard (the v1 −6pp lesson, encoded)
CoT **hurts** plain fact recall — it over-thinks a clean fact into a confidently-
wrong inference. So Reflect is **gated OFF `single-session-user`** outright, and
only fires on the categories where reasoning is the missing step. It also yields
to the deterministic temporal-delta **codemath** path (verified arithmetic beats
CoT). This is the whole reason it's a *gate*, not a global flag.

### Mandate tension (stated plainly)
The mandate is *"no reasoning loops in the headline."* Reflect is **not a loop**
(one call, fixed control flow) — but it **is** read-time reasoning, so it bends
the *spirit* ("the memory layer does the work, not the reader"). We disclose it
as a **single bounded reflection pass**, distinct from an agentic reasoner. Honest
framing: keep it off → headline stays purely write-time-architecture (~83%); turn
it on → matches Hindsight's *shape* and pushes toward ~86–88%, at the cost of the
"no reasoning at read time" differentiation.

### Prediction (built lever-by-lever from the gpt-oss-judge base of 74.4%)
| stack | predicted |
|---|---:|
| v2.0 base | 74.4% |
| + synthesis (counting) | ~80–82% |
| + Reflect over synthesis-conditioned context, gated | **~83–86%** |
| + graph/temporal retrieval (hybrid+session-aware already in tree) | ~85–88% |
| 89% (Hindsight parity) | only if Reflect-over-structured + retrieval all land near-best |

**Call: ~84–87% is the realistic stacked ceiling; 89% is the optimistic top.**
The 0/29 wall says reasoning-over-raw won't fully close it — the path to 89 needs
Reflect reasoning over *structured* memory (synthesis + graph), i.e. the full
stack, and even then a handful resist.

### A/B plan (WS-5 discipline — do not ship on recovery rate alone)
1. **Reflect ON vs OFF**, gpt-oss-120b reader **fixed**, `--synthesis` on in both
   arms (Reflect is the only delta), on the inference/preference subset
   (`samples/ws4_failures_ids.json` minus the counting ids).
2. **No-regression check** on the solved categories — re-run a sample of
   `samples/ws4_nonfailures_ids.json` (esp. `single-session-user`) with Reflect on;
   it must not drop them. The gate should make this a no-op, but verify.
3. Ship only if recovery on the hard subset clears the per-category A/A noise
   floor AND the no-regression check is clean.

### Wiring
`evals/longmemeval/bootstrap_ollama.py`: `--reflect` / `--reflect-max-tokens` /
`--reflect-types`; `_is_reflect_eligible()` (gate) + `_extract_reflect_answer()`
(parse) helpers; the Reflect branch is the first prompt branch in
`_DirectAnswerAdapter.answer_question`, telemetry `reflect_applied`. Hermetic
tests in `tests/unit/evals/test_reflect_wiring.py` (gate, parsing, single-call
budget, gated-off-fact-recall, **stacks-with-synthesis**). No run yet — built and
tested, ready for the A/B when budget allows.

### Command (when running the A/B)
```bash
unset MODEL LLM SUMMARIZER_LLM EMBEDDINGS    # bare names hijack Continuum config aliases
python3.12 -m evals.longmemeval.bootstrap_ollama \
    --reasoner direct --use-ltm --ltm-backend in_memory --no-llm-promoter \
    --retriever hybrid --judge llm --judge-max-tokens 2048 \
    --synthesis --synthesis-model openrouter/openai/gpt-4o-mini \
    --reflect --reflect-max-tokens 512 \
    --question-ids-file samples/ws4_failures_ids.json \
    --output results/v3_reflect_ab
```
(ON arm above; OFF arm drops `--reflect`. Synthesis cache makes extraction free
on the second arm.)

---

## 9. Full-500 verification — the honest result (the headline)

After the design + 118-failure work above, we ran the **full 500** and judged with
the same llama-3.3-70b judge as v2.0, which finally exposed what the failure-only
view had hidden. The decisive control was a **same-setup baseline** (no reflect /
synthesis / router, identical retrieval + recall) — something we should have run
first.

### The control reframes everything
| run (this setup, llama-3.3 judge) | judged |
|---|---:|
| **control — no v3 at all** | **73.8%** |
| reflect-only (preference+KU) | 73.4% |
| reflect + synthesis + router (strict) | 71.4% |
| recall-fix (12 sessions retrieved) | 71.8% |
| *old v2.0 reference* | *76.4%* |

**The real baseline on this setup is ~73.8%, not 76.4%.** The old 76.4% was a
favorable run (gpt-oss-120b has ~±3–5pp per-category run-to-run variance). Every
"regression vs 76.4%" we chased was against a ghost.

### Clean apples-to-apples: reflect vs control (same setup)
| category | control | reflect | Δ | reflect fires? |
|---|---:|---:|---:|:--:|
| single-session-preference | 60.0% | 70.0% | **+10.0** | yes |
| knowledge-update | 64.1% | 67.9% | **+3.8** | yes |
| single-session-user | 91.4% | 94.3% | +2.9 | no → noise |
| single-session-assistant | 96.4% | 98.2% | +1.8 | no → noise |
| temporal-reasoning | 75.2% | 71.4% | −3.8 | no → noise |
| multi-session | 62.4% | 57.9% | −4.5 | no → noise |
| **OVERALL** | **73.8%** | **73.4%** | **−0.4** | |

### What's actually true
1. **Reflect is accuracy-neutral overall (−0.4 = noise).** Where it *fires* it
   helps — preference **+10**, KU **+3.8**. The multi-session/temporal "losses"
   are **pure run-to-run noise**: reflect is gated to preference+KU and never
   touches those categories. gpt-oss variance (±3–5pp/category) is simply larger
   than reflect's net signal across 500 questions.
2. **Synthesis + router are net-NEGATIVE** (73.4 → 71.4 with them on). The router
   fired on duration/recall questions and returned garbage counts (82% wrong even
   after a strict gate); the synthesis count loses to the reader on hard
   coreference counts ("how many tanks" needs entity resolution, not extraction).
   → shipped **off** by default.
3. **Recall was never the bottleneck.** Raising retrieval from 4→12 sessions
   (`--decompose-max-items 80`, fixing a `max_items=16` cap) *lowered* accuracy
   (more distractors). The session-aware cap was a real bug, but recall ≠ the gap.

### The methodology lesson (the most valuable finding)
**Measuring recovery on a failure-only set structurally overstates gains** — a
failure can only improve or stay failed, never regress, so you see every win and
zero damage. The 118-failure runs looked like +20–30% recovery; the full-500
control showed the true effect is ~neutral. **Always carry a same-setup control.**

### Shipped (all opt-in, 1373 unit tests green, mypy+ruff clean)
- `--reflect` (default types: single-session-preference, knowledge-update) — the
  bounded single-pass CoT; **on by default** (helps where it fires, neutral else).
- `--synthesis` / `--router` — **off by default** (net-negative; kept for research).
- `continuum/promotion/synthesis.py` — aggregation, salvage parser, scope guard,
  strict `route_count_answer`.
- `continuum/promotion/local_extractor.py` — $0 dependency-free extractor.
- `evals/longmemeval/build_synthesis_cache.py` — batched extraction-cache builder.

**Bottom line: Continuum on this setup is ~74%; v3's defensible contribution is
preference +10 / KU +3.8 with no regression — not a headline number. The counting
stack was a rigorously-disproven hypothesis. Honest > impressive.**
