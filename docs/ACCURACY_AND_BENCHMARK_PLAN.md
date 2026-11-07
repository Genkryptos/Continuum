# Accuracy & Competitive Benchmark Plan

**Goal.** Two things at once: (1) make Continuum measurably more accurate without
giving up deterministic supersession / bi-temporal correctness, and (2) put
Continuum, Mem0, and Graphiti on the *same* harness across four eval sites so the
resulting numbers are publishable rather than arguable.

**Status:** draft, 2026-08-11. Owner: solo, ~10–15 h/week.
**Update 2026-11-07 — Phase 1 is now measured, not guessed. See §1a below.**

---

## 0. Why this plan exists

Two problems, discovered by reading the competition's source:

1. **Mem0 2.0.17 no longer deletes superseded facts.** The `ADD/UPDATE/DELETE/NONE`
   decision loop (`DEFAULT_UPDATE_MEMORY_PROMPT`, `get_update_memory_messages`) is
   still defined in `mem0/configs/prompts.py` but is called from nowhere in the
   package. The default `add(infer=True)` path is an 8-phase additive pipeline
   whose extraction prompt states its "sole operation is ADD", handling
   contradictions by *linking* (`linked_memory_ids`) rather than overwriting.
   The paper's baseline was measured on **0.1.67** (the version still pinned in
   this repo's environment). The claim is true for that version and false for
   current mem0. Left alone, this reads as a strawman to any reviewer who clones
   mem0 today.

2. **Our own cross-system numbers are not comparable.** BEAM was run with a
   gpt-oss-120b answerer and a gpt-4o-mini judge at top-80; Mem0's published BEAM
   used gpt-5 for *both* answerer and judge at top-200. Those numbers cannot be
   put in the same table. This already caused one withdrawn claim.

Everything below is ordered so that the fairness work lands *before* any
comparative number is generated.

---

## 1a. What the 500-question run of 2026-11-07 actually measured

This plan's Phase 1 listed five candidate accuracy levers (A1–A5) chosen by
reading the competition. A full 500-question LongMemEval-S run has since put
numbers on where the failures are, and they do not distribute the way A1–A5
assumed. Cross-tab of all 155 failures:

| cause | n | concentrated in | reading |
|---|---:|---|---|
| `missing_aggregation` | 55 | **100% multi-session** | fixed turn budget |
| `missing_gold_turn` | 45 | temporal 20, preference 9 | retrieval |
| `wrong_span` | 22 | **know-update 16** | value selection |
| `stale_fact` | 14 | **100% know-update** | supersession |
| `abstained` | 9 | preference 6 | reader conservatism |
| `missing_fact` | 9 | temporal 6 | extraction |

Three categories separate almost perfectly by cause: multi-session is a
budget problem (55 of its 61 failures, and nothing else is), knowledge-update
is a supersession problem (30 of 35), temporal-reasoning is a plain retrieval
problem (20 of 31). Two mechanisms carry **130 of 155** failures.

**The binding constraint, measured.** On *every one* of the 55 aggregation
failures: `items_offered = 8.0`, `items_admitted = 8.0`,
`items_dropped = 0.0`. The reader was never handed more than
`session_top_k × turns_per_session = 8` turns and discarded none. Those rows
need a mean of 2.9 gold turns across 3.0 sessions.

| sessions needed | n | acc | session recall | turn recall |
|---:|---:|---:|---:|---:|
| 2 | 84 | 70.2% | 0.923 | 0.696 |
| 3 | 26 | 34.6% | 0.846 | 0.465 |
| 4 | 17 | 11.8% | 0.691 | 0.280 |

Session recall barely moves; **turn recall collapses**. That gap is the
defect, and it retires this repo's standing claim that multi-hop is
answerer-bound — that claim rests on session recall, which reads fine.

**Two caveats on the run itself**, both of which mean it measured less than
its label:

1. It ran with **`--use-ltm` off** (zero `post_opt_ltm_tokens` on all 500
   rows) while `evals/PREREGISTRATION.md` §3.1 pins it on as "the
   supersession path — the mechanism under test". So the knowledge-update
   number is a *no-supersession* number: `[CURRENT FACT]` items, which
   require LTM, never appeared.
2. Its answerer was `muse-wonder-1.0`, not the pinned `openai/gpt-oss-120b`.
   It is therefore `non_comparable: true` and may not be set beside Mem0-S
   67.6% or Continuum's own ~74% (`docs/report.md` §6), both gpt-oss-120b
   numbers.

**Consequence for this plan.** A1–A5 are not retired, but the ordering below
is no longer the right one: the budget lever (not in A1–A5 at all) dominates
by measured headroom, and A1/A5's supersession work cannot be evaluated until
`--use-ltm` is actually on. The arms and their falsifiers now live in
`evals/PREREGISTRATION.md` §3.4 and P9–P11.

---

## Phase 0 — Level the playing field (blocking, ~10 h)

No comparative claim is made until this phase is done. Its output is a single
harness contract that every system runs under.

### 0.1 Pre-register the run matrix

Write `evals/PREREGISTRATION.md` **before** running anything, fixing:

- answerer model + temperature + seed
- judge model + judge prompt hash
- top-K, context budget, retrieval depth
- dataset split + item count per category
- the metric definition per eval site
- **the prediction for each cell**, and what result would falsify it

Rationale: two findings have already been withdrawn or corrected after the fact
(the ANN-index recall claim; the misremembered Mem0 25/15 figure). A committed
prediction converts a retraction into a legitimate negative result.

### 0.2 Freeze one answerer and one judge across all systems

Single config object consumed by all four runners. The memory layer is the only
variable. Any run where answerer/judge/top-K differ across systems is marked
`non-comparable: true` in its result JSON and may not enter a comparison table.

### 0.3 Pin and record versions

Every result JSON gains a `versions` block: continuum git SHA, `mem0ai` version,
`graphiti-core` version, embedding model, answerer, judge. This is what stops a
0.1.67-vs-2.0.17 mixup from recurring.

### 0.4 Add the two missing system adapters

- `evals/locomo/graphiti_adapter.py` — mirrors the `Mem0LocomoAnswerer` interface
  (`__init__`, `_ensure_ingested`, `answer`); extend `run.py --system` choices to
  `continuum | mem0 | graphiti | all`.
- `bench/mem0_compare.py` — parameterize the mem0 import so both **0.1.67** and
  **2.0.17** can be run (separate venvs, `--mem0-version` tag written into the
  result JSON).

**Gate 0:** `--limit 3` smoke passes for all systems on all four eval sites, with
version blocks populated. No accuracy claims yet.

---

## Phase 1 — Make Continuum more accurate (~25 h)

> **Read `docs/SMALL_MODEL_ACCURACY.md` first.** The goal is not points on a
> leaderboard — it is that a *small* model plus Continuum matches a large model
> plus naive retrieval. That note supplies the mechanism (convert reasoning
> problems into query problems) and adds items **S1–S4**, which outrank A2–A4
> below. A1 is retained as their control arm.

Four changes, ordered by expected yield per hour. Each ships behind a flag with a
paired ablation, so every one either earns its place in the ablation table or gets
reverted. **None of them alters supersession semantics — `current()` stays the
decider and remains deterministic.**

### A1. Annotate instead of only suppressing *(highest leverage)*

**Problem.** `PostgresLTM.search_hybrid` filters on `invalidated_at IS NULL` and
applies the valid-window **only when `q.as_of` is set** (`continuum/stores/postgres/ltm.py:696`).
With no `as_of`, superseded and current facts return mixed and **undated** — the
answerer gets a pile with no way to tell which fact won. This is the mechanism
behind the un-decidered 20% supersession recall.

**Change.** Carry `valid_from` / `valid_to` through into the assembled
`ContextBundle`, and state the validity rule in-band, as Graphiti does:

> Facts are valid between their valid_at and invalid_at dates. Facts with an
> invalid_at of "Present" are currently valid.

**Why it should work.** This is the principled form of the already-measured
`--chrono-sort` result (+25pp on BEAM contradiction, 7.5 → 32.5). Chrono-sort
works because ordering accidentally surfaces both sides of a contradiction;
dated facts plus an explicit rule surface both sides *and* say which one holds.

**Ablation:** `off` / `chrono-sort only` / `dates only` / `dates + rule`.
**Invariant test:** `current()` and as-of results must be byte-identical to
today's. If they move, the change has leaked into the decider — revert.

### A2. Encode the transition into the fact text

Adopt Mem0's transition rule in `continuum/promotion/`. When the promoter detects
supersession, write what was replaced into the content, not only into `valid_to`:

- bad: `"User prefers oat milk lattes"`
- good: `"User switched from almond milk to oat milk after developing an almond sensitivity"`

Belt-and-braces: the fact stays correct even when a downstream consumer strips the
metadata. Costs nothing at read time.

### A3. Split observation date from ingest date

Mem0's sharpest write-time rule: resolve every relative reference ("last week",
"yesterday") against **when the conversation happened**, never against now, and
store the absolute date. Expected to move **event_ordering**, our weakest BEAM
category (27.5% floor, 32.5% chrono-sorted).

### A4. `node_distance` reranker

Graphiti's one read-side idea worth taking, and cheap because `Retriever` already
does graph expansion at step 3. Rank candidates by graph distance from the top
entity hits. Lowest priority — we already have RRF + a cross-encoder, which is
ahead of mem0's default read path.

### A5. Replace the eval's destructive update path with bi-temporal supersession *(first)*

**Verified 2026-08-11 by reading the code — this is worse than "not wired", and it
has a direct bearing on the paper.**

`--use-ltm` does *not* exercise Continuum's bi-temporal supersession. The eval's
LTM store (`evals/longmemeval/continuum_ltm_store.py`) runs
`Mem0Promoter.decide_operations_batch` — an **ADD / UPDATE / DELETE / NOOP decision
loop, i.e. Mem0 0.1.67's model** — and executes it as:

| decision | executed as | effect |
|---|---|---|
| `UPDATE` | `ltm.update(target_id, {"content": merged})` | **mutates the fact in place — old value destroyed** |
| `DELETE` | `ltm.invalidate(target_id, now)` | sets `invalidated_at` (transaction time); `search_hybrid` filters these out entirely |
| `ADD` / `NOOP` | upsert / skip | — |

There is **no `valid_to`, no `superseded_by`, no `current()`, no `as_of`** anywhere
in this path. `invalidate` is transaction-time hard-hiding, not valid-time
supersession: the fact becomes unreachable rather than historical.

**Consequences, stated plainly:**

1. Every LongMemEval number we have — including the **~74%** headline in
   `paper/outline.md:88` and `knowledge-update 51.3%` — was produced by a pipeline
   that behaves like **Mem0 0.1.67**, not like the system the paper describes.
   The paper must not cite LongMemEval as evidence for the bi-temporal claim until
   this is fixed. The `bench/` results (100% vs 30%/27%) *do* exercise the real
   mechanism and are unaffected.
2. `findings/reasoning_loop_2026-06.md` §3 calls this "LTM supersession" and credits
   it with making knowledge-update work. That naming is wrong and should be
   corrected in place — what it measures is destructive update, which is exactly
   the design the paper argues against.

**The upside: the harness already contains a faithful Mem0-style baseline.** So A5
is not "switch a feature on", it is a **controlled A/B of two designs inside one
harness**, everything else held constant:

- **arm A (today):** `Mem0Promoter` decisions → `update` / `invalidate`
- **arm B (new):** bi-temporal supersession → set `valid_to` on the superseded
  fact, keep it retrievable via `as_of`, let `current()` decide what retrieval sees

That is a better experiment than bolting on an external Mem0 adapter, and it is the
single cheapest test of the paper's central claim on a public benchmark. Do it
before A1–A4 and before any Phase-2 cell.

**Gate 1:** A5 plus each of A1–A4 shows a signed effect on at least one eval site
under the frozen harness. Anything that doesn't is reverted and recorded as a negative
result, not left in behind a dead flag.

> **Sequencing note.** A1–A3 are write-time and extraction-time. That is where
> Mem0's own accuracy gains came from; their read side is thinner than ours
> (sigmoid-normalized BM25 + additive entity boost, no cross-encoder). Combined
> with the pg_trgm channel having just been gated for contributing nothing at
> 2.3 s cost, further retrieval-channel work is the low-yield direction.

---

## Phase 2 — The design-space study (~20 h)

> **Framing (deliberate).** The goal is *not* to beat Mem0. Beating a competitor is
> a market objective; this project's objective is to find out whether a design
> choice is right. So the other systems are **controls, not opponents** — they are
> run because they occupy different points on the one axis we care about, and
> without them our own choice is unfalsifiable.
>
> That axis is: **what do you do with a fact that has been superseded?**
>
> | system | answer | consequence |
> |---|---|---|
> | Mem0 0.1.67 | delete it | history destroyed; as-of impossible |
> | Mem0 2.0.17 | keep it, undated, unranked | history kept but indistinguishable from current |
> | Graphiti | keep it, dated, surfaced | correct data, decision delegated to the LLM |
> | **Continuum** | keep it, dated, **deterministically suppressed**, reachable via as-of | decision made by the store |
>
> Four coherent answers to one question, with the whole space covered. Studying
> that properly is better engineering *and* a better paper than any leaderboard
> claim — and it stays interesting even in the cells where Continuum loses.
>
> Practical effect: cells exist to **discriminate designs**, not to fill a table.
> Any cell that cannot change a design conclusion gets cut.

Only after Gates 0 and 1.

| | LOCOMO | LongMemEval | BEAM | `bench/` (supersession, bi-temporal, point-in-time) |
|---|---|---|---|---|
| Continuum (post-Phase-1) | ✅ adapter | ✅ built out | ✅ converter + rubric judge | ✅ |
| Mem0 **0.1.67** (paper baseline) | ✅ adapter | ⬜ adapter | ⬜ | ✅ |
| Mem0 **2.0.17** (current) | ⬜ venv | ⬜ | ⬜ | ⬜ re-run |
| Graphiti | ⬜ adapter | ⬜ adapter | ⬜ | ⬜ |

Existing assets: `evals/locomo/` (both adapters), `evals/longmemeval/` (extensive),
`evals/beam/{convert,rubric_judge}.py`, `bench/` (deterministic suites).
Gaps are the ⬜ cells.

### 2.1 Priority order within the matrix

1. **`bench/` × Mem0 2.0.17** — the deadline item. Directly re-tests the paper's
   central comparison against the version that removed the delete.
2. **`bench/` × Graphiti** — Graphiti has genuine `valid_at`/`invalid_at`/`expired_at`
   and invalidates rather than deletes (`resolve_edge_contradictions` sets
   `invalid_at = resolved_edge.valid_at`). It is the **only** real competitor on
   our home turf. If Continuum cannot beat Graphiti on point-in-time queries, the
   paper's differentiator is narrower than claimed and the framing must change.
3. **The answerer-size ladder** (`docs/SMALL_MODEL_ACCURACY.md` §5a) — run
   Continuum vs the controls at 8B / 20B / 120B and report the gap *as a function
   of answerer size*. This is the single most informative experiment in the plan:
   it tests whether the advantage is structural or borrowed from the answerer.
   A flat Continuum curve against a steep control curve is the result.
4. **LongMemEval × Continuum arms** (direct vs typed-query, per-category) — tests
   the S1–S4 architectural bet on the 344/500 convertible questions.
5. **LOCOMO / BEAM × all** — **optional, and cut by default.** These are leaderboard
   cells. Run them only if a specific design question needs them; "we should have
   a LOCOMO number" is not a design question. BEAM, if run, must use its own
   rubric-nugget judge (0 / 0.5 / 1.0 per nugget, PASS at mean ≥ 0.5) — substring
   scoring is invalid there, since it rewards confident answers that ignore the
   contradiction, the exact failure BEAM exists to detect.

### 2.2 Report per-category, never aggregate-only

**A whole-benchmark score is the wrong unit and will bury the result.** Each of
these suites averages together categories that test Continuum's mechanism
(knowledge-update, contradiction_resolution, temporal ordering) with categories
bounded by the answerer's multi-hop reasoning. `paper/outline.md:212` already
establishes the latter: *"ceiling even at 100% recall → bottleneck is multi-hop
reasoning."* That ceiling is **category-scoped, not benchmark-scoped**. Averaging
across it converts a win into a wash.

Every result table in Phase 2 is per-category. Aggregates may appear only
alongside the category breakdown.

### 2.3 Honest expectations

Stated now, so they can be falsified:

- **Supersession vs Mem0 2.0.17:** our gap **narrows sharply**. Additive mem0 now
  keeps both facts, so its recall of the superseded fact should jump from the
  0.1.67 baseline.
- **Bi-temporal / point-in-time vs Mem0 2.0.17:** our gap **largely holds**.
  Additive storage still gives mem0 no `valid_to` and no as-of filter — it keeps
  both facts and hands the answerer an undated pile, which is precisely the
  failure A1 addresses on our side.
- **vs Graphiti:** genuinely uncertain. Graphiti has the same temporal model.
  Our expected edge is *deterministic* suppression (`current()` as a decider)
  versus Graphiti's default of returning invalidated facts annotated and letting
  the LLM decide. That is a real difference, but it is an argument about
  determinism, not about having the data model.
- **LongMemEval `knowledge-update`: we should win, and have never actually
  competed.** `findings/longmemeval_2026-05.md:126` says it outright — this is
  "the category where Continuum's `superseded_by` schema (built but not wired
  into the eval) should win directly." The headroom is measured: the long-context
  big-model run scores **51.3%** there while every retrieval config sits at
  33–44%, because top-k ranking surfaces the *stale* version of a fact above the
  fresh one. That is exactly what `current()` fixes deterministically. See A5.
- **BEAM `contradiction_resolution`: winnable.** Chrono-sort alone bought +25pp
  (7.5 → 32.5). That is not a reasoning ceiling — it is information availability.
  A model cannot flag a contradiction whose two sides never both reach the
  context. A1 attacks this directly.
- **BEAM `event_ordering`: plausibly winnable** via A3. It is a purely temporal
  task and we are a temporal system; 27.5 → 32.5 is weak because observation-date
  grounding is missing, not because the ceiling is near.
- **Multi-hop categories across all suites: not winnable by memory work.** Bounded
  by the answerer (`paper/outline.md:212`). Do not spend effort here; report and
  move on.
- **LOCOMO aggregate: no strong expectation.** It is Mem0's headline benchmark and
  they have tuned against it.

**Gate 2:** every cell either has a number under the frozen harness or an explicit
"not run, because —". No blank cells in the paper.

---

## Phase 3 — Paper integration (~10 h)

1. **Fix the mem0 baseline claim.** State the version inline: *"Mem0 0.1.67, whose
   update path deletes the superseded fact; version 2.0.17 replaced this with an
   additive pipeline — measured separately in Table N."* This converts the biggest
   review risk into a strength: we tracked the competitor's evolution.
2. **Add Graphiti to related work** as the closest system, with the concrete
   distinction (deterministic suppression vs annotated-and-let-the-LLM-decide).
3. **Ablation table** from Phase 1 (A1–A4), including reverted changes.
4. Remaining known blockers, unchanged: reproducibility pins per table; LaTeX
   port and PDF (needs `brew install tectonic` or Overleaf).

---

## Risk register

| Risk | Mitigation |
|---|---|
| Mem0 2.0.17 closes the supersession gap entirely | Expected and pre-registered. The bi-temporal / point-in-time result is the load-bearing one; supersession was always the weaker claim. |
| Graphiti matches us on point-in-time | Would narrow the thesis to *determinism* rather than *capability*. Run this cell **second**, not last — it should not be discovered late. |
| A1 leaks into `current()` and breaks the 100% result | Byte-identical invariant test on `current()` / as-of is a hard gate on A1. |
| Harness drift reintroduces non-comparable numbers | `non-comparable: true` flag + mandatory `versions` block in every result JSON. |
| Scope creep across four eval sites | Phase 2 is strictly ordered; ship after each cell. `bench/` × mem0 2.0.17 alone is a publishable correction. |

## Sequencing summary

```
Phase 0 (blocking, ~10h) ──► Gate 0: all systems smoke-pass, versions recorded
        │
Phase 1 (~25h) ───────────► Gate 1: each of A1–A4 shows a signed effect
        │
Phase 2 (~20h, ordered) ──► Gate 2: no blank cells
        │
Phase 3 (~10h) ───────────► paper updated
```

Total ~65 h ≈ 5–6 weeks at 10–15 h/week.

**Smallest shippable slice if time collapses:** Phase 0.3 (version pinning) +
Phase 2.1 cell 1 (`bench/` × Mem0 2.0.17) + Phase 3.1 (fix the claim). That alone
protects the paper. Roughly 12 h.
