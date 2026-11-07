# The cost/accuracy curve — where accuracy breaks as context shrinks

*Status: measured, Phase 1 complete · 2026-08-05 · executes `docs/LOW_BUDGET_PLAN.md` Phase 1*

**Headline: the elbow is at 16 000 chars, and it is sharp.** One step below it costs 5.2pp;
two steps cost 12.4pp. The target written down before the run — *within 2pp of the anchor at
8 000 chars* — **is not reachable by budget cuts alone.**

The secondary finding is the more useful one: **the per-category data refutes the routing
plan this ablation was run to inform.** The category designated as the flagship cheap class
turned out to be the second most budget-sensitive in the suite.

---

## 1. Setup

Five points, identical pipeline, identical rows. Only `--max-context-chars` moved.

* **Config:** the v2.0 anchor from `make repro-everything` — `--reasoner direct --use-ltm
  --ltm-backend in_memory --no-llm-promoter --retriever hybrid --session-aware-retrieval
  --session-top-k 12 --turns-per-session 6 --top-k 80 --rerank --rerank-to 24`
* **Reader:** `openai/gpt-oss-120b` via OpenRouter/DeepInfra, seed 0
* **Judge:** `meta-llama/llama-3.3-70b-instruct` (`rescore_with_judge`, which overwrites
  `correct` in place — so `metrics.accuracy` in `judged.json` *is* the judged number)
* **Rows:** `samples/budget_strat_250.json` — frozen, category-proportional to within 0.2pp
* **Cost:** $0.47 total, ~2.5h wall clock
* **Reproduce:** `bash scripts/run_budget_ablation.sh` → `make budget-curve`

### Why there is no 64 000 point

Phase 0 measured the anchor retriever's actual output at p50 26 574 chars / max 44 346. The
`--max-context-chars 64000` flag in the published config **never binds on any row** — it is
byte-identical to running unbounded. Re-measuring it would have cost ~95 min to learn nothing,
so 32 000 is the reference.

**32 000 bound on only 12% of rows** (23.7 of 24 items admitted on average), so it is very
nearly the unbounded control anyway. That is fortunate rather than designed, and it should be
stated whenever the reduction factor is quoted.

---

## 2. The curve

| budget (chars) | avg ctx tokens | p95 | items admitted | judged acc | vs anchor | rows bound | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **32 000** | 4 866 | 6 919 | 23.7 / 24 | **77.2%** | anchor | 12% | $0.172 |
| **16 000** | 3 114 | 3 560 | 17.5 / 24 | **75.6%** | −1.6pp | 82% | $0.124 |
| 8 000 | 1 505 | 1 777 | 10.4 / 24 | 72.0% | −5.2pp | 98% | $0.078 |
| 4 000 | 680 | 892 | 6.3 / 24 | 64.8% | −12.4pp | 100% | $0.053 |
| 2 000 | 300 | 440 | 3.7 / 24 | 53.6% | −23.6pp | 100% | $0.041 |

**Only 16 000 holds within the 2pp target** — 75.6% at 3 114 tokens, a **1.6× reduction**
against the anchor's 4 866. The plan hoped for ~3.5×. It is not there.

Two sanity checks pass:

* **The anchor reproduced.** 77.2% sits inside the ±3–5pp variance band around the published
  ~74–76.4%, so the pipeline is unchanged.
* **chars/token measured 4.55–4.64** across all five points, consistent with Phase 0's 4.70 on
  a different row set, and nowhere near the 4.0 that every prior estimate assumed.

**Items offered was exactly 24.0 at every budget** — the rerank keep count. That is the
ceiling the char budget is cutting *into*, which is why `--rerank-to` remains the other half
of this lever (see §5).

---

## 3. Per-category — and the finding that breaks the plan

Judged accuracy by category. Anchor column absolute; the rest are the same measurement, not
deltas.

| category | n | ctx@32k | 32k | 16k | 8k | 4k | 2k | 32k→2k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 35 | 5 211 | 94.3 | **100.0** | 94.3 | 94.3 | **91.4** | **−2.9** |
| knowledge-update | 39 | 5 217 | 66.7 | 64.1 | 69.2 | 59.0 | 56.4 | −10.3 |
| temporal-reasoning | 66 | 5 048 | 77.3 | 75.8 | 72.7 | 62.1 | 59.1 | −18.2 |
| single-session-assistant | 28 | 3 015 | **100.0** | 96.4 | 89.3 | 82.1 | **75.0** | **−25.0** |
| single-session-preference | 15 | 4 333 | 53.3 | 46.7 | 46.7 | 33.3 | 26.7 | −26.6 |
| multi-session | 67 | 5 194 | 70.1 | 67.2 | 59.7 | 55.2 | **23.9** | **−46.2** |

### 3.1 The routing premise was wrong

`docs/LOW_BUDGET_PLAN.md` §Phase 2 proposed a SMALL budget class containing all three
single-session categories, on the argument that *single-session answers live in one session
and do not need wide expansion regardless of current accuracy*.

The data splits that class in half:

* **`single-session-user` is genuinely robust.** 94.3% at the anchor, **91.4% at 2 000 chars**
  — a 2.9pp loss for a **94% token cut** (4 866 → 300 tokens, 23.7 → 3.7 items). This is the
  only category that behaves the way the plan predicted, and it behaves that way emphatically.
* **`single-session-assistant` is the second most budget-sensitive category in the suite.**
  100% → 75.0%, a 25-point decline, **monotone across all five points**. It was the designated
  flagship of the cheap class — saturated, single-session, obviously safe to starve. It is not.
* **`single-session-preference` breaks worst of the three** (−26.6pp), from an already-weakest
  53.3% baseline.

So neither proposed premise predicts budget tolerance. Not *current accuracy* (assistant was
at 100% and broke). Not *answer span* (all three are single-session and they diverge by 22pp
of decline). **`multi-session` collapsing to 23.9% is the one thing both premises got right.**

### 3.2 Why assistant breaks despite being single-session

Its anchor context is the **smallest of any category** — 3 015 tokens against ~5 200 for the
others. Being already compact, every budget step removes a larger fraction of what it had.
`single-session-user` starts at 5 211 tokens and can afford to lose most of them; assistant
cannot.

That points at a better routing signal than category: **the ratio of budget to the category's
natural context size**, not the category label. Worth testing directly in Phase 2.

### 3.3 Statistical honesty

Per-category cells are small and the deltas must be read with that in mind:

* `single-session-preference` n=15 → **one question = 6.7pp.** Its entire curve is four
  questions wide. Treat it as directional only.
* `single-session-assistant` n=28 → **one question = 3.6pp.** The 32k→16k step is a single
  question. The **full-range −25pp is 7 questions and is monotone across five points**, which
  is what makes it credible; no individual step is.
* `knowledge-update` is **non-monotone** (69.2% at 8k exceeds 66.7% at the anchor). At n=39
  that is one question of noise, not a real inversion. Do not build anything on it.
* `multi-session` (n=67) and `temporal-reasoning` (n=66) are the only cells large enough to
  read step-by-step.

Per §10 of the production module, resolving a 3pp difference at this baseline needs ~3 100
items per arm. **Nothing here resolves 3pp.** What it resolves is the *shape* — a sharp elbow
and a 43-point spread in category sensitivity — and those are far outside noise.

---

## 4. What this means for the plan

### 4.1 Revised Phase 2 budget classes

The table in `docs/LOW_BUDGET_PLAN.md` should not be built as written — it would starve
`single-session-assistant`, which cannot take it.

| class | categories | budget | justification |
|---|---|---:|---|
| **SMALL** | `single-session-user` | 2 000–4 000 | −2.9pp at a 94% token cut |
| **MEDIUM** | `single-session-assistant`, `knowledge-update`, `single-session-preference` | 16 000 | assistant is compact already; the other two are weak and noisy — do not squeeze |
| **LARGE** | `multi-session`, `temporal-reasoning` | 32 000 | −46pp and −18pp respectively at 2k |

Weighted on the 250-row mix: `(35·3 + 82·16 + 133·32) / 250` ≈ **22 300 chars vs 32 000** —
about **1.4×**, not the 1.7× the plan projected from its incorrect class assignment.

### 4.2 The 4× story has to come from Phase 3

Budget cuts alone reach 1.6× at the 2pp bar, and routing adds perhaps 1.4×. Neither gets near
the headline the plan was written around. **Phase 3 — facts instead of raw turns — is now the
load-bearing phase, not a refinement.**

The 2 000-char column is the argument for it: at 3.7 admitted items, `single-session-user`
still answers 91.4% of questions correctly. Something in that regime is working with almost no
context. If atomic LTM facts can carry the other categories the way raw turns carry
`single-session-user`, the compression story is real. If they cannot, the honest headline is
the curve itself.

### 4.3 Phase 4 is structural, not opportunistic

`items_offered` was **exactly 24.0 at every budget** — the rerank keep count, untouched by any
char cap. The char budget only cuts into what rerank already selected. Testing
`--rerank-to 24 / 16 / 12` now measures the *other* half of the same lever, and does it at a
point on the curve where we know what the accuracy cost of losing items looks like.

---

## 5. Reproducing

```bash
python scripts/build_budget_sample.py          # frozen 250-row stratified subset
bash scripts/run_budget_ablation.sh            # 5 points, ~2.5h, ~$0.47, resumable
make budget-curve                              # table + csv + png
```

Raw outputs: `results/budget_{32000,16000,8000,4000,2000}/` — each with `budget_report.json`
(per-row context chars/tokens, items admitted vs dropped, cost) and `judged.json`.

---

## 6. Verdict

**Reported honestly:** the pre-registered target was missed. Within 2pp we achieve 1.6× token
reduction, not the 3.5× the plan aimed at, and the plan's own routing design was refuted by
its own measurement.

**What was gained:** a sharp, reproducible elbow at 16 000 chars; a measured 43-point spread in
category budget-sensitivity that no prior data showed; a corrected chars/token constant; and a
Phase 2 design grounded in data rather than in a plausible-sounding premise about answer span.

The negative result on the target is worth more than the routing table would have been. Had
Phase 2 been built to the original plan, `single-session-assistant` would have been routed to
an 8 000-char class and lost ~11pp, and the aggregate drop would have been attributed to the
router's classifier rather than to the class assignment.
