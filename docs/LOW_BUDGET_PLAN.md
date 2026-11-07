# Accuracy at low budget — measurement + build plan

*Status: proposed · 2026-08-05 · picks up after v2.0 (LongMemEval-S ~76.4% judged) and
`findings/roadmap_v3.md`*

**Thesis to prove:** *the same accuracy at a quarter of the tokens.* Not "+2 more points."
The cost curve is the publishable result and the claim the regulated-infra positioning needs.

**Target to write down now:** hold within **2pp of the 64K-char anchor at an 8–16K-char
average budget** (≈4× token reduction). If the ablation shows that is reachable, it is the
arXiv result and the launch post in one.

---

## 0. Anchor and vocabulary

The anchor config is `repro-everything` ([Makefile:214](Makefile:214)):

```
--provider openrouter --model openai/gpt-oss-120b --openrouter-provider DeepInfra --seed 0
--reasoner direct --use-ltm --ltm-backend in_memory --no-llm-promoter --retriever hybrid
--session-aware-retrieval --session-top-k 12 --turns-per-session 6
--top-k 80 --max-context-chars 64000 --answer-max-tokens 2048 --span-fallback
--decompose-max-items 60 --rerank --rerank-to 24
```

→ ~74% judged (73.6–75.6% across runs); 76.4% on the `results/ws4_full500` run.

**Chars vs tokens — resolve this before anyone writes a number down.** The only real budget
knob today is `--max-context-chars`, and the anchor sets it to **64 000 characters ≈ 16K
tokens**. So the "64K" in the ladder is *chars*, not tokens. This plan runs the ladder in
chars (matching the knob and the anchor) and **reports both axes**, with the token axis
measured from the actual answer prompt rather than divided by four. Restated target: within
2pp of 64K chars at 8–16K chars, i.e. **~16K tokens → ~2–4K tokens**.

### Two premises that need correcting first

**1. Preference is not a safe cheap category.** Judged per-category from
`results/ws4_full500/judged.json` (n=500):

| category | n | judged |
|---|---:|---:|
| single-session-assistant | 56 | **98.2%** |
| single-session-user | 70 | **92.9%** |
| temporal-reasoning | 133 | 75.2% |
| multi-session | 133 | 69.9% |
| knowledge-update | 78 | 66.7% |
| single-session-preference | 30 | **56.7%** |

Assistant (98%) and user (93%) are the strong ones. **Preference is the weakest category in
the suite**, not a 91–96% one. So the router cannot be justified as "these are already
saturated, starve them" — it has to be justified on **span**: single-session answers live in
one session and do not need 12 sessions of expansion *regardless of how accurate they
currently are*. Budget class follows answer span, not current accuracy.

**2. Cutting budget is unusually safe here, for a reason worth stating.**
`findings/roadmap_v3.md` established that of 118 failures, **116 had the answer session
already retrieved** — the residual is reader-bound, not retrieval-bound. That is exactly why
"same accuracy, fewer tokens" is a plausible headline and "more accuracy" is not. Frame every
result in this plan as a **cost** result. Any accuracy gain is a bonus, not the claim.

---

## Phase 0 — Instrumentation (blocker) — **code complete, awaiting the smoke run**

**We cannot currently report average tokens.** This must land before any ablation run, or
Phase 1 produces an accuracy curve with no x-axis.

What is broken:

1. **Prompt tokens never reach the row.** `TelemetryCounter` already tracks
   `answer_prompt_tokens` ([evals/longmemeval/telemetry.py:54](evals/longmemeval/telemetry.py:54)),
   but the direct answerer — the reasoner the anchor uses — sets `last_telemetry` by hand
   ([bootstrap_ollama.py:3389](evals/longmemeval/bootstrap_ollama.py:3389),
   [:3736](evals/longmemeval/bootstrap_ollama.py:3736)) and never opens the counter.
   Only the decomposed path wraps it ([:4199](evals/longmemeval/bootstrap_ollama.py:4199)).
2. **The row field reads from the wrong source.** `pre_opt_total_tokens` comes from
   `opt.get("pre_total", 0)` ([baseline.py:673](evals/longmemeval/baseline.py:673)), which is
   the optimizer's number — 0 in direct mode. Hence `avg_context_tokens_pre_opt: 0.0` in
   every direct run.
3. **The budget is enforced as a byte slice.**
   [bootstrap_ollama.py:3467](evals/longmemeval/bootstrap_ollama.py:3467) does
   `"\n".join(lines)[: self._max_context_chars]` — it cuts mid-line, so the last turn arrives
   truncated mid-sentence. At 64K that rarely matters; at 8K it will silently corrupt the
   tail item and confound the ablation.

Work:

- **P0.1** Wrap the direct answerer in `start_row_telemetry()` / `end_row_telemetry()` the
  same way `answer_question` does, so real provider `usage` counts land in `last_telemetry`.
- **P0.2** Plumb `answer_prompt_tokens` onto the row and into `metrics` as
  `avg_answer_prompt_tokens` + p50/p95. Add `context_chars` alongside it, so the chars↔tokens
  ratio is measured per run instead of assumed at 4.0.
- **P0.2b** **Add the reader's pricing.** `_cost_for` returns `0.0` for any model outside
  `_DEFAULT_PRICING` ([telemetry.py:138](evals/longmemeval/telemetry.py:138)), and that table
  holds only `gpt-4o-mini` / `gpt-4o`. So **every gpt-oss-120b run in `results/` records
  cost ≈ $0** — the `total_cost_usd: 0.0062` on the 500-row `ws4_full500` run is gpt-4o-mini
  judge calls, not the reader. Add `openai/gpt-oss-120b` and
  `meta-llama/llama-3.3-70b-instruct` with their current OpenRouter rates. Non-negotiable:
  the headline of this whole plan is a cost claim, and the harness cannot currently report
  cost for the model making it.
- **P0.3** Replace the byte slice at `:3467` with **whole-item admission**: append lines
  while under budget, drop the item that would overflow. Keep the existing `--max-context-chars`
  semantics; add `--max-answer-prompt-tokens` as a token-native budget that admits items by
  measured token length. Both enforce at the same point.
- **P0.4** Add `--budget-report` to dump per-row `{question_type, context_chars,
  answer_prompt_tokens, items_admitted, items_dropped}` to the output dir.

### What landed

| item | where |
|---|---|
| tokenizer + whole-item admission | `evals/longmemeval/budget.py` (new) |
| counter wrap on the direct reasoner | `bootstrap_ollama.py` — `answer_question` → `_answer_question_direct` |
| whole-item budget at the assembly site | `bootstrap_ollama.py` (was the `[:max_chars]` byte slice) |
| reader + judge pricing, prefixed-id lookup | `telemetry.py` — `_DEFAULT_PRICING`, `_pricing_for` |
| row + metric fields, per-category context cost | `baseline.py` |
| `--max-answer-prompt-tokens`, `--budget-report` | `bootstrap_ollama.py` |
| 21 tests | `tests/unit/evals/test_budget_accounting.py` |

Two things worth knowing before reading Phase-1 output:

- **The measured ratio is ~4.1 chars/token on real turns**, not the 3.9 first sampled and not
  the assumed 4.0. It is now reported per run (`measured_chars_per_token`) and per row, so the
  chars↔tokens conversion in this document never has to be guessed again.
- **`pct_rows_budget_bound` is the run's validity check.** If it reads 0, the retriever never
  offered enough context to fill the budget, and that run is *not* a distinct point on the
  curve — any accuracy delta is noise, not a budget effect. `--budget-report` warns loudly when
  this happens. Check it before reading any Phase-1 number.

**Exit criterion:** run the anchor at n=50 and confirm `avg_context_tokens` and
`total_cost_usd` are both non-zero, `measured_chars_per_token` is reported, and
`pct_rows_budget_bound` is non-zero at 8 000 chars.

```bash
python -m evals.longmemeval.bootstrap_ollama --provider openrouter --model openai/gpt-oss-120b --openrouter-provider DeepInfra --seed 0 --reasoner direct --use-ltm --ltm-backend in_memory --no-llm-promoter --retriever hybrid --session-aware-retrieval --session-top-k 12 --turns-per-session 6 --top-k 80 --max-context-chars 64000 --answer-max-tokens 2048 --span-fallback --decompose-max-items 60 --rerank --rerank-to 24 --budget-report --full --yes --no-smoke --limit 50 --output results/phase0_verify
```

---

## Phase 1 — The budget ablation (the curve)

### What Phase 0 changed about this phase

The n=50 verification measured the anchor retriever's actual output, and two planning
assumptions did not survive it.

**1. The "64K anchor" was never a 64K configuration.** With `--top-k 80 --rerank-to 24`, the
anchor delivers a p50 of **26 574 chars / 5 907 tokens**, max 44 346 chars. The
`--max-context-chars 64000` flag *never binds on any row* — 0% budget-bound. Every row
admitted exactly 24 items, the rerank keep count.

| planned point | % of rows it would clip | verdict |
|---:|---:|---|
| 64 000 | 0% | **no-op** — identical to unbounded |
| 32 000 | 30% | weak |
| 16 000 | 92% | binding |
| 8 000 | 100% | binding |
| 4 000 | 100% | binding |

The original ladder spent its top point on a no-op and its second on a budget clipping under a
third of rows. **Revised ladder: 64 000 (unbounded control) / 32 000 / 16 000 / 8 000 /
4 000 / 2 000.** The 64 000 point stays — it reproduces the published config and anchors the
curve — but it is labelled a control, not a budget. The 2 000 point is added because Phase 3's
facts-only mode targets exactly that regime.

**2. The measured ratio is 4.70 chars/token**, not 4.0 and not the 4.1 seen on synthetic text.
So the anchor is ~5.9K tokens, and the original target ("8–16K chars") was a 1.7–3.3× cut, not
the 4× it was written to mean. Restated: **hold within 2pp of the anchor at ≤ 8 000 chars
(≤ ~1.7K tokens)**, which is a genuine ~3.5× reduction against the *real* anchor.

**3. `--rerank-to` is the real budget knob, not the char cap.** Context size is governed by the
rerank keep count until the char budget drops below ~16 000. That makes **Phase 4 structural
rather than opportunistic** — it is the other half of the same lever. Worth pulling forward if
Phase 1 shows the curve is flat down to 8 000.

### The runs

Six points, same pipeline, only `--max-context-chars` moves:
**64 000 (control) / 32 000 / 16 000 / 8 000 / 4 000 / 2 000**.

Run it on a **fixed stratified subset of n=250** (proportional by category, frozen to a
`--question-ids-file` so every later phase compares against the identical rows), not the full
500. The reason is **wall clock, not money** — at p50 ≈ 9–10s/row, five full-500 runs is ~7h
serial and more under `--rpm` throttling; five 250-runs is ~3.5h, and the per-category cells
still have n≥15. Confirm only the two most interesting points at full 500 afterwards.

The subset is **frozen and committed** at [samples/budget_strat_250.json](samples/budget_strat_250.json)
— proportional to within 0.2pp on every category, smallest cell n=15, regenerable and
verified by test. Do not rebuild it mid-ablation: earlier points would have run on different
rows and stop being comparable, which is the one thing the whole curve depends on.

```bash
make budget-ablation
```

Runs all five points and judges each with `rescore_with_judge` (llama-3.3-70b, the anchor's
judge) — the substring scorer reports 0% on preference and would make the curve unreadable.
Resumable: a point that already has both `budget_report.json` and `judged.json` is skipped, so
an interruption doesn't cost a second run. Pass specific budgets to redo just those:
`bash scripts/run_budget_ablation.sh 16000 8000`.

The runner refuses to start if tiktoken is missing — a curve whose x-axis is a chars/3.9
estimate is not a measurement, and it is better to fail loudly than to publish a caveat.

**Deliverable:** `make budget-curve` → one table and one chart, judged accuracy vs measured
`avg_context_tokens`, **broken out per category**. The per-category breakout is the whole
point: it *is* the router's design input. The aggregate curve alone tells you nothing about
where to spend.

**What to look for:** the elbow. Expect single-session-* flat all the way down and
multi-session/temporal to break first. If multi-session degrades gracefully too, the story is
bigger than routing and Phase 2 gets simpler.

**Read the `bound` column before anything else.** It is `pct_rows_budget_bound` — the share of
rows where the budget actually dropped a retrieved item. A point at 0% is flagged
`NOT BINDING` and excluded from the headline: the retriever never offered enough context to
fill that budget, so it is the *same* configuration with a different number on the flag, and
its accuracy delta is run noise. This is the most likely way to produce a confident wrong
result, which is why the renderer refuses to let such a point win.

---

## Phase 2 — Budget router (the biggest lever)

**Naming:** `--router` is already taken — it is the v3 deterministic count router and requires
`--synthesis` ([bootstrap_ollama.py](evals/longmemeval/bootstrap_ollama.py)). The new flag is
**`--budget-router`**. Do not overload the old one.

The classifiers already exist; this is a mapping, not new inference:

- `evals/longmemeval/task_router.py` — six `TaskMode`s, dataset-hint + wording signals.
- `evals/longmemeval/question_type.py` — pure-regex `QuestionType`.

Both are deterministic and free. **No LLM in the router** — a router that costs an LLM call
undermines the cost claim it exists to make.

Proposed budget classes (calibrate the actual numbers from the Phase-1 elbow, do not hardcode
these before the curve exists):

| class | modes | chars | session-top-k / turns |
|---|---|---:|---|
| **SMALL** | `FACT_LOOKUP`, `ASSISTANT_MEMORY_LOOKUP`, `PREFERENCE_PROFILE` | 8 000 | 3 / 4 |
| **MEDIUM** | `KNOWLEDGE_UPDATE` | 16 000 | 6 / 4 |
| **LARGE** | `MULTI_SESSION_AGGREGATE`, `TEMPORAL_REASONING` | 64 000 | 12 / 6 |

Weighted average on LongMemEval-S's mix (156 single-session + 78 KU + 266 multi/temporal):
`(156·8 + 78·16 + 266·64)/500` ≈ **37K chars vs 64K** — a 1.7× cut before any other change.
That alone does not reach the 4× target; **Phase 3 is what gets LARGE down**, and Phase 2 is
what makes Phase 3 safe to apply.

**Router-accuracy guard.** The router is a classifier and it will mis-route. Measure it
directly: log predicted class vs dataset `question_type`, and report **mis-route rate** and
**accuracy on mis-routed rows**. A router that is 95% accurate but craters the 5% it gets
wrong is worse than no router. Add a **fail-open rule**: if the reader returns an
abstain/IDK under SMALL, re-ask once at LARGE and count the retry tokens in the average.
That converts a routing error from a wrong answer into a cost, which is the right trade.

---

## Phase 3 — Inject facts, not turns, at low budgets

This is the phase that turns the tiered *architecture* into a tiered **cost model** — the part
that is genuinely Continuum's and not a generic RAG knob.

The pieces are already in place:

- `ContinuumLTMHaystackStore` ([continuum_ltm_store.py:118](evals/longmemeval/continuum_ltm_store.py:118))
  extracts atomic facts with supersession and adds them to the retrieval corpus tagged
  `metadata["source"] = "ltm_fact"`.
- The direct assembler **already sorts facts first and labels them** on knowledge-update
  questions — `[CURRENT FACT] …`
  ([bootstrap_ollama.py:3440–3467](evals/longmemeval/bootstrap_ollama.py:3440)).

So the change is: **generalize the KU branch into a budget-pressure branch.**

- **P3.1** `--facts-first` — under a budget class of MEDIUM or below, admit *all* live
  `ltm_fact` items first, then fill the remaining budget with raw turns. Facts are short;
  this is where the 4× comes from.
- **P3.2** `--facts-only-below N` — below N chars, admit facts and MTM summaries only; no
  raw turns at all. This is the direct analogue of Mem0's compression strategy, except the
  facts carry supersession and validity intervals, which Mem0's do not.
- **P3.3** Escalation: when the reader abstains under facts-only, expand to raw turns for
  that row and record it. Report the **expansion rate** — "we only paid for verbatim context
  on X% of questions" is the headline sentence of this phase.

**Ablation to run:** facts-only / facts-first / turns-only at 8K and 4K. Three configs × two
budgets on the same 250 subset.

**Risk to watch:** counting and aggregation questions were **58/118 of the v2.0 failures**
(`findings/roadmap_v3.md`) and are the category most likely to be *hurt* by dropping raw
turns — you cannot count instances you did not admit. Keep multi-session on LARGE + turns
throughout Phase 3 unless the data says otherwise.

---

## Phase 4 — Tighten the rerank tail (opportunistic)

`--rerank-to` is 24 in the anchor, default 4. Test **24 / 16 / 12** at the two budgets that
Phase 1 identifies as interesting.

Report **recall@k-post-rerank**, not just accuracy — the question is how sharp the cutoff can
get before gold context is dropped, and accuracy alone conflates that with reader error. The
harness already tracks `retrieved_session_ids` vs `expected_session_ids` and `partial_recall`
per row, so this is a scoring change, not a pipeline change.

If 12 holds, the hit budget halves for free and Phase 3's budgets all get easier.

---

## Phase 5 — One frontier-reader run (the ceiling)

One run of the best low-budget config **and** one of the 64K anchor, on a frontier reader via
the free tiers already planned (Groq / Cerebras / GitHub Models — `--provider` already
supports the OpenAI-compatible path).

Two questions it answers, both needed for the paper:

1. What does the memory layer deliver when the reader is not the bottleneck? (v2.0 is
   reader-bound at ~74%.)
2. **Does the low-budget config stay close to the anchor on a stronger reader?** This is the
   one that matters — if the gap widens with a better reader, the compression is lossy in a
   way gpt-oss-120b was too weak to notice, and the headline needs qualifying.

Run both, or run neither. A frontier number for only one config is not evidence.

---

## Sequencing, cost, and what ships

Cost basis: gpt-oss-120b via DeepInfra at ≈ **$0.10/M input, $0.50/M output** — *confirm
against OpenRouter's current listing before quoting these anywhere.* At the 64K anchor a row
is ~16K input + ~400 output tokens ≈ **$0.0018**, so a full 500-row run is **~$0.90**. Judge
passes (llama-3.3-70b, ~600 tokens/row) add ~$0.10 per 1 000 rows. **Dollars are not the
constraint here; wall clock is.**

| phase | wall clock | rows | cost | gate |
|---|---|---:|---:|---|
| 0 — instrumentation | ~half a day | 50 | ~$0.10 | tokens *and* cost non-zero |
| 1 — ablation | ~3.5h | 1 250 | ~$1.15 | the per-category curve exists |
| 2 — budget router | 1 day build + ~1h | 500 | ~$0.40 | mis-route rate reported |
| 3 — facts-not-turns | 1–2 days + ~2h | 1 500 | ~$0.55 | expansion rate reported |
| 4 — rerank tail | ~1h | 750 | ~$0.30 | recall@k-post-rerank reported |
| 5 — frontier reader | ~1h | 500 | $0 (free tier) | both configs, or neither |
| full-500 confirmation | ~2h | 1 000 | ~$1.80 | the headline number |

**Total ≈ $4–5; critical path (0→3) ≈ $2.20.** Any estimate in the tens of dollars for this
plan is wrong — an earlier draft of this document said ~$27 by mis-reading the `<$5` ceiling
in `repro-everything` ([Makefile:214](Makefile:214)) as a per-run measurement. It is an upper
bound covering a full run *plus* judge *plus* a LOCOMO smoke.

**0 → 1 → 2 → 3 is the critical path.** 4 and 5 are opportunistic and can slip without
blocking the result.

**Success criterion (write it down now, judge against it honestly):** judged accuracy within
**2pp of the 64K anchor** at an **8–16K average measured context**, on the full 500 with the
same judge. Confirm the winning config at n=500 before claiming it — the 250-row subset is
for finding the config, not for the headline number.

**If it misses:** report the curve anyway. "Accuracy degrades gracefully to X K and breaks at
Y K, and here is which question types break first" is a real finding, and it is the finding
the paper needs either way. A negative result on a measured curve beats no curve.

### Deliverables

1. `findings/budget_curve_2026-08.md` — the ablation, per-category, both axes.
2. `findings/charts/budget_curve.py` — accuracy vs measured tokens.
3. A `make repro-budget` target pinning the winning config, in the style of `repro-everything`.
4. A paper section: *cost–accuracy frontier of a tiered memory*, with the router and
   facts-not-turns ablations as the mechanism.
