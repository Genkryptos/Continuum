# Pre-registration — accuracy & design-space study

*Created 2026-08-11 · repo SHA at registration: **82a7802** · executes Phase 0.1 of
`docs/ACCURACY_AND_BENCHMARK_PLAN.md`*

> **Status: DRAFT — predictions require the author's sign-off before any run.**
> A pre-registration the author did not endorse is worthless. §5 predictions were
> drafted from prior findings; edit them to what you actually believe, then change
> this line to `RATIFIED <date>` and commit **before** running anything in §4.

---

## 1. Why this file exists

Two claims have already been withdrawn or corrected on this project: the ANN-index
recall finding, and a misremembered "Mem0 = 25%/15%" figure that survived a context
compaction. Both were caught, which is good — but both were caught *after* being
believed for a while.

A committed prediction converts a retraction into a legitimate negative result. The
rule is therefore:

**No number produced after this file is ratified may enter a comparison table
unless the run it came from satisfies §2 (admissibility) and its cell appears in
§5 (predictions).**

Adding a cell later is allowed. Adding it *after seeing its result* is not — append
a dated amendment instead, and mark the cell `post-hoc: true`.

---

## 2. Admissibility rules

A run is admissible iff:

1. Its result JSON carries a complete `versions` block (§3.3).
2. Its answerer, judge, top-K, and context budget match §3.1 exactly.
3. Any deviation is recorded as `non_comparable: true` with a reason string.

Runs marked `non_comparable` are still useful for debugging. They may **not** be
placed beside another system's number.

> **Precedent this rule encodes:** BEAM was previously run with a gpt-oss-120b
> answerer + gpt-4o-mini judge at top-80, against Mem0's published gpt-5-answerer
> + gpt-5-judge at top-200 on the 1M split. Those numbers were never comparable.

---

## 3. The frozen harness

### 3.1 ⚠️ The CLI defaults are NOT the winning configuration

`evals/longmemeval/bootstrap_ollama.py` still defaults to the **legacy May config**.
Running it bare reproduces the superseded ~34% ceiling, not the v1 result. Every
value below must be passed explicitly.

| flag | CLI default | **pinned value** | why it matters |
|---|---|---|---|
| `--reasoner` | `legacy` | **`direct`** | `iterative` measured net-negative and was cut; `legacy` is the May path |
| `--retriever` | `cosine` | **`hybrid`** | dense ⊕ BM25 ⊕ RRF |
| `--session-aware-retrieval` | off | **on** | multi-session recall 42% → 85% |
| `--use-ltm` | off | **on** | the supersession path — the mechanism under test |
| `--answer-max-tokens` | `256` | **`2048`** | reasoning models burn the budget thinking and emit nothing |
| `--max-context-chars` | `32000` | **`64000`** | the 8k cap silently dropped answer-bearing turns at recall=1.0 |
| `--top-k` | `8` | **`8`** | unchanged |
| `--rrf-k` | `60` | **`60`** | Cormack 2009 |
| `--seed` | `0` | **`0`** | plus `--openrouter-provider` pinned for reproducibility |

Answerer: **`openai/gpt-oss-120b`** via OpenRouter (the v1 answerer) for all
single-size runs. The ladder (§4.3) varies this axis deliberately and is the only
place it may differ.

### 3.2 Judge

| field | value |
|---|---|
| model | **non-reasoning only** — `llama-3.3-70b-instruct` |
| `JUDGE_SYSTEM_PROMPT` sha256 | `60222fd9751dd5b713c949d826682aeb646f8e0f72c310804fd830bb375c4599` |
| prompt length | 744 chars |
| `max_tokens` | `5` |

**A reasoning model may never be used as the judge.** At `max_tokens=5` gpt-oss-120b
spent its budget thinking and returned "no" for every row, flipping 38/50 correct
answers to wrong. The same failure hit it as Mem0's fact extractor (truncated JSON).
Rule: reasoning models answer; structured/classification sub-tasks use a fast
non-reasoning model.

If the prompt hash changes, every previously-judged number is void. Re-hash and
re-run, or record the old hash alongside the old numbers.

### 3.3 Required `versions` block

```json
{
  "versions": {
    "continuum_sha": "82a7802",
    "mem0ai": "0.1.67",
    "graphiti_core": null,
    "embedder": "bge-m3",
    "answerer": "openai/gpt-oss-120b",
    "judge": "llama-3.3-70b-instruct",
    "judge_prompt_sha256": "60222fd975...",
    "non_comparable": false
  }
}
```

`mem0ai` is currently **0.1.67** in this environment — the paper's baseline. The
2.0.17 arm must run in a separate venv and record its own value. A run that cannot
name its mem0 version is inadmissible; this is the exact confusion that made the
"Mem0 deletes superseded facts" claim version-ambiguous.

---

### 3.4 Retrieval-budget arms (added 2026-11-07)

The 500-question run of 2026-11-07 measured `items_offered = items_admitted =
8.0` and `items_dropped = 0.0` on **every one** of the 55 `missing_aggregation`
failures. The reader was never handed more than
`session_top_k × turns_per_session = 8` turns and discarded none of them, so
those failures cannot be attributed to the reader. Two arms follow, each a
deviation from §3.1 and therefore `non_comparable` against the §3.1 baseline
until run as a matched pair.

| arm | flags added to §3.1 | what it tests |
|---|---|---|
| **A-BUDGET** | `--top-k 24 --session-top-k 8 --turn-rerank global` | whether multi-session accuracy is bounded by the fixed turn budget |
| **A-CHRONO** | `--chrono-sort` | whether presenting turns in timeline order fixes the "both values in context, reader took the retired one" failure |

Two defects found while building these arms, both of which mean a previously
reported result measured something other than its label:

1. **`--chrono-sort` was a silent no-op under `--session-aware-retrieval`.**
   `SessionAwareSemanticRetriever._bundle_topk` overrode the parent's entire
   selection step, which is where `chrono_sort` was applied. Since §3.1 pins
   session-aware retrieval on, the BEAM +25pp chrono-sort finding had never
   been exercised in the pinned configuration. Fixed; regression-tested in
   `tests/unit/evals/test_session_aware_budget.py`.

   **Correction (2026-11-07, after the A-BUDGET + A-CHRONO run): that fix was
   incomplete, and the regression test could not have caught it.** §3.1 also
   pins `--retriever hybrid`, which wrapped the session-aware retriever in
   reciprocal-rank fusion. RRF reads a child's output position as its
   relevance rank and then re-sorts by fused score
   (`continuum/retrieval/rrf.py`), so the chronological order was fed to RRF
   *as relevance* and then discarded. The test asserted on the bare retriever,
   never through the fusion. Two further problems: haystack position is not
   chronological (211 of 500 LongMemEval-S haystacks list sessions out of date
   order), and the A-CHRONO arm therefore did not test its hypothesis. Its
   knowledge-update gain came from the date stamps in defect 2, which act in
   the prompt, downstream of retrieval order. Now: BM25 is fused *inside* the
   session-aware retriever over the whole haystack, before selection; the
   final context is sorted by session timestamp then turn position in the
   answer adapter, after any fusion or rerank; and
   `tests/unit/evals/test_retrieval_fixes.py` asserts on the prompt the reader
   receives through `make_adapter_factory`.
2. **`--ku-recency` instructed the reader to "prefer the latest statement"
   while presenting undated turns in relevance order.** Only the `temporal`
   branch date-stamped turns. The instruction was unactionable, which is a
   better explanation of the 14 `stale_fact` + 16 `wrong_span`
   knowledge-update failures (at 98.7% session recall) than reader error.
   Knowledge-update turns now carry `[YYYY-MM-DD]` and the prompt points at
   it.

`max_items` on the session-aware retriever is now floored at `top_k`. Below
it, the cosine arm of the hybrid supplied fewer candidates to RRF than BM25
did, so raising `--top-k` alone would have shrunk the effective budget rather
than widening it. This also un-breaks `--reranker`, which `bootstrap_ollama.py`
notes is a no-op whenever the retrieved pool is not larger than `rerank_to`.

### 3.5 Audit-fixes arm (added 2026-11-07)

A mechanism-level audit of the 92 failures of the A-BUDGET + A-CHRONO run
found that accuracy is **93.2% on the 395 rows where every gold turn reached
the reader**, and 34.9–38.8% where some did not. The remainder is
turn selection, not the reader. This contradicts `docs/report.md` §5
("reader-bound") and `findings/roadmap_v3.md` §1/§3, both of which were
derived from *session* recall. Fixes, run as one arm (`run_arm_audit_fixes.sh`):

| fix | flag / code | defect |
|---|---|---|
| F1 | `--turn-rerank floor` | `global` let one session take the whole budget; `quota` stopped at `session_top_k × turns_per_session` |
| F1/F2 | `--retriever hybrid` now fuses inside session-aware selection | the wrapper re-sorted the selection (above) |
| F2 | `--chrono-sort`, applied post-fusion, by date | see the correction in §3.4 |
| F4 | `--pref-conditioning --reflect --reflect-types single-session-preference` | both were off: 11 of 16 preference failures answered "I don't know." with every gold turn in context |
| F5 | composite scorer honoured under session-aware retrieval | the override used raw cosine; `--scorer-weights` and `--tau-hours` were inert while being logged as active. The arm passes `--scorer-weights 1,0,0,0` so it reproduces what actually ran; recency is the separate A-RECENCY arm |
| S1 | presupposition clause in every factual prompt | 30 questions name something never mentioned beside a near-miss; 53.3% vs 83.4% |
| S2 | count/list questions keep the dated aggregate prompt | the temporal and KU detectors pre-empted it (22 rows at 63.6%, 6 at 33.3%) |
| S4 | KU earlier-value questions get their own prompt | the latest-wins instruction answers "initially / previous" with the current value |
| M7 | second-person answers match first-person gold, equality stage only | one clear scorer false negative |
| S5 | `versions` block with derived `deviations` | results carried no provenance, so were inadmissible by §2.1 |
| — | `--rerank-to` default 4 → 0 | enabling `--rerank` at `--top-k 24` would have cut the context to 4 turns |

**Not built: a temporal (date-window) retrieval channel.** Hindsight has one
and the audit proposed it. Measured first (`python3.12 -m
evals.audit.temporal_window_selectivity`): on LongMemEval-S, answer sessions
fall inside the question's window no more often than distractor sessions do
(0.57 vs 0.50 for windows under 8 days; 0.89 vs 0.87 above 45), because the
median haystack spans 10 days. A channel with no selectivity adds noise.

**Not re-enabled: `--aggregation-v2`.** `findings/roadmap_v1.1.md` measured it
net-neutral at 7× the tokens, and the audit's evidence (counts are 95.1%
correct at complete evidence; under-counts equal the number of gold turns
retrieved) says the bottleneck is upstream of the prompt.

**Pending arm A-RECENCY:** the audit-fixes arm with the default
`--scorer-weights 0.45,0.25,0.20,0.10` instead of `1,0,0,0`. Recency was never
active under session-aware retrieval before, so its effect is unmeasured.
*Update:* on gold-turn recall (below) recency adds nothing once
`--user-turns-first` is on (0.966 vs 0.968), so the arm is deprioritised.

**Round 2 (same day), from the 45 rows still failing after the audit fixes were
re-run on the 92 prior failures (47 recovered):**

| fix | flag / code | evidence |
|---|---|---|
| user turns first | `--user-turns-first` | 35 of the 45 lacked gold turns. Outside single-session-assistant, 837/840 gold turns are user turns, but assistant replies took ~41% of a 24-turn budget. A wording detector for "what did *you* say" questions catches 54/56 single-session-assistant questions, 0/444 false positives |
| budget | `--top-k 48 --session-top-k 12 --turns-per-session 3` | `python3.12 -m evals.audit.turn_selection_ablation --sample 100` (no LLM): gold-turn recall 0.876 → **0.968**, rows with every gold turn 0.804 → **0.938**, at 18.7k context chars vs 22.9k for the A-BUDGET arm |
| preference routing | wording fallback only when the question has no type label | `--pref-conditioning` sent 35 non-preference questions (32 previously correct) to a prompt with no "not mentioned" path |
| presupposition wording | paraphrase counts as mentioned | the first wording refused 3 rows whose evidence was in context under another name |

**Regression check before the full run** (83 rows: all 30 `_abs`, the 3
paraphrase refusals, 50 previously correct rows incl. 8 from the preference
misroute set; $3.47): `_abs` 16 → 25 of 30; paraphrase refusals 0 → 0 of 3
(the rewording did not move them; they were wrong before too); guard 50 → 46
of 50. Of the 4: one is judge noise (word-for-word the same answer as the last
run, verdict flipped), one is a new presupposition over-refusal ("a woman
selling jam" vs "a jam maker (he)"), one is a preference row whose single
concrete pick the rubric judge rejected, one lost its gold turn in retrieval.
The judge flip is why the arm now passes `--openrouter-provider DeepInfra`:
§3.1 requires a pinned provider and neither run script set one; pinned, the
judge returned the same verdict 3/3 on that pair.

Measured and rejected: scoring a turn by its best sentence (incidental
mentions) moved 48 missed gold turns up and 51 down. Also from the ablation:
`floor` alone at 24 turns *lowers* recall versus the A-BUDGET structure
(0.876 vs 0.901); it pays off only together with user-first.

---

## 4. Datasets and metrics

### 4.1 LongMemEval-S — 500 questions

| category | n |
|---|---:|
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |
| multi-session | 133 |
| knowledge-update | 78 |
| temporal-reasoning | 133 |

**Metric:** LLM-judged accuracy (§3.2), reported **per category**. Substring is
reported alongside as a floor only — it under-counts paraphrase by ~12.4pp and
scores `single-session-preference` at ~0% as a pure artifact.

**Aggregate accuracy may not be reported without the per-category breakdown.**
The suite mixes categories Continuum should win with categories bounded by the
answerer's multi-hop reasoning; averaging across that boundary hides both effects.

### 4.2 `bench/` — deterministic, no LLM

| suite | n | metric |
|---|---:|---|
| supersession (e2e through Postgres) | 50 scenarios | `current()` correctness; stale-fact count |
| bi-temporal | 20 (15 point-in-time + 5 retroactive) | exact-match on as-of answer |

These are answerer-independent, which makes them the load-bearing results.

### 4.3 The answerer-size ladder

Sizes: **8B / 20B / 120B**, all other §3.1 values fixed. Two arms: `direct`
retrieval vs typed-query (once S1–S3 exist). Reported as *accuracy vs answerer
size*, per arm.

### 4.4 Out of scope by default

LOCOMO and BEAM are **not run** unless a specific design question requires them.
"We should have a number" is not a design question. If BEAM is run it must use its
own rubric-nugget judge (0 / 0.5 / 1.0 per nugget, question PASS at mean ≥ 0.5) —
substring/entailment scoring is invalid there, as it rewards confident answers that
ignore the contradiction, the exact failure BEAM exists to detect.

---

## 5. Predictions and falsifiers

**DRAFT — edit before ratifying.** Each row states what is expected, and what
result would prove it wrong. A falsified row is a publishable finding, not a
failure.

| # | cell | prediction | falsified if |
|---|---|---|---|
| P1a | A5 arm B vs arm A on `knowledge-update` | **roughly equal.** Arm A already hides stale facts via `invalidated_at`, and both arms present the same current view by construction. A large gain here would be suspicious, not welcome — investigate the candidate sets before believing it | a large gap either way → the arms are not the controlled comparison they are designed to be; check the drain filter before interpreting anything else |
| P1b | A5 arm B vs arm A on `temporal-reasoning` | **arm B improves.** It is the only arm that retains the superseded fact, so questions needing a prior state become answerable at all | no improvement → retention alone is insufficient without the typed-query path (S1–S3); the two are not independent levers, and A5 should be re-read as a prerequisite rather than a lever |
| P1c | A5 arm A information loss | arm A's `UPDATE` overwrites the target's text, so some prior values are unrecoverable; arm B's `superseded_retained` count should be > 0 on any row with a real contradiction | `superseded_retained == 0` across a full run → the promoter is not emitting DELETE/UPDATE decisions at all, and A5 measures nothing |
| P2 | `bench/` supersession × Mem0 **2.0.17** | gap vs 0.1.67 **narrows sharply** — additive storage retains the superseded fact | gap unchanged → the additive rewrite doesn't affect retrievability, and the 0.1.67 result generalizes |
| P3 | `bench/` bi-temporal × Mem0 **2.0.17** | gap **holds** — no `valid_to`, no as-of filter, so point-in-time stays unanswerable | Mem0 2.0.17 scores >60% point-in-time → additive storage alone is sufficient, and bi-temporal metadata is not the differentiator |
| P4 | `bench/` × **Graphiti** | Continuum ≥ Graphiti on point-in-time; Graphiti ≫ Mem0 | Graphiti matches Continuum → the differentiator is *determinism*, not capability, and the paper must be reframed |
| P5 | Convertibility audit (50 temporal-reasoning questions) | ≥40% are answerable by `as_of`/`timeline`/count | <40% → the typed-query lever is weaker than assumed; **stop and re-scope before building S2** |
| P6 | Typed-query arm on `temporal-reasoning` | improves over direct-retrieval arm at fixed answerer | no improvement → converting reasoning to queries doesn't pay, and S1–S4 are a negative result |
| P7 | **The ladder** | typed-query arm degrades *less* with smaller answerers than the direct arm | both arms degrade equally → the accuracy is coming from the answerer, not the store; this is the core claim and its failure is the most important possible outcome |
| P9 | **A-BUDGET** on `multi-session` | accuracy at 3- and 4-session questions **rises toward the 2-session level** (70.2%). Observed decay is 70.2% / 34.6% / 11.8% at 2/3/4 sessions while *session* recall only falls 0.923 → 0.846 → 0.691 and *turn* recall collapses 0.696 → 0.465 → 0.280; the decay steepens rather than staying geometric, which is the signature of a fixed budget meeting growing demand | the 3-/4-session cells stay flat → multi-session really is reader-bound and the §5 "not predicted" note below stands. This is the decisive test; a flat result retires the budget hypothesis entirely |
| P10 | **A-CHRONO** on `knowledge-update` | the `stale_fact` count **falls**. 17 rows carried `supersession_verdict == "stale"` at `gold_turn_recall == 1.0` — the fresh turn was in context and the answer took the retired value (`four`→`3`, `$400,000`→`$350,000`, `every week`→`Every two weeks`) | `stale_fact` unchanged → ordering and dating are not the mechanism, and the failure is the reader's value selection rather than its access to recency |
| P11 | A-BUDGET cost | prompt tokens scale ~3× with `--top-k 8 → 24`; total run cost rises but **stays under 2×** because output (reasoning) tokens dominate — the 2026-11-07 run billed 984,866 completion tokens against 1,000,328 prompt tokens, i.e. 99.7% of *visible* answer length was hidden reasoning | cost more than doubles → the longer context is inducing proportionally more reasoning, and the budget lever is not free; re-scope to `--turn-rerank global` at `--top-k 8` |
| P8 | Ladder control | the direct arm shows **steep** degradation with model size | direct arm is also flat → something shared bottlenecks both arms and the ladder is void; fix before interpreting P7 |

### Explicitly not predicted

- ~~Multi-hop composition categories. Bounded by the answerer
  (`paper/outline.md:212`: ceiling persists at 100% recall). No memory-layer
  intervention is expected to move these; they are reported, not chased.~~
  **Superseded 2026-11-07 — see P9.** This rested on *session* recall, which
  reads 0.923 at 2 sessions and stays high. *Turn* recall is the binding
  quantity and collapses to 0.280 by 4 sessions. "100% recall" in the cited
  line is session recall, so the ceiling it describes was never evidence that
  the answerer is the constraint. Multi-session is now predicted and chased.
- Any LOCOMO or BEAM aggregate (see §4.4).

---

## 6. Amendments

Append dated entries. Never edit §5 in place after ratification.

| date | change | reason |
|---|---|---|
| 2026-11-07 | §3.4 added (A-BUDGET, A-CHRONO); P9–P11 added; the "multi-hop is answerer-bound" exclusion struck | the 500-row run showed the turn budget saturated on 100% of aggregation failures, and the session-vs-turn recall gap refutes the premise of the exclusion |
| 2026-11-07 | recorded: `--chrono-sort` was a no-op under `--session-aware-retrieval`; `--ku-recency` shipped an unactionable recency instruction | both are cases of a flag whose measured effect was not the effect its name describes; §1 exists for exactly this |
| 2026-11-07 | §3.4 defect 1 corrected (the chrono fix was undone by RRF under `--retriever hybrid`); §3.5 added | the gap audit; three more flags whose measured effect was not their named effect (chrono under hybrid, the inert composite scorer, unconditional KU recency) |
| 2026-11-07 | P10 recorded as confirmed **by the date stamps, not by chronological ordering** | A-CHRONO never delivered timeline order in the pinned configuration, so it cannot be credited |
| 2026-11-07 | recorded: `--answer-max-tokens 2048` is too low for a reasoning answerer | 17/500 gpt-oss-120b answers came back empty, each at exactly 2048 completion tokens — hidden reasoning consumed the cap before any visible text. Any future gpt-oss-120b arm uses ≥ 8192 |

---

## 7. Runs already executed against this file

| date | run | status |
|---|---|---|
| 2026-11-07 | LongMemEval-S 500q, answerer `muse-wonder-1.0`, judge `meta-llama/llama-3.3-70b-instruct`, §3.1 flags otherwise | **`non_comparable: true`** — §3.1 pins the answerer as `openai/gpt-oss-120b`. Reported 69.0%. It may **not** be placed beside Mem0-S 67.6% or Continuum's own ~74%, both of which are gpt-oss-120b numbers (`docs/report.md` §6). Useful for the failure cross-tab that motivated §3.4, and for nothing comparative. |
| 2026-11-07 | A-BUDGET + A-CHRONO (`run_arm_budget_chrono.sh`), same answerer and judge | **`non_comparable: true`** (answerer, `--top-k 24`, `--use-ltm` off; the result also lacks a `versions` block). 81.6%; P9 confirmed (multi-session 54.1→72.2), P10 confirmed by the date stamps (`stale_fact` 14→0), P11 within bound (1.37× cost). Source of the §3.5 audit. |
| 2026-11-07 | §3.5 audit-fixes arm (`run_arm_audit_fixes.sh`), answerer `muse-wonder-1.0`, judge pinned to DeepInfra | **`non_comparable: true`** (answerer, `--top-k 48`, `--use-ltm` off, dirty worktree). 90.8%, $21.12. Gold-turn recall 0.892 → 0.974 vs A-BUDGET. |
| 2026-11-07 | §3.5 audit-fixes arm with the pinned answerer (`run_arm_audit_fixes_oss120b.sh`): `openai/gpt-oss-120b` via DeepInfra | **`non_comparable: true`** (`--top-k 48`, `--use-ltm` off, dirty worktree). 85.8% (llama judge), 85.2% on an independent manual Opus 5.5 grade of all 500 rows (3 disagreements, all stricter), $0.34. Retrieval identical to the muse run, so the 25-row gap is the reader. 17 rows were empty at the 2048-token cap (§6). Artifacts: `findings/longmemeval/audit_fixes_2026-11/`. |
| 2026-11-07 | the 17 empty rows above rerun at `--answer-max-tokens 8192` | 10 recovered, 3 answered wrong, 4 still exhausted 8192. Merged: 87.8% llama judge / 87.2% Opus grade. **Mixed caps (483 rows at 2048, 17 at 8192) — not a single pinned config**; a clean number needs a full run at 8192. Hindsight reports 89.0% at gpt-oss-120b with a gpt-oss-120b judge, so even the clean run would differ on the judge. |
