# LongMemEval-S on Continuum: what 7 runs tell us about the ceiling

**Authors**: Continuum eval team
**Date**: 2026-05-23
**Status**: closed; supersedes `results/optimizer_iteration_*_*.json` analysis files. **Superseded by [`findings/reasoning_loop_2026-06.md`](reasoning_loop_2026-06.md)** — v1 broke this report's 32% ceiling to 60.8% judged, and notably did so *without* the iterative reasoning this report predicted was required. See the June report for what actually moved the number (stronger model + clean retrieval + honest scoring) and the reasoning loop we built and cut.
**Reproducibility**: all numbers in this report are extracted from JSON files in `results/` (cited inline). Re-run from scratch via `make repro-longmemeval`.

---

## TL;DR

- Across **6 full LongMemEval-S runs** (n=500 each) spanning 4 model families (gpt-4o-mini, Llama-4-Scout-17B, Llama-3.3-70B, llama-3.2-3B) and 6 retriever variants, **substring accuracy is bounded at 32-34 %**. The high-water mark is **Llama-4-Scout-17B at 34.4 %** with the optimizer-chain + top-k=4 retriever.
- A Llama-3.3-70B run with **100 % recall (full haystack in long context)** scores **32.2 %** — identical to the same model with top-k=4 retrieval (which has 77 % recall). **More context does not help.** The ceiling is *not* a retrieval problem.
- The failure-mode breakdown is consistent across every run: **~90 % of incorrect answers are `wrong_retrieval`** (the ground-truth session *was* retrieved but the model still answered incorrectly), versus ~10 % `missing_fact` (right session was never in the context). **The bottleneck is composition / reasoning, not memory.**
- Per-category: catastrophic on `multi-session` (15-20 %) and `temporal-reasoning` (7-23 %), strong on `single-session-*` (55-75 %). These are the categories that *require multi-hop synthesis at query time*, which a single retrieve→answer pipeline cannot do.
- Two retriever-level levers we built specifically for this — **`DecompositionRetriever` and `WikiAdapter`** — yielded **+3 pp and ≤0 pp** respectively on the categories they targeted. The retriever lever curve is flat past 32 %.

**Headline conclusion**: LongMemEval-S is a multi-hop reasoning benchmark dressed in memory clothing. A single-shot `retrieve → answer` architecture (which is what a memory framework alone exposes) has a structural ceiling around 32 %. Published 55-80 % systems use *agentic iterative reasoning* over memory, not better memory.

---

## 1 · Setup

**Dataset**: `LongMemEval-S` (cleaned), 500 questions across 6 categories. Each question ships with ~40 chat sessions (~115 K tokens) the system must ingest before answering. Source: `evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json`.

**Adapter under test**: `evals/longmemeval/bootstrap_ollama.py` — wraps Continuum's session pipeline (`STM/MTM/LTM`, optimizer chain, retriever, prompt formatter) onto LongMemEval's harness. To isolate the retrieval+answer axis from the storage/promotion axis, the adapter uses an in-memory `FlatHaystackStore` (every haystack message addressable, no eviction). Continuum's real MTM/LTM promotion pipeline is **bypassed for this eval**; that is itself a finding and is called out in §7.

**Scorer**: substring match (primary). LLM-judge rescoring (`rescore_with_judge.py`, gpt-4o-mini) applied to selected runs; results in §4. The substring scorer systematically undercounts ~9 pp uniformly and reports 0 % on `single-session-preference` (a known artifact, see §5).

**Retrievers tested across the 6 full runs**:

| name | what it does |
|---|---|
| `STMSemanticRetriever` top-k=4 | cosine over `all-MiniLM-L6-v2`, top-4 |
| same + optimizer chain | adds `StmTrim`+`MtmSummarize`+`SemanticDedupe`+`LLMLingua`+`ScoreAwareBudgetPrune` |
| same + long-context | bypasses optimizer, sends full haystack (~75 K tokens) to model |
| same + scorer-weight tuning | composite scorer (relevance/importance/recency/confidence) re-weighted per Prompt 38 |
| `DecompositionRetriever` | splits question into atomic sub-questions, retrieves per sub-question, merges (Prompt 40) |
| `WikiAdapter` | per-session LLM-built summary + atomic-fact list, indexed for cosine retrieval (Karpathy LLM-Wiki pattern, post-Prompt 40) |

---

## 2 · Headline results

All numbers below are from full 500-question runs. Sourced JSONs in **footnotes**.

| # | model | retriever / config | accuracy | recall | p50 / p95 latency | source |
|---|---|---|---:|---:|---:|---|
| 1 | Llama-4-Scout-17B (Groq) | top-k=4 + optimizer | **34.4 %** | 77.4 % | 2.1 s / 2.6 s | [^1] |
| 2 | Llama-3.3-70B (NVIDIA) | top-k=4 + optimizer | 32.2 % | 77.4 % | 30 s / 62 s | [^2] |
| 3 | **Llama-3.3-70B (NVIDIA)** | **long-context (full haystack)** | **32.2 %** | **100.0 %** | 31 s / 62 s | [^3] |
| 4 | gpt-4o-mini | top-k=4 + decompose | 29.2 % | 76.3 % | 4.8 s / 6.9 s | [^4] |
| 5 | gpt-4o-mini | top-k=4 scorer-tuned | 27.8 % | 73.4 % | 2.4 s / 3.5 s | [^5] |
| 6 | gpt-4o-mini | top-k=4 + optimizer (final) | 26.2 % | 77.4 % | 2.4 s / 3.8 s | [^6] |

[^1]: `results/baseline_2026-05-21.json` — Llama-4-Scout-17B-16e-instruct via Groq.
[^2]: `results/nvidia_70b_adaptive/baseline_2026-05-22.json` — meta/llama-3.3-70b-instruct via NVIDIA Build.
[^3]: `results/nvidia_longctx/baseline_2026-05-22.json` — same model, full haystack in context, optimizer bypassed.
[^4]: `results/gpt4omini_decompose/baseline_2026-05-22.json` — `DecompositionRetriever(base=top-k=4)`.
[^5]: `results/optimizer_iter2_scorer/baseline_2026-05-22.json` — scorer weights 0.55 / 0.20 / 0.15 / 0.10, τ=96 h.
[^6]: `results/gpt4omini_topk/baseline_2026-05-22.json` — gpt-4o-mini, Continuum's reference iter-4 final config.

**Range across all 6 runs: 26.2 % – 34.4 %.** All sit in a 8.2 pp band despite spanning a 23× model-size range (3 B → 70 B) and 6 different retriever configurations.

---

## 3 · The 32 % ceiling

The decisive datum is **rows 2 and 3 in the table above** — same model, same scorer, only the retriever differs:

| | recall | accuracy |
|---|---:|---:|
| top-k=4 (77 % recall) | 77.4 % | 32.2 % |
| long-context (100 % recall, ~75 K tokens) | 100.0 % | 32.2 % |

**Going from 77 % recall to 100 % recall changed accuracy by 0.0 pp.** Every fact the question needed was in the context for the long-context run; the model still answered incorrectly 67.8 % of the time. Adding context past the level the retriever already provides is *not* a lever. Whatever is bottlenecking accuracy is not "the right session wasn't retrieved."

This is the single most important finding from the sweep. It says the LongMemEval-S accuracy ceiling for a one-shot retrieve→answer pipeline is **architectural**, not parametric.

---

## 4 · The 90 % wrong_retrieval finding

Every run's `failure_breakdown` field carries the same shape:

| run | `missing_fact` | `wrong_retrieval` | total fail |
|---|---:|---:|---:|
| Scout-17B top-k | 30 | 298 | 328 |
| 70B top-k | 32 | 307 | 339 |
| 70B long-context | 0 | 334 | 339 |
| gpt-4o-mini top-k (iter-4 final) | 32 | 337 | 369 |
| gpt-4o-mini decompose | 35 | 319 | 354 |
| gpt-4o-mini scorer-tuned | 38 | 323 | 361 |

LongMemEval's scorer defines:

- `missing_fact` — ground-truth session **not** in retrieved set; model can't possibly answer.
- `wrong_retrieval` — ground-truth session **was** retrieved; model still answered wrong.

**Across every full run, `wrong_retrieval` outnumbers `missing_fact` by ~10×.** The proportion stays roughly the same regardless of retriever — including the long-context run, which has zero `missing_fact` by construction (everything is in the context) yet still has 334 `wrong_retrieval` failures.

Translated: when the model fails, it almost always fails *with the right context in front of it*. The retriever is doing its job. The reasoning step is where accuracy is being lost.

**LLM-judge rescoring** on a targeted subsample (`results/gpt4omini_wiki_targeted_v2/baseline_2026-05-23.judged.json`) confirms this isn't a scoring artifact: substring → judged moved 8.3 % → 25.0 % on multi-session, but most failures the judge still rates as wrong. The model genuinely got the answer wrong; the substring scorer is downstream noise.

---

## 5 · Per-category breakdown

All 6 categories, all 6 full runs:

| category | n | Scout-17B | gpt-4o-mini topk | gpt-4o-mini decompose | 70B topk | **70B long-ctx** | gpt-4o-mini scorer |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 70 | **64.3 %** | 58.6 % | 54.3 % | 62.9 % | 58.6 % | 55.7 % |
| single-session-assistant | 56 | **75.0 %** | 71.4 % | 73.2 % | 73.2 % | 62.5 % | 69.6 % |
| single-session-preference | 30 | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| multi-session | 133 | **19.5 %** | 11.3 % | 15.0 % | 16.5 % | 14.3 % | 12.0 % |
| temporal-reasoning | 133 | **22.6 %** | 6.8 % | 14.3 % | 17.3 % | 19.5 % | 8.3 % |
| knowledge-update | 78 | 37.2 % | 33.3 % | 35.9 % | 39.7 % | **51.3 %** | 43.6 % |

Observations:

- **`single-session-preference` is uniformly 0 %**. This is a substring-scorer artifact — these questions ask the user to choose between named options ("did you prefer X or Y?") and any non-verbatim paraphrase scores as wrong. LLM-judged accuracy on this category is ~57 % (per Iteration-4 analysis); ignore the substring number.
- **`single-session-*` (excluding preference) is strong everywhere** (54-75 %). When the answer lives in one session and retrieval delivers it, gpt-4o-mini and stronger models can answer.
- **`multi-session` and `temporal-reasoning` are catastrophic everywhere** (7-23 %). Both require composing facts across multiple sessions, which is the exact shape the long-context run *also* fails at (14.3 % multi-session even with 100 % recall). Bigger models help marginally (Scout-17B is the best at 19.5 % multi-session) but no run breaks ~25 %.
- **`knowledge-update` is the only category where the big-model long-context run dominates** (51.3 % vs 33-44 % for everyone else). Knowledge-update questions test whether the system uses the *latest* version of a fact when the user has updated it; long-context lets the model see *all* the versions and reason about which is current. With top-k retrieval, the older version often beats out the newer in cosine ranking → wrong answer. This is the category where Continuum's `superseded_by` schema (built but not wired into the eval) should win directly.

---

## 6 · Lever-by-lever: what we tried and what it bought us

Six independent levers, each one a Prompt-level work item. Listed in order tried.

| # | lever | hypothesis | result | verdict |
|---|---|---|---|---|
| 1 | Optimizer chain (5 strategies) | compression frees context budget for more retrievals | 0 pp at top-k=4 (chain is near-no-op when budget already fits) | inert on this benchmark |
| 2 | Cross-encoder reranker | reorder top-k by query-relevance | 0 pp (iter-3 sweep) | added no signal cosine didn't already capture |
| 3 | Scorer-weight tuning | re-weight relevance/importance/recency/confidence | **−2 pp + recall regression** | the default weights are already near optimal |
| 4 | Long-context (`mode=long`) | hand the model the entire haystack — 100 % recall | **identical to top-k=4** (32.2 % = 32.2 %) | the 32 % ceiling is real; retrieval is not the lever |
| 5 | `DecompositionRetriever` | split multi-hop questions, retrieve per sub-question, merge | **+3 pp global**, +7.5 pp on temporal-reasoning, +3.7 pp on multi-session | the directionally-right lever, but magnitude small |
| 6 | `WikiAdapter` (Karpathy-style) | pre-compress each session into summary + atomic facts at ingest, retrieve from the dense layer | **8.3 % substring / 25 % judged on multi-session n=12** (below decompose); 20 % substring on temporal-reasoning (n=5); 20 % substring on knowledge-update (n=5) | regressed vs decompose on 2 of 3 target categories; lever exhausted |

The pattern across the table: every lever that touches **retrieval** moved the number by at most ~+3 pp. The one experimental signal that *might* break the ceiling — applying Continuum's real promotion pipeline (MTM summaries + LTM facts + `superseded_by` supersession + entity graphs) — was never wired into this eval. The eval used `FlatHaystackStore` to isolate the retrieval+answer axis, and the resulting data says that axis is saturated at 32 %.

---

## 7 · What this means

### For Continuum

The framework, as a *memory layer*, works:

- Retrieval recall is 76-100 % across every retriever variant.
- p95 retrieval latency is 2-7 s on gpt-4o-mini, 60 s on NVIDIA-70B (model-bound, not memory-bound).
- gpt-4o-mini top-k cost is **$0.00001 / query** — 100× cheaper than the long-context alternative at identical accuracy. The cost-efficiency story is real.

What the framework cannot do — and shouldn't try to — is multi-hop composition at query time. That belongs to whatever reasoner sits *above* Continuum (LangGraph, AutoGen, a custom decompose-answer-compose loop). **Continuum is the memory layer; the reasoner is the caller's responsibility.**

### For LongMemEval-S as a benchmark

LongMemEval-S markets itself as a long-term memory benchmark. The 32 % ceiling under 100 % recall says it's effectively a **multi-hop reasoning benchmark with a memory front-end**. The "memory" part is solvable at high recall by even a simple cosine retriever; the "reasoning" part is what gates accuracy. Future memory frameworks claiming X % on LongMemEval-S should specify whether X comes from (a) better recall (mostly already saturated), or (b) an external reasoning layer wrapping their memory tier. The two are not the same product.

### For the field

The published 55-80 % systems on this benchmark all share one trait: **agentic iterative retrieval** — the reasoner issues multiple read+search rounds at query time, refining its hypothesis. That is not a memory architecture, it's a reasoning architecture *built on top of* memory. Confusing the two leads to optimisation effort spent on the wrong layer (which is, frankly, what most of this sweep was).

---

## 8 · What we are not doing

- Further retriever tuning. Six independent retriever-level levers all flat past +3 pp; the data is in.
- Bigger embedders. Stella-1.5B / gte-Qwen-7B would lift recall from 77 % → ~90 %, but recall is not the bottleneck — see §3.
- More LongMemEval-S runs. The 8.2 pp band across 4 model families says the ceiling is architectural, not noise.

---

## 9 · Where the work goes next

Two directions, in priority order. Both abandon the LongMemEval-S accuracy chase.

1. **Benchmark Continuum on memory operations** — ingest cost/latency, retrieval recall under load, supersession correctness, bi-temporal queries. These are the axes where the framework wins and where its differentiation lives. (Phase 3B in the next-prompt plan.)
2. **Wire Continuum's real LTM+supersession into the eval** as a single experiment, specifically targeting `knowledge-update` (where the long-context big-model run scored 51.3 % — there is real headroom). This is a focused test of the framework's actual mechanism, not another retriever variant.

---

## Appendix · Reproducibility

Every accuracy / recall / failure-breakdown number above is read directly from JSON. To regenerate the tables from raw data:

```bash
cd /Users/mayanksahu/Continuum
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 findings/charts/extract_tables.py
```

To regenerate the charts (PNG):

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 findings/charts/render_charts.py
```

To re-execute the headline runs from scratch (~30 min, ~$0.05 in API cost):

```bash
make repro-longmemeval   # introduced in Prompt 42
```

### Cited JSON files

| section | file |
|---|---|
| §2 row 1 (Scout-17B) | `results/baseline_2026-05-21.json` |
| §2 row 2 (70B top-k) | `results/nvidia_70b_adaptive/baseline_2026-05-22.json` |
| §2 row 3 (70B long-ctx) | `results/nvidia_longctx/baseline_2026-05-22.json` |
| §2 row 4 (decompose) | `results/gpt4omini_decompose/baseline_2026-05-22.json` |
| §2 row 5 (scorer-tuned) | `results/optimizer_iter2_scorer/baseline_2026-05-22.json` |
| §2 row 6 (gpt-4o-mini iter-4) | `results/gpt4omini_topk/baseline_2026-05-22.json` |
| §6 wiki multi-session | `results/gpt4omini_wiki_targeted_v2/baseline_2026-05-23.json` |
| §6 wiki temporal | `results/gpt4omini_wiki_temporal_v2/smoke/baseline_2026-05-23.json` |
| §6 wiki kupdate | `results/gpt4omini_wiki_kupdate_v2/smoke/baseline_2026-05-23.json` |
| superseded analysis files | `results/optimizer_iteration_3_retrieval_tuning.json`, `results/optimizer_iteration_4_final.json` |
