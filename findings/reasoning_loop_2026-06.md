# Breaking the 32% ceiling: what actually moved LongMemEval-S (and what didn't)

**Authors**: Continuum eval team
**Date**: 2026-05-31
**Status**: open; **supersedes** [`findings/longmemeval_2026-05.md`](longmemeval_2026-05.md)
**Reproducibility**: numbers below are from JSON in `results/v1_final/` (cited inline). Re-run via `make repro-everything`.

---

## TL;DR

- The [May report](longmemeval_2026-05.md) found a hard **32-34% substring ceiling** on LongMemEval-S across 6 retrievers × 4 model families, and concluded: *"LongMemEval-S is a multi-hop reasoning benchmark dressed in memory clothing… a single-shot retrieve→answer architecture has a structural ceiling around 32%. Published 55-80% systems use agentic iterative reasoning, not better memory."*
- **That conclusion was half right and half wrong.** v1 hits **60.8% judged** (48.4% substring) on the full 500 — nearly double the ceiling — with a **single-shot retrieve→answer** pipeline. The ceiling was real, but **breaking it did not require iterative reasoning.**
- We *built* the agentic iterative reasoner the May report implied we needed (`continuum/reasoning/IterativeReasoner`: decompose → per-sub-q retrieve → verify → compose, budget-capped). **In head-to-head A/Bs it was net-negative** — it lost to dumb direct retrieval on both single-hop (88% direct vs ~50% iterative) and multi-hop (16% vs 6%). **We cut it from v1.**
- What actually moved the number, in order of impact: **(1) a stronger answerer** (gpt-oss-120b vs the May models), **(2) the LLM judge** revealing substring under-counted paraphrases by +12.4pp, **(3) session-aware retrieval + fixing two silent truncation bugs** (8k-char context cap, reasoning-model token starvation) that lifted multi-session from ~16% → 55%, **(4) LTM supersession** making knowledge-update work (98.7% recall).
- **Honest residual**: temporal-reasoning (41.4%) and multi-session aggregation are genuine model-reasoning limits, not memory or retrieval gaps. They're the v1.1 frontier.

**Headline conclusion**: the May ceiling was *not* a reasoning-architecture problem. It was a **model-capability + retrieval-coverage + scoring-artifact** problem. A strong model fed clean, complete context via direct retrieval beats an elaborate reasoning loop over a weaker model. The reasoning machinery we built to break the ceiling actively hurt; the simplest path broke it.

---

## 1 · Setup

**Dataset**: LongMemEval-S (cleaned), 500 questions / 6 categories, ~40 sessions (~115K tokens) per question. Source unchanged from the May report.

**v1 configuration** (`evals/longmemeval/bootstrap_ollama.py`):

| axis | v1 |
|---|---|
| answerer | `openai/gpt-oss-120b` via OpenRouter |
| reasoner | **`--reasoner direct`** — retrieve → hand raw turns to the answerer, one call. (NOT the IterativeReasoner; see §4.) |
| retriever | `--retriever hybrid` (cosine ⊕ BM25, RRF) + `--session-aware-retrieval` (session-first ranking) |
| memory | `--use-ltm` (in-memory LTM + deterministic supersession) |
| context | `--answer-max-tokens 2048 --max-context-chars 64000` |
| scorer | LLM judge (`llama-3.3-70b-instruct`, non-reasoning) — substring reported as a floor |

**Scorer note**: the May report flagged substring under-counts ~9pp and reports 0% on `single-session-preference`. v1 confirms this precisely: the judge flips +73 / −11 rows, +12.4pp overall, and rescues single-session-preference from 6.7% → 50%.

---

## 2 · Headline: before → after

### Overall (LongMemEval-S, n=500)

| | May 2026 (best) | v1 (substring) | v1 (judged) |
|---|---:|---:|---:|
| overall | 34.4% | 48.4% | **60.8%** |
| recall | 77-100% | 91.6% | 91.6% |

### Per-category (judged where available)

| category | n | May (substring) | v1 substring | **v1 judged** |
|---|---:|---:|---:|---:|
| single-session-assistant | 56 | 55-75% | 78.6% | **98.2%** |
| single-session-user | 70 | 55-75% | 78.6% | **94.3%** |
| multi-session | 133 | 15-20% | 50.4% | **54.9%** |
| knowledge-update | 78 | — | 43.6% | **51.3%** |
| single-session-preference | 30 | 0% (artifact) | 6.7% (artifact) | **50.0%** |
| temporal-reasoning | 133 | 7-23% | 30.1% | **41.4%** |

Source: `results/v1_final/baseline_2026-05-31.json` (substring) and `results/v1_final/judged.json` (judged); rendered by `findings/charts/v1_summary.py`.

The categories the May report called "catastrophic" — multi-session (15-20% → 54.9%) and temporal (7-23% → 41.4%) — moved the most. Single-session moved from "strong" to "nearly solved" (94-98%).

---

## 3 · What moved the number — which lever bought which pp

Honest attribution. The gains are *not* evenly distributed and the biggest one is the least architectural.

1. **Stronger answerer (gpt-oss-120b).** The dominant lever. The May models (Scout-17B, llama-3.3-70B) topped out at 34%; swapping the answerer alone — same direct retrieve→answer shape — took single-hop categories to ~88% substring. Most of the headline gain is the model, and we say so plainly.

2. **The LLM judge (+12.4pp), not a model change at all.** Substring was rejecting correct paraphrases wholesale — single-session-preference at 6.7% with 93% recall was pure scoring artifact. The judge (non-reasoning, to avoid the failure in §5) recovered +73 rows. This is *measurement*, not capability — but it's the difference between reporting 48% and 61%.

3. **Session-aware retrieval + two silent truncation bugs.** Multi-session was stuck at ~16% with 42% recall. Three fixes compounded:
   - `--session-aware-retrieval` (rank sessions first, take turns from the top-N) → recall 42% → 85%.
   - A hidden **8000-char context cap** in the direct adapter was dropping answer-bearing turns *even when the session was retrieved* (recall=1.0 but "I don't have that information"). Lifting it to 64k fixed it.
   - **Reasoning-model token starvation**: gpt-oss-120b burned its 200-token answer budget *thinking*, returning truncated/empty answers on longer multi-session outputs. Bumping to 2048 unblocked it.
   Net: multi-session ~16% → 55%.

4. **LTM supersession.** `--use-ltm` made knowledge-update functional — 98.7% recall, 51.3% judged. The supersession path (invalidate stale facts so they don't compete with current ones during retrieval) is the one component that's genuinely differentiated, and it's the LOCOMO story (§6).

---

## 4 · What we built and cut: the IterativeReasoner

The May report implied the path to >32% was agentic iterative reasoning. We built it: `continuum/reasoning/iterative_reasoner.py` — a budget-capped loop (≤6 LLM calls) that classifies intent → decomposes → retrieves per sub-question → verifies claims → composes with an evidence packet, with a refine-on-fail round and abstain semantics. It's ~250 LoC, fully unit-tested, layer-clean (zero `evals/*` imports), wired as `--reasoner iterative`.

**Then we A/B'd it against `--reasoner direct` on identical questions, same model, same scoring:**

| | direct | iterative |
|---|---:|---:|
| single-session-user (judged-adjacent) | ~88% | ~50% |
| multi-session (hard slice) | 16% | 6% |

Iterative lost everywhere. Diagnosis: with a strong model, the decompose → verify → packet → head machinery **loses information** that 100%-recall retrieval already surfaced. The deterministic verifier passed ~all or ~none of candidates (no useful signal); the reasoning heads short-circuited with confident wrong spans before the composer ran. **The scaffolding got in the model's way.**

We cut it from v1. The class stays in the tree (tested, documented) as a negative result and a possible lever for *weaker* answerers, but v1 ships direct mode. This is the central lesson: **don't build the impressive thing; build the thing the data rewards.**

The **abstain head** (return "I don't have enough information" below a confidence threshold) was **gated out** before implementation — `judge.py` scores any abstention as wrong (0.0), so abstaining can only lose points under judged accuracy. The gate caught it; we skipped the feature.

---

## 5 · A cross-cutting lesson: reasoning models break structured-output tasks

Twice this cycle, `gpt-oss-120b` (a reasoning model) silently broke a non-answerer task by burning its token budget on reasoning and emitting nothing parseable:

- **As a judge** (`max_tokens=5` for yes/no): every row came back "no" → flipped 38/50 correct answers to wrong. Fix: judge with a *non-reasoning* model (llama-3.3-70b-instruct); the small token cap is then correct.
- **As Mem0's fact extractor**: returned truncated/unterminated JSON → mem0's parser choked. Same fix: non-reasoning extractor.

**Rule of thumb that fell out of v1**: reasoning models are for *answering*; structured/classification sub-tasks (judging, extraction, yes/no) want a fast non-reasoning model. This is now load-bearing across the eval harness.

---

## 6 · LOCOMO vs Mem0 — **preliminary, not a published number**

We wired a LOCOMO head-to-head (`evals/locomo/`, `make bench-locomo`): same answerer + judge both sides, only the memory layer differs (Continuum direct-retrieval vs Mem0 distill-and-retrieve).

On a 50-question hard-category slice (multi-hop + temporal, no single-hop):

| category | Continuum | Mem0 |
|---|---:|---:|
| multi-hop | 36.8% | 36.8% |
| open-domain | 57.1% | 28.6% |
| temporal | **33.3%** | **4.2%** |
| overall | **38.0%** | 20.0% |

**Why this is preliminary and NOT a launch claim**: Mem0's run was partially handicapped by intermittent extraction failures (incomplete memory), and the slice is hard-category-heavy. We will **not** publish a Mem0-beating headline until a clean, un-handicapped full run — doing otherwise would be the cherry-picked benchmarking this project exists to avoid.

That said, the *direction* is informative and consistent with the thesis: **Mem0 distills to facts and discards the timeline, so temporal reasoning collapses (4% vs 33%).** Continuum keeps raw turns, so sequence survives. The supersession/keep-raw bet shows up exactly where theory predicts. A clean full head-to-head is the v1.1 deliverable.

---

## 7 · Cost & latency receipts

- **Latency** (v1 full 500, direct mode, gpt-oss-120b via OpenRouter): **p50 5.3s, p95 21.9s** per question. One answerer call per question (`avg_llm_calls = 1.0`); the judge is a separate offline pass.
- **LLM calls**: 500 answerer + 500 judge = ~1,000 calls for the full evaluated+judged run.
- **Cost**: the result JSON reports `$0.00` — this is a **tracking gap, not free**. The cost table in `bootstrap_ollama.py` only prices the OpenAI/Groq models, not OpenRouter. Estimate from tokens: gpt-oss-120b answering ~16k input + ≤2k output per question × 500 ≈ a few dollars; the judge (llama-70b, ~8 tokens out) is negligible. **Wiring real OpenRouter cost accounting is a v1.1 cleanup.**

---

## 8 · Where we still lose, and what's next

- **temporal-reasoning (41.4%)** — the genuine weak spot. Date arithmetic ("how many days between X and Y") from raw context is a model-reasoning limit; the judge can't rescue genuine miscounts. Lever: structured temporal extraction, or a reasoning step *scoped to the dates* (not the failed general reasoning loop).
- **multi-session aggregation** — at recall=1.0 the model still miscounts ("3 weddings" → "1"). Completeness of retrieval is solved (85% recall); *counting over* it is not.
- **Cost accounting** — `$0.00` artifact; wire OpenRouter pricing.
- **LOCOMO clean run** — un-handicap Mem0 (lower extraction chunk size, verify near-zero parse errors), run full, publish a fair head-to-head.

---

## 9 · Reproducibility

```bash
# headline LongMemEval-S v1 run + LOCOMO smoke (target: <2h, <$5)
make repro-everything

# or the pieces:
#   full LongMemEval-S, direct mode, judged
OPENROUTER_API_KEY=… python3.12 -m evals.longmemeval.bootstrap_ollama \
    --provider openrouter --model openai/gpt-oss-120b \
    --reasoner direct --use-ltm --no-llm-promoter --retriever hybrid \
    --session-aware-retrieval --session-top-k 12 --turns-per-session 6 \
    --top-k 80 --max-context-chars 64000 --answer-max-tokens 2048 \
    --full --yes --no-smoke --output results/v1_final
python3.12 -m evals.longmemeval.rescore_with_judge \
    --input results/v1_final/baseline_*.json --output results/v1_final/judged.json \
    --provider openrouter --judge-model meta-llama/llama-3.3-70b-instruct
python3.12 findings/charts/v1_summary.py results/v1_final/judged.json
```

All v1 numbers in this report are in `results/v1_final/`.
