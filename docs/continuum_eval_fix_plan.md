# Continuum eval-fix plan — status + execution log

This document tracks the multi-phase fix plan against the live state
of the repo. It is the canonical place to look for "what's done, what's
PARTIAL, what's missing" — kept in sync as work lands.

## STEP 0 — repo audit

### Files in scope

```
evals/longmemeval/
├── adapter.py                ContinuumAdapter base class
├── answer_post.py            cleanup + validator + IDK guard (NEW — Phase A1/A3/A-lite)
├── baseline.py               eval runner, RowResult, scoring entry
├── bootstrap_ollama.py       all LLM clients, retrievers, _DecomposedAnsweringAdapter, CLI
├── candidates.py             typed candidate extraction (NEW — Phase A2/B engine support)
├── content_wiki.py           hand-tuned regex extractor (Codex import — overfit, optional)
├── decompose.py              DecompositionRetriever, build_decompose_prompt, parse_subquestions
├── decomposed_answer.py      sub-answer + synthesis + aggregation JSON prompts
├── judge.py                  LLMJudgeScorer (LLM-only)
├── question_type.py          deterministic 12-type classifier (NEW — Phase A-lite support)
├── rescore_with_judge.py     batch re-judge a results JSON
├── run.py                    top-level eval entry
├── setup.py                  dataset download / vendoring
├── telemetry.py              per-row LLM call counter (NEW — Phase A4 part 1)
└── wiki.py                   legacy Karpathy-style WikiAdapter (orphaned, kept on disk)

scripts/
├── check_bench_regressions.py
└── verify_clean_install.py
                              ← NO build_diagnostic_sample.py yet (Phase A0 MISSING)

tests/unit/evals/
├── test_adaptive_throttle.py
├── test_decompose.py
├── test_longmemeval_baseline.py
├── test_longmemeval_harness.py
├── test_retrieval_mode.py
└── test_scorer_weights.py
                              ← NO tests for question_type, candidates, telemetry,
                                answer_post, hybrid_scorer (this pass adds them)
```

### Per-phase status

| Phase | Item | Status | Where it lives |
|---|---|---|---|
| **A4** | LLM-call telemetry counter | **DONE** | `evals/longmemeval/telemetry.py` |
| A4 | `RowResult.telemetry` field + JSON propagation | **DONE** | `evals/longmemeval/baseline.py` |
| A4 | Per-row JSONL trace file (`logs/eval_trace_*.jsonl`) | **MISSING → adding this pass** | new `evals/longmemeval/trace.py` |
| **A0** | `scripts/build_diagnostic_sample.py` | **MISSING → adding this pass** | new |
| A0 | `--question-ids-file` CLI flag | **MISSING → adding this pass** | `bootstrap_ollama.py` |
| **A1** | Don't-IDK guard | **DONE** | `answer_post.should_block_idk` + adapter `_postprocess` |
| **A2** | Count aggregator override | **DONE** | `candidates.best_count_for_object` + adapter `_postprocess` |
| **A3** | Final answer cleanup | **DONE** | `answer_post.clean_final_answer` |
| **A-lite** | Answer-shape validator + 1-shot repair | **DONE** | `answer_post.validate_answer_shape` + adapter `_postprocess` |
| **B1** | Temporal engine | **MISSING — deferred to next pass** | needs new `evals/longmemeval/temporal.py` |
| **B3** | Knowledge-update engine | **PARTIAL** — `candidates.extract_change_facts` is the input | needs new `evals/longmemeval/knowledge_update.py` |
| **B2** | Preference engine | **PARTIAL** — `candidates.extract_preferences` is the input | needs new `evals/longmemeval/preference.py` |
| **C** | Hybrid scorer (rule + numeric + unit + LLM fallback) | **MISSING → adding this pass** | new `evals/longmemeval/hybrid_scorer.py` |
| C | Cache layer for judge | **PARTIAL — adding cache key in scorer this pass; backing store deferred** | `hybrid_scorer.py` |
| C | CLI: `--use-cache`, `--max-cost-usd`, `--debug-trace`, `--force`, `--categories` | **PARTIAL — adding `--judge`, `--question-ids-file`, `--debug-trace`** | `bootstrap_ollama.py` |
| **Evidence safety** | Soft session dedup | **MISSING — deferred to next pass** | `WikiMemoryRetriever` |
| Evidence safety | Wiki card / source expansion | **MISSING — deferred to next pass** | new helper |
| **Tests** | Tests for new modules (1-8 of spec) | **MISSING → adding this pass** | `tests/unit/evals/test_*.py` |
| Tests | Tests for B1/B3/B2 (10-13 of spec) | **N/A — those modules don't exist yet** | |

### What this pass lands

1. **STEP 0 doc** — this file.
2. **Phase A0** — `scripts/build_diagnostic_sample.py` + `--question-ids-file` CLI flag + composition test.
3. **Phase A4 part 2** — `evals/longmemeval/trace.py` + per-row JSONL trace + schema test.
4. **Phase C** — `evals/longmemeval/hybrid_scorer.py` with 5 stages (exact → normalized → numeric → unit → rule_semantic → optional LLM) + cache key + tests.
5. **CLI extensions** — `--question-ids-file`, `--judge {none,rule,hybrid,llm}`, `--debug-trace` flags.
6. **Tests for A1/A2/A3/A-lite helpers** — `tests/unit/evals/test_answer_pipeline.py` + `test_hybrid_scorer.py` + `test_diagnostic_sample.py`.

### What this pass deliberately defers

| Item | Why deferred |
|---|---|
| **B1 temporal engine** | ~500 LOC of dated-event extraction + duration arithmetic + ordering. Single biggest engineering item. Needs its own focused pass. |
| **B2 preference engine** | Needs preference reasoning over multiple stated/implied facts, not just extraction. Builds on the candidates already there but doesn't fit in this pass. |
| **B3 knowledge-update engine** | Old/new/current resolver over `extract_change_facts` output. Needs question-shape detection for "what changed", "what was", "what is now". Defer. |
| `--use-cache`, `--max-cost-usd`, `--force`, `--categories` flags | Cost guard needs accurate cost-estimation (Phase A4 part 2 supplies the data); cache backend (sqlite or disk JSON) is its own ~100 LOC. Adding `--judge` alone is enough to flip the hybrid scorer on. |
| Soft session dedup + wiki source expansion | Touches the hot retrieval path; needs paired retrieval + answer evaluation. |
| Live diagnostic_50 before/after measurement | Requires API spend (~$2-3); user should run locally after this pass merges and report numbers. |

### Phase A gate (read after this pass)

Will be filled in by the user after running:

```bash
# 1. Build the deterministic sample (once)
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    scripts/build_diagnostic_sample.py --seed 42

# 2. Baseline (before Phase A fixes) — needs a git checkout to pre-Phase-A;
#    OR re-judge an existing 500-row JSON over the diagnostic IDs.

# 3. After-Phase-A run
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    -m evals.longmemeval.bootstrap_ollama \
    --provider openai --model gpt-4o-mini \
    --wiki-memory --wiki-top-k 6 --decompose-answer --llm-judge \
    --judge hybrid --debug-trace \
    --question-ids-file samples/diagnostic_50_ids.json \
    --output results/diagnostic_50_phaseA --limit 50 --full --yes
```

Gate:
- ≥ +6pp over baseline → continue to Phase B
- < +6pp → inspect failure mix in `logs/eval_trace_*.jsonl` before more work

### Phase B gate (after B1/B2/B3 land)

- ≥ +10pp over Phase A baseline → continue to Phase C polish
- < +10pp → pivot recommendation: agentic iterative retrieval

## Execution log

| date | pass | landed |
|---|---|---|
| 2026-05-24 | Phase A core | telemetry counter, question classifier, candidate extractor, answer cleanup, IDK guard, shape validator, count override (this is `_DecomposedAnsweringAdapter._postprocess`) |
| 2026-05-24 | Phase A0 + A4 part 2 + C + tests | diagnostic sample script, JSONL trace, hybrid scorer, CLI flags, test suite |
| 2026-05-24 | Wiring + tests verified | `--question-ids-file` / `--judge` / `--debug-trace` flags wired through `bootstrap_ollama.main_async`; `trace_writer` plumbed into `run_baseline`; 4 new test files (143 evals tests, 71 new) pass; ruff clean; `samples/diagnostic_50_ids.json` generated with seed=42 |

Future passes will append rows here. Each pass should:
- Update the per-phase status table.
- Note any test failures or smoke regressions.
- Record the diagnostic_50 numbers if run.
