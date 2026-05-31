# Changelog

All notable changes to **continuum-memory** are listed here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/) with the understanding
that the public API may still shift before 1.0.

## [1.0.0] — 2026-06-XX

v1 — broke the LongMemEval-S 32% ceiling to **60.8% judged**, and learned the
ceiling was never a reasoning-architecture problem.

### Added
- **Hybrid retrieval**: `continuum/retrieval/bm25.py` (BM25 over a regex
  tokenizer) + `continuum/retrieval/rrf.py` (Reciprocal Rank Fusion);
  `--retriever {cosine,bm25,hybrid}` in the eval harness.
- **LTM supersession for eval**: `continuum/stores/in_memory/ltm.py` +
  `evals/longmemeval/continuum_ltm_store.py` drive STM→facts→LTM with
  bi-temporal invalidation; `--use-ltm`. Knowledge-update recall 98.7%.
- **`SmallLLM`** (`continuum/extraction/small_llm.py`) — sqlite-cached small
  LLM with Ollama + OpenAI-compatible (LM Studio / OpenRouter) backends.
- **`IterativeReasoner`** (`continuum/reasoning/`) — budget-capped
  decompose→verify→compose loop. **Shipped but not used in v1**: A/Bs showed
  it net-negative vs direct retrieval (see findings); retained as a tested
  negative result.
- **Direct answer mode** (`--reasoner direct`) — the v1 winner: retrieve →
  hand raw turns to the answerer, one call.
- **LLM judge as primary metric** + offline re-judge
  (`rescore_with_judge.py`, `--provider openrouter`, `--judge-max-tokens`).
- **OpenRouter provider**; `--answer-max-tokens`, `--max-context-chars`,
  `--session-aware-retrieval`, `--offset`/`--no-smoke` batching,
  `--small-llm same`.
- **LOCOMO benchmark + Mem0 head-to-head** scaffold (`evals/locomo/`,
  `make bench-locomo`) — preliminary; clean run pending.
- **v1 findings report** `findings/reasoning_loop_2026-06.md` (supersedes the
  May report); `findings/charts/v1_summary.py`; `make repro-everything`.

### Changed
- README headline now leads with LongMemEval-S 60.8% judged; the "honest
  evaluation" section pairs the May ceiling report with the June correction.

### Fixed
- Recall always-0 under the iterative adapter (`last_ctx` never set).
- Reasoning models silently breaking structured tasks at low token caps
  (the yes/no judge and Mem0's JSON extractor) — use non-reasoning models.
- Direct-mode 8000-char context cap silently dropping answer-bearing turns.

### Known gaps
- temporal-reasoning 41.4% (genuine model-reasoning limit).
- OpenRouter cost accounting unwired (result JSON reports `$0.00`).
- LOCOMO vs Mem0 not yet a fair published number (Mem0 partially handicapped).

## [0.3.0] — 2026-05-23

Phase 3 — benchmarks, demo, findings, and a publishable repo state.

### Added
- **Four memory-operation benchmarks** under `bench/`:
  - `ingest_throughput.py` — Continuum vs raw list vs mem0 (stub).
  - `retrieval_quality.py` — recall@k vs naive cosine on a 200-session synthetic corpus.
  - `supersession_correctness.py` — **100 % vs 38 %** on 50 scripted update-then-query scenarios.
  - `bi_temporal.py` — **100 % vs 75 %** on 20 "as of date Y" queries (point-in-time + retroactive corrections).
- `bench-ingest` / `bench-retrieval` / `bench-supersession` / `bench-bitemporal` / `bench-all` Make targets.
- **CLI chat-agent demo** at `examples/chat_agent/` with a scripted 60-second walkthrough (`make demo-chat`) showing supersession in real time. No infrastructure required.
- **LongMemEval-S evaluation report** at `findings/longmemeval_2026-05.md` (5 pages, 9 sections, 3 auto-generated charts) documenting the 32 % ceiling across 6 retrievers × 4 model families.
- **Reproducibility artefact** at `findings/longmemeval/repro/` with a `make repro-longmemeval` target that reruns the two headline configurations in ~30 min for ~$0.10 and verifies against a tolerance contract.
- **Four documentation pages** under `docs/`: quickstart, architecture, config reference, operations.
- **Top-level README rewrite** organised around the new value proposition (production-grade memory infrastructure, not a reasoning engine).
- **Screencast storyboard** at `documentation/screencast.md` — frame-by-frame 3-minute script.

### Changed
- Project status `3 - Alpha` → `4 - Beta`.
- `--embedder-device` CLI flag (`auto` / `cpu` / `mps` / `cuda`) on the eval harness to sidestep MPS OOM when running alongside LMStudio.
- Composite scorer reverted to default weights (0.45 / 0.25 / 0.20 / 0.10, τ = 168 h) — Phase 2 tuning experiments did not yield a stable lift.

### Documented
- Architectural ceiling at ~32 % for single-shot `retrieve → answer` pipelines on LongMemEval-S, evidenced across 7 sweeps. This positions Continuum as a memory layer rather than a reasoning engine.

## [0.2.0] — Phase 2

LTM with supersession + bi-temporal columns, fact / entity extraction,
promotion pipeline, policy engine, composite scorer + reranker.

### Added
- `continuum/stores/postgres/ltm.py` with the migration-004 schema (`superseded_by`, `valid_from`, `recorded_at`).
- `continuum/extraction/entity_extractor.py` (GLiNER) + `continuum/extraction/llm_extractor.py` (LLM-driven).
- `continuum/extraction/fact_extractor.py` — atomic-fact extraction over MTM summaries.
- `continuum/promotion/` — Mem0Promoter, triggers, IdleStmFlush watcher.
- `continuum/policies/` — policy engine + 8 default policies (migration 004).
- `continuum/scoring/scorer.py` — composite scorer (relevance / importance / recency / confidence).
- `continuum/retrieval/reranker.py` — cross-encoder reranker.
- `continuum/retrieval/retriever.py` — main `Retriever` orchestrating cosine + reranker + scorer.
- LongMemEval evaluation harness at `evals/longmemeval/`.
- Acceptance tests for Phase 1 + Phase 2 completion.

## [0.1.0] — Phase 1

Initial public scaffold.

### Added
- Async-first `ContinuumSession` orchestrating STM + retriever + responder + background queue.
- STM implementations: in-memory, async-safe (thread-safe), Postgres.
- MTM implementation: `PostgresMTM` with pgvector retrieval.
- Embedding service (sentence-transformers, configurable).
- `BackgroundQueue` for off-foreground bookkeeping.
- Migrations 001-003 (initial schema, pgvector, lexical search).
- Token optimizer chain (5 strategies — Stm trim / MTM summarize / semantic dedupe / LLMLingua / score-aware budget prune).
- pyproject.toml with hatchling build backend, strict mypy + ruff configuration, PEP 561 type marker.
- pytest + pytest-asyncio harness, full unit + integration suites.
- Makefile targets for test / lint / typecheck / format / check / install / clean.
- GitHub Actions CI workflow with matrix Python versions and a Postgres service.
