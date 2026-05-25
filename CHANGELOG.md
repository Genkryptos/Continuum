# Changelog

All notable changes to **continuum-memory** are listed here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/) with the understanding
that the public API may still shift before 1.0.

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
