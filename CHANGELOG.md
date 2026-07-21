# Changelog

All notable changes to **continuum-memory** are listed here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/) with the understanding
that the public API may still shift before 1.0.

## [Unreleased]

### Added
- **Automatic capture (`CONTINUUM_CAPTURE=1`, off by default).** The
  `UserPromptSubmit` hook can now *write*: durable facts you state are stored
  without an explicit `remember`, so memory accrues from conversation. It reads
  **only your prompt** — never Claude's output or the transcript, because
  generated text is not evidence about you. The extractor
  (`continuum.promotion.capture`) is deterministic and precision-biased: no LLM,
  no network, and it refuses actions, questions, hypotheticals, retractions,
  work-artifact subjects ("my build is failing"), transient states, and anything
  credential-shaped — a secret drops its whole sentence. Measured 18/18 durable
  facts kept with **0 false captures out of 47** adversarial negatives.
  `--dry-run` previews a turn before you enable it; `CONTINUUM_CAPTURE_MAX`
  (default 3) caps one turn. (A modifier between the determiner and the head —
  "I have a *dentist* appointment" — walked past the transient-noun filter until
  a real session caught it.)
- **Targeted forgetting — `Memory.forget(contains=…)`.** The maintenance
  policy could not express "that one is wrong, delete it": `superseded_only`
  matches nothing until an LLM decider has retired something, and turning it off
  left age as the only lever, which reaches the whole store. `contains` is a
  case-insensitive substring match, bound not interpolated.
- **Forgetting — `Memory.forget()` / `PostgresLTM.prune()`.** Memory only ever
  grew; there was no eviction of any kind. Pruning is expressed as the same
  bi-temporal close as supersession (`invalidated_at`), so a forgotten fact
  leaves `recall`, `current` and `timeline` **together** and the row survives on
  disk for recovery — never a `DELETE`. Defaults are deliberately timid: only
  facts whose valid-time window has closed (superseded) *and* that nobody has
  read for `unused_for` are eligible, scoped to the namespace, capped by
  `limit`, and **dry-run by default**. `superseded_only=False` widens it to
  facts that are still true. Nothing prunes on a timer, and it is **not** an MCP
  tool — an agent should not be able to decide to forget things about you.
- **MCP observability** — `CONTINUUM_MCP_LOG_LEVEL` (default `WARNING`, quiet).
  At `INFO` every tool call logs one line with its inputs, outcome and latency
  (`tool=recall query=… k=3 hits=1 [18ms]`); a failing tool logs `FAILED` with a
  traceback instead of vanishing. **stderr only** — on the stdio transport stdout
  is the JSON-RPC channel.
- **`Memory.from_postgres(dsn, *, embeddings=True)`** — a production-stack factory
  mirroring `Memory.in_memory()`: Postgres STM/LTM + the full hybrid retriever, so
  `recall` is **relevance-ranked** (dense + sparse), not recency-ranked. The local
  bge-m3 embedder attaches by default (`embeddings=False` for sparse-only, no
  download).
- **`EmbeddingQueryRetriever`** (`continuum.retrieval.embedding_query`) — embeds
  `query.text` before delegating so the LTM dense channel fires; shared by the
  chat REPL path and `from_postgres`.
- **MCP HTTP transport** — `continuum-mcp --http` (Streamable-HTTP, `/mcp`) /
  `--sse`, with `--host`/`--port`, for a standalone always-on server a client
  connects to by URL. `make mcp-serve-http`.
- **`make mcp-eval`** (`scripts/mcp_eval.py`) — deterministic retrieval scorecard
  through the MCP tools (recall@1/@3, supersession, timeline) over a fixed
  scenario with distractors; honors `CONTINUUM_DB_DSN` to eval the Postgres stack.
- **`make mcp-install` / `mcp-smoke` / `mcp-serve` / `mcp-claude`** helpers.

- **Attribute-keyed memory — `current` is now an exact lookup.** `add(text,
  attribute="residence")` tags what a fact is *about*; `current(subject,
  attribute, as_of=…)` then answers it via an exact JSONB tag lookup honouring
  valid time (`PostgresLTM.by_tags`), instead of semantically searching for the
  attribute's *name*. An attribute label ("user residence") is a poor probe for a
  sentence ("I moved from Boston to New York City") — that mismatch, not ranking,
  is why `current` was wrong. `as_of` answers "what was current in March?".
  When the store can answer attributes exactly its answer is final, **including
  "no such fact"** — it no longer invents a value for an attribute it has no data
  for. Untagged facts keep the relevance-ranked fallback.

### Fixed
- **`remember` stored credentials and lied about empty writes.** Pasting a
  config (`password: hunter2`) wrote the secret straight into Postgres — the
  secret filter existed only on the automatic-capture path, not on the explicit
  tool the *model* calls while summarising what a user pasted. It now refuses
  credential-shaped text with a reason. An empty string replied `"stored"` while
  storing nothing; it now says so.
- **An unparseable `occurred_at` was dropped in silence.** `03/15/2026`,
  `last Tuesday` and `2026-13-45` all returned a bare `"stored"`, leaving the
  caller believing the fact was anchored in time — which is what `current` and
  `as_of` reason over. Unparsed dates are now reported in the ack, and a bare
  year (`2019`) is accepted. Ambiguous formats stay refused rather than guessed:
  `03/15` is March 15th to an American and invalid to most of the world.
- **`current` answered with the stale fact after a dated correction.** Found by
  using the thing, not by the suite — which stayed green because both existing
  tests dated *both* facts. Real conversations don't: you state a standing fact
  with no date ("I live in Lisbon"), then correct it with one ("I moved to Porto
  on July 1st"). Writes defaulted `valid_from` to *now*, so the correction
  looked a year **older** than what it replaced and `current` kept answering
  Lisbon — forever. Valid time is now recorded only when the caller states it
  (a NULL means "unstated"), and ordering compares valid time only when every
  candidate has one, else transaction time. This also fixed `timeline` listing
  a correction before the fact it corrects, and `as_of` returning "not found"
  for a period an undated fact covered.
- **Dating a fact buried it in recall.** Recency decay was based on `valid_from`
  (when the fact became true) rather than `created_at` (when we learned it), so
  the only rows carrying valid time — the ones a user had deliberately dated —
  were scored as a year stale, while undated noise written seconds earlier
  scored ~1.0 on recency and outranked them. Decay now measures transaction
  time; valid time still drives `current` / `timeline` / `as_of`, where it
  belongs. Choosing between versions of one fact is supersession's job.
  Paraphrase recall@1 went 70% → 90% (100% @3) and the timeline check 0/1 → 1/1.
- **`EmbeddingService.embed()` now rejects a bare `str`.** A string is iterable,
  so `embed("hello")` embedded it *per character* and `[0]` returned the vector
  for one letter — a well-formed unit vector that ranks nothing correctly.
- **`rank-bm25` was missing from the package dependencies** — it is imported at
  module load by the *default* in-memory LTM, so a clean
  `pip install "continuum-memory[mcp]"` crashed with `ModuleNotFoundError` on the
  first tool call. It was only ever present in `requirements.txt`, which nobody
  installing from PyPI reads. `make build-verify` passed anyway, because a
  session that assembles 0 items never reaches the retrieval stack; that gate now
  installs the `[mcp]` extra and drives a real `remember` → `recall` through the
  freshly installed binary.
- **`scripts/mcp_smoke.py` lost the last reply of a run.** It wrote every request
  and waited for EOF; `mcp` ≥ 1.28 drops in-flight replies when stdin closes, so
  `recall` — the final call — silently came back missing and looked like a
  retrieval failure. It now reads replies as they arrive, like a real client.
- **Retrieval was query-independent on the Postgres path** (core bug, affected
  `session.search()` for every Postgres user — not just MCP). The stores return
  hits *without* their embedding vector, so `Scorer` computed
  `cosine(query.embedding, None) == 0` as the relevance for **every** item and
  ranking collapsed to importance/recency — the same results for any query.
  `Retriever._score_all` now reuses the relevance the hybrid search already
  computed (dense ⊕ sparse ⊕ RRF), min-max normalised across the pool, when an
  item carries no embedding of its own. Custom scorers that don't expose
  `config.weights` keep their own ordering. Measured on a 14-fact scenario with
  distractors: **recall@1 10% → 100%**, recall@3 30% → 100%.
- **MCP server: the backing session was never started**, so the Postgres
  connection pool never opened and the first tool call hung. Started lazily on
  first use, exactly once.
- **`Memory.add()` stored no embedding**, leaving LTM rows invisible to the
  dense channel; `Memory` now embeds on write when an embedder is attached.
- **MCP server now honors `CONTINUUM_DB_DSN`** — `_default_memory()` builds a real
  Postgres-backed session (via `from_postgres`) instead of always falling back to
  in-memory. `CONTINUUM_MCP_EMBEDDINGS=0` disables the embedder; if the backend
  can't be built the server logs and degrades to in-memory rather than failing.

## [2.0.0] — 2026-07-19

First public release. Turns Continuum from a research codebase into an
importable library with a clean API and an MCP server.

### Added
- **Public `Memory` facade** (`from continuum import Memory`) — `add` /
  `remember` / `recall` / `current` (supersession-resolved) / `timeline`
  (bi-temporal history) + sync wrappers. `Memory.in_memory()` for zero-config
  demos/tests; wrap a Postgres-backed `ContinuumSession` for production.
- **MCP server** (`continuum-mcp`, `continuum-memory[mcp]`) — exposes memory as
  `recall`/`remember`/`current`/`timeline` tools to any MCP client (Claude Code,
  Cursor, …). See `docs/mcp.md`.
- **`docs/limitations.md`** — honest known-limitations: accuracy is reader-bound
  (~63% multi-session counting); the disproven deterministic levers (synthesis
  router, evidence distillation, temporal codemath — all net-negative); the
  benchmark-variance + same-setup-control methodology lesson.
- **`examples/`** — `bare_agent.py` (framework-free, runnable) and
  `langgraph_node.py` (Memory as a LangGraph node).

### Notes
- Research/eval levers (synthesis, router, distill, temporal codemath) are NOT
  part of the product API — they remain opt-in flags in `evals/` as tested
  negative results. Only the memory layer + supersession + retrieval + a bounded
  reflect pass + vote-of-N self-consistency are the shipped path.

## [1.0.0] — 2026-05-31

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
