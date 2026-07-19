# Continuum

**Production-grade memory infrastructure for AI agents.** Tiered storage (STM / MTM / LTM), first-class supersession and bi-temporal queries, cost-efficient retrieval. Plugs in under whatever reasoner you're using — LangGraph, AutoGen, custom — and provides the memory layer they all lack.

```bash
git clone https://github.com/Genkryptos/Continuum.git && cd Continuum
make demo-chat       # 60-second walkthrough, no infrastructure, no API key
make bench-all       # the four benchmarks Continuum is built to win
```

---

## What it is

Most "memory" for LLM agents is one of:

* a vector database (great recall, no notion of "current" vs "stale" facts),
* an append-only chat history (no organization, gets expensive fast),
* a manually-curated profile object (works until the user changes their mind).

Continuum is the **operational memory layer**: tiered storage so recent conversation, mid-term summaries, and long-term facts are addressable separately; **supersession** — superseded LTM facts are stamped `invalidated_at` (never deleted), so every read filters on `invalidated_at IS NULL` and the system always knows which version is current; and **bi-temporal columns** — `valid_from`/`valid_to` (world-time) plus `created_at`/`invalidated_at` (system-time) — so it can answer *"what did the user say about X **as of** date Y?"*, including retroactive corrections.

It is *not* a reasoning engine. The framework's value-prop is **what the layer below the reasoner should have been doing all along**, with benchmark numbers below to back it up.

## Why this exists — the headline measurements

The robust, **reproducible** wins are the deterministic ones (they don't move
between runs); the LongMemEval number is honest but carries reader variance.

| benchmark | Continuum | naive baseline |
|---|---|---|
| **Supersession correctness** (50 scripted updates) | **100 %** | 38 % |
| **Bi-temporal "as of date Y"** (20 scripted timelines) | **100 %** | 75 % |
| **LongMemEval-S** (500 Q, judged, gpt-oss-120b) | **~74 %** (73.6 – 75.6 % across repeated runs) | 60.8 % (v1.0) · 34.4 % (May ceiling) |
| ↳ same-setup per-category (control run) | multi-session 62 % · temporal 75 % · KU 64 % · assistant 96 % · user 91 % | — |
| Retrieval recall @ 4 (200-session synthetic corpus) | 55 % | 55 % (tied — recency signal absent) |
| Ingest p50 / session (1 user turn) | 0.18 ms + 6 LLM-extraction calls | 0.00 ms (raw list) |

Sources: LongMemEval-S numbers are documented in the [v1 findings report](findings/reasoning_loop_2026-06.md) and regenerated with `make repro-everything` (raw run outputs are gitignored, not committed); synthetic benchmarks from [`bench/`](bench/), reproducible via `make bench-all` (~60 s, no infra, no API key).

> **Honest headline — read this.** On LongMemEval-S the judged number is **~74 %**
> (73.6 – 75.6 % across repeated same-setup runs). `gpt-oss-120b` has ~±3–5 pp
> per-category run-to-run variance, so **a single run is not a stable estimate**:
> the **76.4 %** in earlier v2.0 notes was a favorable draw, and read-side
> additions (a bounded reflect pass + vote-of-3) are **within that noise of the
> no-additions baseline (73.8 %)** — approximately accuracy-neutral overall.
> Per-category figures swing between runs and should not be read as fixed. The
> **reproducible** wins are the two **deterministic** benches above — supersession
> and bi-temporal "as of date Y", both **100 %** — which don't move. We also
> disproved four "have the memory layer fix the reader" levers (synthesis,
> router, distillation, temporal code-math — all net-negative), and learned the
> methodology lesson that measuring gains only on known failures overstates them.
> Full write-up: [docs/limitations.md](docs/limitations.md) · [docs/report.md](docs/report.md).

> **v2.0** — **72.8% → 76.4% judged** (full-500, gpt-oss-120b), from one
> **architecture-native retrieval lever** — no new model, no compression:
> - **Over-fetch + cross-encoder rerank (WS-4):** over-fetch the candidate pool,
>   then a cross-encoder (`continuum/retrieval/reranker.py`) reorders it and
>   keeps the best 24. This lifts retrieval recall and lands on the
>   **retrieval-bound** category: **multi-session 57.9% → 69.9% (+12pp**, n=133
>   A/B, net +16 questions). Temporal +2.3pp bonus; knowledge-update flat
>   (within noise); reranking **gated off for single-session-preference** (it
>   reorders away the rubric turns). The ss-user −2.9pp that tripped the
>   regression guard is **reader jitter, not a regression** — per-question diff
>   shows the flips are inference questions that vary run-to-run, not dropped
>   context. **Honest framing: the win is multi-session retrieval, not a
>   model change.**
>
> **v1.1** — **60.8% → 72.8% judged** (confirmed full-500, gpt-oss-120b), from
> two **memory-attributable** fixes — no new model, no reasoning loop, just
> surfacing data the schema already had:
> - **Temporal date-surfacing (the driver):** temporal questions were answered
>   with the turn *dates stripped from the prompt* (the model literally replied
>   "no dates available"). Surfacing each turn's date + the reference "now"
>   lifted temporal **41.4% → 72.9% (+32pp)** — confirmed by an n=133 A/B
>   (signal +33pp vs a 0.8pp A/A noise floor).
> - **Supersession-recency:** mark the current (non-invalidated) LTM facts and
>   prefer them over stale turns → knowledge-update **~+4.5pp** (n=78 A/B).
>
> Both gated to their categories, so the solved ones (single-session 96–98%)
> didn't move. Honest note: 90%+ on this benchmark needs a *frontier reader* —
> our gains are what the **memory layer** earns at a fixed mid-tier model. The
> aggregation-prompt and preference levers were tested and **cut as net-neutral
> / unmeasurable** (see [`findings/roadmap_v1.1.md`](findings/roadmap_v1.1.md)).

> **v1.0.0** — Continuum broke the [May 2026 ~34% LongMemEval-S ceiling](findings/longmemeval_2026-05.md) to **60.8% judged** — and did it *without* the iterative reasoning we predicted we'd need. What actually moved the number (a stronger answerer + clean direct retrieval + honest scoring), and the reasoning loop we built and **cut** as net-negative, are documented in [**findings/reasoning_loop_2026-06.md**](findings/reasoning_loop_2026-06.md). A LOCOMO head-to-head vs Mem0 is preliminary (clean run pending) and not yet a published claim.

The supersession + bi-temporal wins aren't tunable parameters — they're consequences of the schema. Append-only stores and vector databases **cannot** reach these numbers without re-implementing this architecture on top.

---

## The 60-second demo

```bash
make demo-chat
```

Runs a scripted walkthrough where the user:

1. introduces themselves with `I live in NYC and just adopted a dog named Rex` →
   Continuum extracts `user.location = NYC` and `user.pets.dog = Rex` into LTM.
2. updates: `I just moved to Boston` →
   the NYC fact is marked **`superseded_by`** the Boston one. Both are still in storage; only Boston is "current".
3. queries `/show ltm` →  current facts only (Boston + Rex).
4. queries `/show ltm --all` →  full history *with* the supersession edge visible.
5. `/query "where do I live"` → returns Boston, not NYC, by construction.

The full demo runs in **63 ms** with the canned LLM (no API key). It's the canonical *show* of what `bench-supersession` proves as a number.

See [`examples/chat_agent/`](examples/chat_agent/) for source + the interactive REPL.

---

## Architecture in one diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                       ContinuumSession                             │
│                                                                    │
│   process_turn(user_msg)                                           │
│        │                                                           │
│        ▼                                                           │
│   ┌────────┐    ┌──────────┐   ┌─────────┐   ┌──────────────┐     │
│   │  STM   │ →  │ Retriever│ → │Optimizer│ → │  Responder   │     │
│   │ recent │    │ cosine + │   │ chain   │   │ your LLM     │     │
│   │ turns  │    │ composite│   │ (compress│   │ + ctx        │     │
│   │        │    │ scorer   │   │  budget) │   │              │     │
│   └────┬───┘    └──────────┘   └─────────┘   └──────┬───────┘     │
│        │                                            │              │
│        │ promotion (background queue)               │ reply        │
│        ▼                                            ▼              │
│   ┌────────┐                                  to caller            │
│   │  MTM   │ ← session summaries (LLM-generated)                   │
│   └────┬───┘                                                       │
│        ▼                                                           │
│   ┌────────┐                                                       │
│   │  LTM   │ ← atomic facts + entities + supersession             │
│   │ pgvector│   + bi-temporal columns (valid_from, invalidated_at)│
│   └────────┘                                                       │
└────────────────────────────────────────────────────────────────────┘
```

Each box is a swappable component. The retriever and optimizer chain are protocol-based; the stores live behind `STMProtocol` / `MTMProtocol` / `LTMProtocol` so the in-memory variant works for local dev and the Postgres variant scales to production.

---

## Quick start (no infra)

**Use it in Python** — the library API (`pip install continuum-memory`):

```python
import asyncio
from continuum import Memory

async def main():
    mem = Memory.in_memory()                        # zero-config — no Postgres, no model download
    async with mem:
        await mem.add("I moved to Boston")
        await mem.add("Actually I'm in NYC now")    # supersession handles the update
        print(await mem.recall("where do I live?"))
        print(await mem.current("user", "residence"))  # the resolved, current value
        print(await mem.timeline("residence"))          # bi-temporal history, oldest→newest

asyncio.run(main())
```

Or plug it into any MCP client (Claude Code, Cursor, …) with zero glue — see
[docs/mcp.md](docs/mcp.md): `pip install "continuum-memory[mcp]"` then `continuum-mcp`.

> Supersession + bi-temporal history are strongest on the Postgres path (below);
> `Memory.in_memory()` is the zero-setup path for demos and tests. Honest scope
> and known limits: [docs/limitations.md](docs/limitations.md).

**From source (demo + benchmarks):**

```bash
git clone https://github.com/Genkryptos/Continuum.git && cd Continuum

# 1. Install Continuum + its dev tooling
pip install -e .

# 2. See the 60-second story
make demo-chat

# 3. Reproduce the benchmarks
make bench-all

# 4. (Optional) reproduce the v1 LongMemEval-S numbers
#    answerer = gpt-oss-120b via OpenRouter; non-reasoning judge.
#    The dataset is fetched on demand (not committed to the repo).
export OPENROUTER_API_KEY=sk-or-…
make repro-everything
```

That's the complete loop. No Postgres needed for the demo or any of the benchmarks. The full production path with Postgres+pgvector is documented under [`continuum/stores/postgres/`](continuum/stores/postgres/) and exercised by the integration tests under `tests/integration/`.

### Production path (Postgres) + verifying your `.env`

```bash
cp .env.example .env       # fill in a provider key and/or CONTINUUM_DB_DSN
make db-up                 # start Postgres + pgvector (docker-compose), wait until ready
make db-migrate            # create the pgvector extension + LTM schema
make check-env             # ✓/✗ report: config loads, provider key, DB reachable + schema, smoke
make run                   # interactive chat REPL on a real Postgres-backed session
```

`make run` (i.e. `python -m continuum.chat`) starts an interactive `you>` REPL on a **real** `ContinuumSession` — `PostgresSTM` + `PostgresLTM` + the hybrid `Retriever` + an OpenRouter responder. Every turn persists to Postgres and is indexed for recall, so quitting and re-running with the same `--session` remembers the conversation. `make run` keeps the embedder **off** (LTM sparse/trigram + STM recency, no model download); use `make run-full` for dense semantic recall (downloads `bge-m3`, ~2.3 GB, once). Useful flags: `--session <id>` (named persistent session), `--model <id>` (OpenRouter model, default `openai/gpt-4o-mini`), `--in-memory` (no DB — also `make run-mem`), `--mock` (deterministic, no LLM call). Inside the REPL: `/help`, `/search <q>`, `/stats`, `/session <id>`, `/exit`. Pass extra flags via `make run ARGS="--session alice"`.

`make check-env` (i.e. `python -m continuum.doctor`) is the "did I set this up right?" command — it loads your config, detects which LLM provider key is present (in `.env` or your shell), probes the configured Postgres DSN for the `pgvector` extension, and runs one in-memory turn end-to-end. A missing key or unreachable DB is a **warning** (the offline path still works); a broken config or a session that won't start is a **failure** (non-zero exit, so it works in CI too). Add `--ping` to validate each provider key against a live read-only API call, or `--full` to actually load the embedder.

> The compose defaults match `.env.example`'s DSN (`postgresql://postgres:postgres@localhost:5432/continuum`). If your `.env` uses different credentials, set `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` (env or shell) before `make db-up` so the two agree.

---

## Make targets

| target | what |
|---|---|
| `make demo-chat` | scripted 60-second chat-agent walkthrough (in-memory) |
| `make run` | interactive chat REPL on a real Postgres-backed session (embedder off) |
| `make run-full` | same, with the embedder for dense recall (downloads bge-m3) |
| `make db-up` / `make db-down` | start / stop local Postgres+pgvector (docker-compose) |
| `make db-migrate` | apply migrations (pgvector ext + LTM schema) |
| `make check-env` | verify your `.env`: config, provider key, DB+schema, smoke |
| `make bench-ingest` | ingest throughput (Continuum vs raw list vs mem0) |
| `make bench-retrieval` | retrieval recall@k vs naive cosine |
| `make bench-supersession` | the killer feature — 100 % vs 38 % |
| `make bench-bitemporal` | "as of date Y" lookups — 100 % vs 75 % |
| `make bench-all` | run all four benchmarks in sequence |
| `make bench-locomo` | LOCOMO + Mem0 head-to-head (preliminary) |
| `make repro-longmemeval` | reproduce the LongMemEval-S evaluation |
| `make repro-everything` | reproduce both headline runs (LongMemEval + benches) |
| `make test` | full test suite (unit + integration) |
| `make test-fast` | unit tests only — no infra required |
| `make check` | format + lint + mypy strict |
| `make build` | build wheel + sdist into `dist/` |
| `make build-verify` | build, then install + smoke-test in a fresh venv |

---

## Documentation

Four focused docs under [`docs/`](docs/):

* [**Quickstart**](docs/quickstart.md) — five commands to a working `ContinuumSession`.
* [**Architecture**](docs/architecture.md) — the three tiers, promotion lifecycle, supersession, bi-temporal.
* [**Config reference**](docs/config.md) — every env-var and YAML knob with defaults.
* [**Operations**](docs/operations.md) — Postgres setup, scaling, observability, common production issues.

If you're new, the recommended path is: this README → `make demo-chat` → `docs/architecture.md` → `docs/quickstart.md`. About 30 minutes if you read everything; 10 if you skim.

## Honest evaluation — the LongMemEval-S reports

Two reports, read in order — the second corrects the first, which is the point:

**[`findings/longmemeval_2026-05.md`](findings/longmemeval_2026-05.md) (May)** — seven full sweeps across four model families and six retrievers found a hard **32-34% substring ceiling**, *even at 100% recall*, and concluded the bottleneck was multi-hop reasoning: a single-shot `retrieve → answer` pipeline couldn't exceed it "without an external reasoning loop."

**[`findings/reasoning_loop_2026-06.md`](findings/reasoning_loop_2026-06.md) (June, v1)** — we built that reasoning loop (the `IterativeReasoner`), A/B'd it, and it was **net-negative** — so we cut it. v1 broke the ceiling to **60.8% judged** with a *single-shot* pipeline anyway. What actually moved the number:

* a stronger answerer (the dominant lever — we say so plainly),
* the LLM judge revealing substring under-counted paraphrases by **+12.4 pp**,
* session-aware retrieval + fixing two silent truncation bugs (multi-session ~16% → 55%),
* LTM **supersession** making knowledge-update work (98.7% recall).

The June report is deliberately honest about what we got wrong (the reasoning-loop prediction), which lever bought which points, where we still lose (temporal-reasoning at 41%), and why the LOCOMO/Mem0 head-to-head is still *preliminary* rather than a published win. Reproduce both headline runs with `make repro-everything`.

The throughline that informs Continuum's positioning: **a strong model fed clean, complete context beats elaborate scaffolding** — be the memory layer that surfaces the right context cheaply; don't try to out-reason the reasoner.

---

## Project layout

```
continuum/                  the framework itself
├── core/                   session orchestration, types, config
├── stores/                 STM / MTM / LTM implementations
│   ├── stm/                in-memory + thread-safe + Postgres
│   ├── in_memory/          in-memory LTM with supersession (eval + local dev)
│   └── postgres/           pgvector-backed MTM + LTM with supersession + bi-temporal
├── retrieval/              composite scorer + BM25 + reciprocal-rank-fusion hybrid
├── extraction/             entity / fact / LLM extractors + cached SmallLLM helper
├── promotion/              Mem0Promoter, triggers, IdleStmFlush
├── optimizer/              token-budget compression chain (5 strategies)
├── scoring/                composite scorer (relevance / importance / recency / confidence)
├── policies/               policy engine + 8 default policies (migration 004)
├── reasoning/              IterativeReasoner — shipped but cut from v1 (tested negative result)
├── embeddings/             embedding service (sentence-transformers)
└── db/                     pgvector upgrade helpers

memory/                     legacy STM engine the framework re-exports (ConversationSTM)

bench/                      memory-operation benchmarks (make bench-all)
├── ingest_throughput.py    Continuum vs raw list vs mem0
├── retrieval_quality.py    recall@k vs naive cosine
├── supersession_correctness.py   100% vs 38%
└── bi_temporal.py          "as of date Y" — 100% vs 75%

evals/                      reproduction harness (datasets fetched on demand)
├── longmemeval/            LongMemEval-S driver (the v1 60.8% result)
└── locomo/                 LOCOMO + Mem0 head-to-head (preliminary)

findings/                   evaluation reports + reproducibility
├── longmemeval_2026-05.md  the May "32% ceiling" report
├── reasoning_loop_2026-06.md   the v1 correction (supersedes it)
└── charts/                 summary-generation scripts

docs/                       quickstart · architecture · config · operations
examples/chat_agent/        the 60-second CLI demo (make demo-chat)
scripts/                    diagnostic-sample + bench-regression + install-verify helpers
migrations/                 numbered Postgres migrations (pgvector, lexical search, policies)

tests/
├── unit/                   per-component unit tests
├── integration/            cross-tier flows + Postgres pgvector
└── acceptance/             phase-completion gates
```

> This `release` branch ships the **essential library + reproduction harness only**.
> Vendored datasets, raw run outputs (`results/`, `bench/results/*.json`,
> `.wiki_cache*`), and earlier prototype code are intentionally excluded — they're
> regenerated on demand and gitignored. Full history lives on `main`.

---

## What Continuum is *not*

* **Not a vector database.** It uses pgvector under the hood but adds tier semantics, supersession, and bi-temporal queries that pgvector alone doesn't provide.
* **Not a reasoning engine.** No agentic loop, no multi-hop retrieval, no chain-of-thought orchestration. Use LangGraph, AutoGen, or your own loop on top — Continuum is what they should plug into for memory.
* **Not a managed service.** It's an open-source Python framework. Self-host on your own Postgres.
* **Not a leaderboard-tuned LongMemEval contender.** We measured honestly; see the report.

---

## Status

Phase 1 (storage + protocols + tests), Phase 2 (extraction + promotion + policies + retrieval), and Phase 3 (benchmarks + demo + findings) are complete. The framework passes strict mypy + full unit / integration / acceptance test gates.

Production readiness (deploy scripts, observability, multi-tenant testing) is the next phase if and when there's interest.

---

## License

MIT.

## Citation

If Continuum's supersession or bi-temporal benchmarks are useful in your work:

```
Continuum: production-grade memory infrastructure for AI agents.
https://github.com/Genkryptos/Continuum, 2026.
```
