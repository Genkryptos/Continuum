# Continuum

**Production-grade memory infrastructure for AI agents.** Tiered storage (STM / MTM / LTM), first-class supersession and bi-temporal queries, cost-efficient retrieval. Plugs in under whatever reasoner you're using — LangGraph, AutoGen, custom — and provides the memory layer they all lack.

```bash
git clone <repo> && cd Continuum
make demo-chat       # 60-second walkthrough, no infrastructure, no API key
make bench-all       # the four benchmarks Continuum is built to win
```

---

## What it is

Most "memory" for LLM agents is one of:

* a vector database (great recall, no notion of "current" vs "stale" facts),
* an append-only chat history (no organization, gets expensive fast),
* a manually-curated profile object (works until the user changes their mind).

Continuum is the **operational memory layer**: tiered storage so recent conversation, mid-term summaries, and long-term facts are addressable separately; **`superseded_by` edges** on LTM facts so the system knows which version is current; **`valid_from` / `recorded_at` bi-temporal columns** so the system can answer *"what did the user say about X **as of** date Y?"* — including retroactive corrections.

It is *not* a reasoning engine. The framework's value-prop is **what the layer below the reasoner should have been doing all along**, with benchmark numbers below to back it up.

## Why this exists — the headline measurements

| benchmark | Continuum | best naive baseline | delta |
|---|---|---|---|
| **Supersession correctness** (50 scripted updates) | **100 %** | 38 % | **+62 pp** |
| **Bi-temporal "as of date Y"** (20 scripted timelines) | **100 %** | 75 % | **+25 pp** |
| Retrieval recall @ 4 (200-session synthetic corpus) | 55 % | 55 % | tied (recency signal absent) |
| Ingest p50 / session (1 user turn) | 0.18 ms + 6 LLM-extraction calls | 0.00 ms (raw list) | framework overhead |

Source: [`bench/`](bench/), reproducible via `make bench-all` (~60 seconds, no infra, no API key).

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
│   │  LTM   │ ← atomic facts + entities + supersession edges        │
│   │ pgvector│   + bi-temporal columns (valid_from, recorded_at)    │
│   └────────┘                                                       │
└────────────────────────────────────────────────────────────────────┘
```

Each box is a swappable component. The retriever and optimizer chain are protocol-based; the stores live behind `STMProtocol` / `MTMProtocol` / `LTMProtocol` so the in-memory variant works for local dev and the Postgres variant scales to production.

[Old ER diagram for the data model →](documentation/lightEr.svg)

---

## Quick start (no infra)

```bash
git clone <repo> && cd Continuum

# 1. Install Continuum + its dev tooling
pip install -e .

# 2. See the 60-second story
make demo-chat

# 3. Reproduce the benchmarks
make bench-all

# 4. (Optional) reproduce the LongMemEval-S evaluation numbers
#    ~30 min, ~$0.10 of OpenAI usage (gpt-4o-mini × 500 questions)
export OPENAI_API_KEY=sk-…
make repro-longmemeval
```

That's the complete loop. No Postgres needed for the demo or any of the benchmarks. The full production path with Postgres+pgvector is documented under [`continuum/stores/postgres/`](continuum/stores/postgres/) and exercised by the integration tests under `tests/integration/`.

---

## Make targets

| target | what |
|---|---|
| `make demo-chat` | scripted 60-second chat-agent walkthrough |
| `make bench-ingest` | ingest throughput (Continuum vs raw list vs mem0) |
| `make bench-retrieval` | retrieval recall@k vs naive cosine |
| `make bench-supersession` | the killer feature — 100 % vs 38 % |
| `make bench-bitemporal` | "as of date Y" lookups — 100 % vs 75 % |
| `make bench-all` | run all four benchmarks in sequence |
| `make repro-longmemeval` | reproduce the LongMemEval-S evaluation |
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

## Honest evaluation — the LongMemEval-S report

[`findings/longmemeval_2026-05.md`](findings/longmemeval_2026-05.md) documents what we learned from running seven full LongMemEval-S sweeps across four model families and six retriever variants.

**Headline finding**: substring accuracy plateaus at **32 %** regardless of model or retriever, *even at 100 % recall* (a Llama-3.3-70B long-context run with the full haystack in context). The ceiling is not retrieval — it's the model's ability to compose multi-hop answers at query time. A single-shot `retrieve → answer` pipeline cannot exceed this on LongMemEval-S without an external reasoning loop.

We document this honestly rather than chase a leaderboard number with techniques that don't fit Continuum's scope. The report includes:

* full results table across 6 runs × 6 categories,
* the failure-mode breakdown (90 % wrong_retrieval — facts present, model failed to compose),
* auto-generated charts at [`findings/charts/`](findings/charts/),
* a reproducibility artifact at [`findings/longmemeval/repro/`](findings/longmemeval/repro/) — `make repro-longmemeval` reproduces the two headline runs in 30 min for $0.10.

The report's conclusion is the same one that informed Continuum's positioning: **be the memory layer; let the caller bring the reasoner.**

---

## Project layout

```
continuum/                  the framework itself
├── core/                   session orchestration, types, config
├── stores/                 STM / MTM / LTM implementations
│   ├── stm/                in-memory + thread-safe + Postgres
│   └── postgres/           pgvector-backed MTM + LTM with supersession + bi-temporal
├── retrieval/              composite-scorer retriever + reranker
├── extraction/             entity + fact + LLM extractors
├── promotion/              Mem0Promoter, triggers, IdleStmFlush
├── optimizer/              token-budget compression chain (5 strategies)
├── scoring/                composite scorer (relevance / importance / recency / confidence)
└── policies/               policy engine + 8 default policies (migration 004)

bench/                      Phase 3B memory-operation benchmarks
├── ingest_throughput.py
├── retrieval_quality.py
├── supersession_correctness.py
└── bi_temporal.py

examples/
└── chat_agent/             the 60-second CLI demo

findings/                   the LongMemEval-S evaluation + reproducibility
├── longmemeval_2026-05.md  technical report
├── charts/                 auto-generated PNGs + extraction script
└── longmemeval/repro/      `make repro-longmemeval` artifact

evals/longmemeval/          the eval driver (used by repro)

tests/
├── unit/                   per-component unit tests
├── integration/            cross-tier flows + Postgres pgvector
└── acceptance/             phase-completion gates

migrations/                 numbered Postgres migrations (pgvector, lexical search, policies)
```

---

## What Continuum is *not*

* **Not a vector database.** It uses pgvector under the hood but adds tier semantics, supersession, and bi-temporal queries that pgvector alone doesn't provide.
* **Not a reasoning engine.** No agentic loop, no multi-hop retrieval, no chain-of-thought orchestration. Use LangGraph, AutoGen, or your own loop on top — Continuum is what they should plug into for memory.
* **Not a managed service.** It's an open-source Python framework. Self-host on your own Postgres.
* **Not a vibrant-tuned LongMemEval contender.** We measured honestly; see the report.

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
https://github.com/<owner>/Continuum, 2026.
```
