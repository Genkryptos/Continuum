# Continuum

[![PyPI](https://img.shields.io/pypi/v/continuum-mcp)](https://pypi.org/project/continuum-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/continuum-mcp)](https://pypi.org/project/continuum-mcp/)
[![CI](https://github.com/Genkryptos/Continuum/actions/workflows/ci.yml/badge.svg)](https://github.com/Genkryptos/Continuum/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21691914.svg)](https://doi.org/10.5281/zenodo.21691914)

**Memory for AI agents that knows what is true now, and what was true then.**

Most agent memory is a vector store: it finds similar text, but it can't tell
you which of two contradicting facts is current, or what the user believed last
March. Continuum keeps every fact with its history. When something changes, the
old fact is retired, not deleted, so your agent always answers with the latest
value and can still answer *"what was true on date Y?"*.

Use it as an **MCP server** (Claude Code, Claude Desktop, Cursor) or as a
**Python library** under any agent framework. It runs on your own Postgres,
with local embeddings, so by default no memory data leaves your machine.

```bash
pip install "continuum-mcp[mcp,postgres,embed]"
```

---

## Results

| | Continuum | Compared with |
|---|---|---|
| **Current value after an update** (50 scripted updates) | **100%** | 38% for a plain store |
| **"What was true on date Y?"** (20 scripted timelines) | **100%** | 75% for a plain store |
| **Point-in-time questions, head-to-head with the Mem0 SDK** ([paper](https://doi.org/10.5281/zenodo.21691914)) | **100%** | 27% for Mem0 |
| **LongMemEval-S accuracy**, 500 questions, gpt-oss-120b reader | **85.8%** | ~74% earlier Continuum · 60.8% v1.0 |
| **Answer-bearing turns returned by MCP `recall`** (default `k=24`) | **98%** | 80% before 2.2.0 |

How to read these:

- **The first three are deterministic.** They test the storage model: once an
  update is recognised (the benchmarks tag each fact with its attribute, as
  `add(..., attribute=...)` does), reads return the current value, or the value
  as of a past date, every time. They don't measure how well updates are
  detected in free text. Being properties of the schema, they don't move
  between runs; reproduce them in seconds with `make bench-supersession
  bench-bitemporal`, with no database and no API key.
- **The LongMemEval number** comes from the eval harness in `evals/longmemeval/`
  (retrieval plus a gpt-oss-120b reader and its answer prompts; llama-3.3-70b
  judge). With the stronger muse-wonder-1.0 reader on identical retrieval it
  reaches **90.8%**. The run used top-k 48 with long-term memory off, which is
  not the published configuration, so it is **not yet comparable** to
  leaderboard numbers. Re-run it with `run_arm_audit_fixes_oss120b.sh`;
  artifacts are in
  [`findings/longmemeval/audit_fixes_2026-11/`](https://github.com/Genkryptos/Continuum/blob/main/findings/longmemeval/audit_fixes_2026-11/).
- **The MCP number measures retrieval, not answers.** On 60 LongMemEval
  haystacks run through `Memory.recall()`, 98% of the turns that hold the answer
  come back in the top 24. Through MCP, your client model writes the answer.
  Details: [`findings/library_recall_2026-11/`](https://github.com/Genkryptos/Continuum/blob/main/findings/library_recall_2026-11/).

---

## Quick start

### As an MCP server

Give Claude Code (or any MCP client) persistent memory with four tools:
`remember`, `recall`, `current` and `timeline`.

```bash
pip install "continuum-mcp[mcp,postgres,embed]"

# point it at a Postgres database with pgvector, and migrate once
python -m continuum.db.migrate --dsn "$CONTINUUM_DB_DSN"

# register it with Claude Code over stdio
claude mcp add continuum -e CONTINUUM_DB_DSN="$CONTINUUM_DB_DSN" -- continuum-mcp
```

Prefer a long-running server? `continuum-mcp --http --port 8000`, then
`claude mcp add -t http continuum http://127.0.0.1:8000/mcp`.

Embeddings run locally on `bge-m3`, so memory itself needs no API key. Full
reference (transports, namespaces, autostart, client configs):
[docs/mcp.md](https://github.com/Genkryptos/Continuum/blob/main/docs/mcp.md).

### As a Python library

```python
import asyncio
from continuum import Memory

async def main():
    async with Memory.in_memory() as mem:  # no Postgres, no model download
        await mem.add("I live in Boston", attribute="residence")
        await mem.add("I moved from Boston to New York City", attribute="residence")

        print(await mem.current("user", "residence"))
        # -> I moved from Boston to New York City

        for fact in await mem.timeline("Boston"):  # every version, oldest first
            print(fact.content)

        hits = await mem.recall("where do I live?")
        print([h.content for h in hits])

asyncio.run(main())
```

For production, `Memory.from_postgres(dsn)` gives you durable storage, dense
recall and bi-temporal queries.

### See it in 60 seconds, no setup

```bash
git clone https://github.com/Genkryptos/Continuum.git && cd Continuum
pip install -e .
make demo-chat
```

A scripted conversation: the user moves from NYC to Boston, and Continuum
retires the old address, answers "where do I live?" with Boston, and still
shows the full history on request. It runs in under a second with a canned LLM.

---

## Why Continuum

- **Supersession.** A contradicting fact retires the old one
  (`invalidated_at` is set; nothing is deleted). Reads see only what is
  current, and the history is still there.
- **Bi-temporal history.** Every fact carries world time (`valid_from` /
  `valid_to`) and system time (`created_at` / `invalidated_at`), so "as of
  date Y" questions work, including corrections made after the fact.
- **Hybrid retrieval.** Dense vectors, trigram lexical search and
  reciprocal-rank fusion in one Postgres query, on pgvector's HNSW index.
- **Tiered memory.** Recent turns, session summaries and long-term facts are
  stored and addressed separately, so context stays cheap as history grows.
- **Safe by default.** `remember` refuses text that looks like a credential.
  Writes to a non-durable store say so instead of reporting a silent success.
  `forget` only retires facts that are already superseded, is a dry run
  unless you say otherwise, and is deliberately not exposed to agents.
- **Built to be swapped.** Stores, retriever, scorer and optimizer sit behind
  protocols. Use the in-memory backends for tests and Postgres in production.

---

## How it works

```
         your agent / MCP client
                   │
        remember · recall · current · timeline
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ Memory                                                      │
│                                                             │
│  STM  recent turns, verbatim                                │
│   │                                                         │
│   ▼   promotion (background)                                │
│  MTM  session summaries                                     │
│   │                                                         │
│   ▼                                                         │
│  LTM  atomic facts on Postgres + pgvector                   │
│       supersession  ·  valid_from / valid_to  ·  namespaces │
│                                                             │
│  Retriever: dense + lexical + RRF  →  scorer  →  top k      │
└─────────────────────────────────────────────────────────────┘
```

Architecture in depth: [docs/architecture.md](https://github.com/Genkryptos/Continuum/blob/main/docs/architecture.md).

---

## Reproduce the numbers

```bash
make bench-all            # every memory-operation benchmark (supersession, bi-temporal, retrieval, ingest)
make mcp-eval             # MCP retrieval quality over a fixed scenario
make repro-everything     # the v2.0 LongMemEval-S run (~74%) + LOCOMO; needs OPENROUTER_API_KEY
./run_arm_audit_fixes_oss120b.sh   # the Nov 2026 run (85.8%); needs OPENROUTER_API_KEY
```

<details>
<summary>All make targets</summary>

| target | what |
|---|---|
| `make demo-chat` | scripted 60-second chat-agent walkthrough (in-memory) |
| `make run` | interactive chat REPL on a real Postgres-backed session (embedder off) |
| `make run-full` | the same, with the embedder for dense recall (downloads bge-m3) |
| `make db-up` / `make db-down` | start / stop local Postgres + pgvector (docker-compose) |
| `make db-migrate` | apply migrations (pgvector extension + LTM schema) |
| `make check-env` | verify your `.env`: config, provider key, DB and schema, smoke turn |
| `make bench-supersession` | current value after updates: 100% vs 38% |
| `make bench-bitemporal` | "as of date Y" lookups: 100% vs 75% |
| `make bench-all` | every memory-operation benchmark in sequence |
| `make bench-locomo` | LOCOMO + Mem0 head-to-head (preliminary) |
| `make mcp-eval` | MCP retrieval quality (recall, supersession, timeline) |
| `make repro-longmemeval` | reproduce the LongMemEval-S evaluation |
| `make repro-everything` | reproduce the v2.0 LongMemEval-S run (~74%) + LOCOMO |
| `make test` / `make test-fast` | full test suite / unit tests only |
| `make check` | format + lint + strict mypy |
| `make build-verify` | build, then install and smoke-test in a fresh venv |

</details>

<details>
<summary>Production setup with Postgres</summary>

```bash
cp .env.example .env       # set CONTINUUM_DB_DSN and, for the chat REPL, a provider key
make db-up                 # Postgres + pgvector via docker-compose
make db-migrate            # create the schema
make check-env             # checks config, provider key, DB reachability and schema
make run                   # interactive REPL on a Postgres-backed session
```

Operations guide (scaling, observability, common issues):
[docs/operations.md](https://github.com/Genkryptos/Continuum/blob/main/docs/operations.md).

</details>

---

## How we got here, honestly

Continuum's LongMemEval history is public, including the parts we got wrong:

| when | LongMemEval-S | what moved it |
|---|---|---|
| May 2026 | 34% ceiling | seven sweeps hit a wall; we predicted only a reasoning loop could break it ([report](https://github.com/Genkryptos/Continuum/blob/main/findings/longmemeval_2026-05.md)) |
| v1.0 | 60.8% | a stronger reader and clean single-shot retrieval; the reasoning loop we built was net-negative and cut ([report](https://github.com/Genkryptos/Continuum/blob/main/findings/reasoning_loop_2026-06.md)) |
| v1.1 – v2.0 | ~74% | surfacing turn dates (temporal 41% → 73%), preferring current facts, reranking; 73.6–75.6% across repeated runs |
| Nov 2026 | 85.8% | turn selection inside retrieved sessions; the misses were turns lost within sessions that had been found ([artifacts](https://github.com/Genkryptos/Continuum/blob/main/findings/longmemeval/audit_fixes_2026-11/)) |

We also tested and dropped four "let memory fix the reader" ideas (synthesis,
routing, distillation, temporal code-math), all net-negative. Known limits:
[docs/limitations.md](https://github.com/Genkryptos/Continuum/blob/main/docs/limitations.md) · full write-up:
[docs/report.md](https://github.com/Genkryptos/Continuum/blob/main/docs/report.md).

---

## Documentation

- [MCP server](https://github.com/Genkryptos/Continuum/blob/main/docs/mcp.md): transports, tools, namespaces, client setup
- [Quickstart](https://github.com/Genkryptos/Continuum/blob/main/docs/quickstart.md): from install to a working session
- [Architecture](https://github.com/Genkryptos/Continuum/blob/main/docs/architecture.md): tiers, promotion, supersession, bi-temporal model
- [Configuration](https://github.com/Genkryptos/Continuum/blob/main/docs/config.md): every environment variable and YAML setting
- [Operations](https://github.com/Genkryptos/Continuum/blob/main/docs/operations.md): Postgres, scaling, observability
- [Limitations](https://github.com/Genkryptos/Continuum/blob/main/docs/limitations.md): what Continuum does not do well yet

## What Continuum is not

- **Not a vector database.** It uses pgvector, and adds current-versus-stale
  semantics and time that a vector store doesn't have.
- **Not a reasoning engine.** No agent loop. Use LangGraph, AutoGen or your own
  loop on top; Continuum is the memory underneath.
- **Not a managed service.** Open source; you run it on your own Postgres.

## Paper

The design, the formal bi-temporal model and the head-to-head with the Mem0 SDK
are in the preprint. On the same scenarios, Mem0 answers latest-value questions
competitively (72%) but only 27% of point-in-time questions, because it deletes
superseded facts; Continuum answers 100%.

> Sahu, M. (2026). *Correctness Under Time: Bi-Temporal, Supersession-Aware
> Memory for AI Agents.* Zenodo. https://doi.org/10.5281/zenodo.21691914

```bibtex
@misc{sahu2026continuum,
  author    = {Sahu, Mayank},
  title     = {Correctness Under Time: Bi-Temporal, Supersession-Aware Memory for AI Agents},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21691914},
  url       = {https://doi.org/10.5281/zenodo.21691914}
}
```

Source: [`paper/continuum.tex`](https://github.com/Genkryptos/Continuum/blob/main/paper/continuum.tex).

## License

[MIT](https://github.com/Genkryptos/Continuum/blob/main/LICENSE).
