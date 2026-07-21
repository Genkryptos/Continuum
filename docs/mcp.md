# Continuum MCP server

Give any MCP client (Claude Code, Claude Desktop, Cursor, …) persistent,
supersession-aware memory with zero glue code.

## Install & run

```bash
pip install "continuum-memory[mcp,postgres,embed]"   # the real thing (see below)
continuum-mcp                       # stdio — an MCP client spawns this itself
continuum-mcp --http --port 8000    # standalone always-on HTTP server (connect by URL)
```

`[mcp]` alone installs a working server, but only the **in-memory** backend:
recall is recency-ranked and everything is lost when the process exits. Add
`postgres` for durability and `embed` for semantic recall — or skip `embed` and
set `CONTINUUM_MCP_EMBEDDINGS=0` for a sparse-only server that still persists
(lexical recall, no ~2.3 GB model download).

Both paths are verified from a fresh venv against the built wheel, not just in
the dev tree: handshake → `remember` → `recall`, with the Postgres run asserted
down to the row (`scripts/mcp_smoke.py`, `make mcp-smoke`).

## Backends

`continuum-mcp` picks its store from the environment:

| Backend | Selected when | Recall | Persistence |
|---|---|---|---|
| **In-memory** (default) | no DSN set | recency only — **no retriever** | ephemeral: resets when the server process exits |
| **Postgres + embedder** | `CONTINUUM_DB_DSN` set | dense + sparse hybrid, **relevance-ranked** | durable: survives restarts |

```bash
export CONTINUUM_DB_DSN=postgresql://user:pass@localhost:5432/continuum
export CONTINUUM_MCP_EMBEDDINGS=0   # optional: sparse-only, skip the ~2.3GB bge-m3 download
continuum-mcp
```

The Postgres backend needs a migrated database (`make db-up && make db-migrate`;
see `docs/config.md`). The in-memory default is fine for demos, but its recall is
recency-ranked — at scale the relevant memory gets buried, which also weakens
`current` (it depends on recall surfacing the fact). Use Postgres for real recall.

### Keep your DSN out of the repo

**`continuum.yaml` is tracked in git.** Its DSN points at the docker-compose
database (`postgres:postgres@…`) on purpose, and it must stay that way. Put your
real connection string in the **environment** — `CONTINUUM_DB_DSN`, your shell
profile, or the MCP client's `env` block — never in the tracked config. Editing
that file to reach your own database is how a machine username or password ends
up in a public commit; it happened to us during this cycle and was caught by
luck, not by review.

Same rule for the client side. `.claude/settings.local.json` is gitignored and is
a fine place for a DSN; `.mcp.json` is *meant* to be committed, so keep the
connection string in the environment and let the server read `CONTINUUM_DB_DSN`
from there. Check which kind of file you are editing before you paste a DSN in.

## When the database is down

With a DSN configured, the server **never silently falls back to in-memory** —
answering from a store that vanishes at exit is worse than an error, because the
caller believes it has durable memory. Instead the tool call returns a clear
`isError` message and the server stays up, so the rest of the session still works.

Set `CONTINUUM_MCP_AUTOSTART` to have it bring the database up on first use:

```bash
export CONTINUUM_MCP_AUTOSTART="brew services start postgresql@16"   # or: docker compose up -d postgres
```

Opt-in only — with the variable unset nothing is ever executed. On a failed
connect the server runs that command once, polls until the database answers, and
retries; if it still can't connect you get the error above rather than a hang.
Because it lives in the server, this works for **any** MCP client, and also
covers the database dying mid-session.

> **Readiness means the database, not the port.** The probe opens a real
> connection rather than checking `host:port`. A port check reports success
> whenever *anything* owns the port — including a different PostgreSQL major
> version that lacks your database entirely (two Homebrew versions both wanting
> 5432 is enough to trigger it). Acting on that false positive silently points
> memory at the wrong store, so `pg_isready`-style checks are not sufficient.

For Claude Code you can additionally warm the backend before the first prompt
with a `SessionStart` hook in `.claude/settings.local.json`, which also removes
the one-off model-load latency from your first memory call.

## Isolation (multi-user / multi-project)

By default all memory lives in one shared store (`namespace = "default"`). To keep
tenants apart on a single database, set a namespace per server:

```bash
export CONTINUUM_MCP_NAMESPACE=alice     # or a project name, an agent id, ...
```

Every write is stamped with it and every read (`recall`, `current`, `timeline`,
graph expansion, and the short-term buffer) is filtered to it, so two namespaces
on the same database can never surface each other's facts. Without this, one
global store was shared across every session, user, and project.

## Supersession (optional, needs an LLM)

Without it, a fact and its later correction both sit in the store competing for
the same query — "pricing is 9 dollars" can outrank "switched to 12 dollars",
because relevance has no idea which is current. Turn it on and each write is
routed through the Mem0 ADD/UPDATE/DELETE decider, which retires or merges the
fact it replaces:

```bash
export CONTINUUM_MCP_SUPERSESSION=1
export OPENROUTER_API_KEY=...                       # or put it in .env
export CONTINUUM_MCP_SUPERSESSION_MODEL=openai/gpt-4o-mini   # optional
```

Measured on a five-write scenario, `recall@1` on changed facts went **1/3 -> 3/3**:
"Pricing will be 9 dollars" was retired, and "Target launch is end of Q3" plus
"Launch slipped to Q4" were merged into a single current fact.

**Off by default, and refused without a key.** The decider only consults the LLM
for the ambiguous similarity band (contradictions sit around cosine 0.6-0.8);
with no LLM it returns NOOP there, and honouring that would silently discard the
new fact. It costs one small completion per write that has near neighbours, and
it rewrites text when merging.

## Automatic recall (Claude Code hook)

By default memory only fires when Claude decides to call `recall`. To make it
fire on **every** turn, register a `UserPromptSubmit` hook that recalls against
the prompt and injects the hits as context:

```jsonc
// .claude/settings.local.json
{ "hooks": { "UserPromptSubmit": [ { "hooks": [ {
  "type": "command",
  "command": "CONTINUUM_DB_DSN=postgresql://…/continuum python -m continuum.mcp.recall_hook",
  "timeout": 8
} ] } ] } }
```

It reads the prompt on stdin and emits `additionalContext`. Design:

- **Never blocks the prompt** — no DB, bad input, or any error prints nothing and
  exits 0, so the turn proceeds untouched.
- **Fast**: a fresh process per prompt cannot load the 2.3GB embedder (~7s each),
  so it uses **sparse** recall — sub-second. Use the same namespace as the server.
- Env: `CONTINUUM_MCP_NAMESPACE` (scope), `CONTINUUM_RECALL_HOOK_K` (default 5),
  `CONTINUUM_RECALL_HOOK_EMBEDDINGS=1` (dense; only worthwhile against a warm
  process, not the per-prompt hook).

## Add to a client

**Claude Desktop / Claude Code** — add to your MCP config (`claude_desktop_config.json`
or `.mcp.json`):

```json
{
  "mcpServers": {
    "continuum": {
      "command": "continuum-mcp"
    }
  }
}
```

**Cursor** — `~/.cursor/mcp.json`:

```json
{ "mcpServers": { "continuum": { "command": "continuum-mcp" } } }
```

**Claude Code, one-liner** (stdio, or point at the always-on HTTP server):

```bash
claude mcp add continuum -- continuum-mcp
claude mcp add continuum --transport http http://127.0.0.1:8000/mcp   # HTTP daemon
```

## Test it

```bash
make mcp-smoke   # handshake + remember→recall round-trip (proves it works, no Claude)
make mcp-eval    # scores retrieval: recall@1/@3, supersession, timeline
make mcp-bench   # latency: p50/p95 per tool + embedder-vs-DB breakdown
```

`mcp-bench` never inherits `CONTINUUM_DB_DSN` — it writes facts, so it would
pollute a real store. It runs in-memory unless you hand it a throwaway:
`make mcp-bench MCP_BENCH_DSN=postgresql://…/continuum_bench`.

Measured on Postgres 16.10 + pgvector 0.8.0 + bge-m3 (CPU, Apple silicon):

| tool | p50 | dominated by |
|---|---:|---|
| `remember` | ~81 ms | embedding the text |
| `recall` | ~87 ms | embedding the query |
| `timeline` | ~8 ms | no embed on a repeated entity |
| `current` | **~1.6 ms** | exact tag lookup — never touches the model |

The embedder is ~77 ms of every semantic call (hybrid retrieval itself is ~7 ms),
so attribute-keyed `current` is ~50× faster than a semantic `recall`. Add ~7 s
one-off on the first tool call of a session for pool open + model load.

`mcp-eval` runs a fixed scenario with distractors. On the in-memory backend
recall@1 is low (recency, no retriever); set `CONTINUUM_DB_DSN` and it measures
the Postgres + embedder stack, where the relevant memory ranks first.

## Tools

| tool | signature | what it does |
|---|---|---|
| `recall` | `recall(query, k=8)` | retrieve up to `k` relevant memories, best-first |
| `remember` | `remember(text, occurred_at?, attribute?)` | store a fact/turn. `occurred_at` = when it became true; `attribute` = what it's *about* |
| `current` | `current(subject, attribute, as_of?)` | the current value after supersession, e.g. `current("user","residence")`; `as_of` asks what was current back then |
| `timeline` | `timeline(entity, since?, until?)` | bi-temporal history for an entity, oldest→newest |

### Tag attributes so `current` is exact

`current` asks about **one attribute**. Tag facts on write and it becomes an
exact lookup honouring valid time, instead of a fuzzy search for the attribute's
*name* (an attribute label like `"user residence"` is a poor semantic probe for a
sentence like *"I moved from Boston to New York City"*):

```python
remember("I live in Boston",              occurred_at="2026-01-10", attribute="residence")
remember("I moved from Boston to NYC",    occurred_at="2026-06-15", attribute="residence")

current("user", "residence")                    # → "I moved from Boston to NYC"
current("user", "residence", as_of="2026-03-01") # → "I live in Boston"  (bi-temporal)
current("user", "employer")                      # → "not found"  (honest — no such fact)
```

If **nothing** in the store is tagged, `current` falls back to relevance-ranked
retrieval — best-effort, and only as good as the probe. But once the store *does*
use attribute tags, its lookup is authoritative: an attribute with no fact
returns "not found" instead of inventing one from an unrelated memory.

## Seeing what it did

The server is silent by default — an MCP client shows you tool *results*, never
why a `recall` came back thin. Turn on the log to find out:

```bash
export CONTINUUM_MCP_LOG_LEVEL=INFO   # WARNING (default) · INFO · DEBUG
```

```
continuum-mcp INFO continuum.mcp.server: tool=remember chars=17 durable=True [87ms]
continuum-mcp INFO continuum.mcp.server: tool=recall query=where do I live k=3 hits=1 [18ms]
```

Every line carries the tool, its inputs, the outcome (`hits=`, `durable=`,
`found=`) and wall-clock latency; a failing tool logs `FAILED` with a traceback
and still returns a proper MCP error to the client. `hits=0` on a query you
expected to match is the signal that retrieval — not the client — is the problem.

Logs go to **stderr only**: on the stdio transport stdout *is* the JSON-RPC
channel, and a single stray line there corrupts the session. In Claude Code they
land in the MCP server log (`claude mcp` shows the path); over HTTP/SSE they go
to the terminal running the daemon.

## Notes

- Only the memory layer is exposed — no research/eval flags.
- `current`/`timeline` are only as strong as the store behind the server; the
  in-memory default is for demos, Postgres is where valid-time history lives.
- Advanced: embed the server in your own process with
  `from continuum.mcp import build_server; build_server(memory=my_memory)`.
