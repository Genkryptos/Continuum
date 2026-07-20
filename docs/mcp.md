# Continuum MCP server

Give any MCP client (Claude Code, Claude Desktop, Cursor, …) persistent,
supersession-aware memory with zero glue code.

## Install & run

```bash
pip install "continuum-memory[mcp]"
continuum-mcp                       # stdio — an MCP client spawns this itself
continuum-mcp --http --port 8000    # standalone always-on HTTP server (connect by URL)
```

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

## Notes

- Only the memory layer is exposed — no research/eval flags.
- `current`/`timeline` are only as strong as the store behind the server; the
  in-memory default is for demos, Postgres is where valid-time history lives.
- Advanced: embed the server in your own process with
  `from continuum.mcp import build_server; build_server(memory=my_memory)`.
