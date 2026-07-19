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
```

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

Untagged facts still work via relevance-ranked retrieval, but that is best-effort.
When the store can answer attributes exactly, its answer is final — including
"no such fact", rather than guessing a wrong value.

## Notes

- Only the memory layer is exposed — no research/eval flags.
- `current`/`timeline` are only as strong as the store behind the server; the
  in-memory default is for demos, Postgres is where valid-time history lives.
- Advanced: embed the server in your own process with
  `from continuum.mcp import build_server; build_server(memory=my_memory)`.
