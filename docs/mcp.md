# Continuum MCP server

Give any MCP client (Claude Code, Claude Desktop, Cursor, …) persistent,
supersession-aware memory with zero glue code.

## Install & run

```bash
pip install "continuum-memory[mcp]"
continuum-mcp          # runs over stdio
```

By default it uses a zero-config in-process store. To back it with Postgres
(where the full supersession decider and bi-temporal history live), set
`CONTINUUM_DB_DSN` (see `docs/config.md`) before launching.

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

## Tools

| tool | signature | what it does |
|---|---|---|
| `recall` | `recall(query, k=8)` | retrieve up to `k` relevant memories, best-first |
| `remember` | `remember(text, occurred_at?)` | store a fact/turn (`occurred_at` is an optional ISO date) |
| `current` | `current(subject, attribute)` | the current value after supersession, e.g. `current("user","residence")` |
| `timeline` | `timeline(entity, since?, until?)` | bi-temporal history for an entity, oldest→newest |

## Notes

- Only the memory layer is exposed — no research/eval flags.
- `current`/`timeline` are only as strong as the store behind the server; the
  in-memory default is for demos, Postgres is where valid-time history lives.
- Advanced: embed the server in your own process with
  `from continuum.mcp import build_server; build_server(memory=my_memory)`.
