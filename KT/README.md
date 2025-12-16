# Web Search MCP Integration (Continuum)

This doc explains how the live web-search MCP server was added, how it’s wired into `AgentMTM`, and how to run it end-to-end. Paths are relative to the repo root.

## Components
- Server: `continuum_mcp/web_search_server.py` (FastMCP server exposing `search_web` and `fetch_page`).
- Client utilities: `helper/web_search.py` (`WebSearchService` for search/fetch/format with ddgs/duckduckgo fallback).
- Agent integration: `agent/AgentMTM.py` optional web-context injection plus capability/time grounding prompts.
- Config: `settings.py` and `.env` flags for enabling, timeouts, and budgets.

## Install
```bash
pip install -r requirements.txt  # includes mcp + ddgs/duckduckgo-search
```

## Run the MCP server
```bash
python -m continuum_mcp.web_search_server
```
- Uses env vars: `WEB_SEARCH_MAX_RESULTS` (default 3), `WEB_SEARCH_TIMEOUT` (default 10s), `WEB_SEARCH_ALLOW_NETWORK` (1/0).
- Module name is `continuum_mcp` to avoid shadowing the pip `mcp` package.

## Enable web search in the agent
Set in `.env` (or export):
```
WEB_SEARCH_ENABLED=1          # turn on web search in the CLI agent
WEB_SEARCH_AUTO=1             # auto-search when MTM returns nothing
WEB_SEARCH_MAX_RESULTS=3      # per-query result cap
WEB_SEARCH_MAX_CONTEXT_TOKENS=512  # truncate injected context to fit budget
WEB_SEARCH_TIMEOUT=10
WEB_SEARCH_ALLOW_NETWORK=1
```
Then run the agent demo:
```bash
python main.py
```
Manual trigger: include `@web` or “web search …” in your message. Auto mode also fires when MTM returns no hits.

## Code snippets
Create a service (anywhere you need web search):
```python
from helper.web_search import WebSearchService

web = WebSearchService(
    max_results=3,
    timeout=10.0,
    allow_network=True,
)
results = web.search("latest space news")
print(web.format_results(results))
```

Wire into the agent (example from `main.py`):
```python
from agent.AgentMTM import AgentMTM
from helper.web_search import WebSearchService

web_search_service = WebSearchService(
    max_results=WEB_SEARCH_MAX_RESULTS,
    timeout=WEB_SEARCH_TIMEOUT,
    allow_network=WEB_SEARCH_ALLOW_NETWORK,
)

agent = AgentMTM(
    model_name=LLM_MODEL,
    mtm_retriever=mtm_retriever,
    stm_callbacks=mtm_callbacks,
    web_search_service=web_search_service,
    auto_web_search=WEB_SEARCH_AUTO,
    web_search_max_results=WEB_SEARCH_MAX_RESULTS,
    max_web_context_tokens=WEB_SEARCH_MAX_CONTEXT_TOKENS,
)
```

Agent prompt shaping (inside `AgentMTM.handle_user_input`):
- Inserts a capability hint when web search is available so the model knows results may be present.
- Inserts a current datetime system message for time-sensitive questions.
- When search is triggered, formats results as a `system` message: `Live web search results: …` and injects it near the top of the prompt, truncating if it would exceed `max_web_context_tokens` or the overall context budget.
- Returns `web_results`/`web_error` in the result dict for observability.

## Trigger logic
- Triggers on substrings: `@web`, `/web`, `web search`, `search the web`.
- If `WEB_SEARCH_AUTO=1`, also triggers when MTM returns no memories for the turn.
- If search fails or exceeds budget, the agent logs a warning and continues without breaking the turn.

## Tests
- Web integration/unit coverage in `test/test_agent_mtm.py`:
  - Verifies web context injection, capability hint, and current datetime prompt.
- Quick syntax check:
  ```bash
  python3 -m py_compile agent/AgentMTM.py helper/web_search.py continuum_mcp/web_search_server.py
  ```

## Troubleshooting
- `ModuleNotFoundError: mcp.server.fastmcp`: reinstall deps (`pip install -r requirements.txt`).
- `ddgs`/`duckduckgo_search` warnings: handled; ensure one of them is installed.
- Search SSL errors: set `WEB_SEARCH_ALLOW_NETWORK=0` to disable network or fix local certs.
- DB column errors: the batch insert now omits `embedding_model` to match the current `mtm_memories` schema.
