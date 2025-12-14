# Continuum
Continuum is an agent centric memory engine providing short-, mid- and long-term memory (STM/MTM/LTM) across sessions. It uses a vector store, scored retrieval and LLM-powered memory evolution, exposed via FastAPI so any agent framework can plug into it.

<picture>
  <!-- in dark mode, show the white-on-transparent image -->
  <source srcset="documentation/DarkEr.svg" media="(prefers-color-scheme: dark)">
  <!-- in light mode, show the dark-on-transparent image -->
  <source srcset="documentation/lightEr.svg" media="(prefers-color-scheme: light)">
  <!-- fallback (will be used if the browser doesn’t support <picture>) -->
  <img src="documentation/lightEr.svg" alt="ER Diagram">
</picture>

## Quick start
1. Create and activate a virtualenv, then install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables (or add them to a `.env` file):
   - `OPENAI_API_KEY` for cloud LLM/embedding calls (optional; dummy client is used otherwise)
   - `MTM_DB_URL` for the MTM Postgres instance (see `vectorDb/docker-compose.yml` for a local dev stack)
   - Optional agent settings in `settings.py` (LLM model, token budgets, embedding model).
3. Launch the CLI demo that wires STM + MTM together:
   ```bash
   python main.py
   ```
   The prompt will show STM token usage after each turn and automatically persists evicted messages to MTM.

## Project layout
- `agent/`: agent loops that orchestrate STM/MTM usage.
- `memory/stm/`: short-term memory implementation with compression hooks.
- `memory/mtm/`: mid-term memory retrieval, storage and summarization services.
- `helper/`: token counting, DB config helpers and small utilities.
- `api/`: FastAPI surface (stubbed in this snapshot) for serving Continuum.
- `vectorDb/docker-compose.yml`: quick-start Postgres/pgvector stack for MTM storage.

## Testing
Unit tests are located under `test/`. Run them with:
```bash
pytest
```
