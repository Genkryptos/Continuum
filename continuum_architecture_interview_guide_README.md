# Continuum Architecture Interview Guide

Generated files:

- `continuum_architecture_interview_guide.html`
- `continuum_architecture_interview_guide.pdf`
- `continuum_architecture_interview_guide_README.md`

The HTML file is the source of truth. The PDF was rendered from that HTML.

## Repository Inspection Performed

The guide was written after inspecting the actual Continuum repository, including:

- `README.md`
- `docs/README.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/config.md`
- `docs/operations.md`
- `docs/continuum_eval_fix_plan.md`
- `documentation/README-legacy.md`
- modern runtime code under `continuum/`
- legacy STM/MTM/agent code under `memory/` and `agent/`
- migrations under `migrations/`
- benchmark scripts and latest benchmark outputs under `bench/`
- LongMemEval harness and wiki experiments under `evals/longmemeval/`
- generated findings under `findings/`
- representative generated results under `results/`
- tests under `tests/`
- `pyproject.toml`, `Makefile`, and `continuum.yaml`
- CLI/API/server entry points including `main.py`, `api/main.py`, `examples/chat_agent/agent.py`, and `continuum_mcp/web_search_server.py`

The document intentionally separates the packaged modern runtime from legacy/demo paths and evaluation-only code.

## PDF Generation

Tool used:

- Headless Google Chrome at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`

Command used from the repository root:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --print-to-pdf=/Users/mayanksahu/Continuum/continuum_architecture_interview_guide.pdf \
  file:///Users/mayanksahu/Continuum/continuum_architecture_interview_guide.html
```

Regenerate by editing `continuum_architecture_interview_guide.html`, then running the command above.

## Assumptions and Classification Notes

- `ContinuumSession` is implemented, but the default quickstart path only wires in-memory STM and a stub responder. MTM, LTM, retriever, promoter, optimizer, and real LLM responder are injected components.
- STM is implemented through both legacy `memory/stm/ConversationSTM.py` and modern async wrappers under `continuum/stores/stm/`.
- MTM is implemented in modern Postgres form under `continuum/stores/postgres/mtm.py`; legacy MTM code exists but is excluded from the wheel.
- LTM is marked partially implemented because schema, store, invalidation, hybrid search, and graph traversal exist, but robust source-traced supersession / knowledge-update semantics are not fully wired.
- Raw memory ledger is marked partially implemented because `memory_episodes` exists in migration 001, but no complete first-class runtime episode store was found.
- LLM Wiki / compiled memory is marked partially implemented because it exists in the LongMemEval evaluation harness, not as a core runtime source-of-truth layer.
- Temporal, preference, and knowledge-update engines are marked planned or partial based on `docs/continuum_eval_fix_plan.md`, policy/candidate code, and schema support.
- Continuum-Code is marked planned because config and policy hooks exist, while docs explicitly say the code-indexing subsystem is not yet implemented.
- `api/main.py` is marked planned/stub because it is empty in this snapshot.
- Production observability is marked partial/recommended: background queue stats, eval telemetry, and logging exist, but no first-class Prometheus/OpenTelemetry integration was found.

## Verification

After generation, verify:

```bash
ls -lh continuum_architecture_interview_guide.html \
       continuum_architecture_interview_guide.pdf \
       continuum_architecture_interview_guide_README.md

python3 -m html.parser continuum_architecture_interview_guide.html

file continuum_architecture_interview_guide.pdf
```

The guide does not claim full project test success; it is a generated architecture artifact based on repository inspection.
