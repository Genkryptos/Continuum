# Chat agent demo — Continuum in 60 seconds

A minimal CLI chat agent that demonstrates Continuum's three core
features without infrastructure:

1. **Tiered memory** — STM (raw turns) + LTM (extracted facts).
2. **Supersession** — when a new fact contradicts an existing one,
   the old one is marked `superseded_by` the new one; retrieval
   filters to current facts only.
3. **Inspectable state** — slash-commands dump each tier so you can
   see exactly what the system remembers and why.

No Postgres, no API keys, no model downloads. Runs in **63 ms** on
the canned-LLM path.

## Run the scripted demo

```bash
bash examples/chat_agent/demo.sh
```

You'll see a six-act walkthrough that:

1. Plants two facts (`user.location = NYC`, `user.pets.dog = Rex`).
2. Confirms chitchat is correctly ignored — no spurious fact extraction.
3. Supersedes the location (`NYC → Boston`) — the *original* fact
   stays in storage, marked `superseded_by`, but `/show ltm` (current-
   facts view) only shows Boston.
4. Demonstrates `/show ltm --all` revealing the full history with the
   supersession edge visible.
5. Runs `/query` calls that respect supersession — the retrieved set
   contains the *current* Boston/Luna facts, never the stale ones.

The whole script runs in well under a second.

## Run interactively

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    -m examples.chat_agent.agent
```

Then type chat messages or slash-commands:

```text
you> Hi! I just moved to Austin for a new job at Globex.
  → extracted user.location = Austin
  → extracted user.employer = Globex
  agent> Got it. (recalled: user.location = Austin; user.employer = Globex)

you> /show ltm
  CURRENT FACTS (n=2)
  [a1b2c3d4] user.location = Austin    valid_from=10:14:22 (3s ago)
  [e5f6a7b8] user.employer = Globex    valid_from=10:14:22 (3s ago)
```

Available commands:

| command | what |
|---|---|
| `/show stm` | last 10 turns (raw conversation) |
| `/show ltm` | facts that haven't been superseded |
| `/show ltm --all` | every fact, with supersession arrows |
| `/query <text>` | retrieve current facts overlapping `<text>` |
| `/help`, `/exit` | obvious |

## Use a real LLM (optional)

```bash
export OPENAI_API_KEY="sk-…"
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    -m examples.chat_agent.agent --llm openai
```

Now the agent's responses come from gpt-4o-mini, conditioned on the
recalled facts. Memory mechanics are identical — only the response
text changes.

## How this maps to the real Continuum framework

This demo deliberately keeps everything in-process so a stranger can
clone the repo and run it without setup. Every concept it shows maps
to a real Continuum component:

| demo concept | real Continuum equivalent |
|---|---|
| `DemoMemory.stm`             | `continuum.stores.stm.InMemorySTM` / `PostgresSTM` |
| `DemoMemory.ltm` with `_Fact` | `continuum.stores.postgres.PostgresLTM` rows (migration 004 schema) |
| `_apply_supersession` | the `superseded_by` FK column + `Mem0Promoter` contradiction detection |
| `extract_facts` (rule-based) | `continuum.extraction.FactExtractor` + `LLMEntityExtractor` (LLM-driven) |
| `retrieve` (substring) | `continuum.retrieval.STMSemanticRetriever` (cosine + composite scorer) |
| `_canned_responder` / `_openai_responder` | the `responder` callback on `ContinuumSession` |

The architecture is identical. Production differences:

* Real LTM uses pgvector for cosine similarity over embeddings, not
  substring overlap.
* Real fact extraction uses an LLM; this demo uses regex so it's
  free and deterministic.
* Real supersession detection is LLM-driven; this demo applies
  supersession on every fact-add for the same attribute, which gives
  the same outcome but skips the detector quality axis (see
  [`bench/supersession_correctness.py`](../../bench/supersession_correctness.py)
  for the schema-only benchmark and [`findings/longmemeval_2026-05.md`](../../findings/longmemeval_2026-05.md)
  for context).

## Customising the demo script

`demo_script.txt` is a plain text file — one input line per row,
lines starting with `#` are comments. Edit it to script your own
walkthrough; the agent will execute every non-comment line in order
just as if you'd typed it.
