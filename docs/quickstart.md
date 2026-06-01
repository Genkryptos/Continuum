# Quickstart

Goal: a working `ContinuumSession` in **five commands**. No Postgres,
no API keys, no model downloads at first — those come in later
sections when you outgrow the in-memory defaults.

## 1 · Clone + install

```bash
git clone <repo> && cd Continuum
pip install -e .
```

`pip install -e .` puts Continuum on your import path in editable mode
and pulls in the runtime deps from `pyproject.toml`. Use a venv or
`uv` if you don't want to pollute the system Python.

## 2 · Verify it works

```bash
make demo-chat
```

Runs the scripted 60-second walkthrough end-to-end. If you see the
supersession output (`→ extracted user.location = Boston (superseded
prior)`), the framework is wired correctly and you can move on.

## 3 · Your first session — in 10 lines

Create `hello.py`:

```python
import asyncio
from continuum.core.config import ContinuumConfig
from continuum.core.session import ContinuumSession

async def main():
    config = ContinuumConfig.load()           # defaults are sane
    async with ContinuumSession(config) as session:
        reply = await session.process_turn("Hi! I just moved to Boston.")
        print("agent>", reply)
        reply = await session.process_turn("Where do I live?")
        print("agent>", reply)

asyncio.run(main())
```

```bash
python hello.py
```

Out of the box you get:

* **STM** = `InMemorySTM` (default, no infra). Holds the conversation.
* **Responder** = a stub that echoes context. To use a real LLM,
  pass `responder=` to `ContinuumSession(...)` — see step 5.
* **No MTM / LTM / retriever** = `process_turn` doesn't try to recall
  long-term facts yet. That's the next step.

## 4 · Add retrieval (still no infra)

```python
from continuum.embeddings.service import EmbeddingService
from continuum.retrieval.retriever import Retriever

# Build a retriever from an in-memory embedder.
embeddings = EmbeddingService(config.embedding)
retriever = Retriever(stm=session.stm, embeddings=embeddings,
                      config=config.retriever)

async with ContinuumSession(config, retriever=retriever) as session:
    await session.process_turn("My dog's name is Rex.")
    await session.process_turn("I love hiking on weekends.")
    reply = await session.process_turn("What's my dog called?")
    print(reply)   # retrieval surfaces the Rex turn from STM
```

This still uses zero infrastructure — the embedder is
`sentence-transformers/all-MiniLM-L6-v2` running on CPU. First call
downloads the model (~90 MB) and caches it under `~/.cache/`.

## 5 · Plug in your LLM

Inject a `responder` so `process_turn` uses your model:

```python
async def my_responder(user_msg: str, ctx) -> str:
    # ctx is a continuum.core.types.ContextBundle; flatten as you see fit.
    context_text = "\n".join(it.content for it in ctx.items) if ctx else ""
    # ...call your provider (OpenAI / Anthropic / local) with prompt + context...
    return "the model's reply"

async with ContinuumSession(config, retriever=retriever,
                            responder=my_responder) as session:
    ...
```

Any async callable matching `(user_msg, ctx) -> str` works. See
[`examples/chat_agent/agent.py`](../examples/chat_agent/agent.py) for
both a canned stub responder and an OpenAI one.

## Five commands — verified

```bash
# 1. clone + install
git clone <repo> && cd Continuum && pip install -e .

# 2. verify (60-second demo)
make demo-chat

# 3. write hello.py (the 10-line snippet above)

# 4. run it
python hello.py

# 5. when ready for production: add Postgres
#    (see docs/operations.md — Postgres setup)
```

If you got through 1-4 in under 10 minutes you're at the bar
[Prompt 53 set](../README.md). Step 5 is *optional* — Continuum runs
fine in-memory for prototyping and demos. The Postgres path unlocks
production scale (millions of facts) and is documented separately in
[Operations](operations.md).

## What you can NOT do yet from this quickstart

* **Persist memory across processes.** The default `InMemorySTM` loses
  state on exit. For that you need either `AsyncSafeSTM` (in-process
  thread safety) or `PostgresSTM`. See [Operations](operations.md).
* **Trigger LTM promotion.** Without a promoter wired in,
  `process_turn` keeps everything in STM. Add a `Promoter` (see
  [Architecture](architecture.md) §promotion-lifecycle).
* **Run the benchmarks.** The bench targets need the framework
  Python with sentence-transformers; if you used `pip install -e .`
  in a venv that path is already configured.

## Common first-time gotchas

| You see | Fix |
|---|---|
| `ModuleNotFoundError: continuum` | `pip install -e .` from the repo root. |
| `RuntimeError: no current event loop` | You're calling an async function from sync code — wrap in `asyncio.run(...)`. |
| Embedder hangs at "loading model" | First call downloads ~90MB to `~/.cache/huggingface/`. Disconnected? Set `TRANSFORMERS_OFFLINE=1` after the first successful run. |
| `make demo-chat` says `python3: not found` | Mac-specific: install Python 3.12 framework or override with `BENCH_PYTHON=$(which python3) make demo-chat`. |

## Small LLM helper

Some Continuum modules (span selection, claim verification, intent
classification) delegate to a tiny local instruct model through
`continuum.extraction.small_llm.SmallLLM`. The default model is
`qwen2.5:1.5b-instruct` served by Ollama.

```bash
# 1. install Ollama (mac):
brew install ollama
ollama serve &

# 2. pull the default small model (~1 GB):
ollama pull qwen2.5:1.5b-instruct
```

Override the model or endpoint with env vars (constructor args take
precedence):

```bash
export CONTINUUM_SMALL_LLM_MODEL=qwen2.5:1.5b-instruct
export CONTINUUM_SMALL_LLM_URL=http://localhost:11434
```

Responses are cached on disk at `~/.cache/continuum/small_llm.db` keyed
on `(method, model, cache_key)`, so repeated passes over the same input
don't re-hit the model.

## Where to next

* You want to understand what just happened → [Architecture](architecture.md).
* You want to tune what was just used → [Config reference](config.md).
* You want to put it in production → [Operations](operations.md).
