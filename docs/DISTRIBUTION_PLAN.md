# Continuum — Distribution & Zero-Setup Plan

**Status:** proposed · **Owner:** Mayank · **Context:** installing the Continuum
MCP server today is a five-step path — `pip install`, run Postgres, migrate,
download a 2.3 GB embedder, register with the client. That friction is the
biggest adoption barrier now that the engine is production-grade. This plan is
how to get to "one command, no setup," ordered by payoff, not by ambition.

---

## The core tension (read this first)

Continuum's value rests on two heavy dependencies, and every zero-setup option is
really a decision about what to do with them:

1. **Postgres + pgvector** — the durable, hybrid, bi-temporal store validated
   this cycle (recall curves, HNSW tuning, the migrations). It cannot be trivially
   embedded in a single binary.
2. **The bge-m3 embedder (~2.3 GB)** — required for dense recall; it bloats any
   bundled artifact and is platform-specific.

There is no free lunch. A true single `.exe` must either **bundle both** (a
multi-GB, per-platform binary) or **replace both** (a weaker embedded store and a
smaller/no embedder). So the plan ships the cheap, high-value wins first and
treats the literal single-file install as a real, separately-scheduled project.

**Guiding principle:** whatever ships as "zero setup" becomes the path people
judge Continuum by, so it should use the *good* backend (pgvector) wherever
possible. That is why the Docker route leads, not the embedded-DB route.

---

## Tier 1 — Docker one-liner (highest payoff, least new code)

**Goal:** anyone with Docker gets the full stack — Postgres+pgvector, migrated,
plus the MCP HTTP server — in one command.

```
docker run -p 8000:8000 ghcr.io/genkryptos/continuum-mcp
claude mcp add continuum --transport http http://127.0.0.1:8000/mcp
```

**Why it is cheap:** the pieces already exist — `continuum-mcp --http`
(Streamable-HTTP transport), `continuum.db.migrate`, and the fixed wheel. This is
a Dockerfile + entrypoint that *migrates then serves*, not new product code.

**Scope:**
- A `Dockerfile` building on `pgvector/pgvector:pg16`, installing the wheel with
  `[mcp,postgres]`, plus an entrypoint that starts Postgres, waits for ready,
  runs `python -m continuum.db.migrate`, then execs `continuum-mcp --http`.
  (Single-container-with-embedded-Postgres for the demo path; a `compose.yml`
  with a separate DB service for anything durable/production.)
- **Default to sparse** (`CONTINUUM_MCP_EMBEDDINGS=0`) so first run has no 2.3 GB
  download. Publish a second `:embed` tag that bakes the model in for dense
  recall.
- Publish to GHCR on tag via a release workflow; document the volume for durable
  data.

**Acceptance:** on a machine with only Docker, `docker run …` → `claude mcp add`
→ a `remember`/`recall` round trip works, with **no** Python, manual Postgres, or
migrate step. A named volume survives a container restart.

**Effort:** S–M. **Risk:** low — no product code changes; the embedded-Postgres
demo container needs a clean shutdown path so data is not lost on stop.

---

## Tier 2 — PyPI + `uvx` (the cheap Python-native complement)

**Goal:** remove the Python-environment friction for developers who prefer not to
run Docker.

```
uvx continuum-mcp            # ephemeral env, no manual venv
# or
pipx install continuum-memory[mcp]
```

**Why now:** the wheel is built and fixed (rank-bm25 + migrations both packaged,
CI-green). The only gap is publishing it. `uvx`/`pipx` then give a one-line run
with no venv management. Does **not** remove the Postgres/embedder friction — it
is the Python-path convenience, complementary to Tier 1.

**Scope:**
- Owner-only: `twine upload dist/*` with the PyPI token (I cannot handle
  credentials). Verify the uploaded wheel is byte-identical to the CI-green one.
- A release workflow that builds and (optionally) publishes on tag, so this is
  never a manual, drift-prone step again.
- One-line docs: `uvx continuum-mcp`, sparse by default.

**Acceptance:** `uvx continuum-mcp --help` works in a clean environment with no
prior install; a Postgres-backed round trip works when a DSN is provided.

**Effort:** S. **Blocks:** the version/tag must be coherent first (already done:
`v2.0.0` re-pointed at the fixed commit). **Risk:** low, but publishing is
irreversible per version — never reuse a version number.

---

## Tier 3 — Embedded backend → true single-file install (the real unlock)

**Goal:** the literal ask — a double-click `.exe` or a one-click install file,
**no external services**. This is blocked by Postgres, so the prerequisite is an
embedded store.

**Fix, in dependency order:**

1. **An embedded vector backend** — `sqlite-vec` or DuckDB+VSS behind the
   existing `LTMProtocol`, so `Memory` can run with no external database. This is
   the load-bearing piece and the real cost: it will **not** reproduce pgvector's
   exact hybrid/HNSW/bi-temporal SQL, so it is a second, slightly-weaker retrieval
   path to build and hold to the same test bar (the harnesses and needle sets
   from the hardening cycle apply directly).
2. **A one-click bundle**, once (1) exists:
   - **Claude Desktop `.dxt` / MCP bundle** — a single file the user installs with
     one click. This is the closest thing to the request and should be the first
     target after the embedded backend.
   - **PyInstaller/Nuitka single exe** — viable only after (1), and even then ship
     it **sparse-only**, because the embedder bloats the binary to multiple GB per
     platform.

**Acceptance:** a non-technical user installs one file, with no Docker, Python, or
Postgres, and gets a working `remember`/`recall` in their client — with the
embedded backend measured against the same needle sets (document the recall gap
vs pgvector honestly).

**Effort:** L. **Risk:** high — a second retrieval backend is real surface area,
and "zero setup" that quietly ships weaker recall would undo the credibility the
hardening cycle bought. Only schedule this if the target audience genuinely will
not touch Docker.

---

## Suggested order

1. **Tier 1 (Docker)** — one command, full pgvector stack, almost no new code.
   Do this first; it covers most developer adoption.
2. **Tier 2 (PyPI + uvx)** — cheap, complementary, and unblocks the ecosystem
   (`pip install` finally works). Owner does the token upload.
3. **Tier 3 (embedded backend → .dxt/exe)** — only if targeting users who will
   not run Docker. It is the only path to a literal single file, and it costs a
   whole second backend to get there.

Tiers 1–2 are a few days of mostly-packaging work using code that already exists.
Tier 3 is a project. Doing 1 and 2 well may make 3 unnecessary for the audience
that matters.
