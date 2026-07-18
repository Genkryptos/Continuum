# Continuum productionization — execution prompts

A set of self-contained, handoff-ready prompts to take Continuum from research
codebase to shipped v2.0. Each prompt assumes a **cold start** — prepend
**Prompt 0 (shared context)** to every one. Execute in order M1 → M5; M4's
secret scan can run in parallel and is a hard release gate.

---

## Prompt 0 — Shared context (prepend to EVERY prompt below)

```
You are working on Continuum, an open-source AI-agent memory library.
Repo: /Users/mayanksahu/Continuum  ·  branch: release-3.0 (pushed)  ·  Python 3.12
Package name (PyPI): continuum-memory.

WHAT IT IS
Persistent memory for AI agents: ingest conversation turns → extract facts →
bi-temporal SUPERSESSION (handle updates/retractions, keep history queryable
"as of date X") → hybrid retrieval. Bi-temporal supersession is the
differentiator (Mem0 ~49% / Zep ~63.8% / Continuum ~75.6% on LongMemEval-S).

HONEST BENCHMARK STATE (do not inflate)
- Best config: reflect (preference+KU) + vote-3 = ~75.6% LongMemEval-S judged
  (llama-3.3-70b judge), gpt-oss-120b reader. Same-setup baseline is 73.8%.
- Preference +20pp is the clean per-category win. ~5k tokens/query.
- RESEARCH LEVERS THAT ARE NET-NEGATIVE AND MUST NOT BE IN THE PRODUCT API:
  synthesis/router (counting), distill, temporal codemath. They stay opt-in
  flags in evals/ only. Only ship: the memory layer + supersession + hybrid
  retrieval + reflect(preference/KU) + vote-n.

HARD CONSTRAINTS
- All work must keep: `pytest tests/unit` green (currently 1393), `mypy` strict
  clean on continuum/, `ruff check` clean on files you touch.
- Never commit or modify: .env, continuum.yaml (local, gitignored). Never print
  secret values.
- Commit style: end messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Branch off main/
  release-3.0; don't push unless asked.
- Prefer editing existing structure over new files; match surrounding style.

KEY FILES
- continuum/core/session.py  — ContinuumSession (process_turn, search,
  process_turn_sync, search_sync): the internal orchestrator to wrap.
- continuum/__init__.py — currently exports only __version__.
- continuum/{promotion,retrieval,extraction,stores,core} — subsystems.
- docs/RELEASE_PLAN.md — the v2.0 release plan (9 workstreams). Align to it.
- findings/roadmap_v3.md — the honest research log (numbers + disproven ideas).
```

---

## Prompt 1 (M1) — Public `Memory` facade

```
GOAL: turn Continuum from "a codebase" into "an importable library" by building
a clean, documented public API. This resolves RELEASE_PLAN WS-A "API surface
audit".

TASK
1. Read continuum/core/session.py fully to learn ContinuumSession's real
   capabilities and constructor (config, stores, embedder, in-memory vs Postgres).
2. Create continuum/memory.py with a `Memory` facade that wraps ContinuumSession:
     Memory(config=None, *, in_memory=False, embeddings=True) -> Memory
     async add(text, *, session_id="default", occurred_at=None) -> None
     async recall(query, *, k=8) -> list[MemoryItem]
     async current(subject, attribute) -> str | None      # supersession-resolved
     async timeline(entity, *, since=None, until=None) -> list[...]  # bi-temporal
     async remember(fact) -> None     # alias/explicit write
   Plus sync variants (add_sync/recall_sync/...) delegating to the existing
   *_sync methods where present.
3. Map each method to the existing ContinuumSession/stores machinery — DO NOT
   reimplement retrieval/supersession; wrap them. If `current`/`timeline` have
   no clean backing call, implement thin adapters over the LTM store's
   supersession + bi-temporal query (find the existing methods first).
4. Export from continuum/__init__.py: `Memory`, `__version__`, and the core
   public types (MemoryItem, MemoryTier, and the config type). Set __all__.
5. Docstrings on every public method with a one-line example.

CONSTRAINTS
- Do NOT surface any research lever (synthesis/router/distill/codemath) on Memory.
- Config must degrade gracefully: `Memory(in_memory=True)` works with zero setup
  (no Postgres, no model download) for the quickstart.

TESTS (hermetic — fakes, no DB, no network, no LLM)
- tests/unit/test_memory_facade.py: add→recall roundtrip, current() returns the
  resolved (not stale) value after an update, timeline() ordering, sync wrappers,
  in_memory construction. Use fakes for stores/embedder like the eval tests do.

DEFINITION OF DONE
- `from continuum import Memory` works; the 5-line quickstart in Prompt 3 runs.
- New tests green; full `pytest tests/unit` green; mypy strict + ruff clean.
- Commit: "feat(api): public Memory facade (add/recall/current/timeline)".
```

---

## Prompt 2 (M2) — MCP server

```
GOAL: expose the Memory facade as an MCP server so any MCP agent (Claude Code,
Cursor, etc.) plugs Continuum in with zero glue. Depends on M1.

TASK
1. Add the MCP Python SDK dependency (`mcp`) to pyproject under an optional
   extra: `continuum-memory[mcp]`.
2. Create continuum/mcp/server.py — a stdio MCP server exposing tools backed by
   `continuum.Memory`:
     - recall(query, k=8) -> list of memories
     - remember(text, occurred_at?) -> ack
     - current(subject, attribute) -> value
     - timeline(entity, since?, until?) -> ordered events
   Typed input schemas; each tool calls the Memory facade; errors returned as
   tool errors, not crashes.
3. Add a console entry point `continuum-mcp = "continuum.mcp.server:main"` in
   pyproject [project.scripts].
4. continuum/mcp/__init__.py exporting the server factory.

CONSTRAINTS
- No research levers exposed. Backing store is configurable (in-memory default
  for zero-config; Postgres via env/config).
- The server must start with `continuum-mcp` and respond to an MCP `list_tools`.

TESTS (hermetic)
- tests/unit/test_mcp_server.py: tool schemas are well-formed; each handler
  calls the (faked) Memory facade and returns the expected shape; a malformed
  arg yields a tool error, not an exception.

DOCS
- docs/mcp.md: the ~10-line snippet to add Continuum to a Claude/Cursor MCP
  config (command: continuum-mcp), and the tool list.

DEFINITION OF DONE
- `continuum-mcp` starts; list_tools returns the 4 tools; tests green; mypy+ruff
  clean. Commit: "feat(mcp): Continuum MCP server (recall/remember/current/timeline)".
```

---

## Prompt 3 (M3) — Docs, quickstart & honest proof

```
GOAL: make Continuum credible and adoptable — a 5-minute quickstart on the new
API and an HONEST proof/limitations section. Feeds RELEASE_PLAN WS-B + WS-C.
Depends on M1 (uses the Memory facade).

TASK
1. README.md — rewrite the top:
   - One-paragraph thesis (memory for agents; bi-temporal supersession is the
     wedge).
   - The three numbers, honestly: LongMemEval-S ~75.6% (architecture-native,
     gpt-oss-120b, reflect+vote-3), ~5k tokens/query, supersession benchmark.
   - 5-minute QUICKSTART using the Memory facade:
       from continuum import Memory
       mem = Memory(in_memory=True)
       await mem.add("I moved to Boston"); await mem.add("Actually I'm in NYC now")
       await mem.current("user","residence")   # -> "NYC"
2. docs/limitations.md — the HONEST known-limitations (this is credibility, not
   weakness). Pull from findings/roadmap_v3.md §9:
   - counting/aggregation is reader-bound (~63% multi-session; synthesis, router,
     and distill all net-negative — deterministic machinery over gpt-oss's
     unreliable intermediate outputs fails).
   - temporal codemath net-negative (bad SPECs → confident wrong answers).
   - the failure-only-measurement methodology lesson (always carry a same-setup
     control).
   - retrieval recall@4 parity note (from RELEASE_PLAN WS-B).
3. examples/ — at minimum: examples/bare_agent.py (Memory + an LLM loop) and
   examples/langgraph_node.py (Memory as a LangGraph memory node). Each runs and
   doubles as a smoke test.
4. Refresh docs/architecture.md and docs/config.md to match v2.0 reality (tiers,
   supersession, retrieval, the Memory facade). Sync config.md with continuum.yaml
   keys.
5. CHANGELOG.md: condense development history into a 2.0.0 entry.

CONSTRAINTS
- Every claimed number must be reproducible via a documented command (cite
  results/ paths or make targets). No number without a source.
- Do not document research flags as product features.

DEFINITION OF DONE
- Quickstart copy-pastes and runs on a clean in_memory Memory. Examples run.
- No inflated claims; limitations page present and honest.
- Commit: "docs: v2.0 quickstart, examples, and honest limitations".
```

---

## Prompt 4 (M4) — Release engineering + security (HARD GATE)

```
GOAL: make the repo releasable. RELEASE_PLAN WS-A. The secret scan is a blocker.

TASK
1. SECURITY (do FIRST, blocks everything):
   - Run a secret scan over the FULL git history (gitleaks or `git log -p |
     grep -E 'sk-(or-|proj-)?[A-Za-z0-9]'`). Report any hits — do NOT print full
     keys. Tell the user which keys to rotate (OpenRouter + OpenAI keys have been
     exposed in local .env this session; confirm .env was never committed).
   - Ensure .gitignore covers: .env, continuum.yaml, website/node_modules,
     website/dist, .synthesis_cache/, results/.
2. Version story: reconcile pyproject.toml version, continuum/__init__.__version__,
   and CHANGELOG to a single 2.0.0 (RELEASE_PLAN notes pyproject drift).
3. CI: .github/workflows/ci.yml — pytest (unit) + ruff + mypy on py3.11 and
   3.12, plus `make check`. Cache deps. Must pass on a clean clone.
4. PyPI packaging: verify `python -m build` produces a valid wheel + sdist for
   continuum-memory; check the name is available/reserved; `twine check dist/*`.
5. Merge prep: open a PR release-3.0 → main summarizing v2.0 (do NOT auto-merge;
   leave for the owner). Prune dead branches list for the owner to confirm.
6. License/NOTICE consistency; add LICENSE headers if the repo convention has them.

CONSTRAINTS
- Never commit secrets or local config. If the secret scan finds a committed
  secret, STOP and report — do not attempt to rewrite history without owner sign-off.

DEFINITION OF DONE
- Secret scan clean (or findings reported with rotation list); CI green on clean
  clone; wheel builds + twine check passes; single 2.0.0 version; release PR open.
- Commit(s): "chore(release): CI, packaging, version reconcile, gitignore hygiene".
```

---

## Prompt 5 (M5) — Launch content

```
GOAL: launch assets. RELEASE_PLAN WS-E. Depends on M1–M4 landed.

TASK
1. website/ — finish the landing page (source lives where the built website/dist
   came from; if only dist exists, reconstruct a minimal static source or edit
   dist directly): thesis, the three honest numbers, quickstart, a benchmark
   table (Continuum vs Mem0/Zep from docs/positioning.md), install + MCP snippet.
   Keep it self-contained static HTML/CSS (no framework needed).
2. Technical report draft (docs/report.md or a paper): supersession semantics,
   the LongMemEval methodology INCLUDING the honest negative results (the three
   disproven deterministic levers + the failure-only-measurement lesson), the
   token-efficiency comparison, reproducibility commands.
3. Launch post (long-form README section or blog): "Memory for agents is a state
   problem, not a retrieval problem" — the supersession thesis, honestly framed.
4. A Show HN / community post draft (r/LocalLLaMA, LangChain discord, AI eng):
   title + 2-paragraph body + the repo link, honest tone.

CONSTRAINTS
- Reuse the exact numbers and sources from M3; no new/uncited claims.
- Static site only (the artifact pipeline blocks external CDN/scripts).

DEFINITION OF DONE
- Landing page renders and states only sourced numbers; report drafted with the
  honest methodology; launch + HN drafts ready for the owner to post.
```

---

## Execution notes
- **Order:** M1 → M2 → M3 → (M4 secret scan in parallel, as a gate) → M5.
- **Every milestone** ends with: full unit suite green, mypy strict clean on
  continuum/, ruff clean, one focused commit.
- **Owner-only actions** (do not automate): rotating keys, merging to main,
  publishing to PyPI, posting launch content.
- **Scope discipline (solo builder):** M1+M2+M3+M4 = shippable v2.0. The paper
  (M5.2) must not block the library release.
