# Continuum v2.0 — Public Release Plan

**Status:** draft · **Owner:** Mayank · **Created:** 2026-07-05
**Goal:** first public release of Continuum ("v2.0 everything") — library on PyPI,
repo public, website live, benchmarks reproducible by strangers.

---

## 0. Release thesis

Continuum debuts publicly at **v2.0.0**: production-grade memory infrastructure
for AI agents, with the three claims nobody else in the category backs with
reproducible numbers:

1. **Supersession correctness 100 %** (vs 38 % baseline) — deterministic, reproducible
2. **Bi-temporal "as of date Y" queries 100 %** (vs 75 % baseline) — deterministic, reproducible
3. **LongMemEval-S ~74 % judged** (full 500, `gpt-oss-120b`; 73.6–75.6 % across
   repeated same-setup runs — a single run is not a stable estimate. The 76.4 %
   in earlier notes was a favorable draw; see [limitations.md](limitations.md).)

Everything in this plan serves one test: *a stranger with a laptop can verify
the claims in under an hour.*

**Scope freeze:** v2.0 ships what is already benchmarked. Typed memory
(migration 005 / kind registry / `code_symbol`), procedural memory, and the
agent harness are **post-launch roadmap** (§7), advertised but not shipped.

---

## 1. WS-A · Release engineering (blockers)

- [ ] **One version story.** Reconcile `pyproject.toml` (`0.3.0`), CHANGELOG
      (`1.0.0`), README ("v2.0"), and branch name (`release-3.0`) to a single
      public **2.0.0**. Internal v1/v2/v3 history becomes "development
      milestones" in the changelog, not public versions.
- [ ] Merge `release-3.0` → `main`; prune dead branches (`release`,
      `release-1.1`, `release-2.0`, feature branches) or mark archived.
- [ ] **API surface audit**: decide what `continuum/__init__.py` exports; that
      is the public API contract under SemVer. Everything else documented as
      internal.
- [ ] **CI (GitHub Actions)**: pytest + ruff + mypy on py3.11/3.12; `make
      bench-all` as a smoke job (no API keys, ~60 s); docker-compose up +
      migration smoke test.
- [ ] **PyPI packaging**: `continuum-memory` name check/reserve, wheel build,
      `pip install continuum-memory` → `make demo-chat` path works from a
      clean venv. TestPyPI dry run first.
- [ ] **Repo hygiene**: gitignore `website/node_modules` + `website/dist`;
      resolve untracked `samples/control_failed_ids.json` (commit or ignore);
      commit or revert the `continuum.yaml` working-tree change.
- [ ] **Secret scan** over full git history (keys, DSNs, personal data) before
      the repo flips public — history is forever. `gitleaks` or
      `trufflehog` over all branches.
- [ ] License headers / NOTICE consistent with `LICENSE`.

## 2. WS-B · Quality gates (the credibility layer)

- [ ] Full test suite green on a clean clone (`pytest`, fresh DB via
      docker-compose).
- [ ] **Fresh benchmark run** of everything the README claims, on the exact
      commit that becomes the release tag. Pin and publish: model ids, judge
      model, dataset hashes, seed, date. Numbers in README must match this run.
- [ ] `make repro-everything` verified start-to-finish by a clean-machine run
      (new user simulation — ideally a friend, or a fresh VM).
- [ ] Regression guard wired into CI for the synthetic benches (they're free);
      LongMemEval stays manual (costs credits) with a documented procedure.
- [ ] Known-limitations section written honestly: retrieval recall@4 tied with
      baseline at 55 % on the synthetic corpus, reader jitter on inference
      questions, ingest cost (6 LLM extraction calls/session). Publishing
      weaknesses is part of the brand.

## 3. WS-C · Docs & developer experience

- [ ] **Quickstart**: 5-minute path — install → demo-chat → first real
      integration. No Postgres required for the first taste (in-memory store).
- [ ] `docs/architecture.md` refreshed to match v2.0 reality (tiers,
      supersession, bi-temporal, over-fetch + rerank).
- [ ] Config reference (`docs/config.md`) synced with `continuum.yaml`.
- [ ] `examples/` — at minimum: bare-Python agent, LangGraph node, "as-of
      query" walkthrough, supersession walkthrough.
- [ ] Public CHANGELOG rewrite for 2.0.0 (development history condensed).
- [ ] CONTRIBUTING.md, issue/PR templates, CODE_OF_CONDUCT.

## 4. WS-D · Integrations (adoption path)

- [ ] **LangGraph adapter at launch** — a `ContinuumMemory` that drops into a
      LangGraph graph; this is the #1 discovery channel.
- [ ] AutoGen adapter as fast-follow (v2.0.x), listed on roadmap at launch.
- [ ] `examples/langgraph_agent.py` doubling as the adapter's test.

## 5. WS-E · Website & launch content

- [ ] Finish `website/` — landing page: thesis, the three numbers, quickstart,
      benchmarks page with **rerun-it-yourself instructions** front and center.
- [ ] **Technical report** (the paper draft): supersession semantics,
      bi-temporal model, LongMemEval trajectory with ablations. Ship as
      `docs/technical-report.pdf` + arXiv submission after launch.
- [ ] Launch post (blog/README-long-form): "Memory for agents is a state
      problem, not a model problem." The 34 % → 76 % story is the narrative.
- [ ] Show HN + relevant communities (r/LocalLLaMA, LangChain discord, AI eng
      slack), spaced over launch week.

## 6. Launch sequence

| step | gate |
|---|---|
| 1. Scope freeze, branch merged to `main` | WS-A blockers done |
| 2. CI green + fresh benchmark run pinned | WS-B done |
| 3. Docs/examples complete, TestPyPI dry run | WS-C done |
| 4. Clean-machine repro by an outsider | pass = go |
| 5. Tag `v2.0.0`, PyPI publish, repo public, website live | same day |
| 6. Launch post + Show HN | day after repo flips public |
| 7. **Launch week = support week** | answer every issue < 24 h |

Suggested pacing (solo, part-time): WS-A+B ≈ 2 weeks, WS-C+D ≈ 2 weeks,
WS-E ≈ 1–2 weeks overlapping. **Target: ~5–6 weeks to public.**

## 7. Post-launch roadmap (advertised at launch, shipped after)

- **v2.1 — Typed memory**: migration 005 (structured JSONB payload +
  multi-space embeddings table), kind registry, `code_symbol` end-to-end with
  tree-sitter ingestion; new benches: code-recall + deterministic code
  supersession.
- **v2.2 — Procedural memory**: strategies/outcomes as a memory kind;
  supersession as belief revision.
- **v2.3 — The improvement benchmark**: agent repeats a job class over
  simulated weeks; measure the learning curve; memory is the only variable.
  (This is the flagship demo: *the agent that gets better because it
  remembers.*)
- **Harness**: scenario-based eval runner (LongMemEval/LoCoMo/supersession as
  scenarios), grown from the internal eval pipeline; standalone repo when it
  earns it.

## 8. Definition of done (v2.0.0)

A stranger can: `pip install continuum-memory` → run the 60-second demo →
follow the quickstart into a real agent → rerun `make bench-all` and see the
synthetic numbers (supersession / bi-temporal 100 %) → (with API credits) rerun
LongMemEval and land at **~74 %** within `gpt-oss-120b` judge/run noise → read
the technical report and understand why. Every claim in the README traces to a
pinned, reproducible run.

## 9. Risks

- **Benchmark drift**: judged numbers move with judge-model updates → pin
  judge model + publish raw outputs alongside scores.
- **Launch-week fragility**: first impressions are permanent for OSS; the
  clean-machine repro (step 4) is the non-negotiable gate.
- **Scope creep**: typed memory is tempting to squeeze in — don't. It ships
  as v2.1 with its own benches, and gives the project a heartbeat after
  launch.
- **Solo bus-factor**: keep launch surface small enough that one person can
  support it (this is why the harness/DAG framework are explicitly out).
