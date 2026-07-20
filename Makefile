# =============================================================================
# Makefile — Continuum development tasks
# =============================================================================
# Usage:
#   make test          Run the full test suite (unit + integration)
#   make test-fast     Unit tests only — no DB, runs in seconds
#   make test-cov      Full suite with HTML + terminal coverage report
#   make lint          ruff check (lint only)
#   make typecheck     mypy --strict (100% type coverage gate)
#   make format        ruff format (auto-format)
#   make check         CI pre-flight: format check + lint + typecheck
#   make help          Show all targets
#
# Requires: Python ≥ 3.11, pip packages from `make install-dev`

.DEFAULT_GOAL := help
.PHONY: test test-fast test-integration test-e2e test-cov benchmark \
        lint typecheck format check \
        install install-dev clean help \
        db-up db-down db-logs db-reset db-clear db-migrate db-migrate-dry check-env check-env-ping run run-full run-mem \
        mcp-install mcp-smoke mcp-eval mcp-bench scale-test mcp-serve mcp-serve-http mcp-claude \
        repro-longmemeval repro-everything bench-ingest bench-retrieval bench-supersession \
        bench-bitemporal bench-locomo bench-all bench-gate demo-chat build build-verify

# ── Toolchain ─────────────────────────────────────────────────────────────────

PYTHON  := python3
PYTEST  := $(PYTHON) -m pytest
RUFF    := $(PYTHON) -m ruff
MYPY    := $(PYTHON) -m mypy

# Targets that import Continuum's framework modules (which transitively
# need PyYAML, httpx, sentence-transformers, etc.) need an interpreter
# with the dev deps installed. We prefer the Python.org framework build
# when available because that's where the runtime deps were pinned
# during the LongMemEval sweep — overrideable via BENCH_PYTHON=path.
BENCH_PYTHON ?= $(shell test -x /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
                && echo /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
                || echo python3)

SRC     := continuum
TESTS   := tests

# MCP HTTP server bind address (override: make mcp-serve-http MCP_PORT=9000)
MCP_HOST ?= 127.0.0.1
MCP_PORT ?= 8000

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run the full test suite (unit + integration)
	$(PYTEST) $(TESTS) -v

test-fast: ## Run only unit tests — no external dependencies required
	$(PYTEST) $(TESTS)/unit -v -m unit

test-integration: ## Run only integration tests — PostgreSQL must be running
	$(PYTEST) $(TESTS)/integration -v -m integration

test-e2e: ## End-to-end: drive the real continuum-mcp binary against Postgres (needs a DB + the [mcp] extra)
	$(BENCH_PYTHON) -m pytest $(TESTS)/e2e -v -m e2e

test-cov: ## Run all tests with HTML + terminal coverage; fail below 80 %
	$(PYTEST) $(TESTS) \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml \
		--cov-fail-under=80 \
		-v
	@echo ""
	@echo "HTML report → htmlcov/index.html"

benchmark: ## Run performance benchmarks (pytest-benchmark)
	$(PYTEST) $(TESTS)/benchmarks -v --benchmark-only --benchmark-sort=mean

# ── Code quality ──────────────────────────────────────────────────────────────

lint: ## Run ruff lint checks on source + tests
	$(RUFF) check $(SRC) $(TESTS)

typecheck: ## Strict mypy type-check (100% type coverage gate)
	$(MYPY) --strict $(SRC)

format: ## Auto-format source + tests with ruff
	$(RUFF) format $(SRC) $(TESTS)
	$(RUFF) check --fix $(SRC) $(TESTS)

check: ## CI pre-flight: format check + lint + strict type-check (no writes)
	$(RUFF) format --check $(SRC) $(TESTS)
	$(RUFF) check $(SRC) $(TESTS)
	$(MYPY) --strict $(SRC)

# ── Setup ─────────────────────────────────────────────────────────────────────

install: ## Install runtime dependencies from requirements.txt
	pip install -r requirements.txt

install-dev: ## Install runtime + full developer toolchain
	pip install -r requirements.txt
	pip install \
		pytest pytest-asyncio pytest-cov pytest-benchmark \
		ruff mypy

# ── Local infra + environment check ───────────────────────────────────────────

COMPOSE := $(shell command -v docker-compose >/dev/null 2>&1 && echo docker-compose || echo "docker compose")

db-up: ## Start local Postgres (pgvector) in the background and wait until ready
	@$(COMPOSE) up -d postgres
	@echo "waiting for Postgres to accept connections…"
	@$(COMPOSE) exec -T postgres sh -c \
		'until pg_isready -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-continuum} >/dev/null 2>&1; do sleep 1; done' \
		|| true
	@echo "Postgres is up (host port = POSTGRES_PORT, default 5432; see docker ps)"
	@echo "next:  make db-migrate   (create the pgvector ext + LTM schema)"
	@echo "       make check-env    (verify your config can reach it)"

db-down: ## Stop the local Postgres container (named data volume is kept)
	@$(COMPOSE) down

db-logs: ## Tail the local Postgres logs
	@$(COMPOSE) logs -f postgres

db-reset: ## Stop Postgres AND delete its data volume (destructive)
	@$(COMPOSE) down -v

db-clear: ## Delete ALL memory records (keeps schema + migration history); prompts unless ARGS=--yes
	@$(BENCH_PYTHON) -m continuum.db.clear $(ARGS)

db-migrate: ## Apply migrations/*.sql to the configured DB (creates pgvector ext + LTM schema)
	@$(BENCH_PYTHON) -m continuum.db.migrate

db-migrate-dry: ## Show which migrations would be applied, without applying them
	@$(BENCH_PYTHON) -m continuum.db.migrate --dry-run

run: ## Start the chat REPL (Postgres-backed; embedder off — no model download)
	@$(BENCH_PYTHON) -m continuum.chat --no-embeddings $(ARGS)

run-full: ## Like run, but WITH the embedder for dense recall (downloads bge-m3 ~2.3GB on first use)
	@$(BENCH_PYTHON) -m continuum.chat $(ARGS)

run-mem: ## Start the chat REPL with in-memory stores (no Postgres needed)
	@$(BENCH_PYTHON) -m continuum.chat --in-memory --no-embeddings $(ARGS)

# ── MCP server ────────────────────────────────────────────────────────────────
# The Continuum MCP server uses STDIO transport: an MCP client (e.g. Claude Code)
# launches its OWN copy per session and talks to it over stdin/stdout. You do NOT
# pre-start it for such a client. `mcp-serve` is for debugging or an HTTP client;
# `mcp-smoke` proves the server works with no client at all.

mcp-install: ## Install the package + MCP extra (editable) — provides the `continuum-mcp` script
	@$(BENCH_PYTHON) -m pip install -e ".[mcp]"

mcp-smoke: ## Prove the MCP server works end-to-end (handshake + remember→recall); no Claude needed
	@$(BENCH_PYTHON) scripts/mcp_smoke.py

mcp-eval: ## Score MCP retrieval quality (recall@1/@3, supersession, timeline) over a fixed scenario. Set CONTINUUM_DB_DSN to eval the Postgres+embedder stack
	@$(BENCH_PYTHON) scripts/mcp_eval.py $(ARGS)

mcp-bench: ## Measure MCP tool latency (p50/p95 + embedder-vs-DB breakdown). Set MCP_BENCH_DSN for a THROWAWAY Postgres; never touches your live store
	@$(BENCH_PYTHON) scripts/mcp_bench.py $(if $(MCP_BENCH_DSN),--dsn $(MCP_BENCH_DSN)) $(ARGS)

scale-test: ## Load 1k/10k/50k rows and measure recall latency + index usage. Requires SCALE_DSN (a THROWAWAY Postgres)
	@$(BENCH_PYTHON) scripts/scale_test.py $(if $(SCALE_DSN),--dsn $(SCALE_DSN)) $(ARGS)

mcp-serve: ## Run the MCP server over stdio in the foreground (debug/manual — a stdio client spawns its own copy)
	@$(BENCH_PYTHON) -m continuum.mcp.server

mcp-serve-http: ## Run a standalone always-on HTTP MCP server (default http://127.0.0.1:8000/mcp) that Claude connects to by URL. Override MCP_HOST / MCP_PORT
	@bin="$$(dirname $(BENCH_PYTHON))/continuum-mcp"; \
	 test -x "$$bin" || { echo "ERROR: $$bin not found — run 'make mcp-install' first"; exit 1; }; \
	 echo "Serving Continuum MCP at http://$(MCP_HOST):$(MCP_PORT)/mcp  (Ctrl-C to stop)"; \
	 echo "Register with:  claude mcp add continuum --transport http http://$(MCP_HOST):$(MCP_PORT)/mcp"; \
	 "$$bin" --http --host $(MCP_HOST) --port $(MCP_PORT)

mcp-claude: ## Register the server with Claude Code (local scope). Use ARGS='--scope user' for all projects
	@bin="$$(dirname $(BENCH_PYTHON))/continuum-mcp"; \
	 test -x "$$bin" || { echo "ERROR: $$bin not found — run 'make mcp-install' first"; exit 1; }; \
	 claude mcp add continuum $(ARGS) -- "$$bin"

check-env: ## Verify .env: config loads, provider key, DB reachable, in-memory smoke
	@$(BENCH_PYTHON) -m continuum.doctor

check-env-ping: ## Like check-env, but also validates each LLM provider key via a live API call
	@$(BENCH_PYTHON) -m continuum.doctor --ping

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean: ## Remove caches, coverage artefacts, and build output
	find . -type d -name __pycache__  -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache  -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache  -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov      -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml

# ── Reproducibility — LongMemEval findings (Prompt 42) ───────────────────────

repro-longmemeval: ## Reproduce the LongMemEval-S headline numbers from findings/longmemeval_2026-05.md
	@bash findings/longmemeval/repro/run_repro.sh

repro-everything: ## Reproduce the v2.0 headline (~74% judged, 73.6–75.6% across runs): full LongMemEval-S (direct + over-fetch + rerank) + judged rescore + LOCOMO smoke (needs OPENROUTER_API_KEY; ~<2h, <$5)
	@test -n "$$OPENROUTER_API_KEY" || ( echo "ERROR: export OPENROUTER_API_KEY first" && exit 1 )
	@echo "==> [1/3] LongMemEval-S v2.0 full run (direct + over-fetch + cross-encoder rerank)…"
	@$(BENCH_PYTHON) -m evals.longmemeval.bootstrap_ollama \
		--provider openrouter --model openai/gpt-oss-120b \
		--openrouter-provider DeepInfra --seed 0 \
		--reasoner direct --use-ltm --ltm-backend in_memory --no-llm-promoter --retriever hybrid \
		--session-aware-retrieval --session-top-k 12 --turns-per-session 6 \
		--top-k 80 --max-context-chars 64000 --answer-max-tokens 2048 --span-fallback \
		--decompose-max-items 60 --rerank --rerank-to 24 \
		--full --yes --no-smoke --output results/v2_final
	@echo "==> [2/3] judged rescore (non-reasoning judge)…"
	@$(BENCH_PYTHON) -m evals.longmemeval.rescore_with_judge \
		--input $$(ls -t results/v2_final/baseline_*.json | head -1) \
		--output results/v2_final/judged.json \
		--provider openrouter --judge-model meta-llama/llama-3.3-70b-instruct
	@$(BENCH_PYTHON) findings/charts/v1_summary.py results/v2_final/judged.json
	@echo "==> [3/3] LOCOMO smoke (continuum side, 50 q)…"
	@test -f evals/locomo/data/locomo10.json || ( mkdir -p evals/locomo/data && \
		curl -L -o evals/locomo/data/locomo10.json \
		  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json )
	@$(BENCH_PYTHON) -m evals.locomo.run --system continuum \
		--provider openrouter --model openai/gpt-oss-120b \
		--judge-provider openrouter --judge-model meta-llama/llama-3.3-70b-instruct \
		--rpm 30 --judge-rpm 15 --limit 50 --output results/locomo_smoke

# ── Memory-operation benchmarks (Phase 3B, Prompt 43+) ───────────────────────

bench-ingest: ## Run the ingest-throughput benchmark (Continuum vs raw vs mem0)
	@$(BENCH_PYTHON) -m bench.ingest_throughput --sessions 50 --turns 6

bench-retrieval: ## Run the retrieval-quality benchmark (recall@k vs naive cosine)
	@$(BENCH_PYTHON) -m bench.retrieval_quality --sessions 200 --queries 30

bench-supersession: ## Run the supersession-correctness benchmark (Continuum's killer feature)
	@$(BENCH_PYTHON) -m bench.supersession_correctness --scenarios 50

bench-bitemporal: ## Run the bi-temporal "as of date Y" benchmark (Continuum-only feature)
	@$(BENCH_PYTHON) -m bench.bi_temporal --scenarios 20

bench-locomo: ## Run the LOCOMO benchmark — Continuum vs Mem0 head-to-head (needs OPENROUTER_API_KEY + GROQ_API_KEY; downloads dataset if absent)
	@test -f evals/locomo/data/locomo10.json || ( \
		echo "downloading LOCOMO dataset…" && mkdir -p evals/locomo/data && \
		curl -L -o evals/locomo/data/locomo10.json \
		  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json )
	@$(BENCH_PYTHON) -m evals.locomo.run \
		--system both --provider openrouter --model openai/gpt-oss-120b \
		--judge-provider groq --judge-model llama-3.3-70b-versatile \
		--output results/locomo_v1

bench-all: bench-ingest bench-retrieval bench-supersession bench-bitemporal ## Run every Phase-3B memory-operation benchmark in sequence

bench-gate: ## Verify the latest bench-all run against the README's contract thresholds
	@$(BENCH_PYTHON) scripts/check_bench_regressions.py

# ── Demos (Phase 3C, Prompt 47+) ─────────────────────────────────────────────

demo-chat: ## Replay the scripted Continuum chat-agent demo (no infra, <1s)
	@bash examples/chat_agent/demo.sh

# ── Packaging (Phase 3E, Prompt 54) ──────────────────────────────────────────

build: ## Build wheel + sdist into dist/ — hatchling
	@$(BENCH_PYTHON) -m build --wheel --sdist
	@echo
	@ls -lh dist/

build-verify: build ## Build, then verify a clean install in a fresh venv
	@TMP_VENV="$$(mktemp -d)/continuum_pip_verify"; \
	echo "=== clean-venv install into $$TMP_VENV ==="; \
	$(BENCH_PYTHON) -m venv "$$TMP_VENV"; \
	"$$TMP_VENV/bin/pip" install --quiet dist/continuum_memory-*.whl; \
	echo "=== smoke import + minimal session ==="; \
	"$$TMP_VENV/bin/python" scripts/verify_clean_install.py; \
	rm -rf "$$TMP_VENV"

# ── Help ──────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo "Continuum — available make targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
