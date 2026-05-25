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
.PHONY: test test-fast test-integration test-cov benchmark \
        lint typecheck format check \
        install install-dev clean help \
        repro-longmemeval bench-ingest bench-retrieval bench-supersession \
        bench-bitemporal bench-all bench-gate demo-chat build build-verify

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

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run the full test suite (unit + integration)
	$(PYTEST) $(TESTS) -v

test-fast: ## Run only unit tests — no external dependencies required
	$(PYTEST) $(TESTS)/unit -v -m unit

test-integration: ## Run only integration tests — PostgreSQL must be running
	$(PYTEST) $(TESTS)/integration -v -m integration

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

# ── Memory-operation benchmarks (Phase 3B, Prompt 43+) ───────────────────────

bench-ingest: ## Run the ingest-throughput benchmark (Continuum vs raw vs mem0)
	@$(BENCH_PYTHON) -m bench.ingest_throughput --sessions 50 --turns 6

bench-retrieval: ## Run the retrieval-quality benchmark (recall@k vs naive cosine)
	@$(BENCH_PYTHON) -m bench.retrieval_quality --sessions 200 --queries 30

bench-supersession: ## Run the supersession-correctness benchmark (Continuum's killer feature)
	@$(BENCH_PYTHON) -m bench.supersession_correctness --scenarios 50

bench-bitemporal: ## Run the bi-temporal "as of date Y" benchmark (Continuum-only feature)
	@$(BENCH_PYTHON) -m bench.bi_temporal --scenarios 20

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
