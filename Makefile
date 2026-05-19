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
        install install-dev clean help

# ── Toolchain ─────────────────────────────────────────────────────────────────

PYTHON  := python3
PYTEST  := $(PYTHON) -m pytest
RUFF    := $(PYTHON) -m ruff
MYPY    := $(PYTHON) -m mypy

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

# ── Help ──────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo "Continuum — available make targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
