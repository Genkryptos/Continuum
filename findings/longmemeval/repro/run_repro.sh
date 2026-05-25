#!/usr/bin/env bash
# findings/longmemeval/repro/run_repro.sh
# =======================================
# Reproduce the two headline LongMemEval-S runs from
# findings/longmemeval_2026-05.md, then verify each against the
# expected numbers in expected_results.json (±2pp tolerance).
#
# Total wall-clock: ~25-35 min (gpt-4o-mini × 2 runs × 500 questions).
# Total API cost: ~$0.10 (gpt-4o-mini at retrieval-budget).
# Requires:
#   * OPENAI_API_KEY in the environment (or OPEN_AI_KEY in .env, the
#     legacy name used in the prior runs).
#   * Python 3.12 framework interpreter at
#     /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
#     (override via PY3 env var).
#   * Dataset at evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json
#     (vendored in the repo).

set -euo pipefail

# ── Paths + interpreter ─────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPRO_DIR="$REPO_ROOT/findings/longmemeval/repro"
RESULTS_DIR="$REPO_ROOT/results/repro"
PY3="${PY3:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"

if [[ ! -x "$PY3" ]]; then
  PY3="$(command -v python3)"
  echo "WARNING: framework python3 not found; falling back to $PY3"
fi

# ── OpenAI key — accept either canonical name ───────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]]; then
    OPENAI_API_KEY="$(grep -E '^OPEN(_AI|AI)_(API_)?KEY=' "$REPO_ROOT/.env" \
                       | head -1 | cut -d= -f2- | tr -d '"')"
    export OPENAI_API_KEY
  fi
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set (and not found in .env)."
  echo "  export OPENAI_API_KEY='sk-…' and re-run."
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$RESULTS_DIR"

# ── Helper: run a single configuration + verify it ──────────────────
run_one() {
  local name="$1"; shift
  local out_dir="$RESULTS_DIR/$name"
  echo
  echo "============================================================"
  echo "  REPRO RUN — $name"
  echo "============================================================"
  rm -rf "$out_dir"
  "$PY3" -m evals.longmemeval.bootstrap_ollama \
    "$@" \
    --output "$out_dir"
  # The harness writes either to $out_dir/baseline_*.json (full run) or
  # to $out_dir/smoke/baseline_*.json (smoke only). With --full we
  # expect the top-level file; fall back to smoke for visibility.
  local result
  if compgen -G "$out_dir/baseline_*.json" > /dev/null; then
    result="$(ls -t "$out_dir"/baseline_*.json | head -1)"
  else
    result="$(ls -t "$out_dir"/smoke/baseline_*.json | head -1)"
  fi
  "$PY3" "$REPRO_DIR/verify.py" --run-name "$name" --result "$result"
}

# ── Read configurations from expected_results.json ──────────────────
# Python emits one "NAME\t<args…>" line per run; bash reads + dispatches.
mapfile -t RUNS < <(
  "$PY3" - <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path("findings/longmemeval/repro/expected_results.json").read_text())
for r in cfg["runs"]:
    args = " ".join(r["command_args"])
    print(f"{r['name']}\t{args}")
PY
)

EXIT=0
for line in "${RUNS[@]}"; do
  name="${line%%$'\t'*}"
  args="${line#*$'\t'}"
  # shellcheck disable=SC2086
  run_one "$name" $args || EXIT=$((EXIT + 1))
done

echo
echo "============================================================"
if [[ "$EXIT" -eq 0 ]]; then
  echo "  ✅ ALL REPRO RUNS WITHIN TOLERANCE"
else
  echo "  ❌ $EXIT run(s) outside tolerance — see logs above"
fi
echo "============================================================"
exit "$EXIT"
