#!/usr/bin/env bash
# scripts/run_budget_ablation.sh
# ==============================
# Phase 1 of docs/LOW_BUDGET_PLAN.md — the cost/accuracy curve.
#
# Five budget points, identical pipeline, identical rows. ONLY
# --max-context-chars moves. Everything else is the v2.0 anchor config
# from `make repro-everything`.
#
#   32000 / 16000 / 8000 / 4000 / 2000 chars
#   (~6.8k / 3.4k / 1.7k / 0.9k / 0.4k tokens at the MEASURED 4.7
#    chars/token — not the 4.0 that was assumed before Phase 0)
#
# No 64000 point. Phase 0 measured the anchor retriever at p50 26.6k /
# max 44.3k chars, so --max-context-chars 64000 never binds on any row:
# it is byte-identical to running unbounded, and re-measuring it costs
# ~95 min to learn nothing. The originally planned ladder spent its top
# point there.
#
# CAVEAT worth carrying into the writeup: 32000 is therefore the
# reference, and it is NOT unbounded — it clips the top ~30% of rows
# (mildly; p75 is 33.1k chars). Deltas below are "vs the 32k point", and
# the unbounded reference is the Phase 0 measurement in
# results/phase0_verify/, taken on a different row set. If the paper
# needs a matched unbounded number on this subset, run:
#   bash scripts/run_budget_ablation.sh 64000
#
# The real ceiling is --rerank-to (24), not the char cap: every row in
# Phase 0 admitted exactly 24 items. Below ~16000 chars the budget takes
# over. That makes the rerank tail (Phase 4) the other half of this lever.
#
# Each point is judged with llama-3.3-70b. The substring scorer reports
# 0% on single-session-preference (a known artifact) and would make the
# curve unreadable, so judged accuracy is the only number to read.
#
# Wall clock ~3.5h for all five points; cost ~$1.15 total. Resumable:
# a point whose budget_report.json already exists is skipped, so you can
# re-run after an interruption without paying twice.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   bash scripts/run_budget_ablation.sh
#   bash scripts/run_budget_ablation.sh 16000 8000     # just these points
#
# Then:
#   python findings/charts/budget_curve.py results/budget_*/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY3="${PY3:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
if [[ ! -x "$PY3" ]]; then
  PY3="$(command -v python3)"
  echo "WARNING: framework python3 not found; falling back to $PY3"
fi

SAMPLE="${SAMPLE:-samples/budget_strat_250.json}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/llama-3.3-70b-instruct}"
BUDGETS=("$@")
if [[ ${#BUDGETS[@]} -eq 0 ]]; then
  BUDGETS=(32000 16000 8000 4000 2000)
fi

# ── Preconditions ───────────────────────────────────────────────────
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]]; then
    OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' "$REPO_ROOT/.env" \
                          | head -1 | cut -d= -f2- | tr -d '"'"'"' ')"
    export OPENROUTER_API_KEY
  fi
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set (and not found in .env)." >&2
  exit 2
fi

if [[ ! -f "$SAMPLE" ]]; then
  echo "ERROR: sample $SAMPLE not found. Build it first:" >&2
  echo "  $PY3 scripts/build_budget_sample.py" >&2
  exit 2
fi

# tiktoken is what makes the x-axis a measurement rather than an
# estimate. Refuse to run a curve we'd have to caveat.
if ! "$PY3" -c 'from evals.longmemeval.budget import tokens_measured; raise SystemExit(0 if tokens_measured() else 1)'; then
  echo "ERROR: tiktoken is not installed for $PY3." >&2
  echo "  Token counts would be chars/3.9 estimates, not measurements." >&2
  echo "  Install it, then re-run:  $PY3 -m pip install tiktoken" >&2
  exit 2
fi

N_ROWS="$("$PY3" -c "import json,sys; print(len(json.load(open('$SAMPLE'))['question_ids']))")"
echo "ablation: ${#BUDGETS[@]} points x ${N_ROWS} rows  (sample: $SAMPLE)"
echo

# ── One budget point ────────────────────────────────────────────────
run_point() {
  local chars="$1"
  local out_dir="results/budget_${chars}"

  if [[ -f "$out_dir/budget_report.json" && -f "$out_dir/judged.json" ]]; then
    echo "== ${chars} chars — already complete, skipping"
    return 0
  fi

  echo "============================================================"
  echo "  BUDGET POINT — ${chars} chars"
  echo "============================================================"
  mkdir -p "$out_dir"

  # The v2.0 anchor config. Only --max-context-chars varies.
  "$PY3" -m evals.longmemeval.bootstrap_ollama \
    --provider openrouter --model openai/gpt-oss-120b \
    --openrouter-provider DeepInfra --seed 0 \
    --reasoner direct --use-ltm --ltm-backend in_memory --no-llm-promoter \
    --retriever hybrid \
    --session-aware-retrieval --session-top-k 12 --turns-per-session 6 \
    --top-k 80 --max-context-chars "$chars" --answer-max-tokens 2048 \
    --span-fallback --decompose-max-items 60 --rerank --rerank-to 24 \
    --question-ids-file "$SAMPLE" \
    --budget-report --full --yes --no-smoke \
    --output "$out_dir"

  local baseline
  baseline="$(ls -t "$out_dir"/baseline_*.json | head -1)"
  echo "-- judging ${chars} --"
  "$PY3" -m evals.longmemeval.rescore_with_judge \
    --input "$baseline" \
    --output "$out_dir/judged.json" \
    --provider openrouter --judge-model "$JUDGE_MODEL"
  echo
}

for chars in "${BUDGETS[@]}"; do
  run_point "$chars"
done

echo "============================================================"
echo "  ALL POINTS COMPLETE"
echo "============================================================"
"$PY3" findings/charts/budget_curve.py results/budget_*/
