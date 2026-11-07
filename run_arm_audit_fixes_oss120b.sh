#!/usr/bin/env bash
# Audit-fixes arm with the §3.1 answerer: openai/gpt-oss-120b via OpenRouter.
#
# Identical to run_arm_audit_fixes.sh (90.8% with muse-wonder-1.0) except the
# answerer, so the two runs isolate the answerer axis. This is also the
# answerer of the reader-matched table in docs/report.md §6 (Hindsight ~89%,
# Continuum ~74%, Mem0-S 67.6%).
#
# Still non_comparable to the §3.1 baseline — versions.deviations will list
# --top-k 48 and --use-ltm off. Against Hindsight's 89.0 the answerer now
# matches, but the judge does not: theirs was gpt-oss-120b, ours is
# llama-3.3-70b-instruct (§3.2).
#
# --openrouter-provider DeepInfra pins BOTH the answerer and the judge to one
# backend (one flag serves both); DeepInfra serves both models. Unpinned,
# OpenRouter's provider switching measured ~44% answer variance on
# gpt-oss-120b at temperature 0.
#
# Overrides: OUT=... DATASET=... ./run_arm_audit_fixes_oss120b.sh [extra flags]
set -euo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-results/arm_audit_fixes_oss120b_20261107}"
LOG="$OUT.log"
DATASET="${DATASET:-evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json}"
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

set -a
. ./.env
set +a

: "${OPENROUTER_API_KEY:?missing from ./.env — the answerer and judge need it}"

nohup "$PY" -m evals.longmemeval.bootstrap_ollama \
  --dataset "$DATASET" \
  --provider openrouter --model openai/gpt-oss-120b \
  --openrouter-provider DeepInfra \
  --rpm 60 \
  --judge llm \
  --judge-provider openrouter \
  --judge-model meta-llama/llama-3.3-70b-instruct \
  --reasoner direct \
  --retriever hybrid \
  --session-aware-retrieval \
  --answer-max-tokens 2048 \
  --max-context-chars 64000 \
  --rrf-k 60 \
  --seed 0 \
  --top-k 48 \
  --session-top-k 12 \
  --turns-per-session 3 \
  --turn-rerank floor \
  --user-turns-first \
  --chrono-sort \
  --scorer-weights 1,0,0,0 \
  --pref-conditioning \
  --reflect \
  --reflect-types single-session-preference \
  --reflect-max-tokens 2048 \
  --full --yes --no-smoke \
  --output "$OUT" \
  "$@" \
  > "$LOG" 2>&1 &
RUN_PID=$!

# Hold both sleep assertions for the life of the run. On battery the system
# assertion (-s) is not honoured, so a closed lid still sleeps the Mac.
nohup caffeinate -is -w "$RUN_PID" > /dev/null 2>&1 &
CAF_PID=$!

sleep 3
if ! kill -0 "$RUN_PID" 2>/dev/null; then
  echo "FAILED to start — last lines of $LOG:"
  tail -20 "$LOG"
  exit 1
fi

echo "run PID        $RUN_PID"
echo "caffeinate PID $CAF_PID"
case "$LOG" in /*) echo "log            $LOG" ;; *) echo "log            $(pwd)/$LOG" ;; esac
