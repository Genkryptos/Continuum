#!/usr/bin/env bash
# A-BUDGET + A-CHRONO arm on LongMemEval-S (500 questions).
# See evals/PREREGISTRATION.md §3.4 for what these flags are testing.
#
# Deviates from the §3.1 pinned config on three axes, so any number this
# produces is non_comparable against another system:
#   answerer muse-wonder-1.0   (pinned: openai/gpt-oss-120b)
#   --top-k 24                 (pinned: 8)
#   --use-ltm OFF              (pinned: on — ~17h of one-time ingest)
set -euo pipefail
cd "$(dirname "$0")"

OUT="results/arm_budget_chrono_20261107"
LOG="$OUT.log"
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

set -a
. ./.env
. /Users/mayanksahu/Hackathon/repart/.env
set +a

: "${OPENROUTER_API_KEY:?missing from ./.env — the judge needs it}"
: "${MODEL_API_KEY:?missing from the hackathon .env — the answerer needs it}"

nohup "$PY" -m evals.longmemeval.bootstrap_ollama \
  --provider meta --model muse-wonder-1.0 \
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
  --top-k 24 \
  --session-top-k 8 \
  --turn-rerank global \
  --chrono-sort \
  --full --yes --no-smoke \
  --output "$OUT" \
  > "$LOG" 2>&1 &
RUN_PID=$!

# Hold both sleep assertions for the life of the run, so a closed-lid or
# idle Mac cannot kill in-flight HTTPS mid-row.
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
echo "log            $(pwd)/$LOG"
