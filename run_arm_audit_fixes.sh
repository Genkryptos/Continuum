#!/usr/bin/env bash
# Audit-fixes arm on LongMemEval-S (500 questions). Every fix from the
# 2026-11-07 gap audit, on top of the A-BUDGET + A-CHRONO arm:
#   --turn-rerank floor          per-session floor, then global fill (F1)
#   --user-turns-first           user turns ahead of assistant replies unless
#                                the question asks what the assistant said;
#                                837/840 gold turns are user turns, assistant
#                                replies took ~41% of the budget
#   --top-k 48 --session-top-k 12 --turns-per-session 3
#                                chosen by evals/audit/turn_selection_ablation.py
#                                (100-row sample, no LLM): gold-turn recall
#                                0.876 -> 0.968, rows complete 0.804 -> 0.938,
#                                with less context (18.7k chars) than the last
#                                full run (22.9k)
#   --retriever hybrid           BM25 now fused INSIDE session-aware selection,
#                                over the whole haystack (F1/F2)
#   --chrono-sort                applied after fusion, by date not haystack
#                                position (F2; 211/500 haystacks are unsorted)
#   --scorer-weights 1,0,0,0     the composite scorer is now honoured under
#                                session-aware retrieval (F5); relevance-only
#                                reproduces what the last run actually used,
#                                so recency stays a separate, unconfounded A/B
#   --pref-conditioning --reflect --reflect-types single-session-preference
#                                the preference prompt and the bounded
#                                preference reasoning pass were both off (F4)
#   --openrouter-provider DeepInfra
#                                pins the judge to one backend (§3.1 requires
#                                it). Unpinned, OpenRouter spread the judge
#                                across providers and it flipped its verdict
#                                on a word-for-word identical answer. Affects
#                                only the judge; the answerer is Meta's API.
# Plus code-level fixes with no flag: count questions keep the dated
# aggregate prompt (S2), KU earlier-value questions get their own prompt
# (S4), a presupposition check in every factual prompt (S1), second-person
# answers match first-person gold (M7), and a `versions` block (S5).
#
# Still non_comparable: the result JSON's versions.deviations lists why
# (answerer muse-wonder-1.0, --top-k 48, --use-ltm off).
#
# Overrides: OUT=... DATASET=... ./run_arm_audit_fixes.sh [extra flags]
set -euo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-results/arm_audit_fixes_20261107}"
LOG="$OUT.log"
DATASET="${DATASET:-evals/longmemeval/LongMemEval/data/longmemeval_s_cleaned.json}"
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

set -a
. ./.env
. /Users/mayanksahu/Hackathon/repart/.env
set +a

: "${OPENROUTER_API_KEY:?missing from ./.env — the judge needs it}"
: "${MODEL_API_KEY:?missing from the hackathon .env — the answerer needs it}"

nohup "$PY" -m evals.longmemeval.bootstrap_ollama \
  --dataset "$DATASET" \
  --provider meta --model muse-wonder-1.0 \
  --judge llm \
  --judge-provider openrouter \
  --judge-model meta-llama/llama-3.3-70b-instruct \
  --openrouter-provider DeepInfra \
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
