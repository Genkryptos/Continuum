#!/usr/bin/env bash
# examples/chat_agent/demo.sh
# ===========================
# Replay the scripted walkthrough through the chat agent. The script
# itself lives at demo_script.txt — edit that to change what gets
# typed; this wrapper just picks the right Python and points the agent
# at it.
#
# Runtime: <60 seconds, no API key, no infrastructure.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PY3="${PY3:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
if [[ ! -x "$PY3" ]]; then
  PY3="$(command -v python3)"
fi

cd "$REPO_ROOT"
exec "$PY3" -m examples.chat_agent.agent \
  --script "$HERE/demo_script.txt" \
  --llm "${DEMO_LLM:-mock}"
