#!/bin/bash
set -euo pipefail

ROOT="/Users/rziiiii/.openclaw/workspace"
cd "$ROOT"

mkdir -p log

LOG_FILE="$ROOT/log/ralph-supervisor.log"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ralph run" >> "$LOG_FILE"

  if "$ROOT/.venv_ralph/bin/ralph" run \
    --agents codex \
    --max 1000000 \
    --no-timeout \
    --test-cmd "python3 -m py_compile src/soxl_quant/*.py production/*.py tests/*.py" \
    >> "$LOG_FILE" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ralph exited cleanly; restarting in 5s" >> "$LOG_FILE"
  else
    code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ralph exited with code $code; restarting in 5s" >> "$LOG_FILE"
  fi

  sleep 5
done
