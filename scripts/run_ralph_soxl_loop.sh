#!/bin/bash
set -euo pipefail

ROOT="/Users/rziiiii/.openclaw/workspace"
cd "$ROOT"

mkdir -p log

export PATH="$ROOT/.venv_ralph/bin:$PATH"

exec "$ROOT/.venv_ralph/bin/ralph" run \
  --agents codex \
  --max 20 \
  --timeout 10800 \
  --test-cmd "python3 -m py_compile src/soxl_quant/*.py production/*.py tests/*.py"
