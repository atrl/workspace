#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYMPHONY_ROOT="${SYMPHONY_ROOT:-$HOME/code/symphony}"
WORKFLOW_PATH="${WORKFLOW_PATH:-$ROOT/WORKFLOW.md}"
LOGS_ROOT="${LOGS_ROOT:-$ROOT/log}"

if [[ -z "${LINEAR_API_KEY:-}" ]]; then
  echo "LINEAR_API_KEY is required" >&2
  exit 1
fi

mkdir -p "$LOGS_ROOT"

if [[ ! -d "$SYMPHONY_ROOT/.git" ]]; then
  git clone https://github.com/openai/symphony "$SYMPHONY_ROOT"
fi

cd "$SYMPHONY_ROOT/elixir"
mise trust
mise install
mise exec -- mix setup
mise exec -- mix build
exec mise exec -- ./bin/symphony "$WORKFLOW_PATH" --logs-root "$LOGS_ROOT"
