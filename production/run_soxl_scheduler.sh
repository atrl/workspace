#!/bin/bash
# Load private LaunchAgent environment, then run the repo-owned scheduler.

set -euo pipefail

ROOT="/Users/rziiiii/.openclaw/workspace"
ENV_FILE="${SOXL_LAUNCH_ENV_FILE:-${HOME}/.openclaw/soxl.env}"

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:${HOME}/.npm-global/bin:${HOME}/.npm/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"

cd "$ROOT"
exec "$ROOT/qlib_venv/bin/python" "$ROOT/production/soxl_daily_scheduler.py" "$@"
