#!/bin/bash
# Refresh local dashboard artifacts without placing orders or deploying remotely.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${SOXL_DASHBOARD_ROOT:-${SOXL_SCHEDULER_ROOT:-${DEFAULT_ROOT}}}"
ENV_FILE="${SOXL_LAUNCH_ENV_FILE:-${HOME}/.openclaw/soxl.env}"
DASHBOARD_PYTHON="${SOXL_DASHBOARD_PYTHON:-${SOXL_SCHEDULER_PYTHON:-${ROOT}/qlib_venv/bin/python}}"

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:${HOME}/.npm-global/bin:${HOME}/.npm/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$ROOT"
mkdir -p runtime/soxl/paper_trading

echo "[dashboard-refresh] run started at $(date -Iseconds)"
"$DASHBOARD_PYTHON" scripts/refresh_dashboard_model_replays.py --python "$DASHBOARD_PYTHON"
"$DASHBOARD_PYTHON" scripts/build_backtest_dashboard.py
"$DASHBOARD_PYTHON" scripts/build_dashboard_v2_data.py
"$DASHBOARD_PYTHON" scripts/check_dashboard_static_export.py
"$DASHBOARD_PYTHON" scripts/check_dashboard_v2_data.py
echo "[dashboard-refresh] all done"
