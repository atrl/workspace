#!/bin/bash
# Refresh replayable model curves and rebuild local dashboard artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${SOXL_DASHBOARD_ROOT:-${DEFAULT_ROOT}}"
PYTHON_BIN="${SOXL_DASHBOARD_PYTHON:-${ROOT}/qlib_venv/bin/python}"

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:${HOME}/.npm-global/bin:${HOME}/.npm/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$ROOT"

echo "[dashboard-refresh] run started at $(date -Iseconds)"
"$PYTHON_BIN" "$ROOT/scripts/refresh_dashboard_model_replays.py" --python "$PYTHON_BIN"
"$PYTHON_BIN" "$ROOT/scripts/build_dashboard_bundle.py"
"$PYTHON_BIN" "$ROOT/scripts/check_dashboard_static_export.py"
echo "[dashboard-refresh] all done"
