#!/bin/bash
# Load private LaunchAgent environment, then run the repo-owned scheduler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${SOXL_SCHEDULER_ROOT:-${DEFAULT_ROOT}}"
ENV_FILE="${SOXL_LAUNCH_ENV_FILE:-${HOME}/.openclaw/soxl.env}"
SCHEDULER_PYTHON="${SOXL_SCHEDULER_PYTHON:-${ROOT}/qlib_venv/bin/python}"

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:${HOME}/.npm-global/bin:${HOME}/.npm/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"

arg_present() {
    local needle="$1"
    shift
    for arg in "$@"; do
        if [ "$arg" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

append_value_arg() {
    local flag="$1"
    local value="$2"
    if [ -n "$value" ]; then
        SCHEDULER_ARGS+=("$flag" "$value")
    fi
}

truthy_env() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

SCHEDULER_ARGS=("$@")
if ! arg_present "--runtime-composer" "${SCHEDULER_ARGS[@]}"; then
    RUNTIME_COMPOSER="${SOXL_RUNTIME_COMPOSER:-none}"
    if [ -n "$RUNTIME_COMPOSER" ] && [ "$RUNTIME_COMPOSER" != "none" ]; then
        SCHEDULER_ARGS+=("--runtime-composer" "$RUNTIME_COMPOSER")
        append_value_arg "--composer-profile" "${SOXL_RUNTIME_COMPOSER_PROFILE:-}"
        append_value_arg "--slow-profile" "${SOXL_RUNTIME_SLOW_PROFILE:-}"
        append_value_arg "--fast-profile" "${SOXL_RUNTIME_FAST_PROFILE:-}"
        if truthy_env "${SOXL_RUNTIME_BULL_PARTICIPATION_GUARD:-}"; then
            SCHEDULER_ARGS+=("--bull-participation-guard")
        fi
        append_value_arg "--bull-ma-window" "${SOXL_RUNTIME_BULL_MA_WINDOW:-}"
        append_value_arg "--bull-momentum-window" "${SOXL_RUNTIME_BULL_MOMENTUM_WINDOW:-}"
        append_value_arg "--bull-high-window" "${SOXL_RUNTIME_BULL_HIGH_WINDOW:-}"
        append_value_arg "--bull-min-periods" "${SOXL_RUNTIME_BULL_MIN_PERIODS:-}"
        append_value_arg "--bull-min-momentum" "${SOXL_RUNTIME_BULL_MIN_MOMENTUM:-}"
        append_value_arg "--bull-max-drawdown" "${SOXL_RUNTIME_BULL_MAX_DRAWDOWN:-}"
    fi
fi

cd "$ROOT"
exec "$SCHEDULER_PYTHON" "$ROOT/production/soxl_daily_scheduler.py" "${SCHEDULER_ARGS[@]}"
