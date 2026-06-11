#!/bin/bash
# Load private LaunchAgent environment, then refresh weekly runtime selection evidence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${SOXL_WEEKLY_WF_ROOT:-${SOXL_SCHEDULER_ROOT:-${DEFAULT_ROOT}}}"
ENV_FILE="${SOXL_LAUNCH_ENV_FILE:-${HOME}/.openclaw/soxl.env}"
WEEKLY_PYTHON="${SOXL_WEEKLY_WF_PYTHON:-${SOXL_SCHEDULER_PYTHON:-${ROOT}/qlib_venv/bin/python}}"

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

cd "$ROOT"

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

RUNTIME_COMPOSER="${SOXL_RUNTIME_COMPOSER:-none}"
if [ -n "$RUNTIME_COMPOSER" ] && [ "$RUNTIME_COMPOSER" != "none" ]; then
    COMPOSER_PROFILE="${SOXL_RUNTIME_COMPOSER_PROFILE:-}"
    SLOW_PROFILE="${SOXL_RUNTIME_SLOW_PROFILE:-}"
    FAST_PROFILE="${SOXL_RUNTIME_FAST_PROFILE:-}"
    COMPOSER_OUTPUT_DIR="${SOXL_WEEKLY_WF_OUTPUT_DIR:-runtime/model/card/${COMPOSER_PROFILE}}"

    if [ "$RUNTIME_COMPOSER" != "budget_usage" ]; then
        echo "[weekly-wf] unsupported runtime composer=${RUNTIME_COMPOSER}" >&2
        exit 2
    fi
    if [ -z "$COMPOSER_PROFILE" ] || [ -z "$SLOW_PROFILE" ] || [ -z "$FAST_PROFILE" ]; then
        echo "[weekly-wf] incomplete runtime composer env" >&2
        exit 2
    fi

    echo "[weekly-wf] run started at $(date -Iseconds) mode=runtime_composer composer=${RUNTIME_COMPOSER} profile=${COMPOSER_PROFILE} slow=${SLOW_PROFILE} fast=${FAST_PROFILE}"

    COMPOSER_ARGS=(
        "$ROOT/scripts/review_budget_usage_runtime_promotion.py"
        --profile "$COMPOSER_PROFILE"
        --slow-profile "$SLOW_PROFILE"
        --fast-profile "$FAST_PROFILE"
        --output-dir "$COMPOSER_OUTPUT_DIR"
    )
    if truthy_env "${SOXL_RUNTIME_BULL_PARTICIPATION_GUARD:-}"; then
        COMPOSER_ARGS+=(--slow-model "soxl_${SLOW_PROFILE}_actor.pt" --fast-model "soxl_${FAST_PROFILE}_actor.pt")
    fi
    "$WEEKLY_PYTHON" "${COMPOSER_ARGS[@]}"

    echo "[weekly-wf] all done"
    exit 0
fi

PROFILE="${SOXL_WEEKLY_WF_PROFILE:-balanced_rule_residual_v22}"
FAMILY="${SOXL_WEEKLY_WF_FAMILY:-balanced}"
OUTPUT_DIR="${SOXL_WEEKLY_WF_OUTPUT_DIR:-runtime/model/card/${PROFILE}}"
MODEL_SUMMARY="${SOXL_WEEKLY_WF_MODEL_SUMMARY:-${OUTPUT_DIR}/soxl_${PROFILE}_summary.json}"
OVERLAY_REVIEW="${SOXL_WEEKLY_WF_OVERLAY_REVIEW:-runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json}"

echo "[weekly-wf] run started at $(date -Iseconds) profile=${PROFILE} family=${FAMILY}"

"$WEEKLY_PYTHON" "$ROOT/scripts/walkforward_backtest_validate.py" \
    --profile "$PROFILE" \
    --output-dir "$OUTPUT_DIR"

"$WEEKLY_PYTHON" "$ROOT/scripts/review_runtime_selection.py" \
    --profile "$PROFILE" \
    --model-summary "$MODEL_SUMMARY" \
    --overlay-review "$OVERLAY_REVIEW" \
    --output-dir "$OUTPUT_DIR" \
    --family "$FAMILY" \
    --persist-model-promotion

echo "[weekly-wf] all done"
