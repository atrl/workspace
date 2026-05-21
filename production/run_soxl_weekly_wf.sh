#!/bin/bash
# Load private LaunchAgent environment, then refresh weekly runtime selection evidence.

set -euo pipefail

ROOT="/Users/rziiiii/.openclaw/workspace"
ENV_FILE="${SOXL_LAUNCH_ENV_FILE:-${HOME}/.openclaw/soxl.env}"

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

cd "$ROOT"

PROFILE="${SOXL_WEEKLY_WF_PROFILE:-balanced_rule_residual_v22}"
FAMILY="${SOXL_WEEKLY_WF_FAMILY:-balanced}"
OUTPUT_DIR="${SOXL_WEEKLY_WF_OUTPUT_DIR:-runtime/model/card/${PROFILE}}"
MODEL_SUMMARY="${SOXL_WEEKLY_WF_MODEL_SUMMARY:-${OUTPUT_DIR}/soxl_${PROFILE}_summary.json}"
OVERLAY_REVIEW="${SOXL_WEEKLY_WF_OVERLAY_REVIEW:-runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json}"

echo "[weekly-wf] run started at $(date -Iseconds) profile=${PROFILE} family=${FAMILY}"

"$ROOT/qlib_venv/bin/python" "$ROOT/scripts/walkforward_backtest_validate.py" \
    --profile "$PROFILE" \
    --output-dir "$OUTPUT_DIR"

"$ROOT/qlib_venv/bin/python" "$ROOT/scripts/review_runtime_selection.py" \
    --profile "$PROFILE" \
    --model-summary "$MODEL_SUMMARY" \
    --overlay-review "$OVERLAY_REVIEW" \
    --output-dir "$OUTPUT_DIR" \
    --family "$FAMILY" \
    --persist-model-promotion

echo "[weekly-wf] all done"
