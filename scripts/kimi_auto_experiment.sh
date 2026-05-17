#!/usr/bin/env bash
# =============================================================================
# SOXL Auto-Experiment Orchestrator
# =============================================================================
# Creates a tmux session and launches Kimi CLI to execute the next [auto]-tagged
# task from docs/active_workpad.md.
#
# Usage:
#   scripts/kimi_auto_experiment.sh           # normal run (checks guards)
#   scripts/kimi_auto_experiment.sh --force   # skip guards (for testing)
#
# Scheduling: systemd timer or cron
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="/tmp/kimi-auto-experiment.lock"
TMUX_SESSION="kimi-auto-exp"
LOG_FILE="${WORKSPACE}/runtime/harness/auto_experiment.log"
PROMPT_FILE="${WORKSPACE}/.codex/prompts/auto_experiment.md"
WORKPAD="${WORKSPACE}/docs/active_workpad.md"
GPU_UTIL_THRESHOLD=50
FORCE=false

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] $*" | tee -a "${LOG_FILE}"
}

cleanup() {
    rm -f "${LOCK_FILE}"
    if [[ -n "${TMP_PROMPT_FILE:-}" && -f "${TMP_PROMPT_FILE}" ]]; then
        rm -f "${TMP_PROMPT_FILE}"
    fi
}

# ---------------------------------------------------------------------------
# Guard 1: Lock file
# ---------------------------------------------------------------------------
if [[ -f "${LOCK_FILE}" ]]; then
    pid="$(cat "${LOCK_FILE}" 2>/dev/null || echo "unknown")"
    if kill -0 "${pid}" 2>/dev/null; then
        log "SKIP: Lock file exists and process ${pid} is still running."
        exit 0
    else
        log "WARN: Stale lock file found (PID ${pid} dead). Removing."
        rm -f "${LOCK_FILE}"
    fi
fi

# ---------------------------------------------------------------------------
# Guard 2: Existing tmux session
# ---------------------------------------------------------------------------
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    log "SKIP: tmux session '${TMUX_SESSION}' already exists."
    exit 0
fi

# ---------------------------------------------------------------------------
# Guard 3: GPU utilization (skip if --force)
# ---------------------------------------------------------------------------
if [[ "${FORCE}" != true ]]; then
    if command -v nvidia-smi &>/dev/null; then
        gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        if [[ "${gpu_util}" =~ ^[0-9]+$ && "${gpu_util}" -gt "${GPU_UTIL_THRESHOLD}" ]]; then
            log "SKIP: GPU utilization ${gpu_util}% > threshold ${GPU_UTIL_THRESHOLD}%."
            exit 0
        fi
        log "INFO: GPU utilization ${gpu_util}%."
    else
        log "INFO: nvidia-smi not found, skipping GPU check."
    fi
fi

# ---------------------------------------------------------------------------
# Guard 4: Workpad must contain [auto] tag
# ---------------------------------------------------------------------------
if [[ ! -f "${WORKPAD}" ]]; then
    log "SKIP: Workpad not found at ${WORKPAD}."
    exit 0
fi

if ! grep -q '\[auto\]' "${WORKPAD}"; then
    log "SKIP: No [auto] tagged task found in workpad."
    exit 0
fi

log "INFO: Found [auto] task in workpad. Proceeding."

# ---------------------------------------------------------------------------
# Guard 5: Prompt file must exist
# ---------------------------------------------------------------------------
if [[ ! -f "${PROMPT_FILE}" ]]; then
    log "ERROR: Prompt file not found at ${PROMPT_FILE}."
    exit 1
fi

# ---------------------------------------------------------------------------
# Acquire lock
# ---------------------------------------------------------------------------
echo $$ > "${LOCK_FILE}"
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Build the prompt (base64-encode to avoid shell escaping issues)
# ---------------------------------------------------------------------------
PROMPT_B64="$(base64 -w0 "${PROMPT_FILE}")"
TMP_PROMPT_FILE="$(mktemp /tmp/kimi_prompt.XXXXXX)"

# ---------------------------------------------------------------------------
# Create tmux session and launch Kimi
# ---------------------------------------------------------------------------
log "INFO: Creating tmux session '${TMUX_SESSION}' and launching Kimi CLI."

# Create detached tmux session with the working directory
tmux new-session -d -s "${TMUX_SESSION}" -c "${WORKSPACE}"

# Send commands to the tmux session
tmux send-keys -t "${TMUX_SESSION}" "cd '${WORKSPACE}'" C-m

# Decode prompt and launch kimi.  We use a temp file + command substitution
# to safely pass the multi-line prompt without shell-escaping issues.
tmux send-keys -t "${TMUX_SESSION}" "echo '${PROMPT_B64}' | base64 -d > '${TMP_PROMPT_FILE}'" C-m
tmux send-keys -t "${TMUX_SESSION}" "kimi -p \"\$(cat '${TMP_PROMPT_FILE}')\" --yolo" C-m
tmux send-keys -t "${TMUX_SESSION}" "rm -f '${TMP_PROMPT_FILE}'" C-m

# Add a completion hook: when kimi exits, log and close the session
tmux send-keys -t "${TMUX_SESSION}" "echo '[auto-experiment] Session finished at \$(date)' >> '${LOG_FILE}'" C-m
tmux send-keys -t "${TMUX_SESSION}" "exit" C-m

log "INFO: tmux session '${TMUX_SESSION}' launched."
log "INFO: Attach with: tmux attach -t ${TMUX_SESSION}"

# ---------------------------------------------------------------------------
# Non-blocking exit (parent script returns immediately; tmux runs in bg)
# ---------------------------------------------------------------------------
exit 0
