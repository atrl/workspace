#!/bin/bash
# SOXL scheduler health check — 查看定时任务执行进度

set -u

ROOT="${SOXL_SCHEDULER_HEALTH_ROOT:-$(pwd)}"
OS_NAME="${SOXL_SCHEDULER_HEALTH_OS:-$(uname -s)}"
HEALTH_PYTHON="${SOXL_SCHEDULER_HEALTH_PYTHON:-${ROOT}/qlib_venv/bin/python}"
SCHEDULER_LABEL="com.soxl.daily.scheduler"
PREOPEN_SCHEDULER_LABEL="com.soxl.preopen.scheduler"
WEEKLY_WF_SCHEDULER_LABEL="com.soxl.weekly.wf"
LAUNCHAGENTS_DIR="${SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR:-${HOME:-}/Library/LaunchAgents}"
LAUNCH_ENV_FILE="${SOXL_LAUNCH_ENV_FILE:-${HOME:-}/.openclaw/soxl.env}"
HEALTH_WEEKDAY="${SOXL_SCHEDULER_HEALTH_WEEKDAY:-$(date +%u)}"
HEALTH_HHMM="${SOXL_SCHEDULER_HEALTH_HHMM:-$(date +%H%M)}"
HEALTH_DATE="${SOXL_SCHEDULER_HEALTH_DATE:-$(date +%Y-%m-%d)}"
HEALTH_PREOPEN_CUTOFF_EPOCH="${SOXL_SCHEDULER_HEALTH_PREOPEN_CUTOFF_EPOCH:-}"
HEALTH_POSTCLOSE_CUTOFF_EPOCH="${SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH:-}"
PREOPEN_HHMM="2100"
PREOPEN_TIME_LABEL="21:00"
PREOPEN_WINDOW_MINUTES="${SOXL_SCHEDULER_HEALTH_PREOPEN_WINDOW_MINUTES:-90}"
POSTCLOSE_HHMM="0615"
POSTCLOSE_TIME_LABEL="06:15"
POSTCLOSE_WINDOW_MINUTES="${SOXL_SCHEDULER_HEALTH_POSTCLOSE_WINDOW_MINUTES:-90}"
SELECTED_MODEL_PROFILE=""
PAPER_EXPECTED_LAST_TRADING_DAY=""
RUNTIME_COMPOSER_STATE="default"
RUNTIME_COMPOSER=""
RUNTIME_COMPOSER_PROFILE=""
RUNTIME_SLOW_PROFILE=""
RUNTIME_FAST_PROFILE=""
RUNTIME_BULL_PARTICIPATION_GUARD="false"
RUNTIME_COMPOSER_MISSING=""
ORDINARY_PAPER_PROOF_STATUS=""
ORDINARY_PAPER_PROOF_SELECTED_MODEL=""
ORDINARY_PAPER_PROOF_EXECUTION=""
DATA_GATE_STATUS="unknown"
DATA_GATE_DETAIL=""
DATA_REGISTRY_GATE_STATUS="unknown"
DATA_REGISTRY_GATE_DETAIL=""
MODEL_GATE_STATUS="unknown"
MODEL_GATE_DETAIL=""
BROKER_GATE_STATUS="unknown"
BROKER_GATE_DETAIL=""
SCHEDULER_BROKER="paper"
DASHBOARD_GATE_STATUS="unknown"
DASHBOARD_GATE_DETAIL=""
VERCEL_REMOTE_STATUS="unknown"
VERCEL_AUTH_STATUS="unknown"
VERCEL_PROJECT_STATUS="unknown"
VERCEL_DEPLOYMENT_URL="none"
VERCEL_ENV_FILE_APPLIED_KEYS="unknown"
POSTCLOSE_TARGET_STATUS="unknown"
POSTCLOSE_TARGET_DETAIL=""

file_epoch() {
    path="$1"
    if epoch=$(stat -f %m "$path" 2>/dev/null); then
        echo "$epoch"
        return
    elif epoch=$(stat -c %Y "$path" 2>/dev/null); then
        echo "$epoch"
        return
    fi
    echo ""
}

format_mtime() {
    path="$1"
    epoch=$(file_epoch "$path")
    if [ -z "$epoch" ]; then
        echo "unknown"
        return
    fi

    if date -r "$epoch" "+%H:%M %Y-%m-%d" 2>/dev/null; then
        return
    fi
    if date -d "@$epoch" "+%H:%M %Y-%m-%d" 2>/dev/null; then
        return
    fi
    echo "unknown"
}

file_size() {
    du -h "$1" | awk '{print $1}'
}

missing_log_status() {
    log="$1"
    case "$log" in
        runtime/soxl/wf_weekly.log|runtime/soxl/wf_weekly.err)
            echo "pending (not created yet)"
            ;;
        *)
            echo "NOT FOUND"
            ;;
    esac
}

number_value() {
    value=$(printf "%s" "$1" | sed 's/^0*//')
    if [ -z "$value" ]; then
        echo 0
    else
        echo "$value"
    fi
}

hhmm_to_minutes() {
    hhmm="$1"
    case "$hhmm" in
        [0-9][0-9][0-9][0-9])
            ;;
        *)
            echo ""
            return
            ;;
    esac
    hour=${hhmm%??}
    minute=${hhmm#??}
    hour_num=$(number_value "$hour")
    minute_num=$(number_value "$minute")
    echo $((hour_num * 60 + minute_num))
}

marker_hhmm() {
    marker="$1"
    printf "%s" "$marker" | sed -n 's/.*run started at [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T\([0-9][0-9]\):\([0-9][0-9]\):.*/\1\2/p' | tail -1
}

marker_in_schedule_window() {
    marker="$1"
    schedule_hhmm="$2"
    window_minutes="$3"
    marker_time=$(marker_hhmm "$marker")
    marker_minutes=$(hhmm_to_minutes "$marker_time")
    schedule_minutes=$(hhmm_to_minutes "$schedule_hhmm")
    window_num=$(number_value "$window_minutes")
    if [ -z "$marker_minutes" ] || [ -z "$schedule_minutes" ]; then
        return 1
    fi
    window_end=$((schedule_minutes + window_num))
    if [ "$window_end" -lt 1440 ]; then
        [ "$marker_minutes" -ge "$schedule_minutes" ] && [ "$marker_minutes" -le "$window_end" ]
    else
        wrapped_end=$((window_end - 1440))
        [ "$marker_minutes" -ge "$schedule_minutes" ] || [ "$marker_minutes" -le "$wrapped_end" ]
    fi
}

postclose_due_today() {
    weekday_num=$(number_value "$HEALTH_WEEKDAY")
    hhmm_num=$(number_value "$HEALTH_HHMM")
    postclose_num=$(number_value "$POSTCLOSE_HHMM")
    [ "$weekday_num" -ge 2 ] && [ "$weekday_num" -le 6 ] && [ "$hhmm_num" -ge "$postclose_num" ]
}

preopen_due_today() {
    weekday_num=$(number_value "$HEALTH_WEEKDAY")
    hhmm_num=$(number_value "$HEALTH_HHMM")
    preopen_num=$(number_value "$PREOPEN_HHMM")
    [ "$weekday_num" -ge 1 ] && [ "$weekday_num" -le 5 ] && [ "$hhmm_num" -ge "$preopen_num" ]
}

preopen_cutoff_epoch() {
    if [ -n "$HEALTH_PREOPEN_CUTOFF_EPOCH" ]; then
        echo "$HEALTH_PREOPEN_CUTOFF_EPOCH"
        return
    fi
    cutoff_text="${HEALTH_DATE} ${PREOPEN_TIME_LABEL}"
    if date -j -f "%Y-%m-%d %H:%M" "$cutoff_text" +%s 2>/dev/null; then
        return
    fi
    if date -d "$cutoff_text" +%s 2>/dev/null; then
        return
    fi
    echo ""
}

postclose_cutoff_epoch() {
    if [ -n "$HEALTH_POSTCLOSE_CUTOFF_EPOCH" ]; then
        echo "$HEALTH_POSTCLOSE_CUTOFF_EPOCH"
        return
    fi
    cutoff_text="${HEALTH_DATE} ${POSTCLOSE_TIME_LABEL}"
    if date -j -f "%Y-%m-%d %H:%M" "$cutoff_text" +%s 2>/dev/null; then
        return
    fi
    if date -d "$cutoff_text" +%s 2>/dev/null; then
        return
    fi
    echo ""
}

preopen_success_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -Eq "^\[scheduler\] (core done|all done)$" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(preopen_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -z "$cutoff_epoch" ] || [ -z "$log_epoch" ]; then
        return 0
    fi
    if [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    if grep -q "\[scheduler\] run started at " "$path" 2>/dev/null; then
        candidate_markers=$(preopen_success_candidate_markers "$path")
        if [ -n "$candidate_markers" ]; then
            [ -n "$(preopen_last_success_target_marker "$path")" ]
            return
        fi
    fi
    if grep -q "\[scheduler\] run started at " "$path" 2>/dev/null; then
        awk '
            /\[scheduler\] run started at / { seen = 1; ok = 0; next }
            seen && /^\[scheduler\] (core done|all done)$/ { ok = 1 }
            END { exit ok ? 0 : 1 }
        ' "$path"
        return
    fi
    return 0
}

preopen_current_success_outside_window_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -Eq "^\[scheduler\] (core done|all done)$" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(preopen_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -n "$cutoff_epoch" ] && [ -n "$log_epoch" ] && [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    marker=$(preopen_last_success_candidate_marker "$path")
    if [ -n "$marker" ] && ! marker_in_schedule_window "$marker" "$PREOPEN_HHMM" "$PREOPEN_WINDOW_MINUTES"; then
        return 0
    fi
    return 1
}

postclose_success_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -Eq "^\[scheduler\] (core done|all done)$" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(postclose_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -z "$cutoff_epoch" ] || [ -z "$log_epoch" ]; then
        return 0
    fi
    if [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    if [ -n "$(postclose_last_success_slot_marker "$path")" ]; then
        return 0
    fi
    return 1
}

postclose_current_success_outside_window_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -Eq "^\[scheduler\] (core done|all done)$" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(postclose_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -n "$cutoff_epoch" ] && [ -n "$log_epoch" ] && [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    marker=$(postclose_last_success_candidate_marker "$path")
    if [ -n "$marker" ] && ! marker_in_schedule_window "$marker" "$POSTCLOSE_HHMM" "$POSTCLOSE_WINDOW_MINUTES"; then
        return 0
    fi
    return 1
}

postclose_cached_dry_run_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -q "\[scheduler\] all done (with cached dry-run)" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(postclose_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -z "$cutoff_epoch" ] || [ -z "$log_epoch" ]; then
        return 0
    fi
    if [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    if grep -q "\[scheduler\] run started at " "$path" 2>/dev/null; then
        awk '
            /\[scheduler\] run started at / { seen = 1; cached = 0; next }
            seen && /\[scheduler\] all done \(with cached dry-run\)/ { cached = 1 }
            END { exit cached ? 0 : 1 }
        ' "$path"
        return
    fi
    return 0
}

postclose_current_success_without_slot_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -Eq "^\[scheduler\] (core done|all done)$" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(postclose_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -n "$cutoff_epoch" ] && [ -n "$log_epoch" ] && [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    if [ -n "$(postclose_last_success_slot_marker "$path")" ]; then
        return 1
    fi
    awk '
        /^\[scheduler\] run started at / { marker = $0; ok = 0; next }
        /^\[scheduler\] (core done|all done)$/ { ok = 1 }
        END {
            if (ok && marker != "" && marker !~ "schedule_slot=postclose") {
                exit 0
            }
            exit 1
        }
    ' "$path"
}

postclose_current_success_without_trigger_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -Eq "^\[scheduler\] (core done|all done)$" "$path" 2>/dev/null; then
        return 1
    fi
    cutoff_epoch=$(postclose_cutoff_epoch)
    log_epoch=$(file_epoch "$path")
    if [ -n "$cutoff_epoch" ] && [ -n "$log_epoch" ] && [ "$log_epoch" -lt "$cutoff_epoch" ]; then
        return 1
    fi
    if [ -n "$(postclose_last_success_slot_marker "$path")" ]; then
        return 1
    fi
    awk '
        /^\[scheduler\] run started at / { marker = $0; ok = 0; next }
        /^\[scheduler\] (core done|all done)$/ { ok = 1 }
        END {
            if (ok && marker != "" &&
                marker ~ "schedule_slot=postclose" &&
                marker !~ "schedule_trigger=launchagent_postclose" &&
                marker !~ "schedule_trigger=cron_postclose") {
                exit 0
            }
            exit 1
        }
    ' "$path"
}

marker_has_postclose_trigger() {
    marker="$1"
    marker_has_text "$marker" "schedule_trigger=launchagent_postclose" ||
        marker_has_text "$marker" "schedule_trigger=cron_postclose"
}

marker_has_preopen_trigger() {
    marker="$1"
    marker_has_text "$marker" "schedule_trigger=launchagent_preopen" ||
        marker_has_text "$marker" "schedule_trigger=cron_preopen"
}

postclose_last_run_marker() {
    path="$1"
    if [ ! -f "$path" ]; then
        return
    fi
    awk '/^\[scheduler\] run started at / { marker = $0 } END { print marker }' "$path"
}

postclose_success_candidate_markers() {
    path="$1"
    if [ ! -f "$path" ]; then
        return
    fi
    awk -v health_date="$HEALTH_DATE" '
        /^\[scheduler\] run started at / {
            marker = $0
            next
        }
        /^\[scheduler\] (core done|all done)$/ {
            if (marker != "" &&
                marker ~ "schedule_slot=postclose" &&
                (marker ~ "schedule_trigger=launchagent_postclose" ||
                 marker ~ "schedule_trigger=cron_postclose") &&
                marker ~ health_date "T") {
                print marker
            }
        }
    ' "$path"
}

postclose_last_success_candidate_marker() {
    path="$1"
    postclose_success_candidate_markers "$path" | tail -1
}

postclose_last_success_slot_marker() {
    path="$1"
    candidate_markers=$(postclose_success_candidate_markers "$path")
    if [ -z "$candidate_markers" ]; then
        return
    fi
    success_marker=""
    while IFS= read -r candidate_marker; do
        if marker_in_schedule_window "$candidate_marker" "$POSTCLOSE_HHMM" "$POSTCLOSE_WINDOW_MINUTES"; then
            success_marker="$candidate_marker"
        fi
    done <<EOF
$candidate_markers
EOF
    printf "%s\n" "$success_marker"
}

scheduler_last_run_marker() {
    path="$1"
    if [ ! -f "$path" ]; then
        return
    fi
    awk '/^\[scheduler\] run started at / { marker = $0 } END { print marker }' "$path"
}

preopen_success_candidate_markers() {
    path="$1"
    if [ ! -f "$path" ]; then
        return
    fi
    awk -v health_date="$HEALTH_DATE" '
        /^\[scheduler\] run started at / {
            marker = $0
            next
        }
        /^\[scheduler\] (core done|all done)$/ {
            if (marker != "" &&
                marker ~ "schedule_slot=preopen" &&
                (marker ~ "schedule_trigger=launchagent_preopen" ||
                 marker ~ "schedule_trigger=cron_preopen") &&
                marker ~ health_date "T") {
                print marker
            }
        }
    ' "$path"
}

preopen_last_success_candidate_marker() {
    path="$1"
    preopen_success_candidate_markers "$path" | tail -1
}

preopen_last_success_target_marker() {
    path="$1"
    candidate_markers=$(preopen_success_candidate_markers "$path")
    if [ -z "$candidate_markers" ]; then
        return
    fi
    success_marker=""
    while IFS= read -r candidate_marker; do
        if marker_in_schedule_window "$candidate_marker" "$PREOPEN_HHMM" "$PREOPEN_WINDOW_MINUTES"; then
            success_marker="$candidate_marker"
        fi
    done <<EOF
$candidate_markers
EOF
    printf "%s\n" "$success_marker"
}

marker_has_text() {
    marker="$1"
    needle="$2"
    case "$marker" in
        *"$needle"*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

print_scheduler_service() {
    echo "--- Scheduler Service ---"
    if [ "$OS_NAME" = "Darwin" ]; then
        if ! command -v launchctl >/dev/null 2>&1; then
            echo "  LaunchAgent ${PREOPEN_SCHEDULER_LABEL}: launchctl unavailable"
            echo "  LaunchAgent ${SCHEDULER_LABEL}: launchctl unavailable"
            return
        fi
        launchctl_output=$(launchctl list 2>/dev/null || true)
        for label in "$PREOPEN_SCHEDULER_LABEL" "$SCHEDULER_LABEL" "$WEEKLY_WF_SCHEDULER_LABEL"; do
            if printf "%s\n" "$launchctl_output" | grep -q "$label"; then
                echo "  LaunchAgent ${label}: loaded"
            else
                echo "  LaunchAgent ${label}: not loaded"
            fi
            repo_plist="${ROOT}/production/${label}.plist"
            installed_plist="${LAUNCHAGENTS_DIR}/${label}.plist"
            if [ ! -f "$repo_plist" ]; then
                echo "  LaunchAgent ${label} plist: repo file missing"
            elif [ ! -f "$installed_plist" ]; then
                echo "  LaunchAgent ${label} plist: not installed"
            elif cmp -s "$repo_plist" "$installed_plist"; then
                echo "  LaunchAgent ${label} plist: matches repo"
            else
                echo "  LaunchAgent ${label} plist: differs from repo"
            fi
        done
        return
    fi

    if command -v systemctl >/dev/null 2>&1; then
        if systemctl is-active cron >/dev/null 2>&1; then
            echo "  cron: active"
        else
            echo "  cron: $(systemctl is-active cron 2>/dev/null || echo unknown)"
        fi
    elif command -v service >/dev/null 2>&1; then
        service cron status 2>/dev/null | head -1 | sed 's/^/  /'
    else
        echo "  cron: service checker unavailable"
    fi
}

env_file_has_nonempty_assignment() {
    var_name="$1"
    env_file="$2"
    if [ ! -f "$env_file" ]; then
        return 1
    fi
    awk -v key="$var_name" '
        BEGIN { found = 0 }
        /^[[:space:]]*#/ { next }
        {
            line = $0
            sub(/^[[:space:]]*export[[:space:]]+/, "", line)
            pattern = "^[[:space:]]*" key "[[:space:]]*="
            if (line ~ pattern) {
                sub(pattern "[[:space:]]*", "", line)
                gsub(/^[\047"]|[\047"]$/, "", line)
                gsub(/[[:space:]]+$/, "", line)
                lower = tolower(line)
                if (line != "" && lower != "replace_me" && lower != "replace-me") {
                    found = 1
                    exit
                }
            }
        }
        END { exit found ? 0 : 1 }
    ' "$env_file"
}

env_file_assignment_value() {
    var_name="$1"
    env_file="$2"
    if [ ! -f "$env_file" ]; then
        return
    fi
    awk -v key="$var_name" '
        /^[[:space:]]*#/ { next }
        {
            line = $0
            sub(/^[[:space:]]*export[[:space:]]+/, "", line)
            pattern = "^[[:space:]]*" key "[[:space:]]*="
            if (line ~ pattern) {
                sub(pattern "[[:space:]]*", "", line)
                gsub(/[[:space:]]+$/, "", line)
                gsub(/^[\047"]|[\047"]$/, "", line)
                print line
                exit
            }
        }
    ' "$env_file"
}

configured_env_value() {
    value="$1"
    if [ -z "$value" ]; then
        return 1
    fi
    normalized=$(printf "%s" "$value" | tr '[:upper:]' '[:lower:]')
    [ "$normalized" != "replace_me" ] && [ "$normalized" != "replace-me" ]
}

print_vercel_project_env_status() {
    current_project="${VERCEL_PROJECT_ID:-}"
    current_org="${VERCEL_ORG_ID:-}"
    launch_project="false"
    launch_org="false"
    if env_file_has_nonempty_assignment "VERCEL_PROJECT_ID" "$LAUNCH_ENV_FILE"; then
        launch_project="true"
    fi
    if env_file_has_nonempty_assignment "VERCEL_ORG_ID" "$LAUNCH_ENV_FILE"; then
        launch_org="true"
    fi

    if configured_env_value "$current_project" && configured_env_value "$current_org"; then
        echo "  VERCEL_PROJECT: configured in current environment"
    elif [ "$launch_project" = "true" ] && [ "$launch_org" = "true" ]; then
        echo "  VERCEL_PROJECT: configured in launch env file"
    elif { configured_env_value "$current_project" || [ "$launch_project" = "true" ]; } && { configured_env_value "$current_org" || [ "$launch_org" = "true" ]; }; then
        echo "  VERCEL_PROJECT: configured across current environment and launch env file"
    else
        echo "  VERCEL_PROJECT: missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file"
    fi
}

env_or_launch_value() {
    var_name="$1"
    value="${!var_name:-}"
    if [ -n "$value" ]; then
        printf "%s" "$value"
        return
    fi
    env_file_assignment_value "$var_name" "$LAUNCH_ENV_FILE"
}

configured_scheduler_arg_value() {
    flag="$1"
    default="$2"
    plist="${LAUNCHAGENTS_DIR}/${SCHEDULER_LABEL}.plist"
    if [ ! -f "$plist" ]; then
        plist="${ROOT}/production/${SCHEDULER_LABEL}.plist"
    fi
    if [ ! -f "$plist" ]; then
        echo "$default"
        return
    fi
    python3 - "$plist" "$flag" "$default" <<'PY' 2>/dev/null
import plistlib
import sys

try:
    with open(sys.argv[1], "rb") as f:
        payload = plistlib.load(f)
except Exception:
    print(sys.argv[3])
    raise SystemExit

args = list(payload.get("ProgramArguments", []))
try:
    idx = args.index(sys.argv[2])
    print(args[idx + 1])
except Exception:
    print(sys.argv[3])
PY
}

configured_scheduler_broker() {
    configured_scheduler_arg_value "--broker" "paper"
}

configured_scheduler_account_notional() {
    configured_scheduler_arg_value "--account-notional" ""
}

truthy_config_value() {
    value=$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')
    case "$value" in
        1|true|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

load_runtime_composer_env() {
    RUNTIME_COMPOSER=$(env_or_launch_value "SOXL_RUNTIME_COMPOSER")
    RUNTIME_COMPOSER_PROFILE=""
    RUNTIME_SLOW_PROFILE=""
    RUNTIME_FAST_PROFILE=""
    RUNTIME_BULL_PARTICIPATION_GUARD="false"
    RUNTIME_COMPOSER_MISSING=""

    if [ -z "$RUNTIME_COMPOSER" ] || [ "$RUNTIME_COMPOSER" = "none" ]; then
        RUNTIME_COMPOSER_STATE="default"
        return
    fi

    RUNTIME_COMPOSER_PROFILE=$(env_or_launch_value "SOXL_RUNTIME_COMPOSER_PROFILE")
    RUNTIME_SLOW_PROFILE=$(env_or_launch_value "SOXL_RUNTIME_SLOW_PROFILE")
    RUNTIME_FAST_PROFILE=$(env_or_launch_value "SOXL_RUNTIME_FAST_PROFILE")
    RUNTIME_BULL_PARTICIPATION_GUARD=$(env_or_launch_value "SOXL_RUNTIME_BULL_PARTICIPATION_GUARD")
    [ -z "$RUNTIME_BULL_PARTICIPATION_GUARD" ] && RUNTIME_BULL_PARTICIPATION_GUARD="false"

    [ -z "$RUNTIME_COMPOSER_PROFILE" ] && RUNTIME_COMPOSER_MISSING="${RUNTIME_COMPOSER_MISSING} SOXL_RUNTIME_COMPOSER_PROFILE"
    [ -z "$RUNTIME_SLOW_PROFILE" ] && RUNTIME_COMPOSER_MISSING="${RUNTIME_COMPOSER_MISSING} SOXL_RUNTIME_SLOW_PROFILE"
    [ -z "$RUNTIME_FAST_PROFILE" ] && RUNTIME_COMPOSER_MISSING="${RUNTIME_COMPOSER_MISSING} SOXL_RUNTIME_FAST_PROFILE"
    if [ -n "$RUNTIME_COMPOSER_MISSING" ]; then
        RUNTIME_COMPOSER_MISSING=$(printf "%s" "$RUNTIME_COMPOSER_MISSING" | sed 's/^ //; s/ /,/g')
        RUNTIME_COMPOSER_STATE="incomplete"
        return
    fi

    RUNTIME_COMPOSER_STATE="enabled"
}

print_scheduler_private_env() {
    echo "--- Scheduler Private Env ---"
    if [ -f "$LAUNCH_ENV_FILE" ]; then
        echo "  launch env file: present path=${LAUNCH_ENV_FILE}"
    else
        echo "  launch env file: missing path=${LAUNCH_ENV_FILE}"
    fi

    if configured_env_value "${VERCEL_TOKEN:-}"; then
        echo "  VERCEL_TOKEN: configured in current environment"
    elif env_file_has_nonempty_assignment "VERCEL_TOKEN" "$LAUNCH_ENV_FILE"; then
        echo "  VERCEL_TOKEN: configured in launch env file"
    else
        echo "  VERCEL_TOKEN: missing in current environment and launch env file"
    fi
    print_vercel_project_env_status
    SCHEDULER_BROKER=$(configured_scheduler_broker)
    echo "  scheduled broker: ${SCHEDULER_BROKER}"
    SCHEDULER_ACCOUNT_NOTIONAL=$(configured_scheduler_account_notional)
    if [ -n "$SCHEDULER_ACCOUNT_NOTIONAL" ]; then
        echo "  account notional: ${SCHEDULER_ACCOUNT_NOTIONAL}"
    fi

    load_runtime_composer_env
    if [ "$RUNTIME_COMPOSER_STATE" = "default" ]; then
        echo "  runtime composer: default single-profile path"
        return
    fi

    if [ "$RUNTIME_COMPOSER_STATE" = "incomplete" ]; then
        echo "  runtime composer: incomplete composer=${RUNTIME_COMPOSER} missing=${RUNTIME_COMPOSER_MISSING}"
        return
    fi

    echo "  runtime composer: enabled composer=${RUNTIME_COMPOSER} profile=${RUNTIME_COMPOSER_PROFILE} slow=${RUNTIME_SLOW_PROFILE} fast=${RUNTIME_FAST_PROFILE} bull_guard=${RUNTIME_BULL_PARTICIPATION_GUARD}"
}

json_field() {
    payload="$1"
    dotted_path="$2"
    SOXL_JSON_FIELD_PAYLOAD="$payload" python3 - "$dotted_path" <<'PY' 2>/dev/null
import json
import os
import sys

path = sys.argv[1].split(".")
text = os.environ.get("SOXL_JSON_FIELD_PAYLOAD", "")
start = text.find("{")
end = text.rfind("}")
if start < 0 or end < start:
    sys.exit(0)
try:
    obj = json.loads(text[start : end + 1])
except Exception:
    sys.exit(0)
value = obj
for part in path:
    if not isinstance(value, dict) or part not in value:
        sys.exit(0)
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

json_array_csv() {
    payload="$1"
    dotted_path="$2"
    SOXL_JSON_FIELD_PAYLOAD="$payload" python3 - "$dotted_path" <<'PY' 2>/dev/null
import json
import os
import sys

path = sys.argv[1].split(".")
text = os.environ.get("SOXL_JSON_FIELD_PAYLOAD", "")
start = text.find("{")
end = text.rfind("}")
if start < 0 or end < start:
    sys.exit(0)
try:
    obj = json.loads(text[start : end + 1])
except Exception:
    sys.exit(0)
value = obj
for part in path:
    if not isinstance(value, dict) or part not in value:
        sys.exit(0)
    value = value[part]
if isinstance(value, list):
    print(",".join(str(item) for item in value))
PY
}

print_paper_health() {
    echo "--- Paper Trading Health ---"
    if [ ! -x "$HEALTH_PYTHON" ]; then
        echo "  unavailable: python not executable at ${HEALTH_PYTHON}"
        return
    fi

    load_runtime_composer_env
    SCHEDULER_BROKER=$(configured_scheduler_broker)
    echo "  broker target: ${SCHEDULER_BROKER}"
    paper_health_args=(production/soxl_paper_trading.py --healthcheck --skip-data-update --no-notify --broker "$SCHEDULER_BROKER")
    SCHEDULER_ACCOUNT_NOTIONAL=$(configured_scheduler_account_notional)
    if [ -n "$SCHEDULER_ACCOUNT_NOTIONAL" ]; then
        paper_health_args+=(--account-notional "$SCHEDULER_ACCOUNT_NOTIONAL")
    fi
    if [ "$RUNTIME_COMPOSER_STATE" = "enabled" ]; then
        paper_health_args+=(
            --runtime-composer "$RUNTIME_COMPOSER"
            --composer-profile "$RUNTIME_COMPOSER_PROFILE"
            --slow-profile "$RUNTIME_SLOW_PROFILE"
            --fast-profile "$RUNTIME_FAST_PROFILE"
        )
        if truthy_config_value "$RUNTIME_BULL_PARTICIPATION_GUARD"; then
            paper_health_args+=(--bull-participation-guard)
        fi
    fi

    health_output=$(
        cd "$ROOT" &&
        "$HEALTH_PYTHON" "${paper_health_args[@]}" 2>&1
    )
    data_ok=$(json_field "$health_output" "data.ok")
    last_date=$(json_field "$health_output" "data.last_date")
    expected=$(json_field "$health_output" "data.expected_last_trading_day")
    PAPER_EXPECTED_LAST_TRADING_DAY="$expected"
    stale_days=$(json_field "$health_output" "data.stale_days")
    model_ok=$(json_field "$health_output" "model.ok")
    model_name=$(json_field "$health_output" "model.name")
    SELECTED_MODEL_PROFILE=$(printf "%s" "$model_name" | awk '{print $1}')
    broker_ok=$(json_field "$health_output" "broker.ok")
    broker_msg=$(json_field "$health_output" "broker.msg")

    if [ -z "$last_date" ]; then
        echo "  data: unknown"
        DATA_GATE_STATUS="unknown"
        DATA_GATE_DETAIL="last_date=unknown expected=${expected:-unknown}"
    elif [ "$data_ok" = "true" ]; then
        echo "  data: ok last_date=${last_date} expected=${expected}"
        DATA_GATE_STATUS="ok"
        DATA_GATE_DETAIL="last_date=${last_date} expected=${expected}"
    else
        echo "  data: stale last_date=${last_date} expected=${expected} stale_days=${stale_days}"
        DATA_GATE_STATUS="stale"
        DATA_GATE_DETAIL="last_date=${last_date} expected=${expected} stale_days=${stale_days}"
    fi

    if [ -n "$model_name" ]; then
        if [ "$model_ok" = "true" ]; then
            echo "  model: ok ${model_name}"
            MODEL_GATE_STATUS="ok"
            MODEL_GATE_DETAIL="${model_name}"
        else
            echo "  model: failed ${model_name}"
            MODEL_GATE_STATUS="failed"
            MODEL_GATE_DETAIL="${model_name}"
        fi
    else
        echo "  model: unknown"
        MODEL_GATE_STATUS="unknown"
        MODEL_GATE_DETAIL="unknown"
    fi

    if [ -n "$broker_msg" ]; then
        if [ "$broker_ok" = "true" ]; then
            echo "  broker: ok ${broker_msg}"
            BROKER_GATE_STATUS="ok"
            BROKER_GATE_DETAIL="${broker_msg}"
        else
            echo "  broker: failed ${broker_msg}"
            BROKER_GATE_STATUS="failed"
            BROKER_GATE_DETAIL="${broker_msg}"
        fi
    else
        echo "  broker: unknown"
        BROKER_GATE_STATUS="unknown"
        BROKER_GATE_DETAIL="unknown"
    fi
}

print_data_registry_health() {
    echo "--- Data Registry Health ---"
    checker="${ROOT}/scripts/update_data_registry.py"
    if [ ! -f "$checker" ]; then
        echo "  unavailable: registry checker missing"
        DATA_REGISTRY_GATE_STATUS="unknown"
        DATA_REGISTRY_GATE_DETAIL="registry checker missing"
        return
    fi
    if [ ! -x "$HEALTH_PYTHON" ]; then
        echo "  unavailable: python not executable at ${HEALTH_PYTHON}"
        DATA_REGISTRY_GATE_STATUS="unknown"
        DATA_REGISTRY_GATE_DETAIL="python not executable at ${HEALTH_PYTHON}"
        return
    fi

    registry_output=$(
        cd "$ROOT" &&
        "$HEALTH_PYTHON" scripts/update_data_registry.py --check 2>&1
    )
    registry_status=$?
    printf "%s\n" "$registry_output" | sed 's/^/  /'
    detail=$(compact_text "$registry_output")
    [ -z "$detail" ] && detail="no output"
    if [ "$registry_status" -ne 0 ]; then
        echo "  status: failed"
        DATA_REGISTRY_GATE_STATUS="failed"
        DATA_REGISTRY_GATE_DETAIL="$detail"
    else
        echo "  status: ok"
        DATA_REGISTRY_GATE_STATUS="ok"
        DATA_REGISTRY_GATE_DETAIL="$detail"
    fi
}

print_ordinary_paper_proof_status() {
    profile="$1"
    selected_model="$2"
    executed="$3"
    execution_status="$4"

    ORDINARY_PAPER_PROOF_SELECTED_MODEL="$selected_model"
    ORDINARY_PAPER_PROOF_EXECUTION="${execution_status:-unknown}"

    if [ "$executed" = "true" ]; then
        ORDINARY_PAPER_PROOF_STATUS="ok"
        echo "  ordinary paper proof: ok latest_executed_profile=${profile} selected_model=${selected_model}"
    elif [ "$execution_status" = "skipped" ]; then
        ORDINARY_PAPER_PROOF_STATUS="ok"
        echo "  ordinary paper proof: ok latest_non_dry_profile=${profile} selected_model=${selected_model} execution=${execution_status}"
    elif [ "$execution_status" = "duplicate_skipped" ]; then
        ORDINARY_PAPER_PROOF_STATUS="pending"
        echo "  ordinary paper proof: pending latest_non_dry_profile=${profile} selected_model=${selected_model} execution=${execution_status}"
        echo "  ordinary paper proof detail: duplicate-skip proves same-date idempotent inference only; next new-date scheduled non-dry run must produce executed or skipped order proof"
    else
        ORDINARY_PAPER_PROOF_STATUS="pending"
        echo "  ordinary paper proof: pending latest_non_dry_profile=${profile} selected_model=${selected_model} execution=${execution_status:-unknown}"
    fi
}

print_postclose_proof_detail() {
    if [ -z "$ORDINARY_PAPER_PROOF_STATUS" ]; then
        return
    fi
    echo "  postclose proof: ordinary paper proof ${ORDINARY_PAPER_PROOF_STATUS} selected_model=${ORDINARY_PAPER_PROOF_SELECTED_MODEL} execution=${ORDINARY_PAPER_PROOF_EXECUTION}"
}

print_postclose_target_status() {
    postclose_log="$1"
    if [ "$RUNTIME_COMPOSER_STATE" != "enabled" ]; then
        return
    fi

    marker=$(postclose_last_success_slot_marker "$postclose_log")
    if [ -z "$marker" ]; then
        marker=$(postclose_last_run_marker "$postclose_log")
        if [ -z "$marker" ]; then
            echo "  postclose target: unknown latest_success_marker has no run-start marker"
            POSTCLOSE_TARGET_STATUS="unknown"
            POSTCLOSE_TARGET_DETAIL="latest_success_marker has no run-start marker"
            return
        fi

        if ! marker_has_text "$marker" "schedule_slot=postclose"; then
            echo "  postclose target: pending latest_success_marker lacks schedule_slot=postclose"
            POSTCLOSE_TARGET_STATUS="pending"
            POSTCLOSE_TARGET_DETAIL="latest_success_marker lacks schedule_slot=postclose"
            return
        fi
    fi

    if ! marker_has_postclose_trigger "$marker"; then
        echo "  postclose target: pending latest_success_marker lacks schedule_trigger=postclose"
        POSTCLOSE_TARGET_STATUS="pending"
        POSTCLOSE_TARGET_DETAIL="latest_success_marker lacks schedule_trigger=postclose"
        return
    fi

    if ! marker_in_schedule_window "$marker" "$POSTCLOSE_HHMM" "$POSTCLOSE_WINDOW_MINUTES"; then
        echo "  postclose target: pending latest_success_marker outside postclose window"
        POSTCLOSE_TARGET_STATUS="pending"
        POSTCLOSE_TARGET_DETAIL="latest_success_marker outside postclose window"
        return
    fi

    if marker_has_text "$marker" "runtime_composer=${RUNTIME_COMPOSER}" &&
        marker_has_text "$marker" "composer_profile=${RUNTIME_COMPOSER_PROFILE}" &&
        marker_has_text "$marker" "slow_profile=${RUNTIME_SLOW_PROFILE}" &&
        marker_has_text "$marker" "fast_profile=${RUNTIME_FAST_PROFILE}"; then
        echo "  postclose target: ok runtime_composer=${RUNTIME_COMPOSER} composer_profile=${RUNTIME_COMPOSER_PROFILE}"
        POSTCLOSE_TARGET_STATUS="ok"
        POSTCLOSE_TARGET_DETAIL="runtime_composer=${RUNTIME_COMPOSER} composer_profile=${RUNTIME_COMPOSER_PROFILE}"
    else
        echo "  postclose target: pending latest_success_marker lacks runtime_composer=${RUNTIME_COMPOSER} composer_profile=${RUNTIME_COMPOSER_PROFILE}"
        POSTCLOSE_TARGET_STATUS="pending"
        POSTCLOSE_TARGET_DETAIL="latest_success_marker lacks runtime_composer=${RUNTIME_COMPOSER} composer_profile=${RUNTIME_COMPOSER_PROFILE}"
    fi
}

print_latest_signal_artifact() {
    echo "--- Latest Signal Artifact ---"
    paper_dir="${ROOT}/runtime/soxl/paper_trading"
    if [ ! -d "$paper_dir" ]; then
        echo "  latest signal: unavailable"
        return
    fi
    latest_summary=$(find "$paper_dir" -mindepth 2 -maxdepth 2 -type f -name summary.json 2>/dev/null | sort | tail -1)
    if [ -z "$latest_summary" ] || [ ! -f "$latest_summary" ]; then
        echo "  latest signal: unavailable"
        return
    fi

    payload=$(cat "$latest_summary")
    date_value=$(json_field "$payload" "date")
    profile=$(json_field "$payload" "profile")
    dry_run=$(json_field "$payload" "dry_run")
    executed=$(json_field "$payload" "executed")
    execution_status=$(json_field "$payload" "execution_result.status")
    target=$(json_field "$payload" "signal.target_position")
    risk_gate=$(json_field "$payload" "risk_gate.approved")
    reconcile=$(json_field "$payload" "reconcile.action_required")
    rel_path=${latest_summary#"$ROOT"/}
    if [ "$risk_gate" = "true" ]; then
        risk_status="approved"
    elif [ "$risk_gate" = "false" ]; then
        risk_status="blocked"
    else
        risk_status="unknown"
    fi
    if [ -z "$reconcile" ]; then
        reconcile="unknown"
    fi
    selected_summary=$(
        if [ -n "${SELECTED_MODEL_PROFILE}" ]; then
            find "$paper_dir" -mindepth 2 -maxdepth 5 -type f -name summary.json 2>/dev/null | sort | while IFS= read -r candidate; do
                candidate_payload=$(cat "$candidate")
                candidate_profile=$(json_field "$candidate_payload" "profile")
                candidate_dry_run=$(json_field "$candidate_payload" "dry_run")
                if [ "$candidate_profile" = "$SELECTED_MODEL_PROFILE" ] && [ "$candidate_dry_run" = "false" ]; then
                    echo "$candidate"
                fi
            done | tail -1
        fi
    )

    echo "  latest signal: date=${date_value} profile=${profile} dry_run=${dry_run} executed=${executed}"
    echo "  target=${target} risk_gate=${risk_status} reconcile=${reconcile}"
    echo "  artifact=${rel_path}"
    if [ -n "${SELECTED_MODEL_PROFILE}" ] && [ -n "$profile" ] && [ "$dry_run" = "false" ]; then
        if [ "$profile" = "$SELECTED_MODEL_PROFILE" ]; then
            if [ "$executed" = "true" ]; then
                echo "  runtime proof: ok latest_executed_profile=${profile} selected_model=${SELECTED_MODEL_PROFILE}"
                print_ordinary_paper_proof_status "$profile" "$SELECTED_MODEL_PROFILE" "$executed" "$execution_status"
            elif [ "$risk_gate" = "true" ] && { [ "$execution_status" = "skipped" ] || [ "$execution_status" = "duplicate_skipped" ]; }; then
                echo "  runtime proof: ok latest_non_dry_profile=${profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${execution_status}"
                print_ordinary_paper_proof_status "$profile" "$SELECTED_MODEL_PROFILE" "$executed" "$execution_status"
            else
                echo "  runtime proof: pending latest_non_dry_profile=${profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${execution_status:-unknown}"
                print_ordinary_paper_proof_status "$profile" "$SELECTED_MODEL_PROFILE" "$executed" "$execution_status"
            fi
        elif [ "$executed" = "true" ]; then
            if [ -n "$selected_summary" ] && [ "$selected_summary" != "$latest_summary" ]; then
                selected_payload=$(cat "$selected_summary")
                selected_profile=$(json_field "$selected_payload" "profile")
                selected_executed=$(json_field "$selected_payload" "executed")
                selected_status=$(json_field "$selected_payload" "execution_result.status")
                selected_risk_gate=$(json_field "$selected_payload" "risk_gate.approved")
                selected_rel_path=${selected_summary#"$ROOT"/}
                if [ "$selected_executed" = "true" ]; then
                    echo "  runtime proof: ok latest_executed_profile=${selected_profile} selected_model=${SELECTED_MODEL_PROFILE}"
                    print_ordinary_paper_proof_status "$selected_profile" "$SELECTED_MODEL_PROFILE" "$selected_executed" "$selected_status"
                    echo "  runtime proof artifact=${selected_rel_path}"
                elif [ "$selected_risk_gate" = "true" ] && { [ "$selected_status" = "skipped" ] || [ "$selected_status" = "duplicate_skipped" ]; }; then
                    echo "  runtime proof: ok latest_non_dry_profile=${selected_profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${selected_status}"
                    print_ordinary_paper_proof_status "$selected_profile" "$SELECTED_MODEL_PROFILE" "$selected_executed" "$selected_status"
                    echo "  runtime proof artifact=${selected_rel_path}"
                else
                    echo "  runtime proof: pending latest_non_dry_profile=${selected_profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${selected_status:-unknown}"
                    print_ordinary_paper_proof_status "$selected_profile" "$SELECTED_MODEL_PROFILE" "$selected_executed" "$selected_status"
                    echo "  runtime proof artifact=${selected_rel_path}"
                fi
            else
                echo "  runtime proof: pending latest_executed_profile=${profile} selected_model=${SELECTED_MODEL_PROFILE}"
                if [ -n "$PAPER_EXPECTED_LAST_TRADING_DAY" ] && [ "$date_value" = "$PAPER_EXPECTED_LAST_TRADING_DAY" ]; then
                    echo "  runtime proof detail: latest executed date already matches expected data date; next non-dry-run execution must use selected model"
                fi
            fi
        fi
    fi
}

print_preopen_target_status() {
    preopen_log="$1"
    marker=$(preopen_last_success_target_marker "$preopen_log")
    if [ -z "$marker" ]; then
        marker=$(scheduler_last_run_marker "$preopen_log")
    fi
    if [ -z "$marker" ]; then
        echo "  preopen target: unknown latest_success_marker has no run-start marker"
        return
    fi

    missing=""
    marker_has_text "$marker" "schedule_slot=preopen" || missing="${missing} schedule_slot=preopen"
    marker_has_preopen_trigger "$marker" || missing="${missing} schedule_trigger=preopen"
    marker_has_text "$marker" "dry_run=True" || missing="${missing} dry_run=True"
    marker_has_text "$marker" "skip_data_update=True" || missing="${missing} skip_data_update=True"

    load_runtime_composer_env
    if [ "$RUNTIME_COMPOSER_STATE" = "enabled" ]; then
        marker_has_text "$marker" "runtime_composer=${RUNTIME_COMPOSER}" || missing="${missing} runtime_composer=${RUNTIME_COMPOSER}"
        marker_has_text "$marker" "composer_profile=${RUNTIME_COMPOSER_PROFILE}" || missing="${missing} composer_profile=${RUNTIME_COMPOSER_PROFILE}"
        marker_has_text "$marker" "slow_profile=${RUNTIME_SLOW_PROFILE}" || missing="${missing} slow_profile=${RUNTIME_SLOW_PROFILE}"
        marker_has_text "$marker" "fast_profile=${RUNTIME_FAST_PROFILE}" || missing="${missing} fast_profile=${RUNTIME_FAST_PROFILE}"
    fi

    if [ -n "$missing" ]; then
        missing=$(printf "%s" "$missing" | sed 's/^ //; s/ /,/g')
        echo "  preopen target: pending latest_success_marker lacks ${missing}"
        return
    fi

    if ! marker_in_schedule_window "$marker" "$PREOPEN_HHMM" "$PREOPEN_WINDOW_MINUTES"; then
        echo "  preopen target: pending latest_success_marker outside preopen window"
        return
    fi

    detail="schedule_slot=preopen schedule_trigger=preopen dry_run=True skip_data_update=True"
    if [ "$RUNTIME_COMPOSER_STATE" = "enabled" ]; then
        detail="${detail} runtime_composer=${RUNTIME_COMPOSER} composer_profile=${RUNTIME_COMPOSER_PROFILE}"
    fi
    echo "  preopen target: ok ${detail}"
}

print_preopen_status() {
    echo "--- Pre-open Run Status ---"
    preopen_log="${ROOT}/runtime/soxl/paper_trading/scheduler_preopen.log"
    if preopen_success_after_cutoff "$preopen_log"; then
        echo "  preopen status: success marker found"
        print_preopen_target_status "$preopen_log"
    elif preopen_current_success_outside_window_after_cutoff "$preopen_log"; then
        echo "  preopen status: due but current success marker is outside preopen window"
        print_preopen_target_status "$preopen_log"
    elif preopen_due_today; then
        if [ -f "$preopen_log" ] && grep -q "\[scheduler\] all done" "$preopen_log" 2>/dev/null; then
            echo "  preopen status: due but no current success marker found"
        else
            echo "  preopen status: due but no success marker found"
        fi
    else
        echo "  preopen status: pending until ${PREOPEN_TIME_LABEL} Mon-Fri Asia/Shanghai"
    fi
}

print_postclose_status() {
    echo "--- Post-close Run Status ---"
    postclose_log="${ROOT}/runtime/soxl/paper_trading/scheduler_postclose.log"
    if postclose_success_after_cutoff "$postclose_log"; then
        echo "  postclose status: success marker found"
        print_postclose_target_status "$postclose_log"
        print_postclose_proof_detail
    elif postclose_current_success_outside_window_after_cutoff "$postclose_log"; then
        echo "  postclose status: due but current success marker is outside postclose window"
        print_postclose_target_status "$postclose_log"
        print_postclose_proof_detail
    elif postclose_current_success_without_slot_after_cutoff "$postclose_log"; then
        echo "  postclose status: due but current success marker lacks schedule_slot=postclose"
        print_postclose_target_status "$postclose_log"
        print_postclose_proof_detail
    elif postclose_current_success_without_trigger_after_cutoff "$postclose_log"; then
        echo "  postclose status: due but current success marker lacks schedule_trigger=postclose"
        print_postclose_target_status "$postclose_log"
        print_postclose_proof_detail
    elif postclose_cached_dry_run_after_cutoff "$postclose_log"; then
        echo "  postclose status: due but only cached dry-run fallback completed"
    elif postclose_due_today; then
        if [ -f "$postclose_log" ] && grep -q "\[scheduler\] all done" "$postclose_log" 2>/dev/null; then
            echo "  postclose status: due but no current success marker found"
            print_postclose_proof_detail
        else
            echo "  postclose status: due but no success marker found"
        fi
    else
        echo "  postclose status: pending until ${POSTCLOSE_TIME_LABEL} Tue-Sat Asia/Shanghai"
    fi
}

print_dashboard_static_export() {
    echo "--- Dashboard Static Export ---"
    checker="${ROOT}/scripts/check_dashboard_static_export.py"
    if [ ! -f "$checker" ]; then
        echo "  unavailable: integrity script missing"
        return
    fi
    if [ ! -x "$HEALTH_PYTHON" ]; then
        echo "  unavailable: python not executable at ${HEALTH_PYTHON}"
        return
    fi

    dashboard_output=$(
        cd "$ROOT" &&
        "$HEALTH_PYTHON" scripts/check_dashboard_static_export.py 2>&1
    )
    dashboard_status=$?
    printf "%s\n" "$dashboard_output" | sed 's/^/  /'
    if [ "$dashboard_status" -ne 0 ]; then
        echo "  status: failed"
        DASHBOARD_GATE_STATUS="failed"
        DASHBOARD_GATE_DETAIL="$(compact_text "$dashboard_output")"
    else
        echo "  status: ok"
        DASHBOARD_GATE_STATUS="ok"
        DASHBOARD_GATE_DETAIL="$(compact_text "$dashboard_output" | sed 's/^ok //')"
    fi
}

compact_text() {
    printf "%s" "$1" | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//'
}

print_vercel_freshness() {
    payload="$1"
    evidence_generated_at=$(json_field "$payload" "static_export.generated_at")
    local_index="${ROOT}/dashboard_data/index.json"
    if [ ! -f "$local_index" ]; then
        echo "  freshness: unknown local dashboard index not found"
        return
    fi

    local_payload=$(cat "$local_index")
    local_generated_at=$(json_field "$local_payload" "generated_at")
    if [ -z "$local_generated_at" ] || [ -z "$evidence_generated_at" ]; then
        echo "  freshness: unknown local_generated_at=${local_generated_at:-unknown} evidence_generated_at=${evidence_generated_at:-unknown}"
    elif [[ "$local_generated_at" > "$evidence_generated_at" ]]; then
        echo "  freshness: stale local_generated_at=${local_generated_at} evidence_generated_at=${evidence_generated_at}"
    else
        echo "  freshness: current local_generated_at=${local_generated_at} evidence_generated_at=${evidence_generated_at}"
    fi
}

print_vercel_remote_deployment() {
    echo "--- Vercel Remote Deployment ---"
    evidence="${ROOT}/runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json"
    if [ ! -f "$evidence" ]; then
        echo "  status: unavailable"
        echo "  evidence: ${evidence#"$ROOT"/} not found"
        return
    fi

    payload=$(cat "$evidence")
    status=$(json_field "$payload" "status")
    auth_status=$(json_field "$payload" "auth.status")
    project_status=$(json_field "$payload" "project.status")
    env_file_present=$(json_field "$payload" "env_file.present")
    env_file_applied_keys=$(json_array_csv "$payload" "env_file.applied_keys")
    deployment_url=$(json_field "$payload" "deployment_url")
    dashboard_latest=$(json_field "$payload" "static_export.latest")
    paper_trading_end=$(json_field "$payload" "static_export.paper_trading_end")
    detail=$(json_field "$payload" "detail")
    rel_path=${evidence#"$ROOT"/}

    if [ -z "$status" ]; then
        status="unknown"
    fi
    if [ -z "$auth_status" ]; then
        auth_status="unknown"
    fi
    if [ -z "$project_status" ]; then
        project_status="unknown"
    fi
    if [ -z "$deployment_url" ]; then
        deployment_url="none"
    fi
    VERCEL_REMOTE_STATUS="$status"
    VERCEL_AUTH_STATUS="$auth_status"
    VERCEL_PROJECT_STATUS="$project_status"
    VERCEL_DEPLOYMENT_URL="$deployment_url"
    echo "  status: ${status}"
    echo "  auth: ${auth_status}"
    echo "  project: ${project_status}"
    if [ -n "$env_file_present" ]; then
        if [ "$env_file_present" = "true" ]; then
            env_file_status="present"
        else
            env_file_status="missing"
        fi
        [ -z "$env_file_applied_keys" ] && env_file_applied_keys="none"
        VERCEL_ENV_FILE_APPLIED_KEYS="$env_file_applied_keys"
        echo "  env_file: ${env_file_status} applied_keys=${env_file_applied_keys}"
    fi
    echo "  deployment_url: ${deployment_url}"
    if [ -n "$dashboard_latest" ] || [ -n "$paper_trading_end" ]; then
        echo "  dashboard_latest=${dashboard_latest} paper_trading_end=${paper_trading_end}"
    fi
    print_vercel_freshness "$payload"
    echo "  evidence=${rel_path}"
    if [ -n "$detail" ]; then
        echo "  detail: $(compact_text "$detail")"
    fi
}

next_run_day_label() {
    schedule="$1"
    cutoff_hhmm="$2"
    weekday_num=$(number_value "$HEALTH_WEEKDAY")
    hhmm_num=$(number_value "$HEALTH_HHMM")
    cutoff_num=$(number_value "$cutoff_hhmm")

    case "$schedule" in
        preopen)
            if [ "$weekday_num" -ge 1 ] && [ "$weekday_num" -le 5 ] && [ "$hhmm_num" -lt "$cutoff_num" ]; then
                echo "today"
            else
                echo "next eligible day"
            fi
            ;;
        postclose)
            if [ "$weekday_num" -ge 2 ] && [ "$weekday_num" -le 6 ] && [ "$hhmm_num" -lt "$cutoff_num" ]; then
                echo "today"
            else
                echo "next eligible day"
            fi
            ;;
        weekly)
            if [ "$weekday_num" -eq 6 ] && [ "$hhmm_num" -lt "$cutoff_num" ]; then
                echo "today"
            else
                echo "next Saturday"
            fi
            ;;
    esac
}

print_next_runs() {
    echo "--- Next Scheduled Runs ---"
    echo "  pre-open: $(next_run_day_label preopen 2100) 21:00 Mon-Fri Asia/Shanghai — US pre-open dry-run + dashboard + Vercel deploy attempt"
    echo "  post-close: $(next_run_day_label postclose "$POSTCLOSE_HHMM") ${POSTCLOSE_TIME_LABEL} Tue-Sat Asia/Shanghai — US post-close data + ${SCHEDULER_BROKER} simulated trading + dashboard + Vercel deploy attempt"
    echo "  weekly: $(next_run_day_label weekly 0700) 07:00 Sat Asia/Shanghai — weekly walk-forward after post-close refresh"
}

append_gate_issue() {
    issue="$1"
    if [ -z "$GOAL_GATE_ISSUES" ]; then
        GOAL_GATE_ISSUES="$issue"
    else
        GOAL_GATE_ISSUES="${GOAL_GATE_ISSUES},${issue}"
    fi
}

print_goal_gate_summary() {
    echo "--- Production Goal Gates ---"
    GOAL_GATE_ISSUES=""

    echo "  data: ${DATA_GATE_STATUS} ${DATA_GATE_DETAIL}"
    [ "$DATA_GATE_STATUS" != "ok" ] && append_gate_issue "data_freshness"

    echo "  data registry: ${DATA_REGISTRY_GATE_STATUS} ${DATA_REGISTRY_GATE_DETAIL}"
    [ "$DATA_REGISTRY_GATE_STATUS" != "ok" ] && append_gate_issue "data_registry_freshness"

    echo "  model: ${MODEL_GATE_STATUS} ${MODEL_GATE_DETAIL}"
    [ "$MODEL_GATE_STATUS" != "ok" ] && append_gate_issue "model_health"

    echo "  broker: ${BROKER_GATE_STATUS} ${BROKER_GATE_DETAIL}"
    [ "$BROKER_GATE_STATUS" != "ok" ] && append_gate_issue "broker_health"

    echo "  dashboard static: ${DASHBOARD_GATE_STATUS} ${DASHBOARD_GATE_DETAIL}"
    [ "$DASHBOARD_GATE_STATUS" != "ok" ] && append_gate_issue "dashboard_static_export"

    case "$VERCEL_REMOTE_STATUS" in
        deployed)
            vercel_gate_status="ok"
            ;;
        ready_for_deploy)
            vercel_gate_status="pending"
            append_gate_issue "scheduled_vercel_deploy"
            ;;
        blocked_no_credentials|blocked_no_project_link)
            vercel_gate_status="blocked"
            append_gate_issue "external_vercel_setup"
            ;;
        deploy_failed|static_export_failed)
            vercel_gate_status="blocked"
            append_gate_issue "vercel_deploy"
            ;;
        unavailable|unknown)
            vercel_gate_status="unknown"
            append_gate_issue "vercel_evidence"
            ;;
        *)
            vercel_gate_status="pending"
            append_gate_issue "vercel_deploy"
            ;;
    esac
    echo "  Vercel remote: ${vercel_gate_status} status=${VERCEL_REMOTE_STATUS} auth=${VERCEL_AUTH_STATUS} project=${VERCEL_PROJECT_STATUS} deployment_url=${VERCEL_DEPLOYMENT_URL} env_file_applied_keys=${VERCEL_ENV_FILE_APPLIED_KEYS}"

    if [ "$RUNTIME_COMPOSER_STATE" = "enabled" ]; then
        if [ -z "$POSTCLOSE_TARGET_DETAIL" ]; then
            POSTCLOSE_TARGET_STATUS="pending"
            POSTCLOSE_TARGET_DETAIL="no post-close target evidence"
        fi
        echo "  post-close composer target: ${POSTCLOSE_TARGET_STATUS} ${POSTCLOSE_TARGET_DETAIL}"
        [ "$POSTCLOSE_TARGET_STATUS" != "ok" ] && append_gate_issue "scheduled_postclose_target"
    else
        echo "  post-close composer target: not_applicable runtime_composer=${RUNTIME_COMPOSER_STATE}"
    fi

    if [ -z "$ORDINARY_PAPER_PROOF_STATUS" ]; then
        ordinary_status="unknown"
        ordinary_detail="selected_model=${SELECTED_MODEL_PROFILE:-unknown} execution=unknown"
    else
        ordinary_status="$ORDINARY_PAPER_PROOF_STATUS"
        ordinary_detail="selected_model=${ORDINARY_PAPER_PROOF_SELECTED_MODEL} execution=${ORDINARY_PAPER_PROOF_EXECUTION}"
    fi
    echo "  ordinary paper proof: ${ordinary_status} ${ordinary_detail}"
    [ "$ordinary_status" != "ok" ] && append_gate_issue "scheduled_paper_proof"

    if [ -z "$GOAL_GATE_ISSUES" ]; then
        echo "  overall: ok"
    else
        echo "  overall: pending ${GOAL_GATE_ISSUES}"
    fi
}

echo "=== SOXL Scheduler Health Check ==="
echo "Time: $(date)"
echo "Root: ${ROOT}"
echo ""

print_scheduler_service
echo ""

print_scheduler_private_env
echo ""

# 检查日志文件
echo "--- Log Files ---"
for log in runtime/soxl/paper_trading/scheduler_preopen.log runtime/soxl/paper_trading/scheduler_preopen.err runtime/soxl/paper_trading/scheduler_postclose.log runtime/soxl/paper_trading/scheduler_postclose.err runtime/soxl/wf_weekly.log runtime/soxl/wf_weekly.err; do
    path="${ROOT}/${log}"
    if [ -f "$path" ]; then
        size=$(file_size "$path")
        mtime=$(format_mtime "$path")
        echo "  $log: $size (last: $mtime)"
    else
        echo "  $log: $(missing_log_status "$log")"
    fi
done
echo ""

# Check the latest completion marker without classifying cached fallback as success.
echo "--- Last Completion Marker ---"
for log in runtime/soxl/paper_trading/scheduler_preopen.log runtime/soxl/paper_trading/scheduler_postclose.log; do
    path="${ROOT}/${log}"
    if [ -f "$path" ]; then
        bn=$(basename "$log" .log)
        last=$(grep -E "\[scheduler\] all done" "$path" 2>/dev/null | tail -1)
        if [ -n "$last" ]; then
            echo "  $bn: $last"
        else
            echo "  $bn: no success marker found"
        fi
    fi
done
weekly_log="${ROOT}/runtime/soxl/wf_weekly.log"
if [ -f "$weekly_log" ]; then
    last=$(grep -E "\[weekly-wf\] all done" "$weekly_log" 2>/dev/null | tail -1)
    if [ -n "$last" ]; then
        echo "  wf_weekly: $last"
    else
        echo "  wf_weekly: no success marker found"
    fi
else
    echo "  wf_weekly: pending until first weekly run"
fi
echo ""

print_paper_health
echo ""

print_data_registry_health
echo ""

print_latest_signal_artifact
echo ""

print_dashboard_static_export
echo ""

print_vercel_remote_deployment
echo ""

print_preopen_status
echo ""

print_postclose_status
echo ""

print_next_runs
echo ""

print_goal_gate_summary
