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
HEALTH_POSTCLOSE_CUTOFF_EPOCH="${SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH:-}"
POSTCLOSE_HHMM="0615"
POSTCLOSE_TIME_LABEL="06:15"
SELECTED_MODEL_PROFILE=""
PAPER_EXPECTED_LAST_TRADING_DAY=""

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

postclose_due_today() {
    weekday_num=$(number_value "$HEALTH_WEEKDAY")
    hhmm_num=$(number_value "$HEALTH_HHMM")
    postclose_num=$(number_value "$POSTCLOSE_HHMM")
    [ "$weekday_num" -ge 2 ] && [ "$weekday_num" -le 6 ] && [ "$hhmm_num" -ge "$postclose_num" ]
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

postclose_success_after_cutoff() {
    path="$1"
    if [ ! -f "$path" ] || ! grep -q "^\[scheduler\] all done$" "$path" 2>/dev/null; then
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
            /\[scheduler\] run started at / { seen = 1; ok = 0; next }
            seen && /^\[scheduler\] all done$/ { ok = 1 }
            END { exit ok ? 0 : 1 }
        ' "$path"
        return
    fi
    return 0
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
                if (line != "") {
                    found = 1
                    exit
                }
            }
        }
        END { exit found ? 0 : 1 }
    ' "$env_file"
}

print_scheduler_private_env() {
    echo "--- Scheduler Private Env ---"
    if [ -f "$LAUNCH_ENV_FILE" ]; then
        echo "  launch env file: present path=${LAUNCH_ENV_FILE}"
    else
        echo "  launch env file: missing path=${LAUNCH_ENV_FILE}"
    fi

    if [ -n "${VERCEL_TOKEN:-}" ]; then
        echo "  VERCEL_TOKEN: configured in current environment"
    elif env_file_has_nonempty_assignment "VERCEL_TOKEN" "$LAUNCH_ENV_FILE"; then
        echo "  VERCEL_TOKEN: configured in launch env file"
    else
        echo "  VERCEL_TOKEN: missing in current environment and launch env file"
    fi
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

print_paper_health() {
    echo "--- Paper Trading Health ---"
    if [ ! -x "$HEALTH_PYTHON" ]; then
        echo "  unavailable: python not executable at ${HEALTH_PYTHON}"
        return
    fi

    health_output=$(
        cd "$ROOT" &&
        "$HEALTH_PYTHON" production/soxl_paper_trading.py --healthcheck --skip-data-update --no-notify 2>&1
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
    elif [ "$data_ok" = "true" ]; then
        echo "  data: ok last_date=${last_date} expected=${expected}"
    else
        echo "  data: stale last_date=${last_date} expected=${expected} stale_days=${stale_days}"
    fi

    if [ -n "$model_name" ]; then
        if [ "$model_ok" = "true" ]; then
            echo "  model: ok ${model_name}"
        else
            echo "  model: failed ${model_name}"
        fi
    else
        echo "  model: unknown"
    fi

    if [ -n "$broker_msg" ]; then
        if [ "$broker_ok" = "true" ]; then
            echo "  broker: ok ${broker_msg}"
        else
            echo "  broker: failed ${broker_msg}"
        fi
    else
        echo "  broker: unknown"
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
            elif [ "$risk_gate" = "true" ] && { [ "$execution_status" = "skipped" ] || [ "$execution_status" = "duplicate_skipped" ]; }; then
                echo "  runtime proof: ok latest_non_dry_profile=${profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${execution_status}"
            else
                echo "  runtime proof: pending latest_non_dry_profile=${profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${execution_status:-unknown}"
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
                    echo "  runtime proof artifact=${selected_rel_path}"
                elif [ "$selected_risk_gate" = "true" ] && { [ "$selected_status" = "skipped" ] || [ "$selected_status" = "duplicate_skipped" ]; }; then
                    echo "  runtime proof: ok latest_non_dry_profile=${selected_profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${selected_status}"
                    echo "  runtime proof artifact=${selected_rel_path}"
                else
                    echo "  runtime proof: pending latest_non_dry_profile=${selected_profile} selected_model=${SELECTED_MODEL_PROFILE} execution=${selected_status:-unknown}"
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

print_postclose_status() {
    echo "--- Post-close Run Status ---"
    postclose_log="${ROOT}/runtime/soxl/paper_trading/scheduler_postclose.log"
    if postclose_success_after_cutoff "$postclose_log"; then
        echo "  postclose status: success marker found"
    elif postclose_cached_dry_run_after_cutoff "$postclose_log"; then
        echo "  postclose status: due but only cached dry-run fallback completed"
    elif postclose_due_today; then
        if [ -f "$postclose_log" ] && grep -q "\[scheduler\] all done" "$postclose_log" 2>/dev/null; then
            echo "  postclose status: due but no current success marker found"
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
    else
        echo "  status: ok"
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
    if [ -z "$deployment_url" ]; then
        deployment_url="none"
    fi
    echo "  status: ${status}"
    echo "  auth: ${auth_status}"
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
    echo "  post-close: $(next_run_day_label postclose "$POSTCLOSE_HHMM") ${POSTCLOSE_TIME_LABEL} Tue-Sat Asia/Shanghai — US post-close data + paper trading + dashboard + Vercel deploy attempt"
    echo "  weekly: $(next_run_day_label weekly 0700) 07:00 Sat Asia/Shanghai — weekly walk-forward after post-close refresh"
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

print_latest_signal_artifact
echo ""

print_dashboard_static_export
echo ""

print_vercel_remote_deployment
echo ""

print_postclose_status
echo ""

print_next_runs
