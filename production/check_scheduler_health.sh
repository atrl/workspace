#!/bin/bash
# SOXL scheduler health check — 查看定时任务执行进度

set -u

ROOT="${SOXL_SCHEDULER_HEALTH_ROOT:-$(pwd)}"
OS_NAME="${SOXL_SCHEDULER_HEALTH_OS:-$(uname -s)}"
HEALTH_PYTHON="${SOXL_SCHEDULER_HEALTH_PYTHON:-${ROOT}/qlib_venv/bin/python}"
SCHEDULER_LABEL="com.soxl.daily.scheduler"
PREOPEN_SCHEDULER_LABEL="com.soxl.preopen.scheduler"
LAUNCHAGENTS_DIR="${SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR:-${HOME:-}/Library/LaunchAgents}"
HEALTH_WEEKDAY="${SOXL_SCHEDULER_HEALTH_WEEKDAY:-$(date +%u)}"
HEALTH_HHMM="${SOXL_SCHEDULER_HEALTH_HHMM:-$(date +%H%M)}"

format_mtime() {
    path="$1"
    epoch=""
    if epoch=$(stat -f %m "$path" 2>/dev/null); then
        :
    elif epoch=$(stat -c %Y "$path" 2>/dev/null); then
        :
    else
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
    [ "$weekday_num" -ge 2 ] && [ "$weekday_num" -le 6 ] && [ "$hhmm_num" -ge 535 ]
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
        for label in "$PREOPEN_SCHEDULER_LABEL" "$SCHEDULER_LABEL"; do
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
    stale_days=$(json_field "$health_output" "data.stale_days")
    model_ok=$(json_field "$health_output" "model.ok")
    model_name=$(json_field "$health_output" "model.name")
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

print_postclose_status() {
    echo "--- Post-close Run Status ---"
    postclose_log="${ROOT}/runtime/soxl/paper_trading/scheduler_postclose.log"
    if [ -f "$postclose_log" ] && grep -q "\[scheduler\] all done" "$postclose_log" 2>/dev/null; then
        echo "  postclose status: success marker found"
    elif postclose_due_today; then
        echo "  postclose status: due but no success marker found"
    else
        echo "  postclose status: pending until 05:35 Tue-Sat Asia/Shanghai"
    fi
}

echo "=== SOXL Scheduler Health Check ==="
echo "Time: $(date)"
echo "Root: ${ROOT}"
echo ""

print_scheduler_service
echo ""

# 检查日志文件
echo "--- Log Files ---"
for log in runtime/soxl/paper_trading/scheduler_preopen.log runtime/soxl/paper_trading/scheduler_preopen.err runtime/soxl/paper_trading/scheduler_postclose.log runtime/soxl/paper_trading/scheduler_postclose.err runtime/soxl/wf_weekly.log; do
    path="${ROOT}/${log}"
    if [ -f "$path" ]; then
        size=$(file_size "$path")
        mtime=$(format_mtime "$path")
        echo "  $log: $size (last: $mtime)"
    else
        echo "  $log: NOT FOUND"
    fi
done
echo ""

# 检查最近一次成功执行
echo "--- Last Successful Run ---"
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
echo ""

print_paper_health
echo ""

print_postclose_status
echo ""

# 检查下次执行时间
echo "--- Next Scheduled Runs (today) ---"
echo "  21:00 Mon-Fri Asia/Shanghai — US pre-open dry-run + dashboard"
echo "  05:35 Tue-Sat Asia/Shanghai — US post-close data + paper trading + dashboard"
echo ""
echo "  (Saturday 07:00 Asia/Shanghai — weekly walk-forward after post-close refresh)"
