#!/bin/bash
# SOXL scheduler health check — 查看定时任务执行进度

set -u

ROOT="${SOXL_SCHEDULER_HEALTH_ROOT:-$(pwd)}"
OS_NAME="${SOXL_SCHEDULER_HEALTH_OS:-$(uname -s)}"
SCHEDULER_LABEL="com.soxl.daily.scheduler"

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

print_scheduler_service() {
    echo "--- Scheduler Service ---"
    if [ "$OS_NAME" = "Darwin" ]; then
        if ! command -v launchctl >/dev/null 2>&1; then
            echo "  LaunchAgent ${SCHEDULER_LABEL}: launchctl unavailable"
            return
        fi
        if launchctl list 2>/dev/null | grep -q "${SCHEDULER_LABEL}"; then
            echo "  LaunchAgent ${SCHEDULER_LABEL}: loaded"
        else
            echo "  LaunchAgent ${SCHEDULER_LABEL}: not loaded"
        fi
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

echo "=== SOXL Scheduler Health Check ==="
echo "Time: $(date)"
echo "Root: ${ROOT}"
echo ""

print_scheduler_service
echo ""

# 检查日志文件
echo "--- Log Files ---"
for log in runtime/soxl/paper_trading/scheduler_preopen.log runtime/soxl/paper_trading/scheduler_postclose.log runtime/soxl/wf_weekly.log; do
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

# 检查下次执行时间
echo "--- Next Scheduled Runs (today) ---"
echo "  21:00 Mon-Fri Asia/Shanghai — US pre-open dry-run + dashboard"
echo "  05:35 Tue-Sat Asia/Shanghai — US post-close data + paper trading + dashboard"
echo ""
echo "  (Saturday 07:00 Asia/Shanghai — weekly walk-forward after post-close refresh)"
