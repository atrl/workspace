#!/bin/bash
# SOXL scheduler health check — 查看定时任务执行进度

echo "=== SOXL Scheduler Health Check ==="
echo "Time: $(date)"
echo ""

# 检查 cron 服务
echo "--- Cron Service ---"
systemctl is-active cron 2>/dev/null || service cron status 2>/dev/null | head -1
echo ""

# 检查日志文件
echo "--- Log Files ---"
for log in runtime/soxl/paper_trading/scheduler_am.log runtime/soxl/paper_trading/scheduler_pm.log runtime/soxl/paper_trading/scheduler_eve.log runtime/soxl/wf_weekly.log; do
    if [ -f "$log" ]; then
        size=$(ls -lh "$log" | awk '{print $5}')
        mtime=$(ls -l --time-style=+%H:%M\ %Y-%m-%d "$log" | awk '{print $6, $7}')
        echo "  $log: $size (last: $mtime)"
    else
        echo "  $log: NOT FOUND"
    fi
done
echo ""

# 检查最近一次成功执行
echo "--- Last Successful Run ---"
for log in runtime/soxl/paper_trading/scheduler_am.log runtime/soxl/paper_trading/scheduler_pm.log runtime/soxl/paper_trading/scheduler_eve.log; do
    if [ -f "$log" ]; then
        bn=$(basename "$log" .log)
        last=$(grep -E "\[scheduler\] all done" "$log" 2>/dev/null | tail -1)
        if [ -n "$last" ]; then
            echo "  $bn: ✅ $last"
        else
            echo "  $bn: ⚠️  no success marker found"
        fi
    fi
done
echo ""

# 检查下次执行时间
echo "--- Next Scheduled Runs (today) ---"
echo "  09:30 — pre-market signal generation"
echo "  16:35 — post-market full pipeline"
echo "  21:00 — evening check (skip data update)"
echo ""
echo "  (Saturday 03:00 — weekly walk-forward)"
