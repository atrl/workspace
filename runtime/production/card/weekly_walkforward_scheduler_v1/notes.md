# weekly_walkforward_scheduler_v1

Date: 2026-05-21

## Goal

Close the mismatch where scheduler health printed a weekly walk-forward next-run
line and checked `runtime/soxl/wf_weekly.log`, but the repo only owned pre-open
and post-close LaunchAgents. The legacy crontab weekly command also pointed at a
stale `/home/rziiiii/dev/workspace` path and the dropped `balanced_trade_lock`
profile.

## Change

- Added `production/run_soxl_weekly_wf.sh`.
- Added `production/com.soxl.weekly.wf.plist`.
- Updated `production/check_scheduler_health.sh` to check
  `com.soxl.weekly.wf` and `runtime/soxl/wf_weekly.err`.
- Updated `production/crontab.txt` to delegate weekly walk-forward to the
  repo-owned wrapper instead of the stale legacy command.

The wrapper defaults to:

- profile: `balanced_rule_residual_v22`
- family: `balanced`
- walk-forward output: `runtime/model/card/balanced_rule_residual_v22`
- overlay review:
  `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json`

It runs corrected walk-forward validation first, then refreshes the runtime
selection review with `--persist-model-promotion`.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py tests/test_scheduler_health.py -q
# 16 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.weekly.wf: loaded
# LaunchAgent com.soxl.weekly.wf plist: matches repo
# weekly: next Saturday 07:00 Sat Asia/Shanghai — weekly walk-forward after post-close refresh
```

## Decision

Weekly walk-forward is now a repo-owned scheduled surface, not just a health
display line or stale crontab note. The weekly job is installed locally and will
produce `runtime/soxl/wf_weekly.log` on the next Saturday run.
