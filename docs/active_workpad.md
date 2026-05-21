---
date: '2026-05-21'
title: 'Production Scheduler + Selective Release Execution Loop'
kind: workpad
status: active
direction: execution
---

# Production Scheduler + Paper Dashboard Loop

## Current Goal

1. Production scheduler, paper-trading state, dashboard rebuild, and Vercel static packaging were fixed and pushed in `c7cf210`.
2. Current production continuation: fix the local-time schedule so data refresh and paper execution run after the US close from an `Asia/Shanghai` environment.
3. Keep model optimization paused until the production loop can repeat data refresh -> inference -> paper state -> dashboard under the corrected schedule.
4. Resume model optimization from the SOXX-aware / selective-release frontier only after production timing is stable.

## 2026-05-20 Loop Evidence

### Root-Cause Evidence

- `runtime/soxl/paper_trading/scheduler.err` showed repeated scheduler failures where the child paper runner stderr contained `Too Many Requests`, `Rate limited`, or `No overlapping dates for SOXL/SOXX/VIX`.
- `production/soxl_daily_scheduler.py` only raised `RuntimeError("paper trading failed with exit code 1")`, so the cached-data retry branch could not see the child stderr and did not run.
- Current data state before local sync:
  - canonical panel: `4071` rows, ending `2026-05-15`
  - qlib calendar: `4071` rows, ending `2026-05-15`
  - raw SOXL/SOXX/VIX mirrors: `4057` rows, ending `2026-04-27`
- `scripts/build_backtest_dashboard.py` scanned `runtime/soxl/paper_trading/`, but skipped all CSVs with fewer than `10` rows, so a new simulated account could not appear in the dashboard.
- The paper dashboard append path used the pre-order account snapshot after order placement, allowing impossible rows such as `action=BUY` with `position=0.0`.

### Changes Made

- `production/soxl_daily_scheduler.py`
  - Added `SchedulerStepError` carrying child stdout/stderr.
  - Added transient data-source classification for rate limits, DNS/network failures, and no-overlap fetches.
  - Cached-data retry now triggers from child stderr and returns normally on success.
- `scripts/build_backtest_dashboard.py`
  - Keeps short histories for `paper_trading` sources while preserving the `len(raw_rows) >= 10` filter for post-train review histories.
- `production/soxl_paper_trading.py`
  - Added `_append_execution_daily_csv()` so cumulative paper CSV rows use the broker snapshot after execution.
- Data surfaces:
  - Ran local-only sync from the canonical panel.
  - Raw SOXL/SOXX/VIX mirrors, canonical panel, and qlib calendar now all end at `2026-05-15`.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_update_market_data.py tests/test_dashboard.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
```

Observed:

- Focused scheduler/paper/update/dashboard tests: `13 passed`.
- Cached-data scheduler dry-run completed with `as_of=2026-05-15` and wrote a dry-run paper artifact for signal date `2026-05-14`.
- The dry-run was blocked by risk gate because the auto-selected `balanced` actor has `walk-forward promotion gate not evaluated` and target step `0.850 > 0.350`.
- Dashboard rebuild wrote `25` versions and now includes `paper_trading` in `dashboard_data/index.json`.

### Continuation Evidence

- `auto_select_profile(profile_family="balanced")` previously selected `balanced` because `best_model_for_family()` fell back to `soxl_balanced_actor.pt` when no promoted candidate was found; runtime then rejected the selected model as `walk-forward promotion gate not evaluated`.
- `best_model_for_family()` now refuses unpromoted fallback models and only returns actors whose promotion sidecar passes `promotion_gate_status()`.
- Local promotion evidence for `balanced_rule_residual_v19` was aligned with the registry intent (`status=canonical`, `wf_status=completed`), so cached-data inference now auto-selects `balanced_rule_residual_v19`.
- Paper mode now persists `PaperBroker` cash/shares in `runtime/soxl/paper_trading/paper_broker_state.json` and clips large first-step target changes to the runtime step limit instead of blocking a cold account forever.
- The daily drawdown circuit-breaker path now replaces the frozen risk-gate decision with a blocked copy instead of mutating the dataclass in place.
- One local non-dry-run paper execution filled a clipped BUY from `0%` toward `35%`, appended the cumulative dashboard CSV from post-execution broker state, and persisted broker state; a follow-up dry-run read the realized `33.5%` position and approved the next step toward `65%`.
- `production/soxl_daily_scheduler.py --smoke` now runs cached-data dry-run inference plus dashboard rebuild without notifications or order placement.
- Static dashboard packaging is Vercel-ready: `dashboard.html`, `dashboard_data/index.json`, `dashboard_data/model_versions.json`, and 25 `dashboard_data/versioned/*/SOXL.json` files are trackable; `vercel.json` rewrites `/` to `/dashboard.html` and sets revalidation headers for dashboard data.

Additional verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_best_model_for_family_refuses_unpromoted_fallback tests/test_runtime.py::test_best_model_for_family_selects_highest_promoted_score -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_risk_gate_clips_large_position_step_when_enabled tests/test_paper_trading.py::test_paper_broker_state_roundtrip tests/test_paper_trading.py::test_post_execution_audit_uses_realized_paper_position tests/test_paper_trading.py::test_block_risk_decision_preserves_frozen_decision_contract -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_smoke_mode_runs_cached_dry_run_with_dashboard tests/test_vercel_dashboard.py -q
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
curl -I http://127.0.0.1:8766/dashboard.html
curl -I http://127.0.0.1:8766/dashboard_data/index.json
```

Final verification:

- Focused regression set: `29 passed`.
- `production/soxl_daily_scheduler.py --smoke`: completed cached-data inference, auto-selected `balanced_rule_residual_v19`, approved the dry-run risk gate, and rebuilt `dashboard_data/` with `25` versions.
- `scripts/check_harness.py`: `passed=71 failures=0`.
- Local HTTP checks returned `200 OK` for `dashboard.html` and `dashboard_data/index.json`.
- Browser screenshot verification was attempted through Computer Use, but macOS Accessibility/Screen Recording permissions were still pending; no screenshot evidence was produced in this pass.

## Next Action

1. Production-loop evidence is committed and pushed in `c7cf210`.
2. Current active loop is the production schedule timezone guard below.

## 2026-05-21 Production Schedule Timezone Guard

### Root Cause

- The active environment is `Asia/Shanghai`, but the SOXL cron and macOS
  LaunchAgent schedule still used local `09:30` and `16:35` rows while describing
  them as US pre-open / post-close.
- Local `16:35` China time is before the US close, so the data refresh could run
  in a structurally stale provider window.
- The existing `21:00` local row was also a non-dry paper run, which could
  duplicate simulated orders on the same `as_of` date.

### Changes

- `production/crontab.txt`
  - `21:00 Mon-Fri Asia/Shanghai`: cached-data `--dry-run` pre-open inference
    plus dashboard.
  - `05:35 Tue-Sat Asia/Shanghai`: full post-close data refresh, paper trading,
    and dashboard.
  - weekly walk-forward moved to `07:00 Sat Asia/Shanghai`, after post-close
    refresh.
- `production/com.soxl.daily.scheduler.plist`
  - full post-close LaunchAgent moved to `05:35 Tue-Sat`.
- `production/check_scheduler_health.sh` and `production/FUTU_SEMI_AUTO.md`
  - schedule text and log names now match the corrected local-time plan.
- `tests/test_production_schedule.py`
  - parses cron and LaunchAgent config and blocks stale local `09:30` / `16:35`
    SOXL scheduler rows.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py -q
# 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_production_schedule.py tests/test_dashboard.py -q
# 20 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update
# completed; selected balanced_rule_residual_v19; rebuilt dashboard_data with 25 versions

curl -I http://127.0.0.1:8767/dashboard.html
curl -I http://127.0.0.1:8767/dashboard_data/index.json
# both returned 200 OK
```

Visual evidence:

- `runtime/production/card/scheduler_timezone_v1/dashboard_visual_2026-05-21.png`
  was captured through Playwright using local Chrome at `1440 x 1000`.
- Pixel check: `94.84%` non-white pixels, so the screenshot is not an empty
  render.
- Computer Use remained blocked by macOS Accessibility / Screen Recording
  permission prompts, so the repo evidence uses Playwright/Chrome instead.

### Next Action

1. Commit and push the schedule guard.
2. Improve the scheduler health check so the corrected macOS LaunchAgent setup
   is observable before the first live run.
3. On the next live post-close window, run or inspect the `05:35 Tue-Sat`
   scheduler output and confirm whether online data extends beyond
   `2026-05-15`.

## 2026-05-21 Scheduler Health Observability

### Root Cause

- `production/check_scheduler_health.sh` still inspected Linux cron even though
  the active local schedule is installed as macOS LaunchAgent
  `com.soxl.daily.scheduler`.
- On the current host the service section printed blank, which hid whether the
  scheduler was actually loaded.
- The script also used GNU `ls --time-style`, which is not reliable on macOS
  once the corrected schedule logs exist.

### Changes

- `production/check_scheduler_health.sh`
  - detects Darwin and reports LaunchAgent loaded / not loaded through
    `launchctl list`
  - keeps a Linux cron branch for cron deployments
  - adds `SOXL_SCHEDULER_HEALTH_ROOT` and `SOXL_SCHEDULER_HEALTH_OS` seams for
    deterministic tests
  - uses portable `stat/date` helpers for log timestamps
- `tests/test_scheduler_health.py`
  - verifies macOS LaunchAgent reporting and success-marker extraction from
    `scheduler_preopen.log` / `scheduler_postclose.log`.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 21 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.daily.scheduler: loaded
# scheduler_preopen.log: NOT FOUND
# scheduler_postclose.log: NOT FOUND
```

### Next Action

1. Commit and push the health-observability guard.
2. Split the macOS LaunchAgent path so pre-open dry-run inference is installed,
   and align LaunchAgent log paths with scheduler health output.
3. Wait for or manually run the first real `05:35 Tue-Sat Asia/Shanghai`
   post-close scheduler pass, then use the health script to confirm log success
   markers and data freshness.

## 2026-05-21 macOS LaunchAgent Split

### Root Cause

- `production/com.soxl.daily.scheduler.plist` was correctly scheduled for
  `05:35 Tue-Sat Asia/Shanghai`, but still wrote legacy `scheduler.log` /
  `scheduler.err`.
- `production/check_scheduler_health.sh` expected `scheduler_postclose.log`, so
  the first live LaunchAgent run would not have appeared in the health output.
- The `21:00 Mon-Fri Asia/Shanghai` cached-data dry-run inference/dashboard path
  existed only in `production/crontab.txt`; the macOS recommended path had no
  pre-open LaunchAgent.

### Changes

- `production/com.soxl.daily.scheduler.plist`
  - post-close full pipeline
  - stdout/stderr: `scheduler_postclose.log` / `scheduler_postclose.err`
- `production/com.soxl.preopen.scheduler.plist`
  - new pre-open dry-run inference/dashboard LaunchAgent
  - args: `--broker paper --dry-run --no-notify --skip-data-update`
  - stdout/stderr: `scheduler_preopen.log` / `scheduler_preopen.err`
- `production/check_scheduler_health.sh`
  - reports both `com.soxl.preopen.scheduler` and `com.soxl.daily.scheduler`
  - lists pre-open and post-close stdout/stderr logs
- `production/FUTU_SEMI_AUTO.md`
  - documents installing/loading/unloading both LaunchAgents.

### External State

```bash
cp production/com.soxl.daily.scheduler.plist production/com.soxl.preopen.scheduler.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist
launchctl list | grep 'com.soxl.*scheduler'
# - 0 com.soxl.preopen.scheduler
# - 0 com.soxl.daily.scheduler
```

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py tests/test_scheduler_health.py -q
# 4 passed

launchctl start com.soxl.preopen.scheduler
# completed cached-data dry-run inference and dashboard rebuild

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler: loaded
# scheduler_preopen.log: 4.0K
# scheduler_preopen: [scheduler] all done
```

### Next Action

1. Commit and push the LaunchAgent split.
2. Guard the post-close cached fallback so transient data-source failure cannot
   mutate paper broker state with stale cached data.
3. Use the first real `05:35 Tue-Sat Asia/Shanghai` post-close run to prove
   online data refresh freshness and post-close paper/dashboard logging.

## 2026-05-21 Cached Fallback Dry-Run Guard

### Root Cause

- `production/soxl_daily_scheduler.py` retried transient data-source failures
  with `--skip-data-update`.
- The retry preserved the original non-dry-run execution mode.
- If the post-close data source failed, scheduler could therefore place a paper
  order using stale cached data.

### Change

- Cached fallback now appends both `--skip-data-update` and `--dry-run`.
- Retry label is `paper trading (cached dry-run)`.
- Success marker is `[scheduler] all done (with cached dry-run)`.
- Notification copy now states no order was placed.
- `tests/test_daily_scheduler.py` proves the retry command is dry-run.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_retries_data_source_failure_as_cached_dry_run -q
# initially failed on old behavior, then passed after the guard

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 22 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update
# completed cached-data dry-run; skipped order placement; rebuilt dashboard_data with 25 versions

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0
```

### Decision

- Fresh post-close data update succeeds: full paper trading can proceed.
- Transient data-source failure: keep cached signal/dashboard visibility, but do
  not move paper broker state.

## 2026-05-21 Paper Trading Freshness Health

### Root Cause

- `production/check_scheduler_health.sh` showed LaunchAgent load state and log
  success markers, but did not show whether the paper-trading loop itself was
  operating on fresh enough data.
- After the pre-open LaunchAgent was manually verified, the script could report a
  healthy scheduler surface while the canonical panel still ended at
  `2026-05-15`.
- The post-close online-refresh proof is still pending until the first real
  `05:35 Tue-Sat Asia/Shanghai` run.

### Change

- `production/soxl_paper_trading.py`
  - healthcheck now reports structured data freshness:
    `last_date`, `expected_last_trading_day`, and `stale_days`.
- `production/check_scheduler_health.sh`
  - added a `Paper Trading Health` section that runs the paper healthcheck,
    parses the JSON payload out of mixed stdout/stderr, and prints data, model,
    and broker status.
  - stale data is displayed as health evidence; the shell health script still
    exits successfully so it can be used for observability.
- `tests/test_paper_trading.py` and `tests/test_scheduler_health.py`
  - cover stale-day reporting and non-zero healthcheck parsing.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_freshness_status_reports_expected_date_and_lag -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_macos_launchagent_and_logs -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 23 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler: loaded
# scheduler_preopen: [scheduler] all done
# scheduler_postclose.log: NOT FOUND
# data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

### Decision

- The production health surface now distinguishes scheduler installation from
  paper-trading readiness.
- Current blocker remains online data freshness, not model selection or paper
  broker availability.
- Next action is to inspect the first real `05:35 Tue-Sat Asia/Shanghai`
  post-close scheduler pass and confirm whether data advances beyond
  `2026-05-15`; if the provider fails, the scheduler should fall back to cached
  dry-run only.

## 2026-05-21 Stale Execution Guard

### Root Cause

- The previous health work made stale data visible, but the paper-trading main
  path still did not enforce the freshness result before qlib/model/order flow.
- If a post-close data update returned successfully while leaving the canonical
  panel stale, a non-dry-run paper session could still continue toward broker
  state mutation.
- Current time during this pass was `2026-05-21 02:15 CST`, before the
  `05:35 Tue-Sat Asia/Shanghai` post-close window, so the live online-refresh
  proof is still pending.

### Change

- `production/soxl_paper_trading.py`
  - checks CSV freshness immediately after the data-update step and before
    `init_qlib()`.
  - blocks non-dry-run execution when data is stale.
  - permits stale data only for dry-run visibility and prints an explicit
    warning.
- `production/soxl_daily_scheduler.py`
  - classifies stale post-update data as a fallback condition.
  - retries with cached data as `--dry-run`, so paper broker state and
    cumulative CSV cannot move on stale data.
- `tests/test_paper_trading.py` and `tests/test_daily_scheduler.py`
  - cover the stale guard and scheduler fallback.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_blocks_non_dry_run_when_data_remains_stale -q
# red: reached qlib init before the guard
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_retries_stale_post_update_data_as_cached_dry_run -q
# red: scheduler exited instead of retrying
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 25 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update --skip-dashboard
# stale data warning, selected balanced_rule_residual_v19, skipped order placement

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
# first pass blocked: Cached data is stale for non-dry-run execution
# retry pass completed as cached dry-run
# [paper] dry-run: skipped order placement
# [scheduler] all done (with cached dry-run)

production/check_scheduler_health.sh
# data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

### Decision

- Stale data is now an execution blocker, not only an observability warning.
- The scheduler can still publish a cached signal/dashboard dry-run when fresh
  data is unavailable, but it cannot mutate paper broker state until data is
  fresh.
- Next proof remains the real `05:35 Tue-Sat Asia/Shanghai` post-close run.

## 2026-05-21 Freshness Market-Close Boundary

### Root Cause

- `_last_trading_day()` used the local `Asia/Shanghai` calendar date and a
  simple previous-day rule.
- At `2026-05-21 02:23 CST`, New York was still
  `2026-05-20 14:23 EDT`, so the `2026-05-20` US session was not closed.
- Healthcheck was therefore too strict before the `05:35 Tue-Sat
  Asia/Shanghai` post-close run, reporting `expected=2026-05-20` when the latest
  completed US trading day was still `2026-05-19`.

### Change

- `production/soxl_paper_trading.py`
  - freshness expectation now uses `America/New_York`.
  - current New York trading date becomes expected only after `17:30 ET`.
  - before that cutoff, freshness expects the previous weekday.
- `tests/test_paper_trading.py`
  - covers the `02:23 CST` pre-close boundary, the `05:35 CST` post-close
    boundary, and Monday pre-open previous-Friday behavior.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_last_trading_day_waits_for_new_york_post_close_cutoff tests/test_paper_trading.py::test_last_trading_day_uses_previous_friday_before_monday_us_close -q
# red: _last_trading_day() had no time injection and used local calendar date
# green: 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 27 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --healthcheck --skip-data-update --no-notify
# data: last_date=2026-05-15 expected_last_trading_day=2026-05-19 stale_days=4

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update --skip-dashboard
# stale data warning with expected=2026-05-19; skipped order placement

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
# first pass blocked stale non-dry-run data at expected=2026-05-19
# retry pass completed as cached dry-run
# [paper] dry-run: skipped order placement

production/check_scheduler_health.sh
# data: stale last_date=2026-05-15 expected=2026-05-19 stale_days=4
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

### Decision

- The health and execution guard now align with the local schedule: before the
  post-close window, they do not expect the still-open US trading day.
- The first real `05:35 Tue-Sat Asia/Shanghai` run remains the needed proof for
  online data refresh and non-dry-run paper execution.

## 2026-05-21 Provider Failover

### Root Cause

- `scripts/update_market_data.py` already supported `yfinance` and `yahoo_api`,
  but `production/soxl_paper_trading.py` hard-coded the yfinance path.
- At `2026-05-21 02:30 CST`, online dry-run probes still could not refresh data:
  - `yfinance`: `No overlapping dates for SOXL/SOXX/VIX`
  - `yahoo_api`: Yahoo HTTP `429 Too Many Requests`
- The live blocker remains provider availability, but the production runner
  should attempt both repo-owned providers before falling back to cached dry-run.

### Change

- `production/soxl_paper_trading.py`
  - added `--data-provider {auto,yfinance,yahoo_api}`.
  - default `auto` tries yfinance first, then yahoo_api for retryable
    provider failures.
- `production/soxl_daily_scheduler.py`
  - added matching `--data-provider` and passes it through to the paper runner.
- `production/FUTU_SEMI_AUTO.md`
  - documents provider auto fallback and explicit provider selection.
- Tests cover provider fallback and scheduler argument propagation.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_yahoo_api -q
# red: _run_data_update did not exist
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_passes_data_provider_to_paper_runner -q
# red: scheduler did not accept --data-provider
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 29 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-dashboard --data-provider auto
# yfinance failed: No overlapping dates for SOXL/SOXX/VIX
# yahoo_api failed: 429 Too Many Requests
# scheduler retried cached dry-run
# [paper] dry-run: skipped order placement
# [scheduler] all done (with cached dry-run)

PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
# wrote dashboard.html, dashboard_data/index.json, dashboard_data/model_versions.json, and 25 versioned exports

curl -I http://127.0.0.1:8768/dashboard.html
curl -I http://127.0.0.1:8768/dashboard_data/index.json
curl -I http://127.0.0.1:8768/dashboard_data/model_versions.json
# all returned 200 OK
```

### Decision

- Production online refresh now tries both supported providers by default.
- Current environment still cannot extend data beyond `2026-05-15`; provider
  failure is handled by cached dry-run, not paper-state mutation.
- The first real `05:35 Tue-Sat Asia/Shanghai` run remains the live proof point.

## 2026-05-21 Polygon + CBOE Data Recovery

### Root Cause

- `--data-provider auto` was more robust after adding `yahoo_api`, but both
  repo-owned Yahoo-family paths still failed in the current environment:
  `yfinance` returned `No overlapping dates for SOXL/SOXX/VIX`, and Yahoo chart
  API returned HTTP `429 Too Many Requests`.
- Futu OpenD was not reachable at `127.0.0.1:11111`, so it could not be used as
  the immediate market-data recovery path.
- A live probe showed Polygon could return SOXL/SOXX daily aggregates through
  `2026-05-19`, while Polygon VIX was not authorized. CBOE's public VIX history
  CSV returned VIX through `2026-05-19`.

### Change

- `scripts/update_market_data.py`
  - added provider `polygon`
  - fetches SOXL/SOXX OHLCV from Polygon using `POLYGON_API_KEY`
  - fetches VIX close from CBOE's public history CSV
- `production/soxl_paper_trading.py`
  - default `auto` provider order is now `yfinance`, `yahoo_api`, `polygon`
- `production/soxl_daily_scheduler.py`
  - accepts and forwards `--data-provider polygon`
- `production/soxl_daily_scheduler.py`
  - now replays child stderr even when a step exits successfully, so provider
    fallback messages remain visible in scheduler logs.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_update_market_data.py::test_update_master_csv_polygon_provider_uses_cboe_vix tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_polygon tests/test_daily_scheduler.py::test_scheduler_accepts_polygon_data_provider -q
# red first, then 3 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_run_replays_child_stderr_on_success -q
# red first, then 1 passed

PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --provider polygon --dry-run --skip-qlib
# before=4071 after=4073 new_rows=2 last_date=2026-05-19

PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --provider polygon
# canonical panel, raw SOXL/SOXX/VIX mirrors, and qlib calendar now end at 2026-05-19

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
# update_data_registry: OK symbols=6 panels=10 qlib=6

production/check_scheduler_health.sh
# data: ok last_date=2026-05-19 expected=2026-05-19
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --data-provider auto
# yfinance failed, yahoo_api failed, polygon succeeded
# data fresh: last_date=2026-05-19 expected=2026-05-19 stale_days=0
# as_of=2026-05-19, signal date=2026-05-18, target=50.00%, risk_gate approved=True
# dry-run skipped order placement and rebuilt dashboard_data with 25 versions

curl -I http://127.0.0.1:8769/dashboard.html
curl -I http://127.0.0.1:8769/dashboard_data/index.json
curl -I http://127.0.0.1:8769/dashboard_data/model_versions.json
# all returned 200 OK

npx -y playwright screenshot --channel chrome --viewport-size=1440,1000 --wait-for-timeout=2000 \
  'http://127.0.0.1:8769/dashboard.html?ts=polygon-cboe-v1' \
  runtime/production/card/polygon_cboe_provider_v1/dashboard_visual_2026-05-21.png
# captured 1440 x 1000 screenshot; non-white pixels: 99.35%
```

### Decision

- The production data freshness blocker is cleared for the currently expected
  completed US trading day (`2026-05-19`).
- The first natural `05:35 Tue-Sat Asia/Shanghai` post-close run still needs to
  prove the live schedule can advance after the next completed session.
- Model optimization remains paused until that live schedule proof is observed,
  then should resume from the SOXX-aware / selective-release frontier rather
  than reopening broad PPO tuning.

## 2026-05-21 Paper Full-Loop Recovery

### Baseline Before Loop

- Current time: `2026-05-21 03:01 CST`, before the natural `05:35
  Tue-Sat Asia/Shanghai` post-close LaunchAgent window.
- `production/check_scheduler_health.sh` showed:
  - `data: ok last_date=2026-05-19 expected=2026-05-19`
  - model: `balanced_rule_residual_v19`
  - broker: paper broker ready
- `runtime/soxl/paper_trading/paper_broker_state.json` was still dated
  `2026-05-14`, with `1800` shares and `33.53%` position.
- `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv` had only `2`
  cumulative paper rows and the dashboard paper version ended at `2026-05-14`.

### Command

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --no-notify --data-provider auto
```

Observed:

- `yfinance` failed with `No overlapping dates for SOXL/SOXX/VIX`.
- `yahoo_api` failed with Yahoo HTTP `429` on `^VIX`.
- `polygon` succeeded with `last_date=2026-05-19`.
- freshness gate passed: `last_date=2026-05-19 expected=2026-05-19 stale_days=0`.
- auto-selected model: `balanced_rule_residual_v19`.
- signal date: `2026-05-18`, target `50.00%`, price `$151.75`.
- risk gate approved.
- paper order submitted and filled.
- dashboard rebuilt.

### Evidence After Loop

- `runtime/soxl/paper_trading/paper_broker_state.json`
  - date: `2026-05-18`
  - shares: `3000`
  - position: `48.56%`
  - account value: `937490.75`
- `runtime/soxl/paper_trading/2026-05-18/reconcile.json`
  - expected position: `50.00%`
  - actual position: `48.56%`
  - action required: `none`
  - note: `position aligned`
- `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv`
  - rows increased from `2` to `3`
  - latest row: `2026-05-18`, action `BUY`, position `48.56%`
- `dashboard_data/index.json`
  - `paper_trading` window now ends at `2026-05-18` with `3` rows.
- Dashboard static verification after rebuild:
  - `dashboard.html`, `dashboard_data/index.json`, and
    `dashboard_data/model_versions.json` returned `200 OK` from local HTTP.
  - `runtime/production/card/paper_full_loop_v1/dashboard_visual_2026-05-21.png`
    captured at `1440 x 1000`; non-white pixels: `99.35%`.

### Decision

- The local paper broker state, cumulative CSV append, reconciliation, and
  dashboard paper version now advance under the full scheduler path with fresh
  data.
- The only remaining production proof is the natural `05:35 Tue-Sat
  Asia/Shanghai` LaunchAgent run after the next completed US session.
- Model optimization should still wait for that natural schedule proof, then
  resume from the SOXX-aware / selective-release frontier.

## 2026-05-20 Selective Release Order-Generation Diagnostic

### Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `selective_release_walkforward_v1`
- Role: `diagnose`
- Target window: `wf_2023_2024` eligible release days, with `wf_2022` as the defensive no-regression check
- Accepted trade-off: a small recovery-window drawdown increase is acceptable only if `changed_days > 0` while `wf_2022` remains unchanged
- Changed layer: `execution`
- External reference:
  - retained release gate family: `RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, `CICC_RETURN_VOLUME_SYNC`
  - prior fixed-actor counterfactual: `selective_release_recovery_v1`
  - walk-forward absorption evidence: `selective_release_walkforward_v1`
- Hypothesis:
  The retained release gates are firing on recovery days, but the order-generation path absorbs the overlay. A gate-ready rebalance-threshold bypass or minimum target lift should translate the same signal into changed orders if thresholding is the primary bottleneck.
- Acceptance criteria:
  - `wf_2023_2024 changed_days > 0` for at least one diagnostic variant
  - `wf_2022 eligible_days == 0` and unchanged return / MaxDD
  - no new PPO retrain or feature injection in this loop
- Expected trade-off:
  Higher recovery-window exposure may modestly widen `wf_2023_2024` drawdown if it finally creates orders.
- Failure mode:
  Eligible days remain fully absorbed, proving the blocker is deeper than the visible rebalance threshold or simple target lift.
- Comparison-table row key: `selective_release_order_generation_v1`

### Historical Contrast

1. Closest success case:
   `selective_release_recovery_v1` improved fixed full-window `test_2023_2024` from `57.12% / 0.719` to `82.10% / 0.877` without changing `test_2022`.
2. Closest failure case:
   `selective_release_walkforward_v1` found `6-18` eligible `wf_2023_2024` release days, but all existing gates had `changed_days=0`.
3. Why this loop is not a duplicate:
   It changes the execution mapping after gate readiness by adding explicit target lift and rebalance-threshold bypass variants; it does not rename the gate or retrain the actor.

### Changes Made

- `src/soxl_quant/backtest.py`
  - factored `selective_release_gate_ready()`
  - added optional `min_target_delta` support to `apply_selective_release_overlay()`
  - added `selective_release_rebalance_threshold()` so gate-ready overlays can bypass or scale the rebalance threshold
  - passed `current_pos` into the selective-release overlay path
- `scripts/evaluate_selective_release_overlay.py`
  - added `selective_release_recovery_threshold_bypass_v1`
  - added `selective_release_recovery_target_lift_v1`
- `scripts/evaluate_selective_release_walkforward.py`
  - added an order-generation diagnostic section to the durable review
- `tests/test_selective_release_overlay.py`
  - covered target-lift behavior and gate-scoped rebalance-threshold bypass

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py::test_selective_release_target_lift_can_force_change_from_current_position tests/test_selective_release_overlay.py::test_selective_release_rebalance_bypass_only_applies_when_gate_is_ready -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py tests/test_selective_release_walkforward.py -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

### Evidence

- Review: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`
- CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.csv`
- JSON: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.json`

| actor_window | diagnostic variant | eligible_days | changed_days | absorbed_days | return / max_dd | readout |
|---|---|---:|---:|---:|---:|---|
| `wf_2022` | `selective_release_recovery_threshold_bypass_v1` | 0 | 0 | 0 | `-34.11% / 38.27%` | defensive gate unchanged |
| `wf_2022` | `selective_release_recovery_target_lift_v1` | 0 | 0 | 0 | `-34.11% / 38.27%` | defensive gate unchanged |
| `wf_2023_2024` | `selective_release_recovery_threshold_bypass_v1` | 18 | 0 | 18 | `-20.21% / 38.54%` | threshold bypass still absorbed |
| `wf_2023_2024` | `selective_release_recovery_target_lift_v1` | 18 | 0 | 18 | `-20.21% / 38.54%` | target lift still absorbed |

### Decision

- Do **not** promote the selective-release overlay to runtime governance.
- Do **not** retrain another release-pack PPO branch yet.
- This loop rules out simple rebalance-threshold bypass and simple target lift as sufficient fixes; the next valid step is per-day execution-state instrumentation on eligible `wf_2023_2024` dates.

Next instrumentation should capture `current_pos`, target before/after overlay, `min_rebalance`, trade lock, buy cooldown, position cap state, and lot rounding before testing another execution rule.

## 2026-05-20 Eligible-Day Execution Trace

### Scope

This pass completes the instrumentation requested by the previous diagnostic. It does not test a new execution rule yet.

### Changes Made

- `src/soxl_quant/backtest.py`
  - added optional `trace_decisions=True` support to `backtest_profile()`
  - records per-day execution state without changing default backtest behavior
  - captures `current_pos`, target before/after overlay, guarded target, `min_rebalance`, trade lock, buy cooldown, cap config fields, lot rounding, order count, and primary blocker inputs
- `scripts/evaluate_selective_release_walkforward.py`
  - writes eligible-day trace artifacts for selective-release walk-forward variants
  - adds `primary_blocker` classification so downstream blockers do not obscure upstream guards
- `tests/test_selective_release_walkforward.py`
  - covers eligible-day trace filtering and primary-blocker priority

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_walkforward.py::test_eligible_decision_trace_rows_keep_gate_ready_execution_state -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_walkforward.py::test_execution_primary_blocker_prioritizes_upstream_guards tests/test_selective_release_walkforward.py::test_eligible_decision_trace_rows_keep_gate_ready_execution_state -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

### Evidence

- Review: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`
- Trace CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_execution_trace.csv`
- Trace JSON: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_execution_trace.json`

Key trace readout on actionable eligible `wf_2023_2024` days:

| variant | traced / eligible | primary blockers | readout |
|---|---:|---|---|
| `selective_release_recovery_threshold_bypass_v1` | `16 / 18` | `trade_lock=15`, `ordered=1` | threshold bypass reaches `min_rebalance=0`, but trade lock collapses the target back to current position on almost every actionable eligible day |
| `selective_release_recovery_target_lift_v1` | `16 / 18` | `trade_lock=12`, `min_rebalance=3`, `ordered=1` | target lift can raise target on some dates, but trade lock is still the dominant absorber; the remaining min-rebalance cases are already near the actor target |

The trace covers `16` of `18` eligible days because the first eligible dates are before the actor has enough lookback to make actionable decisions.

### Decision

- The blocker is now narrower than generic execution absorption: **release-gated trade lock is the dominant upstream absorber**.
- Do not tune release thresholds, lower global rebalance thresholds, or retrain PPO next.
- Next valid loop is a release-gated trade-lock bypass / shortening variant that only applies when the retained release gate is ready, while keeping `wf_2022` unchanged.

## 2026-05-20 Release-Gated Trade-Lock Bypass Loop

### Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `selective_release_walkforward_v1`
- Role: `diagnose`
- Target window: `wf_2023_2024`, with `wf_2022` as the hard no-regression check
- Accepted trade-off: order generation is necessary but not sufficient; `wf_2023_2024` return / MaxDD must not degrade materially after bypassing trade lock
- Changed layer: `execution`
- External reference:
  - retained release gate family: `RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, `CICC_RETURN_VOLUME_SYNC`
  - eligible-day trace showing trade lock as dominant absorber
- Hypothesis:
  If trade lock is the dominant absorber, a release-gated trade-lock bypass should turn eligible days into changed orders without reopening `wf_2022`.
- Acceptance criteria:
  - `wf_2023_2024 changed_days > 0`
  - `wf_2022` unchanged because eligible_days remains `0`
  - `wf_2023_2024` return or Sharpe improves without material MaxDD widening
- Expected trade-off:
  More trades and slightly higher average exposure are acceptable only if recovery quality improves.
- Failure mode:
  Bypass creates churn and exposure but worsens recovery-window return / drawdown.
- Comparison-table row key: `selective_release_trade_lock_bypass_v1`

### Changes Made

- `src/soxl_quant/backtest.py`
  - added `selective_release_trade_lock_days()` to gate trade-lock overrides through the retained release predicate
  - applies the override to both active trade-lock blocking and newly assigned trade-lock days
  - records `effective_trade_lock` in decision trace
- `scripts/evaluate_selective_release_overlay.py`
  - added `selective_release_recovery_trade_lock_bypass_v1`
- `scripts/evaluate_selective_release_walkforward.py`
  - includes the trade-lock bypass variant in diagnostic and trace summaries
- `tests/test_selective_release_overlay.py`
  - covers that trade-lock override only applies when the retained gate is ready

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py::test_selective_release_trade_lock_override_only_applies_when_gate_is_ready -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

### Evidence

| actor_window | variant | eligible_days | changed_days | trades | return | max_dd | decision |
|---|---|---:|---:|---:|---:|---:|---|
| `wf_2022` | `selective_release_recovery_trade_lock_bypass_v1` | 0 | 0 | 22 | `-34.11%` | `38.27%` | defensive gate unchanged |
| `wf_2023_2024` | `selective_release_recovery_trade_lock_bypass_v1` | 18 | 183 | 32 | `-21.43%` | `40.28%` | creates orders, worsens recovery window |

Baseline `wf_2023_2024` was `-20.21% / -0.277 / 38.54% / 22 trades`, so the bypass costs `-1.22pp` return and `+1.74pp` MaxDD while adding `10` trades.

### Decision

- Do **not** promote `selective_release_recovery_trade_lock_bypass_v1`.
- The trace diagnosis was correct: trade lock was blocking orders. The unrestricted bypass is still too blunt and turns signal activation into extra churn without improving recovery quality.
- Next valid variant must be narrower than full bypass, for example a release-gated trade-lock shortening or buy-cooldown-aware rule that only applies when current exposure is below the release target.

## 2026-05-20 Release-Gated Trade-Lock Shortening Loop

### Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `selective_release_walkforward_v1`
- Role: `diagnose`
- Target window: `wf_2023_2024`, with `wf_2022` as the hard no-regression check
- Accepted trade-off: lower churn than full bypass is useful only if recovery return / Sharpe improves without material MaxDD widening
- Changed layer: `execution`
- External reference:
  - eligible-day trace showing trade lock as dominant absorber
  - full trade-lock bypass failure showing unrestricted release creates churn
- Hypothesis:
  A release-gated trade-lock cap should allow some recovery-window orders while preserving the retained gate's `wf_2022` no-eligible defense.
- Acceptance criteria:
  - `wf_2023_2024 changed_days > 0`
  - `wf_2022` unchanged because eligible_days remains `0`
  - `wf_2023_2024` return or Sharpe improves without material MaxDD widening
  - trades remain materially below the full bypass variant
- Failure mode:
  Trade-lock shortening improves return but reopens drawdown / exposure risk enough to block promotion.
- Comparison-table row key: `selective_release_trade_lock_short_v1`

### Changes Made

- `src/soxl_quant/backtest.py`
  - added `trade_lock_cap` support to `selective_release_trade_lock_days()`
  - cap semantics only shorten an existing/new trade lock; they do not raise `0` lock days to `1`
- `scripts/evaluate_selective_release_overlay.py`
  - added `selective_release_recovery_trade_lock_short_v1` with retained gate, `rebalance_threshold_override=0.0`, and `trade_lock_cap=1`
- `scripts/evaluate_selective_release_walkforward.py`
  - includes the short variant in order-generation diagnostics and review output
- `tests/test_selective_release_overlay.py`
  - covers that `trade_lock_cap=1` lowers `8` days to `1` but preserves `0` days

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py::test_selective_release_trade_lock_override_only_applies_when_gate_is_ready -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

### Evidence

| actor_window | variant | eligible_days | changed_days | trades | return | sharpe | max_dd | avg_pos | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `wf_2022` | `selective_release_recovery_trade_lock_short_v1` | 0 | 0 | 22 | `-34.11%` | `-0.926` | `38.27%` | `0.3216` | defensive gate unchanged |
| `wf_2023_2024` baseline | `baseline` | 0 | 0 | 22 | `-20.21%` | `-0.277` | `38.54%` | `0.4764` | control |
| `wf_2023_2024` | `selective_release_recovery_trade_lock_short_v1` | 18 | 175 | 24 | `-17.87%` | `-0.229` | `40.65%` | `0.4620` | improves return, fails MaxDD gate |

Eligible-day trace for `wf_2023_2024`: short variant traced `16 / 18` eligible rows, with primary blockers `trade_lock=12`, `ordered=3`, `buy_cooldown=1`. Full bypass traced `ordered=11` and trades `32`, so the cap is cleaner but still not clean enough for promotion.

### Decision

- Do **not** promote `selective_release_recovery_trade_lock_short_v1`.
- The cap is a useful diagnostic: it improves `wf_2023_2024` return by `+2.34pp` with only `+2` trades and keeps `wf_2022` unchanged.
- It still widens `wf_2023_2024` MaxDD by `+2.11pp`, so the next valid rule must add a drawdown / exposure-gap guard before release-gated trade-lock shortening can become a governed runtime overlay.

## 2026-05-20 Guarded Release Trade-Lock Shortening Loop

### Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `selective_release_trade_lock_short_v1`
- Role: `diagnose`
- Target window: `wf_2023_2024`, with `wf_2022` as the hard no-regression check
- Accepted trade-off: keep the shortening variant's recovery improvement only if MaxDD returns near baseline
- Changed layer: `execution`
- External reference:
  - unguarded shortening improved recovery return but widened MaxDD
  - eligible-day trace showed release orders should only fire when current exposure is demonstrably below the release target and current account drawdown is still low
- Hypothesis:
  Guarding both target boost and execution relaxations with `max_current_drawdown=0.03` and `min_release_exposure_gap=0.10` should prevent late/high-drawdown release orders while retaining useful low-drawdown recovery entries.
- Acceptance criteria:
  - `wf_2022` unchanged because eligible_days remains `0`
  - `wf_2023_2024` return or Sharpe improves vs baseline
  - `wf_2023_2024` MaxDD is not materially wider than baseline
  - `overlay_execution_ready` is sparse and traceable on eligible days
- Closest success case:
  `selective_release_recovery_v1` fixed-actor overlay improved recovery while leaving `test_2022` unchanged.
- Closest failure case:
  `selective_release_recovery_trade_lock_short_v1` improved recovery but widened MaxDD.
- Why this is not a duplicate:
  This loop gates both the target boost and the trade-lock / rebalance relaxations with path drawdown and exposure-gap context; it is not another release-threshold tweak.
- Comparison-table row key: `selective_release_trade_lock_guarded_short_v1`

### Changes Made

- `src/soxl_quant/backtest.py`
  - added `selective_release_execution_guard_ready()`
  - extended guarded overlay semantics so `max_current_drawdown` / `min_release_exposure_gap` can block target boost, rebalance-threshold override, and trade-lock cap
  - added `overlay_execution_ready` and `overlay_exposure_gap` to decision trace
- `scripts/evaluate_selective_release_overlay.py`
  - added `selective_release_recovery_trade_lock_guarded_short_v1`
- `scripts/evaluate_selective_release_walkforward.py`
  - includes the guarded variant in diagnostics and reports execution-ready counts
- `tests/test_selective_release_overlay.py`
  - covers guard-ready logic, guarded execution relaxations, and guarded target boost blocking

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py::test_selective_release_execution_guard_requires_drawdown_room_and_exposure_gap tests/test_selective_release_overlay.py::test_selective_release_execution_relaxations_respect_guard_context -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py::test_selective_release_overlay_guard_blocks_boost_when_drawdown_or_gap_fails -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py tests/test_selective_release_walkforward.py -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

### Evidence

| actor_window | variant | eligible_days | execution_ready | changed_days | trades | return | sharpe | max_dd | avg_pos | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `wf_2022` | `selective_release_recovery_trade_lock_guarded_short_v1` | 0 | 0 | 0 | 22 | `-34.11%` | `-0.926` | `38.27%` | `0.3216` | defensive gate unchanged |
| `wf_2023_2024` baseline | `baseline` | 0 | 0 | 0 | 22 | `-20.21%` | `-0.277` | `38.54%` | `0.4764` | control |
| `wf_2023_2024` | `selective_release_recovery_trade_lock_guarded_short_v1` | 18 | 1 / 16 traced | 185 | 21 | `-19.63%` | `-0.233` | `42.91%` | `0.4833` | improves return slightly, fails MaxDD gate |

Trace readout: the guarded variant made only one eligible row execution-ready, but that early 2024 path change was still enough to raise `wf_2023_2024` MaxDD by `+4.37pp`. It is therefore not a viable promotion candidate.

### Decision

- Do **not** promote `selective_release_recovery_trade_lock_guarded_short_v1`.
- Do not continue threshold-only guard tuning on this mechanism; the single low-drawdown allowed release already creates an unacceptable later drawdown path.
- Next valid work should first attribute the drawdown path caused by the allowed release order, then test a distinct mechanism such as explicit post-release de-risking / target cap rather than another static entry guard.

## 2026-05-20 Guarded Release Drawdown Attribution Loop

### Scope

This loop does not add a new trading rule. It turns the guarded-shortening failure into a reproducible path-level attribution artifact so the next execution mechanism can target the drawdown cause directly.

### Changes Made

- `scripts/evaluate_selective_release_walkforward.py`
  - added `max_drawdown_event()` and `build_drawdown_attribution_rows()`
  - compares each walk-forward variant's account and position path against the same-window baseline
  - writes `selective_release_walkforward_drawdown_attribution.csv/json`
  - adds a `Drawdown Path Attribution` section to the walk-forward review
- `tests/test_selective_release_walkforward.py`
  - covers max-drawdown event extraction
  - covers release-order-to-trough attribution fields on a synthetic guarded path

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_walkforward.py::test_max_drawdown_event_identifies_peak_before_trough tests/test_selective_release_walkforward.py::test_drawdown_attribution_links_guarded_release_order_to_trough_path -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

### Evidence

- Attribution CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_drawdown_attribution.csv`
- Attribution JSON: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_drawdown_attribution.json`
- Updated review: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`

| actor_window | variant | delta_max_dd | release_order_dates | eligible_order_dates | peak -> trough | avg_pos_delta_release_to_trough | max_pos_delta_release_to_trough | trough account delta | readout |
|---|---|---:|---|---|---|---:|---:|---:|---|
| `wf_2023_2024` | `selective_release_recovery_trade_lock_bypass_v1` | `+1.74pp` | `2024-03-28;2024-04-01;2024-04-08;2024-04-09;2024-05-14;2024-05-15;2024-05-16;2024-05-22;2024-07-09;2024-10-08;2024-10-09` | same | `2024-06-18 -> 2024-12-10` | `+0.0636` | `+0.2937` | `-2.76%` | release orders precede wider drawdown |
| `wf_2023_2024` | `selective_release_recovery_trade_lock_short_v1` | `+2.11pp` | `2024-03-28;2024-05-13;2024-05-15` | same | `2024-07-10 -> 2024-12-19` | `+0.0762` | `+0.3196` | `+3.71%` | release orders precede wider drawdown |
| `wf_2023_2024` | `selective_release_recovery_trade_lock_guarded_short_v1` | `+4.37pp` | `2024-03-28` | `2024-03-28;2024-05-13;2024-07-09` | `2024-06-18 -> 2024-08-07` | `+0.0204` | `+0.3140` | `-0.48%` | release order precedes wider drawdown |

### Decision

- The static guarded-shortening rule is still dropped.
- The guarded path confirms that the first execution-ready release order on `2024-03-28` sits before the `2024-08-07` MaxDD trough; later gate-eligible orders also occur on `2024-05-13` and `2024-07-09`.
- The next valid execution test should be a distinct mechanism: post-release de-risking or release-state target cap. It should cap/reduce exposure after release activation if the path starts to draw down, rather than only deciding whether the entry is allowed.

## Prior Data/Release Goal

1. Refresh the repo-owned local data surfaces from the canonical SOXL/SOXX/VIX panel.
2. Run one narrow release/re-entry feature training loop on the retained residual frontier.

## Scope

This pass is a `data` revalidation plus one `factor` / `model_selection` ablation loop.
It is not a broad mode retune, reward retune, or macro-feature retry.

## Baselines And Role

- Canonical baseline: `balanced_rule_residual_v19`
- Direct comparison baseline: `balanced_rule_residual_v22`
- Role: `ablate`
- Target window: fixed-split `test_2022` / `test_2023_2024`, with `test_proxy` as the secondary check
- Accepted trade-off: allow modest `test_2020` giveback if recovery improves without breaking `test_2022` avg_pos / MaxDD gates
- Changed layer: `feature`
- External reference:
  - rulebase restart timing proxy `RB_COOLDOWN_PROXY`
  - transition release signal `TRANS_VIX_DIVERGENCE`
  - CICC low-frequency price-volume factor `CICC_RETURN_VOLUME_SYNC`
- Comparison-table row key: `balanced_rule_residual_v22_release_v1`

## Closest Historical Cases

1. Closest success case:
   `dwell_hold_v2` fixed-actor overlay improved recovery under low churn, and `balanced_rule_residual_v22` retained acceptable defensive behavior on fixed split.
2. Closest failure case:
   `balanced_rule_residual_v19_transition` broad transition-feature retrain and `balanced_macro_v1` broad macro pack both improved recovery narratives while reopening unwanted bear-market exposure / proxy degradation.
3. Why this loop is not a duplicate:
   It does not reopen the full `transition_signals` or `macro_signals` packs. It only curates three retained release/re-entry factors with prior standalone evidence into a narrow pack, keeping the retained v22 residual stack unchanged otherwise.

## Evidence

### Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
```

### Observed State Before Sync

- `data/raw/symbols/SOXL.csv` ended at `2026-04-01`
- `data/processed/panels/soxl_soxx_vix_full.csv` already ended at `2026-04-27`
- `data/qlib/soxl/calendars/day.txt` already ended at `2026-04-27`

### Observed State After Sync

- `data/raw/symbols/SOXL.csv` synced to `2026-04-27`
- `data/raw/symbols/SOXX.csv` created/synced to `2026-04-27`
- `data/raw/external/VIX.csv` created/synced to `2026-04-27`
- `data/qlib/soxl/` rebuilt from the canonical panel
- `data/datasets.yaml` regenerated successfully

### Live Refresh Blocker

Both `requests` and `yfinance` failed with DNS resolution errors against Yahoo endpoints in this automation environment. The blocker here is not rate limiting; it is lack of outbound name resolution / network access from the current run.

## 2026-05-10 Recheck

### Commands

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
paths = {
    "panel": Path("data/processed/panels/soxl_soxx_vix_full.csv"),
    "raw_soxl": Path("data/raw/symbols/SOXL.csv"),
    "raw_soxx": Path("data/raw/symbols/SOXX.csv"),
    "raw_vix": Path("data/raw/external/VIX.csv"),
}
for name, path in paths.items():
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    print(name, len(df), df[date_col].iloc[-1])
cal = Path("data/qlib/soxl/calendars/day.txt")
lines = [line.strip() for line in cal.read_text().splitlines() if line.strip()]
print("qlib_calendar", len(lines), lines[-1])
PY

PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import requests
try:
    requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/SOXL?interval=1d&period1=0&period2=1",
        timeout=20,
    ).raise_for_status()
    print("unexpected_success")
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

### Observed State

- Canonical panel, `raw/` mirrors, and qlib calendar all still end at `2026-04-27`.
- Live refresh remains blocked on `query1.finance.yahoo.com` name resolution.
- There is no newer local data to unlock inside this sandboxed automation run.

## Narrow Release-Pack Loop

- New profile: `balanced_rule_residual_v22_release_v1`
- Feature pack: `release_signals_v1`
  - `RB_COOLDOWN_PROXY`
  - `TRANS_VIX_DIVERGENCE`
  - `CICC_RETURN_VOLUME_SYNC`
- Hypothesis:
  A narrow release pack can teach the retained residual actor to re-risk earlier in recovery windows without repeating the broad-transition failure mode of lifting `test_2022` exposure.
- Acceptance criteria:
  - keep `test_2022 avg_pos <= 0.35`
  - keep `test_2022 maxdd <= 55%`
  - keep `test_proxy sharpe >= 0.50`
  - improve `test_2023_2024` return or Sharpe over `balanced_rule_residual_v22`
- Expected trade-off:
  mild `test_2020` giveback is acceptable if recovery improves materially without proxy drawdown blowout
- Failure mode:
  valid / proxy improve while `test_2022 avg_pos` or proxy MaxDD rises in the same way as prior broad transition / macro retrains
- Planned command:

```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v22_release_v1 --skip-data-refresh --seed 20260411
```

### Training Readout

- Training completed successfully after the terminal-date backtest fix in `src/soxl_quant/backtest.py`.
- Validation still liked the mechanism:
  - best valid snapshot: `return=61.0%`, `sharpe=1.662`, `maxdd=27.84%`
  - iter `20` was close (`60.3% / 1.684 / 26.89%`)
  - iter `40` degraded (`52.9% / 1.635 / 27.04%`)
- This is weak evidence that "train longer" is the answer. The mechanism signal is already visible by iter `0-20`; later training does not resolve the trade-off.

### Corrected Replay Backtest

Replay method: `scripts/run_segment_backtest.py` under the current harness after the terminal-date fix.

| window | `balanced_rule_residual_v22` | `balanced_rule_residual_v22_release_v1` | readout |
|---|---|---|---|
| `test_proxy` | `1099.1% / 0.976 / 53.21% / 0.498 / 169 trades` | `1350.3% / 1.010 / 53.26% / 0.520 / 172 trades` | proxy and upside improve; exposure lifts slightly |
| `test_2020` | `37.3% / 0.812 / 45.75% / 0.408 / 23 trades` | `42.7% / 0.860 / 44.95% / 0.430 / 22 trades` | mild bull improvement |
| `test_2022` | `-41.8% / -1.335 / 43.63% / 0.241 / 21 trades` | `-46.3% / -1.480 / 48.45% / 0.252 / 19 trades` | defense worsens in the wrong direction |
| `test_2023_2024` | `180.2% / 1.336 / 31.91% / 0.486 / 52 trades` | `234.3% / 1.479 / 32.80% / 0.509 / 47 trades` | recovery improves materially |

### Decision

- Keep `balanced_rule_residual_v22_release_v1` as **ablation evidence**, not a replacement candidate.
- The curated trio improves recovery, but it does so by teaching a broader `risk-on` bias rather than a selective release rule:
  - `test_2022 avg_pos` stays below the hard gate (`0.252 < 0.35`) but still drifts upward vs `v22`
  - `test_2022 maxdd` widens by `4.82pp`
  - proxy max drawdown is flat, so the pack is not adding cleaner timing, only more exposure
- Durable review: `runtime/model/card/balanced_rule_residual_v22_release_v1/v22_release_v1_backtest_review.md`

## Bottleneck Reset

Current project bottleneck is **not**:

- another macro feature injection
- another looser mode envelope
- another more-conservative risk-budget ablation
- another “maybe training not enough” guess

Current project bottleneck **is**:

- **selective re-entry / recovery timing after de-risking under low-churn constraints**
- **execution/runtime mapping quality, not generic threshold loosening**
- **factor usage quality, not raw factor availability**: the retained release factors are good enough to move behavior, but direct RL injection makes them act like ungated bullish pressure

Evidence trail:

- `risk_budget_usage_v1` has excellent 2022 defense but still recovers too slowly versus SOXL buy-and-hold
- `balanced_rule_residual_v19` / `v22` remain defensively acceptable but also lag 2023-2024 upside
- `balanced_macro_v1` improved recovery but worsened bear-market exposure, so it does not solve the real trade-off
- `runtime/execution/knowledge.md` already has `显式 release/re-entry overlay` as the next valid execution line
- `runtime/model/card/balanced_rule_residual_v19/v19_overlay_promotion_review.md` shows `fast_reentry` improves `2023-2024` return over `base` (`59.3%` vs `46.1%`) but still trails `dwell_hold_v2` on proxy Sharpe / drawdown (`0.949 / 51.06%` vs `1.064 / 44.42%`)
- `runtime/model/card/balanced_rule_residual_v19/v19_dwell_stress_merge_review.md` shows `regime_route_v1` further improves recovery (`61.8%`) and 2022 drawdown (`32.61%`) but at much higher churn (`464` trades vs `121` for `dwell_hold_v2`)
- `runtime/model/card/risk_budget_usage_v1/risk_budget_usage_v1_full_review.md` shows the composed counterfactual is strong, while the native PPO actor is much more conservative, so part of the bottleneck is training/runtime mismatch rather than missing macro context
- `runtime/model/card/balanced_rule_residual_v22_release_v1/v22_release_v1_backtest_review.md` shows the new release pack improves `2023-2024` materially but worsens `2022`, which means the current model is not learning a conditional release classifier from these signals alone

## Next Action

1. Keep local data sync on the canonical panel path; do not treat data freshness as the primary blocker until a network-enabled run can extend the canonical panel beyond `2026-04-27`.
2. Open the next narrow optimization loop around **gated selective release/re-entry classification**, using `RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, and `CICC_RETURN_VOLUME_SYNC` as release gates inside a low-churn fixed-actor overlay before trying another retrain.
3. Treat retraining as a second-stage step only if the fixed-actor gated release counterfactual improves `2023-2024` without reopening `2022` drawdown.

## 2026-05-14 Automation Run

### Data Refresh

Commands:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import requests
try:
    requests.get("https://query1.finance.yahoo.com/v8/finance/chart/SOXL?interval=1d&period1=0&period2=1", timeout=20).raise_for_status()
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

Observed state:

- Local canonical panel, raw mirrors, and qlib calendar remain aligned at `2026-04-27`.
- `data/datasets.yaml` was regenerated at `2026-05-14T02:02:18Z`.
- Live Yahoo access is still blocked by DNS / name resolution in this automation environment (`query1.finance.yahoo.com` cannot resolve).

### Bottleneck Brainstorm

The current bottleneck is not data freshness or insufficient PPO iterations. The
release pack already moved behavior in `balanced_rule_residual_v22_release_v1`,
but ungated RL injection acted like broad bullish pressure. The right next test
was therefore a fixed-actor execution separation test: keep the v22 actor fixed,
add only a low-churn release gate, and require the gate to use retained factor
evidence (`RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, `CICC_RETURN_VOLUME_SYNC`).

Feature distribution check:

- `CICC_RETURN_VOLUME_SYNC` is materially higher in `2023-2024` than `2022`
  (median `0.6238` vs `0.2756`), making it a useful recovery discriminator.
- `regime_risk` is lower in `2023-2024` than `2022` (median `0.3721` vs `0.6821`),
  so a `max_regime_risk` gate can avoid bear-market release.
- `RB_COOLDOWN_PROXY` is also stronger in `2023-2024` (median `0.1383`, p75 `0.3684`)
  than `2022` (median `0.0000`, p75 `0.1361`).

### Selective Release Overlay Loop

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_overlay.py --profile balanced_rule_residual_v22
```

Evidence:

- Review: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_review.md`
- CSV: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_results.csv`
- JSON: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_results.json`

Best counterfactual: `selective_release_recovery_v1`

| window | baseline v22 fixed actor | selective release | readout |
|---|---:|---:|---|
| `test_proxy` | `1132.19% / 1.043 / 56.48% / avg_pos 0.5226 / 172 trades` | `1415.10% / 1.105 / 54.30% / avg_pos 0.5230 / 171 trades` | better return, Sharpe, and drawdown |
| `test_2020` | `131.13% / 2.160 / 20.92% / avg_pos 0.5356 / 27 trades` | unchanged | no bull-market side effect |
| `test_2022` | `-30.97% / -0.764 / 40.64% / avg_pos 0.3304 / 21 trades` | unchanged | bear defense not reopened |
| `test_2023_2024` | `57.12% / 0.719 / 34.87% / avg_pos 0.5402 / 50 trades` | `82.10% / 0.877 / 36.95% / avg_pos 0.5510 / 50 trades` | recovery improved with modest drawdown cost |

Decision:

- Keep `selective_release_recovery_v1` as the retained execution counterfactual.
- Do not start another PPO retrain yet; this fixed-actor result is stronger evidence that the bottleneck is runtime gating / release classification.
- Next action is walk-forward actor validation for the same overlay, then convert to a governed execution overlay/profile only if the benefit survives.

## 2026-05-15 Automation Recheck

### Data Refresh

Commands:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import requests
try:
    requests.get("https://query1.finance.yahoo.com/v8/finance/chart/SOXL?interval=1d&period1=0&period2=1", timeout=20).raise_for_status()
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

Observed state:

- Local-only sync succeeded and rebuilt `data/qlib/soxl`.
- Canonical panel, raw SOXL/SOXX/VIX mirrors, and qlib calendar all remain aligned at `2026-04-27`.
- Live Yahoo refresh is still blocked by DNS / name resolution in this automation environment.

### Walk-Forward Selective Release Validation

Command:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

Evidence:

- Review: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`
- CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.csv`
- JSON: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.json`

Key result:

| actor_window | baseline | best prior gate | readout |
|---|---:|---:|---|
| `wf_2020` | `113.17% / 2.015 / 19.47% / avg_pos 0.5322` | unchanged | no release signal days |
| `wf_2022` | `-34.11% / -0.926 / 38.27% / avg_pos 0.3216` | unchanged | no release signal days, defense not reopened |
| `wf_2023_2024` | `-20.21% / -0.277 / 38.54% / avg_pos 0.4764` | unchanged | release signals fire, but execution absorbs the overlay |

Detailed readout:

- `selective_release_cicc_v1` has `16` eligible days in `wf_2023_2024`, but `0` changed days.
- `selective_release_tight_v1` has `6` eligible days in `wf_2023_2024`, but `0` changed days.
- `selective_release_recovery_v1` has `18` eligible days in `wf_2023_2024`, but `0` changed days.

Decision:

- Do **not** promote `selective_release_recovery_v1` into runtime/profile governance yet.
- The fixed full-window actor benefit from 2026-05-14 does not survive the walk-forward actor check as an executable change.
- The updated bottleneck is narrower: release factors can identify 2024 recovery days, but the current execution mapping can absorb the boost without changing actual orders.

Next action:

1. Keep data refresh on local canonical sync until a network-enabled run can extend beyond `2026-04-27`.
2. Diagnose target/position absorption on `wf_2023_2024` eligible days before retraining.
3. If the next fixed-actor test changes orders, constrain it to an execution mechanism such as target-lift vs rebalance-threshold bypass; do not add another ungated PPO feature branch.

---

## Archived Snapshot: 2026-04-28 Macro Evaluation

## 背景

在 `balanced` profile 基础上添加宏观领先指标特征包（credit spread, yield curve, DXY, oil），构建 `balanced_macro_v1`，与相同 seed（42）的 baseline `balanced` 做公平对比。

## 实验设计

| 维度 | balanced (baseline) | balanced_macro_v1 |
|------|---------------------|-------------------|
| 特征数 | 59 | 69 (+10 macro) |
| 网络 | MLP [128,64] | MLP [128,64] |
| Seed | 42 | 42 |
| 训练迭代 | 80 | 80 |
| 特征包 | — | `macro_signals` |

## 结果对比

| 指标 | balanced | balanced_macro_v1 | 差异 |
|------|----------|-------------------|------|
| **Valid best** | | | |
| Return | 55.7% | 60.6% | +4.9pp |
| Sharpe | 1.49 | 1.47 | -0.02 |
| MaxDD | 28.4% | 32.7% | +4.3pp |
| Score | **82.10** | **86.15** | +4.05 |
| **test_proxy** | | | |
| Return | **605.2%** | 450.3% | **-154.9pp** |
| Sharpe | **0.89** | 0.78 | **-0.11** |
| MaxDD | 55.3% | 58.3% | +3.0pp |
| avg_pos | 0.519 | 0.594 | +0.075 |
| **test_2020** | | | |
| Return | 126.4% | 144.4% | +18.0pp |
| Sharpe | 2.37 | 2.26 | -0.11 |
| MaxDD | 23.7% | 25.3% | +1.6pp |
| avg_pos | 0.547 | 0.576 | +0.029 |
| **test_2022** | | | |
| Return | -36.7% | -35.5% | +1.2pp |
| Sharpe | -1.23 | -1.00 | +0.23 |
| MaxDD | 38.8% | 39.5% | +0.7pp |
| **avg_pos** | **0.328** ✅ | **0.368** ❌ | **+0.040** |
| **test_2023_2024** | | | |
| Return | 62.4% | **82.5%** | **+20.1pp** |
| Sharpe | 0.79 | **0.87** | **+0.08** |
| MaxDD | 39.2% | 38.6% | -0.6pp |
| avg_pos | 0.560 | 0.620 | +0.060 |

## Promotion Gate

| 检查项 | balanced | balanced_macro_v1 |
|--------|----------|-------------------|
| test_2022_max_dd < 55% | 38.8% ✅ | 39.5% ✅ |
| test_2022_avg_pos < 0.35 | **0.328** ✅ | **0.368** ❌ |
| test_proxy_sharpe > 0.50 | 0.89 ✅ | 0.78 ✅ |
| **结果** | **eligible** | **BLOCKED** |

## 关键发现

1. **Macro 模型在 2022 bear market 更激进，而非更防御**：
   - test_2022 avg_pos: 0.368 vs 0.328（+12%）
   - 宏观指标（credit spread 扩大、yield curve 倒挂）理论上应暗示风险，但 RL 代理反而持仓更高
   - 这是 promotion gate 失败的根本原因

2. **Macro 模型在 2023-2024 recovery 期表现更好**：
   - Return: 82.5% vs 62.4%（+20pp）
   - 宏观特征可能帮助识别 bull market 的早期信号

3. **test_proxy 整体收益下降**：
   - 605.2% → 450.3%（-26%）
   - Sharpe: 0.89 → 0.78
   - 宏观特征在全周期引入了噪声，降低了整体效率

4. **Valid score 反而更高**：
   - 82.10 → 86.15
   - 说明宏观特征在 valid set（2015-2019）过拟合了

## 假设分析

为什么宏观特征没有帮助 bear market 防御？

| 假设 | 可能性 | 验证方式 |
|------|--------|----------|
| Macro 信号与 price action 相比太弱，被淹没 | 高 | 单独回测 macro 信号的 IC |
| Z-score 处理丢失了方向信息 | 中 | 尝试 raw level + delta 组合 |
| 代理需要更长时间学习宏观关联 | 低 | 已训练 80 iter，足够 |
| 宏观特征与现有特征高度相关 | 中 | 计算特征间相关性 |
| 需要 macro-aware position cap | 高 | 尝试将 macro 信号接入 position cap |

## 结论

`balanced_macro_v1` **不通过 promotion gate**，不建议替换 baseline。

宏观特征在当前形式下（z-score + delta 的 10 维输入）对 RL 代理是**噪声而非信号**。虽然 valid score 和 2023-2024 收益改善，但核心缺陷（bear market 防御能力下降）无法容忍。

## 下一步选项

1. **Option A: 放弃 macro 方向** — 资源投入回报比低，现有特征已足够
2. **Option B: 改进 macro 预处理** — 尝试 regime-aware encoding（bull/bear/neutral 标签代替连续 z-score）
3. **Option C: Macro-aware position cap** — 不直接输入 macro 特征到 policy，而是将 macro 信号接入 position cap 系统，硬性限制 bear market 最大仓位
4. **Option D: 单独验证 macro 信号 IC** — 先确认宏观指标本身有预测力，再决定如何接入

推荐：**Option D → 若 IC 验证通过 → Option C**

---

## Phase 1: IC 验证结果 ✅ 完成

### 方法
- 加载 SOXL 日度收益率（2020-01-03 ~ 2026-04-01，1569 个交易日）
- 构建 14 个宏观信号变体（raw level, z-score 252, delta 21, ROC 63）
- 计算信号(t) 与 forward return(t+1, +5, +10, +21) 的 Spearman rank correlation (IC)
- 额外按 bull/bear regime 拆分

### 关键结果

| 信号 | IC(1d) | IC(5d) | IC(21d) | IR(5d) | Pos Rate(5d) |
|------|--------|--------|---------|--------|--------------|
| MACRO_CREDIT_SPREAD | -0.003 | +0.024 | +0.006 | +0.40 | 49.8% |
| MACRO_YC_10Y2Y | +0.022 | +0.020 | +0.017 | +0.40 | 49.9% |
| MACRO_YC_10Y3M | -0.017 | -0.009 | +0.006 | -0.23 | 37.2% |
| MACRO_CREDIT_SPREAD_ZSCORE_252 | -0.015 | +0.012 | -0.002 | +0.18 | 50.1% |
| **MACRO_YC_10Y2Y_ZSCORE_252** | +0.016 | **+0.026** | **+0.027** | **+0.74** | **62.6%** |
| MACRO_YC_10Y3M_ZSCORE_252 | -0.023 | -0.012 | +0.018 | -0.28 | 30.5% |
| MACRO_DXY_ZSCORE_252 | +0.020 | +0.015 | +0.003 | +0.32 | 55.2% |
| **MACRO_OIL_ZSCORE_252** | **-0.039** | **-0.040** | -0.021 | **-1.00** | 12.5% |
| MACRO_CREDIT_SPREAD_DELTA_21 | -0.014 | +0.006 | +0.004 | +0.19 | 49.7% |
| MACRO_YC_10Y2Y_DELTA_21 | +0.005 | +0.006 | +0.021 | +0.14 | 50.1% |
| MACRO_CREDIT_SPREAD_ROC_63 | -0.007 | +0.013 | -0.014 | +0.27 | 50.1% |
| MACRO_YC_10Y2Y_ROC_63 | +0.016 | +0.005 | +0.015 | +0.11 | 28.0% |
| MACRO_DXY_ROC_63 | +0.017 | +0.021 | -0.012 | +0.45 | 53.4% |
| MACRO_OIL_ROC_63 | -0.031 | -0.028 | -0.015 | -0.44 | 29.0% |

### 发现

1. **所有信号的 |IC| < 0.04，属于极弱信号**
   - 最佳信号 MACRO_YC_10Y2Y_ZSCORE_252 的 IC(5d) = +0.026
   - 最佳信号 MACRO_OIL_ZSCORE_252 的 IC(1d) = -0.039
   - 因子投资中 |IC| < 0.05 通常认为无可用的 standalone alpha

2. **高 IR 来自低波动而非高预测力**
   - MACRO_YC_10Y2Y_ZSCORE_252 IR=0.74 看起来很吸引人
   - 但这是 IC std=0.035 带来的（信号非常稳定但幅度极小）
   - 实际可交易的 alpha 几乎为零

3. **Regime 间信号方向翻转 — 致命问题**
   - MACRO_CREDIT_SPREAD_ZSCORE_252: bull IC=+0.027, bear IC=-0.053
   - MACRO_CREDIT_SPREAD_DELTA_21: bull IC=-0.042, bear IC=+0.014
   - 同一信号在 bull/bear 中预测方向相反！
   - 这解释了为什么 RL 代理无法学习：信号关系是 regime-dependent，而 simple MLP 无法隐式建模 regime 条件

4. **数据样本期太短**
   - 仅 2020-2026（6 年），包含一次 bear market（2022）
   - 无法对 regime-specific IC 做出统计可靠的结论

## Phase 2: 决策

**结论：宏观信号无 standalone alpha，永久放弃 direct policy input 方向。**

| 检查标准 | 阈值 | 最佳信号表现 | 判定 |
|----------|------|-------------|------|
| \|mean IC\| | ≥ 0.05 | 0.026-0.039 | ❌ 未达标 |
| IC IR | ≥ 0.30 | 0.74 (但 IC 太小) | ⚠️ 伪达标 |
| Regime 稳定性 | 符号一致 | 方向翻转 | ❌ 未达标 |
| 样本期 | ≥ 10 年 | 6 年 | ❌ 不足 |

**决策：** 执行 **Option A** — 永久放弃 macro 特征作为 policy input。

**理由：**
- IC 验证确认宏观信号对 SOXL 收益无可靠的 standalone 预测力
- Regime 方向翻转意味着即使 IC 略高，也无法被简单 MLP 利用
- 已有 `regime_risk` 等 CORE 特征代理了 regime 信息，添加宏观特征是冗余噪声
- `balanced_macro_v1` 的训练失败（bear market 更激进）与 IC 结果一致

**不再尝试的方向：**
- ❌ 不同 macro 预处理（regime encoding, momentum, cross-sectional rank）
- ❌ Macro-aware position cap（底层信号无预测力，cap 也无意义）
- ❌ 更多宏观数据源（gold, unemployment, CPI 等）

---

## 附：Dashboard 数据区间问题排查 ✅

### 用户反馈
1. "重新计算后 2026 年就没有数据了"
2. "每次热 compute 后都会限制时间区间"
3. "而且没有到今天的最新数据"

### 根因分析

**根因 1: `TEST_SEG` 硬编码为 `("2020-01-01", "2026-03-31")`**
- `src/soxl_quant/settings.py:41`
- `dashboard_server.py` 的 Run Backtest 使用 `_segment_frame(dataset, "test")`
- qlib 的 test segment 被 TEST_SEG 限制，导致 backtest 无法使用 2026-03-31 之后的数据
- 即使 qlib 数据本身已更新到 2026-04-27，backtest 仍被截断

**根因 2: `build_backtest_dashboard.py` 使用 stale raw CSV**
- `load_ohlc_and_indicators()` 从 `data/raw/symbols/SOXL.csv` 加载
- Raw CSV 最新日期: 2026-04-01
- 但 processed CSV `soxl_soxx_vix_full.csv` 已更新到 2026-04-27

**根因 3: yfinance rate limit 导致 raw 数据更新失败**
- `scripts/update_market_data.py` 连续失败：`Too Many Requests`
- 需要等待 rate limit 重置后重试

### 修复

1. **更新 `TEST_SEG` 为 `("2020-01-01", "2026-12-31")`**
   - `src/soxl_quant/settings.py`
   - `src/soxl_quant/app/evaluate.py`（改用 TEST_SEG 变量引用）
   - `scripts/compare_mode_routers.py`
   - `scripts/analyze_learned_router_runtime.py`
   - `src/soxl_quant/backtest.py` / `data.py` 文档同步更新

2. **更新 `build_backtest_dashboard.py` 优先使用 processed CSV**
   - `load_ohlc_and_indicators()` 现在优先读取 `data/processed/panels/soxl_soxx_vix_full.csv`
   - VIX 也优先从 processed CSV 读取

3. **已重建 dashboard**
   - `dashboard.html` 和 `dashboard_data/` 已更新
   - OHLC overlay 现在覆盖到 2026-04-27

### 待办
- [ ] yfinance rate limit 重置后运行 `update_market_data.py` 更新 raw CSV
- [ ] 用新 TEST_SEG 重新运行关键 profile 的 backtest，生成新的 `*_soxl_daily.csv`（延伸到 2026-04-27）
- [ ] 重新运行 `build_backtest_dashboard.py` 更新 versioned backtest 数据

## 清理工作

- [x] 保留 `balanced_macro_v1` 档案（profile.yaml + summary + actor checkpoint）作为失败证据
- [x] 保留 `macro_signals` feature pack 代码（data.py 中的 `_builtin_macro_signals_pack`）但不再激活
- [x] IC 分析脚本和结果存档：`scripts/compute_macro_ic.py`, `runtime/model/card/macro_ic_analysis/`
- [ ] 从 `BUILTIN_FEATURE_PACKS` 中移除 `macro_signals`（或保留但不推荐）
- [ ] 更新 experiment-landscape-review 中 `balanced_macro_v1` 状态为 `dropped`（已 done）

## 实验档案

- `balanced_macro_v1` profile: `runtime/model/card/balanced_macro_v1/profile.yaml`
- `balanced` baseline (seed 42): `runtime/model/card/balanced/soxl_balanced_summary.json`
- `balanced_macro_v1` summary: `runtime/model/card/balanced_macro_v1/soxl_balanced_macro_v1_summary.json`
- IC 分析脚本: `scripts/compute_macro_ic.py`
- IC 结果: `runtime/model/card/macro_ic_analysis/ic_results.json`
- IC 摘要: `runtime/model/card/macro_ic_analysis/ic_summary.txt`

---

## 2026-05-17 Discord 通知集成

### 变更

- `production/notify.py`: 新增 `DiscordNotifier` 和 `MultiNotifier`
  - 支持 `DISCORD_BOT_TOKEN` / `DISCORD_CHANNEL_ID` 环境变量
  - 自动从 `.env.discord` 读取 token（环境变量优先）
  - `get_notifier(include_discord=True)` 同时发送 Bark/macOS + Discord
  - Cloudflare 403 修复：添加 `User-Agent` header
- `production/soxl_paper_trading.py`: 新增 `--discord` 参数
- `production/soxl_daily_scheduler.py`: 新增 `--discord` 参数
- `production/composed_session.py`: 新增 `--discord` 参数

### 权限验证

| 频道 | ID | 状态 |
|------|-----|------|
| `#general` | 1474219222904082606 | ✅ 可读可写 |
| `#tradingagent` | 1475287062553956565 | ✅ 可读可写 (默认目标) |
| `#moon` | 1476061424382906561 | ✅ 可读可写 |
| `#options` | 1476879729624612926 | ✅ 可读可写 |
| `#being` | 1483370904372379768 | ✅ 可读可写 |
| `#soxl` | 1485666024865927309 | ❌ Missing Access |
| `#daily` | 1504487456572182639 | ❌ Missing Access |

Bot 无法访问 `#soxl` 和 `#daily`（需要服务器管理员在频道权限中添加 bot）。当前默认频道设为 `#tradingagent`。

### 环境变量更新

`~/.zshrc` 中的 `DISCORD_BOT_TOKEN` 已更新为最新有效 token。

### 测试命令

```bash
# 仅 Discord
PYTHONPATH=. qlib_venv/bin/python production/notify.py --discord --title "Test" --body "Hello"

# Bark + Discord
PYTHONPATH=. qlib_venv/bin/python production/notify.py --discord --title "Test" --body "Hello"

# Paper trading with Discord
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --discord --dry-run
```

### 已知限制

- `soxl` / `daily` 频道需管理员手动授权 bot 的 `Send Messages` 权限
- 消息长度受 Discord 2000 字符限制（自动截断）

---

## 2026-05-20 Release-State Post-Release Cap Loop

### Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `selective_release_walkforward_v1`
- Role: `diagnose`, now upgraded to promotion-review candidate
- Target window: `wf_2023_2024`, with `wf_2022` as the hard defensive no-regression check
- Accepted trade-off: more trades are acceptable only if `wf_2023_2024` return improves and MaxDD falls below baseline
- Changed layer: `execution`
- External reference:
  - retained release gate family: `RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, `CICC_RETURN_VOLUME_SYNC`
  - drawdown-path attribution from `selective_release_drawdown_attribution_v1`
- Hypothesis:
  The release gate is useful, but the drawdown path needs stateful post-release exposure management. After a release order activates state, cap exposure and relax only sell-side de-risking if the release path starts drawing down.
- Acceptance criteria:
  - `wf_2022` remains unchanged
  - `wf_2023_2024` return improves versus baseline
  - `wf_2023_2024` MaxDD improves versus baseline
  - fixed full-window actor counterfactual is not disturbed
- Expected trade-off:
  Trade count can rise modestly as long as drawdown improves and the rule remains stateful rather than a broad trade-lock bypass.
- Failure mode:
  Post-release cap either becomes a no-op on walk-forward actors or cuts recovery exposure enough to lose the return improvement.
- Comparison-table row key: `selective_release_post_release_cap_v1`

### Changes Made

- `src/soxl_quant/backtest.py`
  - added post-release state tracking after execution-ready release orders
  - added `selective_release_post_release_target_cap()`
  - added sell-only `selective_release_post_release_rebalance_threshold()` and `selective_release_post_release_trade_lock_days()`
  - added trace fields for post-release state days, drawdown, de-risk readiness, and cap target
- `scripts/evaluate_selective_release_overlay.py`
  - added `selective_release_recovery_post_release_cap_v1`
- `scripts/evaluate_selective_release_walkforward.py`
  - added post-release cap to all-variant walk-forward runs, execution trace summaries, attribution, and durable review text
- `tests/test_selective_release_overlay.py`
  - covered cap activation after release-state drawdown and sell-only de-risk relaxations
- `tests/test_selective_release_walkforward.py`
  - covered inclusion of the post-release cap variant in all-variant walk-forward runs

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_overlay.py tests/test_selective_release_walkforward.py -q
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_overlay.py --profile balanced_rule_residual_v22
```

### Evidence

- Review: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`
- Walk-forward CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.csv`
- Trace CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_execution_trace.csv`
- Attribution CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_drawdown_attribution.csv`
- Fixed actor overlay CSV: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_results.csv`

| actor_window | variant | return | Sharpe | MaxDD | trades | eligible_days | changed_days | readout |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `wf_2022` | baseline | `-34.11%` | `-0.926` | `38.27%` | `22` | `0` | `0` | defensive baseline |
| `wf_2022` | `selective_release_recovery_post_release_cap_v1` | `-34.11%` | `-0.926` | `38.27%` | `22` | `0` | `0` | unchanged |
| `wf_2023_2024` | baseline | `-20.21%` | `-0.277` | `38.54%` | `22` | `0` | `0` | control |
| `wf_2023_2024` | `selective_release_recovery_trade_lock_guarded_short_v1` | `-19.63%` | `-0.233` | `42.91%` | `21` | `18` | `185` | drop, drawdown widened |
| `wf_2023_2024` | `selective_release_recovery_post_release_cap_v1` | `-11.37%` | `-0.133` | `27.66%` | `27` | `18` | `185` | candidate |

Attribution readout:

- First execution-ready release order: `2024-03-28`
- Cap variant eligible ordered dates: `2024-03-28;2024-05-14`
- Variant peak/trough: `2024-06-18 -> 2024-08-07`
- Delta MaxDD versus baseline: `-10.88pp`
- Release-to-trough avg position delta versus baseline: `-0.0373`
- Account value at trough versus baseline: `+11.58%`

Fixed actor overlay readout:

- `selective_release_recovery_post_release_cap_v1` is a no-op across fixed full-window `test_proxy`, `test_2020`, `test_2022`, and `test_2023_2024`.
- This keeps the prior fixed-actor counterfactual intact while improving only the saved walk-forward path where release-state de-risking activates.

### Decision

- Keep `selective_release_recovery_post_release_cap_v1` as the active promotion-review candidate.
- Do not promote it directly until runtime governance / packaging checks confirm the profile boundary and trace semantics.
- Do not spend the next loop on another release entry threshold, static guard, or release-pack PPO retrain.

## 2026-05-21 Promotion Governance Packaging Loop

### Scope

This loop does not promote the post-release cap to production runtime. It packages the candidate for review, centralizes the overlay config, and makes the harness reject governance drift that was previously invisible.

### Changes Made

- `src/soxl_quant/policy/execution_overlays.py`
  - centralizes all named selective-release overlay configs
  - exposes `POST_RELEASE_CAP_VARIANT`, `overlay_config()`, and `overlay_variants()`
- `scripts/evaluate_selective_release_overlay.py` and `scripts/evaluate_selective_release_walkforward.py`
  - now read overlay configs from the repo-owned registry instead of hard-coded local lists
- `scripts/review_selective_release_promotion.py`
  - writes `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.{json,md}`
  - verifies `wf_2022` unchanged, `wf_2023_2024` return improved, `wf_2023_2024` MaxDD improved, and fixed full-window actor no-op
  - records `runtime_auto_select_allowed=false`
- Profile governance
  - fixed the registry path from stale `runtime/harness/cards/...` to canonical `runtime/harness/card/registry/profile.yaml`
  - kept historical registry rows allowed while requiring every current `profile.yaml` to have a registry row
  - added missing registry rows for current legacy profile files
  - added profile registry, overlay registry, and promotion-review tests to `scripts/check_harness.py`

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_profile_registry.py tests/test_execution_overlay_registry.py tests/test_selective_release_promotion_review.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/review_selective_release_promotion.py
```

### Evidence

- Promotion review: `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.md`
- Promotion review JSON: `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json`
- Overlay registry: `src/soxl_quant/policy/execution_overlays.py`

Promotion-review result:

| check | result |
|---|---|
| `wf_2022_unchanged` | pass |
| `wf_2023_2024_return_improved` | pass, delta `+8.8381pp` |
| `wf_2023_2024_maxdd_improved` | pass, delta `-10.8815pp` |
| `fixed_actor_noop` | pass across `test_proxy`, `test_2020`, `test_2022`, `test_2023_2024` |
| `runtime_auto_select_allowed` | `false` |

### Decision

- `selective_release_recovery_post_release_cap_v1` is now a packaged promotion-review candidate.
- It is explicitly **not** eligible for scheduler auto-selection until the profile-bound runtime path and model-promotion sidecar are both approved.
- The next runtime packaging step is the profile-bound runtime overlay design / dry-run replay path below.

## 2026-05-21 Profile-Bound Runtime Overlay Dry-Run

Question: can the post-release cap be wired into the runtime profile and paper runner without letting the scheduler auto-select it?

Answer: yes. `balanced_rule_residual_v22` now binds the overlay, the runtime signal path loads release features and emits overlay fields, and production dry-run still blocks on the model-promotion gate.

### Changes Made

- `src/soxl_quant/config.py`
  - added `ExecutionOverlayConfig` to profile loading.
- `runtime/model/card/balanced_rule_residual_v22/profile.yaml`
  - binds `execution_overlay.name=selective_release_recovery_post_release_cap_v1`
  - keeps `execution_overlay.auto_select_allowed=false`
  - uses `runtime/execution/state/selective_release_post_release_cap_v1.json` for live state.
- `src/soxl_quant/runtime.py`
  - adds release feature packs for profile-bound selective-release overlays
  - applies the retained overlay and post-release cap during signal construction
  - persists/loads runtime overlay state and records overlay diagnostics in `SignalIntent`.
- `production/soxl_paper_trading.py`
  - advances and saves overlay state after non-dry-run order execution.
- `scripts/review_selective_release_promotion.py`
  - promotion review now records `runtime_profile_bound=true` while preserving `runtime_auto_select_allowed=false`.

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_profile_bound_runtime_overlay_loads_release_signals_but_blocks_auto_select tests/test_runtime.py::test_runtime_overlay_caps_post_release_target_after_state_drawdown tests/test_runtime.py::test_runtime_overlay_state_activates_after_release_order_and_roundtrips -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_profile_bound_runtime_overlay_relative_state_path_loads_default_state -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_selective_release_promotion_review.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/review_selective_release_promotion.py
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --profile balanced_rule_residual_v22 --skip-data-update --dry-run --no-notify
```

### Evidence

- Focused runtime + promotion review tests passed.
- Dry-run signal date: `2026-05-14`
- Dry-run profile/model: `balanced_rule_residual_v22` / `soxl_balanced_rule_residual_v22_actor.pt`
- Signal readout:
  - `action=2`
  - `target_position=0.75`
  - `target_before_overlay=0.75`
  - `execution_overlay=selective_release_recovery_post_release_cap_v1`
  - `overlay_gate_ready=false`
  - `overlay_execution_ready=false`
  - `post_release_state_days=0`
- Risk gate readout:
  - `approved=false`
  - blocker: `walk-forward promotion gate not evaluated`
  - step limit clipped target to `0.6852543622`
- Promotion review now says:
  - `runtime_profile_bound=true`
  - `runtime_auto_select_allowed=false`
  - reason: profile binds overlay but keeps `auto_select_allowed=false` and the base actor has not passed the model promotion gate.

### Decision

- The profile-bound runtime overlay / dry-run replay path is now in place.
- Do not turn on scheduler auto-selection from this result.
- Next valid work is either explicit model-promotion sidecar governance for v22/post-release-cap runtime use, or a separate narrow model-optimization experiment if auto-selection approval is deferred.

## 2026-05-21 Runtime Auto-Select Overlay Gate

Question: if the base `balanced_rule_residual_v22` sidecar later gets marked promoted, can scheduler auto-selection still bypass the profile overlay approval flag?

Answer: no after this loop. A red test showed `best_model_for_family("balanced")` would choose a higher-scored promoted `balanced_rule_residual_v22` actor even though its profile has `execution_overlay.auto_select_allowed=false`. Runtime auto-selection now skips such profiles and keeps `balanced_rule_residual_v19` as the selected production profile unless v22 overlay approval is explicit.

### Changes Made

- `src/soxl_quant/runtime.py`
  - added `profile_name_for_model_path()`
  - added `runtime_auto_select_allowed()`
  - `best_model_for_family()` now requires both a passing model promotion sidecar and profile-level runtime auto-selection approval.
- `tests/test_runtime.py`
  - added a regression test where promoted v22 has a higher score than v19 but is skipped because the profile-bound overlay is not approved for auto-selection.

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_best_model_for_family_skips_overlay_profile_without_auto_select_approval -q
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from src.soxl_quant.runtime import auto_select_profile
print(auto_select_profile("balanced"))
PY
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
```

### Evidence

- Red test before fix: `best_model_for_family("balanced")` selected `soxl_balanced_rule_residual_v22_actor.pt` over v19 when both summaries were marked promoted.
- Green test after fix: promoted-but-not-profile-approved v22 is skipped.
- Current live auto-select still returns `balanced_rule_residual_v19` / `soxl_balanced_rule_residual_v19_actor.pt`.
- Scheduler dry-run still auto-selected `balanced_rule_residual_v19`, generated the `2026-05-14` signal, approved the risk gate, and skipped order placement because `--dry-run` was set.

### Decision

- Keep `balanced_rule_residual_v22.execution_overlay.auto_select_allowed=false`.
- Do not create or edit a v22 promoted model sidecar in this loop.
- Next valid work is now a model-side decision: either a governed promotion-sidecar review for v22+post-release-cap, or a separate narrow model optimization branch if that approval is deferred.

## 2026-05-21 Runtime Selection Sidecar Review

Question: can v22/post-release-cap be approved for scheduled runtime selection now that the overlay review, profile binding, and auto-select profile gate exist?

Answer at this checkpoint: no. The repo added a model-side runtime selection sidecar and initially blocked automatic selection because the base v22 model sidecar had not persisted corrected walk-forward promotion evidence and the profile overlay flag remained `auto_select_allowed=false`. The later corrected v22 gate loop below supersedes the model-side blocker but keeps the profile flag blocker.

### Changes Made

- `scripts/review_runtime_selection.py`
  - reads the v22 model summary, post-release-cap overlay review, runtime profile flag, and current auto-selected profile
  - writes `runtime_selection_review.json` and `runtime_selection_review.md`
  - exits successfully even when the review blocks selection, because the artifact is the outcome.
- `tests/test_runtime_selection_review.py`
  - covers blocked and approved review outcomes
  - covers the script success semantics for a blocked review
  - covers current `balanced` auto-selection context for v22.
- `scripts/check_harness.py`
  - includes the new review tests and script in harness verification.

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime_selection_review.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/review_runtime_selection.py
```

### Evidence

- Runtime selection review JSON: `runtime/model/card/balanced_rule_residual_v22/runtime_selection_review.json`
- Runtime selection review MD: `runtime/model/card/balanced_rule_residual_v22/runtime_selection_review.md`
- Decision: `blocked`
- Initial passing checks:
  - `model_fixed_split_eligible=true`
  - `overlay_review_candidate=true`
  - `overlay_runtime_profile_bound=true`
- Initial blocking checks:
  - `model_walk_forward_evaluated=false`
  - `model_promotion_eligible=false`
  - `profile_overlay_auto_select_allowed=false`
- Current auto-selected profile remained `balanced_rule_residual_v19`.

### Decision

- Do not turn on `balanced_rule_residual_v22.execution_overlay.auto_select_allowed`.
- Do not manually patch `soxl_balanced_rule_residual_v22_summary.json`.
- The next valid model-side step was to produce and persist corrected walk-forward model-promotion evidence through the repo script, not to edit the summary by hand.

## 2026-05-21 SOXX Relative Weakness Cap Fixed-Actor Diagnostic

Goal: resume SOXX-aware model optimization without another broad PPO retrain. Test whether SOXX should be used only as an execution-side de-risking gate for the existing `balanced_rule_residual_v22` actor, after the direct `balanced_rule_residual_v22_soxx` actor failed the fixed-split promotion gate on `test_2022_avg_pos`.

### Experiment Governance

- Canonical baseline: `balanced_rule_residual_v22` fixed actor.
- Direct comparison baseline: `balanced_rule_residual_v22_soxx`, which injected `soxx_signals` into the actor observation and failed fixed-split promotion because `test_2022_avg_pos=0.380930 > 0.35`.
- Role: `diagnose`.
- Target window and accepted trade-off: fixed split `test_2022` bear-defense / avg-position first; `test_2023_2024` recovery can lose a small amount only if bear exposure improves materially. No runtime promotion without later walk-forward evidence.
- Changed layer: `execution`.
- External reference: repo-owned SOXX underlying ETF dataset and prior `SIGNED_TE_REVERSION_10` tracking-error reversion factor card.
- Hypothesis: SOXX direct actor features caused excessive 2022 exposure; a fixed-actor cap should only reduce target exposure when SOXX is below its short MA, SOXL/SOXX ratio momentum is weak, and signed tracking-error reversion is not strong enough to justify a recovery exception.
- Acceptance criteria: the overlay must reduce or preserve `test_2022` avg position and max drawdown versus the fixed actor without a large `test_2023_2024` return give-up; any candidate remains diagnostic until walk-forward execution absorption is proven.
- Expected trade-off: fewer high-risk bear-market holdings, possible missed recovery entries.
- Failure mode: the cap fires too late or too broadly, hurting recovery while not improving `test_2022`.
- Comparison-table row key: `soxx_relative_weakness_cap_v1`.

### History Check

1. Closest historical success: selective-release fixed-actor overlay improved recovery/proxy slices before execution absorption blocked runtime promotion.
2. Closest historical failure: direct `balanced_rule_residual_v22_soxx` actor feature injection improved some recovery metrics but failed `test_2022_avg_pos`.
3. Why this is not a duplicate: this loop does not retrain or add SOXX to actor state; it tests a defensive execution cap with a signed tracking-error exception, separating model behavior from execution mapping.

### Changes Made

- `src/soxl_quant/policy/execution_overlays.py`
  - added `soxx_relative_weakness_cap_v1` registry config.
  - added pure SOXX weakness gate and cap helpers.
- `src/soxl_quant/backtest.py`
  - qlib strategy now applies `type=soxx_relative_weakness_cap` overlays.
- `scripts/evaluate_soxx_relative_overlay.py`
  - fixed-actor diagnostic grid for baseline vs SOXX relative weakness cap.
  - writes CSV/JSON/Markdown review artifacts.
- `tests/test_execution_overlay_registry.py`
  - covers SOXX overlay registry copy semantics, gating, regime band, no-boost behavior, and signed tracking-error exception.
- `scripts/check_harness.py`
  - compiles the new SOXX overlay evaluation script.

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_execution_overlay_registry.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_soxx_relative_overlay.py
```

### Evidence

- Review: `runtime/execution/card/soxx_relative_weakness_cap_v1/soxx_relative_overlay_review.md`
- JSON: `runtime/execution/card/soxx_relative_weakness_cap_v1/soxx_relative_overlay_results.json`
- Decision: `candidate`
- Fixed-actor results:
  - `test_proxy`: `1621.98% / Sharpe 1.138 / MaxDD 54.73%` vs baseline `1511.97% / 1.112 / 56.48%`; `cap_changed_days=77`.
  - `test_2023_2024`: `72.61% / Sharpe 0.817 / MaxDD 36.89%` vs baseline `57.12% / 0.719 / 34.87%`; return `+15.48pp`, MaxDD `+2.02pp`, `cap_changed_days=19`.
  - `test_2022`: unchanged at `-30.97% / Sharpe -0.764 / MaxDD 40.64% / avg_pos 0.3304`; `eligible_days=44` but `cap_changed_days=0`.

### Decision

- Keep `soxx_relative_weakness_cap_v1` as a fixed-actor diagnostic candidate.
- Do not promote to runtime selection from this artifact alone.
- Next valid follow-up is walk-forward execution absorption validation for this SOXX cap, because `test_2022` eligible days did not change target exposure.

## 2026-05-21 SOXX Relative Weakness Cap Walk-Forward Absorption Validation

Goal: validate whether `soxx_relative_weakness_cap_v1` survives beyond fixed full-window evidence when applied to saved `balanced_rule_residual_v22` walk-forward actors.

### Experiment Governance

- Canonical baseline: each saved `balanced_rule_residual_v22` walk-forward actor run with no SOXX cap.
- Direct comparison baseline: fixed full-window `soxx_relative_weakness_cap_v1` diagnostic, plus the failed direct `balanced_rule_residual_v22_soxx` actor-feature branch.
- Role: `diagnose`.
- Target window and accepted trade-off: `wf_2022` must not widen MaxDD or average exposure; `wf_2023_2024` must show real target/position/order change before any recovery improvement is trusted.
- Changed layer: `execution`.
- External reference: repo-owned SOXX underlying ETF data, `soxx_signals`, and prior `SIGNED_TE_REVERSION_10` factor evidence.
- Hypothesis: the SOXX weakness cap may provide a more direct defensive execution mechanism than selective-release boost gates because it lowers target exposure instead of trying to force re-entry; if the cap is absorbed on saved actors, the fixed-window candidate is not promotable.
- Acceptance criteria: report `eligible_days`, `cap_changed_days`, `changed_days`, and eligible-order counts per walk-forward actor; keep as candidate only if `wf_2023_2024` improves without materially worsening drawdown and at least one walk-forward window has real changed target/position behavior.
- Expected trade-off: possible recovery return improvement with modest drawdown widening, or no-op if the actor already sits below cap on eligible dates.
- Failure mode: eligible days appear but `cap_changed_days=0` or `changed_days=0`, proving execution absorption; or the cap changes positions but worsens recovery/drawdown.
- Comparison-table row key: `soxx_relative_weakness_walkforward_v1`.

### History Check

1. Closest historical success: `selective_release_recovery_post_release_cap_v1` improved saved walk-forward recovery and drawdown after attribution found a post-release path problem.
2. Closest historical failure: `selective_release_walkforward_v1` had eligible days but `changed_days=0`, proving fixed-window overlay gains can disappear through walk-forward execution absorption.
3. Why this is not a duplicate: this tests a SOXX-driven target cap, not a release boost or trade-lock bypass, and it explicitly measures cap target changes separately from position-path changes.

### Changes Made

- `scripts/evaluate_soxx_relative_walkforward.py`
  - validates `soxx_relative_weakness_cap_v1` on saved walk-forward actors.
  - records `feature_eligible_days`, executable `eligible_days`, `cap_changed_days`, position `changed_days`, eligible orders, and absorption counts.
- `tests/test_soxx_relative_walkforward.py`
  - covers SOXX eligibility mask, eligible trace extraction, cap-change tracking, primary blocker tagging, and the rule that eligibility without position change cannot pass review.
- `scripts/check_harness.py`
  - includes the new test module and compiles the new walk-forward review script.

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_soxx_relative_walkforward.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_soxx_relative_walkforward.py
```

### Evidence

- Review: `runtime/execution/card/soxx_relative_weakness_walkforward_v1/soxx_relative_walkforward_review.md`
- JSON: `runtime/execution/card/soxx_relative_weakness_walkforward_v1/soxx_relative_walkforward_results.json`
- Trace: `runtime/execution/card/soxx_relative_weakness_walkforward_v1/soxx_relative_walkforward_execution_trace.csv`
- Decision: `blocked`
- Walk-forward readout:
  - `wf_2020`: return `106.66%` vs baseline `113.17%`; MaxDD unchanged at `19.47%`; `15` executable eligible days, `12` cap-changed days, `39` position-changed days. The cap reduces bull-window upside.
  - `wf_2022`: unchanged at `-34.11% / Sharpe -0.926 / MaxDD 38.27% / avg_pos 0.3216`; `66` feature eligible days, `44` executable eligible days, `0` cap-changed days, `0` position-changed days.
  - `wf_2023_2024`: unchanged at `-20.21% / Sharpe -0.277 / MaxDD 38.54% / avg_pos 0.4764`; `34` feature eligible days, `27` executable eligible days, `6` cap-changed days, `0` position-changed days.
  - Primary absorber on executable eligible trace: `trade_lock` dominates (`wf_2023_2024`: `22/27`; `wf_2022`: `31/44`).

### Decision

- Block `soxx_relative_weakness_cap_v1` for runtime binding and model-promotion support.
- The fixed full-window improvement does not survive saved walk-forward actors.
- Do not tune this cap threshold as the next model optimization loop; the failure is execution absorption plus no recovery improvement, not insufficient SOXX feature availability.

## 2026-05-21 Paper Same-Day Idempotency Guard

Goal: prevent scheduled or manual reruns from submitting another paper order for
the same signal date after a non-dry-run paper summary has already recorded an
executed order.

### Baseline

- Current paper summary:
  `runtime/soxl/paper_trading/2026-05-18/summary.json`
  - `date=2026-05-18`
  - `broker=paper`
  - `dry_run=false`
  - `executed=true`
- Current broker state:
  - `date=2026-05-18`
  - `shares=3000`
  - `position=0.4856047894`
  - `account_value=937490.7536`
- Current cumulative paper dashboard CSV:
  - `3` rows
  - latest row `2026-05-18`, action `BUY`, position `0.485605`

### Change

- `production/soxl_paper_trading.py`
  - checks the existing per-date paper summary before drawdown checks, order
    placement, post-execution audit, dashboard CSV append, or broker-state save.
  - skips duplicate order placement when the existing summary matches the same
    signal date, broker, `dry_run=false`, and `executed=true`.
- `tests/test_paper_trading.py`
  - proves a same-day rerun exits before `place_target_position`,
    `_append_execution_daily_csv`, or `_save_paper_broker_state`.

### Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_skips_duplicate_non_dry_run_for_existing_executed_summary -q
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --no-notify --skip-data-update
```

### Evidence

- TDD red: the focused test failed before the guard because the duplicate rerun
  reached `place_target_position`.
- TDD green: the focused test passed after the guard.
- Real current-state rerun:
  - data remained fresh: `last_date=2026-05-19 expected=2026-05-19 stale_days=0`
  - model auto-select remained `balanced_rule_residual_v19`
  - signal date remained `2026-05-18`
  - risk gate remained approved
  - paper trading printed:
    `already executed for 2026-05-18 broker=paper; skipping duplicate order`
  - scheduler still rebuilt dashboard and exited successfully.
- Post-rerun state:
  - broker state still `date=2026-05-18`, `shares=3000`,
    `position=0.4856047894`.
  - cumulative paper CSV still has exactly `3` rows.
  - dashboard paper summary still ends at `2026-05-18` with `3` rows.
- Verification gates:
  - `tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_vercel_dashboard.py tests/test_dashboard.py`: `29 passed`.
  - `scripts/check_harness.py`: `passed=112 failures=0`.
  - `scripts/build_runtime_index.py --check`: `runtime index OK`.
  - `git diff --check`: clean.
  - `production/check_scheduler_health.sh`: LaunchAgents loaded, data ok
    `last_date=2026-05-19 expected=2026-05-19`, model ok
    `balanced_rule_residual_v19`, broker ok.

### Decision

- Keep this guard in the simulated paper path before relying on the natural
  post-close LaunchAgent. It prevents repeated manual or scheduled same-day
  passes from inching the paper account toward the target multiple times.
- The remaining production proof is still the first natural
  `05:35 Tue-Sat Asia/Shanghai` LaunchAgent run after the next completed US
  trading session.

## 2026-05-21 v22 Corrected Runtime Selection Gate

Goal: resolve whether v22/post-release-cap is blocked by missing model evidence
or only by the explicit profile overlay approval flag.

### Baseline

- Current time was `2026-05-21 03:18 CST`, before the natural
  `05:35 Tue-Sat Asia/Shanghai` post-close LaunchAgent window.
- `production/check_scheduler_health.sh` showed both LaunchAgents loaded,
  `scheduler_postclose.log` not yet present, data fresh for the expected
  completed US session (`last_date=2026-05-19 expected=2026-05-19`), model ok
  on `balanced_rule_residual_v19`, and broker ok.
- `runtime/model/card/balanced_rule_residual_v22/runtime_selection_review.json`
  still treated the v22 model side as unevaluated because corrected
  main-backtest walk-forward evidence had not been loaded into the promotion
  sidecar.

### Changes

- `scripts/walkforward_backtest_validate.py`
  - loads the canonical actor path via `model_artifact_path(profile)` instead
    of the legacy `runtime/model/card/soxl_<profile>_actor.pt` path.
  - expands feature packs through `feature_columns_for_packs(...)`.
- `scripts/review_runtime_selection.py`
  - loads corrected main-backtest walk-forward reports and maps
    `relative_outperformance` into the promotion-gate input.
  - can persist the effective promotion report back to the model summary
    sidecar through `--persist-model-promotion`.
  - distinguishes missing walk-forward evidence from failed walk-forward
    evidence.
- `scripts/review_selective_release_promotion.py`
  - handles relative `--output-dir` paths in the same way as the runtime
    selection review script.

### Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/walkforward_backtest_validate.py --profile balanced_rule_residual_v22 --output-dir runtime/model/card/balanced_rule_residual_v22
PYTHONPATH=. qlib_venv/bin/python scripts/review_runtime_selection.py --profile balanced_rule_residual_v22 --model-summary runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json --overlay-review runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json --output-dir runtime/model/card/balanced_rule_residual_v22 --family balanced --persist-model-promotion
PYTHONPATH=. qlib_venv/bin/python scripts/review_selective_release_promotion.py --profile balanced_rule_residual_v22 --output-dir runtime/execution/card/selective_release_post_release_cap_v1 --variant selective_release_recovery_post_release_cap_v1
```

### Evidence

Corrected main-backtest walk-forward metrics for base `balanced_rule_residual_v22`:

| window | return | Sharpe | MaxDD | trades | B&H return | relative |
|---|---:|---:|---:|---:|---:|---:|
| `wf_2020` | `+136.65%` | `2.160` | `20.16%` | `22` | `+59.78%` | `+76.87pp` |
| `wf_2022` | `-48.40%` | `-1.239` | `54.67%` | `26` | `-86.59%` | `+38.18pp` |
| `wf_2023_2024` | `+50.98%` | `1.011` | `30.55%` | `29` | `-2.60%` | `+53.58pp` |

Persisted model promotion sidecar:

- `eligible=true`
- `walk_forward_evaluated=true`
- `wf_2023_2024_return=50.9777 > 0`
- `wf_2023_2024_sharpe=1.0114 > 0`
- `wf_2023_2024_relative=53.5811 > 15`

Runtime selection review after persisting corrected evidence:

- Decision remains `blocked`.
- Passing checks now include `model_walk_forward_evaluated=true` and
  `model_promotion_eligible=true`.
- The only blocking reason is
  `profile overlay auto_select_allowed=false`.
- Current auto-selected profile remains `balanced_rule_residual_v19`.

Runtime auto-select gate check:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from src.soxl_quant.runtime import auto_select_profile, latest_model_path, promotion_gate_status, runtime_auto_select_allowed
profile, path = auto_select_profile('balanced')
print('auto_select', profile, path.name)
v22 = latest_model_path('balanced_rule_residual_v22')
print('v22_promotion', promotion_gate_status(v22))
print('v22_runtime_auto', runtime_auto_select_allowed('balanced_rule_residual_v22'))
PY
# auto_select balanced_rule_residual_v19 soxl_balanced_rule_residual_v19_actor.pt
# v22_promotion (True, 'model passed promotion gate')
# v22_runtime_auto (False, 'profile overlay selective_release_recovery_post_release_cap_v1 is not approved for runtime auto-selection')
```

### Decision

- The v22 model-side promotion blocker is cleared through corrected
  main-backtest walk-forward evidence.
- The v22/post-release-cap runtime candidate is still intentionally blocked by
  `execution_overlay.auto_select_allowed=false`.
- Do not enable the profile flag in this loop; the remaining action is an
  explicit production promotion decision after the natural scheduler proof, not
  another model-evidence loop.

## 2026-05-21 LaunchAgent Installed Plist Drift Guard

Goal: make the natural `05:35 Tue-Sat Asia/Shanghai` post-close proof safer by
confirming the loaded macOS LaunchAgents are backed by the current repo plists,
not stale copies in `~/Library/LaunchAgents`.

### Root Cause

- `production/check_scheduler_health.sh` already reported whether
  `com.soxl.preopen.scheduler` and `com.soxl.daily.scheduler` were loaded.
- It did not compare the installed plist files with the repo-owned versions.
- A stale installed plist could therefore still use old times, old arguments, or
  old log paths while health reported `loaded`.

### Change

- `production/check_scheduler_health.sh`
  - added `SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR` for deterministic tests.
  - compares `production/com.soxl.preopen.scheduler.plist` and
    `production/com.soxl.daily.scheduler.plist` against the installed plist
    copies on Darwin.
  - reports `matches repo`, `differs from repo`, `not installed`, or
    `repo file missing` for each LaunchAgent.
- `tests/test_scheduler_health.py`
  - now covers installed plist drift.
  - no longer depends on pytest-only fixtures, so the file can run under the
    repo-owned `scripts/check_harness.py` direct-call runner.
- `scripts/check_harness.py`
  - now includes `tests/test_scheduler_health.py`.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_installed_launchagent_plist_drift -q
# red first: health output lacked plist match/drift lines
# green: 1 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py --test-file tests/test_scheduler_health.py
# red first: old scheduler-health test used pytest tmp_path fixture and failed in direct-call harness
# green: passed=122 failures=0

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.preopen.scheduler plist: matches repo
# LaunchAgent com.soxl.daily.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler plist: matches repo
# scheduler_postclose.log: NOT FOUND
# data: ok last_date=2026-05-19 expected=2026-05-19
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

### Decision

- The installed macOS LaunchAgent configs now match the repo-owned schedule and
  log-path definitions.
- The natural post-close proof is still pending because current time during this
  pass was `2026-05-21 03:49 CST`, before the `05:35` window.

## 2026-05-21 Post-close Due Status Guard

Goal: make `production/check_scheduler_health.sh` directly state whether the
post-close run is still pending, has a success marker, or is due but missing a
success marker.

### Root Cause

- The health script listed `scheduler_postclose.log: NOT FOUND`, but before
  `05:35 Tue-Sat Asia/Shanghai` that is expected.
- After `05:35`, the same missing log becomes a production problem.
- Operators needed to infer this timing manually from the wall clock.

### Change

- `production/check_scheduler_health.sh`
  - added an explicit `Post-close Run Status` section.
  - reports `pending until 05:35 Tue-Sat Asia/Shanghai` before the scheduled
    post-close window.
  - reports `due but no success marker found` after the scheduled time if
    `scheduler_postclose.log` lacks `[scheduler] all done`.
  - reports `success marker found` when the post-close log contains a scheduler
    success marker.
  - exposes `SOXL_SCHEDULER_HEALTH_WEEKDAY` and `SOXL_SCHEDULER_HEALTH_HHMM`
    for deterministic tests.
- `tests/test_scheduler_health.py`
  - covers the due-without-success case.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_postclose_due_without_success_marker -q
# red first: status line did not exist
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 3 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.preopen.scheduler plist: matches repo
# LaunchAgent com.soxl.daily.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler plist: matches repo
# scheduler_postclose.log: NOT FOUND
# data: ok last_date=2026-05-19 expected=2026-05-19
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai
```

### Decision

- Before `05:35`, a missing `scheduler_postclose.log` is now classified as
  pending rather than ambiguous.
- After `05:35`, the same health script will classify a missing success marker
  as due/missing, which is the next natural proof checkpoint.

## 2026-05-21 Vercel Dashboard Static Export Integrity Guard

Goal: make the Vercel dashboard package checkable as a complete static bundle,
not just a set of files that happen to exist.

### Root Cause

- `tests/test_vercel_dashboard.py` verified the root rewrite and tracked files,
  but did not prove `dashboard_data/index.json` referenced existing versioned
  payloads.
- `vercel.json` set no-cache headers on `/dashboard.html` and
  `/dashboard_data/:path*`, but not on the rewritten root `/`.
- `production/check_scheduler_health.sh` showed paper data freshness but did
  not state whether the dashboard export that Vercel serves was internally
  consistent.

### Change

- Added `scripts/check_dashboard_static_export.py`.
  - validates `vercel.json` root rewrite and no-cache headers for `/`,
    `/dashboard.html`, and `/dashboard_data/:path*`.
  - validates `dashboard.html`, `dashboard_data/index.json`, and
    `dashboard_data/model_versions.json` exist.
  - validates every version in `dashboard_data/index.json` has a matching
    `dashboard_data/versioned/<version>/SOXL.json` payload.
  - validates row counts against `version_summaries`, and keeps `paper_trading`
    visible in the status line.
- Added the root `/` cache header to `vercel.json`.
- `production/check_scheduler_health.sh` now includes a `Dashboard Static Export`
  section and runs the static export integrity checker.
- `tests/test_vercel_dashboard.py` and `tests/test_scheduler_health.py` cover the
  new guard.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_vercel_dashboard.py -q
# red first: missing root no-cache header and missing check script
# green: 3 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_macos_launchagent_and_logs -q
# red first: health output had no Dashboard Static Export section
# green: 1 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 03:13:41 versions=25 missing=0 model_versions=90 latest=balanced_rule_residual_v22_soxx paper_trading_rows=3 paper_trading_end=2026-05-18 paper_trading

production/check_scheduler_health.sh
# Dashboard Static Export: missing=0, versions=25, model_versions=90
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_vercel_dashboard.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_dashboard.py -q
# 36 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=124 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK
```

### Decision

- The static dashboard package is internally consistent and Vercel-ready for the
  current export.
- Natural post-close LaunchAgent proof remains pending until the `05:35 Tue-Sat
  Asia/Shanghai` run.

## 2026-05-21 Post-close Success Freshness Guard

Goal: prevent a manual or old post-close success marker from satisfying the
natural `05:35 Tue-Sat Asia/Shanghai` proof.

### Root Cause

- Before `05:35`, the live health output correctly showed
  `postclose status: pending until 05:35 Tue-Sat Asia/Shanghai`.
- `print_postclose_status()` only checked whether `scheduler_postclose.log`
  contained `[scheduler] all done`.
- If a manual `launchctl start com.soxl.daily.scheduler` or stale log wrote that
  marker before the scheduled cutoff, a later health check could incorrectly
  treat it as the natural post-close run.

### Change

- `production/check_scheduler_health.sh`
  - added portable log mtime extraction via `file_epoch()`.
  - added a current-day `05:35` cutoff calculation with
    `SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH` for deterministic tests.
  - now counts post-close success only when `scheduler_postclose.log` contains
    `[scheduler] all done` and the log mtime is not older than the cutoff.
  - after the due time, an old marker reports
    `due but no current success marker found`.
- `tests/test_scheduler_health.py`
  - added a regression where a pre-cutoff success marker exists but must not
    satisfy the post-close proof.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_ignores_old_postclose_success_marker_after_due -q
# red first: old marker was accepted as success

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 4 passed

production/check_scheduler_health.sh
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_vercel_dashboard.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_dashboard.py -q
# 37 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=125 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export ... missing=0 ... paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK
```

### Decision

- Do not manually start the post-close LaunchAgent before `05:35`; doing so
  would no longer satisfy the scheduled-run proof anyway.
- The next proof checkpoint remains the natural `05:35 Tue-Sat Asia/Shanghai`
  run, now guarded against stale/manual success markers.

## 2026-05-21 Post-close Log Segment Guard

Goal: make the natural post-close proof robust to LaunchAgent's append-only log
behavior.

### Root Cause

- LaunchAgent appends scheduler stdout to `scheduler_postclose.log`.
- The previous freshness guard compared log mtime to the current-day `05:35`
  cutoff and checked for any `[scheduler] all done` line.
- If an old success line remained in the file and a new post-close attempt only
  appended a failure after the cutoff, the file mtime would become fresh and the
  old success line could still produce a false success.

### Change

- `production/soxl_daily_scheduler.py`
  - prints `[scheduler] run started at ... broker=... dry_run=...` at the start
    of every scheduler invocation.
- `production/check_scheduler_health.sh`
  - when run-start markers exist, only accepts `[scheduler] all done` if it
    appears after the latest `[scheduler] run started at ...` marker.
  - still keeps the cutoff mtime guard, so a pre-cutoff success cannot satisfy
    a due post-close proof.
- `tests/test_scheduler_health.py`
  - covers the append-only false-positive case: old success, latest run start,
    then failure.
- `tests/test_daily_scheduler.py`
  - verifies the scheduler emits the run-start marker.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_prints_run_start_marker -q
# red first: scheduler output only had all done

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_requires_success_after_current_postclose_run_start -q
# red first: old success line still satisfied health

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_daily_scheduler.py -q
# 13 passed

production/check_scheduler_health.sh
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_vercel_dashboard.py tests/test_production_schedule.py tests/test_paper_trading.py tests/test_dashboard.py -q
# 39 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=126 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export ... missing=0 ... paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK
```

### Decision

- The health proof now requires a success marker in the current scheduler run
  segment, not merely anywhere in the append-only log.
- Natural post-close LaunchAgent proof remains pending; current live health
  time during this pass was `2026-05-21 04:25 CST`.

## 2026-05-21 Paper Signal Latest-Bar Fix

Goal: fix the remaining simulated-paper lag where the runner had fresh data for
one date but generated the prior date's signal.

### Root Cause

- `production/soxl_paper_trading.py` reads the latest available bar from
  `data/processed/panels/soxl_soxx_vix_full.csv` and passes it as `as_of` to
  `plan_manual_session()`.
- `production/FUTU_SEMI_AUTO.md` documents this as "runs `plan_manual_session()`
  for the latest available bar".
- `plan_manual_session()` instead treated `as_of` as an excluded execution date
  and selected `full_df.index < as_of`.
- Current live evidence showed the mismatch:
  - data freshness: `last_date=2026-05-19 expected=2026-05-19`
  - paper dashboard and broker state: latest executed row still `2026-05-18`
  - cached dry-run before this fix would therefore remain one signal day behind
    the latest closed bar.

### Change

- `src/soxl_quant/runtime.py`
  - `plan_manual_session()` now parses `as_of` as the latest closed bar and uses
    `full_df.index <= as_of`.
  - This keeps post-close execution non-leaking because the broker action still
    happens after the close; it only stops discarding the freshly available bar.
- `tests/test_paper_trading.py`
  - adds a regression proving `plan_manual_session(..., as_of="2026-05-19")`
    builds the signal from `2026-05-19`, not `2026-05-18`.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_plan_manual_session_uses_latest_available_bar_as_signal_date -q
# red first: signal.date was 2026-05-18
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 37 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --dry-run --skip-data-update --no-notify
# data fresh: last_date=2026-05-19 expected=2026-05-19 stale_days=0
# as_of=2026-05-19
# output=runtime/soxl/paper_trading/2026-05-19
# summary signal.date=2026-05-19, dry_run=true, executed=false

production/check_scheduler_health.sh
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai
```

### Decision

- Treat `as_of` in the automated paper runner as the latest closed bar, matching
  the production scheduler's data-refresh semantics.
- The natural post-close proof is still pending because current live health
  time during this pass was `2026-05-21 04:33 CST`, before the `05:35`
  LaunchAgent window.

## 2026-05-21 Latest Signal Artifact Health

Goal: make the scheduled inference layer visible even before a non-dry-run paper
order appends a new dashboard row.

### Root Cause

- `production/check_scheduler_health.sh` showed:
  - data freshness
  - promoted model selection
  - broker readiness
  - dashboard static export state
  - post-close success marker state
- It did not show the latest persisted inference artifact under
  `runtime/soxl/paper_trading/<date>/summary.json`.
- After the latest-bar fix, a cached dry-run proved `2026-05-19` inference
  existed, but health still surfaced the dashboard paper row ending at
  `2026-05-18`; this made the "定时推理更新" layer look unproven unless the
  operator manually inspected runtime files.

### Change

- `production/check_scheduler_health.sh`
  - adds `Latest Signal Artifact`.
  - reports the newest paper summary date, profile, `dry_run`, `executed`,
    target position, risk-gate status, reconcile status, and artifact path.
- `tests/test_scheduler_health.py`
  - adds a regression proving health reports the latest summary artifact.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_latest_signal_artifact -q
# red first: health output had no Latest Signal Artifact section
# green: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 6 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 38 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=127 failures=0

production/check_scheduler_health.sh
# latest signal: date=2026-05-19 profile=balanced_rule_residual_v19 dry_run=true executed=false
# target=0.3999999985 risk_gate=approved reconcile=manual_review
# artifact=runtime/soxl/paper_trading/2026-05-19/summary.json
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai
```

### Decision

- Keep latest-signal visibility in scheduler health so inference freshness,
  paper execution freshness, and dashboard export freshness can be diagnosed as
  separate layers.
- Natural post-close LaunchAgent proof remains pending; current live health time
  during this pass was `2026-05-21 04:39 CST`.

## 2026-05-21 Natural Post-Close LaunchAgent Readout

Goal: inspect the first real `05:35 Tue-Sat Asia/Shanghai` post-close
LaunchAgent run and make health reporting precise enough that a cached dry-run
fallback cannot masquerade as a full paper-trading success.

### Readout

- The natural LaunchAgent ran at `2026-05-21T05:35:04+08:00`.
- It tried the full post-close path with `broker=paper`, `dry_run=False`,
  `skip_data_update=False`, and `data_provider=auto`.
- Provider results:
  - `yfinance`: `No overlapping dates for SOXL/SOXX/VIX`
  - `yahoo_api`: Yahoo `429 Too Many Requests` on `^VIX`
  - `polygon`: initially failed because `POLYGON_API_KEY` was not present in
    the LaunchAgent environment
- The scheduler correctly fell back to cached dry-run:
  `[scheduler] all done (with cached dry-run)`.
- A manual `launchctl kickstart -k gui/$(id -u)/com.soxl.daily.scheduler` after
  installing the wrapper/private env file no longer failed on missing
  `POLYGON_API_KEY`.
- The manual kickstart still did not complete full paper execution because the
  data surface remained stale:
  `last_date=2026-05-19 expected=2026-05-20 stale_days=1`.

### Changes

- `production/check_scheduler_health.sh`
  - matches only exact `[scheduler] all done` lines as full post-close success.
  - detects `[scheduler] all done (with cached dry-run)` separately.
  - now reports `postclose status: due but only cached dry-run fallback completed`
    for the latest run segment.
- `production/run_soxl_scheduler.sh`
  - new LaunchAgent wrapper that loads
    `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` before invoking the
    scheduler.
- `production/com.soxl.daily.scheduler.plist` and
  `production/com.soxl.preopen.scheduler.plist`
  - now call the wrapper as their first `ProgramArguments` entry.
- `production/FUTU_SEMI_AUTO.md`
  - documents the private env file and reload sequence.

### External State

```bash
cat > ~/.openclaw/soxl.env
# contains: export POLYGON_API_KEY='...'
chmod 600 ~/.openclaw/soxl.env

cp production/com.soxl.daily.scheduler.plist production/com.soxl.preopen.scheduler.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist
launchctl list | grep 'com.soxl.*scheduler'
# - 0 com.soxl.preopen.scheduler
# - 0 com.soxl.daily.scheduler
```

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_does_not_count_cached_dry_run_as_postclose_success \
  tests/test_scheduler_health.py::test_scheduler_health_requires_success_after_current_postclose_run_start \
  tests/test_scheduler_health.py::test_scheduler_health_ignores_old_postclose_success_marker -q
# 3 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py -q
# 3 passed

SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh \
  --broker paper --dry-run --skip-data-update --no-notify --skip-dashboard
# [scheduler] all done

production/check_scheduler_health.sh
# data: stale last_date=2026-05-19 expected=2026-05-20 stale_days=1
# latest signal: date=2026-05-19 profile=balanced_rule_residual_v19 dry_run=true executed=false
# dashboard_static_export ... status: ok
# postclose status: due but only cached dry-run fallback completed
```

Final gates after documentation/index refresh:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 43 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=128 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 05:52:40 versions=25 missing=0 model_versions=90 latest=balanced_rule_residual_v22_soxx paper_trading_rows=3 paper_trading_end=2026-05-18 paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK

git diff --check
# clean
```

### Decision

- The scheduler timing, wrapper install, private LaunchAgent environment, cached
  fallback guard, latest-signal health, and dashboard static export path are now
  observable.
- The production loop is still not fully proven for the `2026-05-20` US session:
  providers did not produce a fresh `2026-05-20` row during the natural or manual
  post-close attempts.
- Do not resume model optimization yet. The next meaningful loop is another
  post-close data/provider readout; only a full non-dry paper execution with
  fresh data should count as production-stable.

## 2026-05-21 Provider Coverage Diagnostics

Goal: after another full wrapper scheduler attempt still fell back to cached
dry-run, make the data-provider bottleneck leg-level instead of only reporting
the merged panel `last_date`.

### Readout

Command:

```bash
SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh \
  --broker paper --no-notify
```

Result:

- Ran at `2026-05-21T06:01:51+08:00`.
- `yfinance`: `No overlapping dates for SOXL/SOXX/VIX`.
- `yahoo_api`: Yahoo `429 Too Many Requests` on `^VIX`.
- `polygon`: completed the fetch/write path but produced `new_rows=0` and
  `last_date=2026-05-19`.
- The scheduler again fell back to cached dry-run; paper broker state stayed at
  `date=2026-05-18`, `shares=3000`, `position=0.4856047894`.

### Change

- `scripts/update_market_data.py`
  - now returns `source_ranges` for `SOXL`, `SOXX`, and `VIX`.
  - now returns `overlap_last_date`.
  - prints a line like:
    `source_ranges SOXL=... SOXX=... VIX=... overlap_last_date=...`
    before stale/freshness decisions.
- `tests/test_update_market_data.py`
  - added a regression proving a lagging leg is visible in
    `source_ranges` and `overlap_last_date`.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_update_market_data.py::test_update_master_csv_reports_provider_leg_coverage_when_vix_lags -q
# red first: KeyError: 'source_ranges'
# green: 1 passed

SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh \
  --broker paper --dry-run --data-provider polygon --no-notify --skip-dashboard
# [update] source_ranges SOXL=2026-05-11..2026-05-19(7) SOXX=2026-05-11..2026-05-19(7) VIX=2026-05-11..2026-05-19(7) overlap_last_date=2026-05-19
# [paper] WARNING: using stale data for dry-run only: last_date=2026-05-19 expected=2026-05-20 stale_days=1
# [scheduler] all done
```

Final gates:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_update_market_data.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 46 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=129 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 06:02:28 versions=25 missing=0 model_versions=90 latest=balanced_rule_residual_v22_soxx paper_trading_rows=3 paper_trading_end=2026-05-18 paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK

git diff --check
# clean

production/check_scheduler_health.sh
# data: stale last_date=2026-05-19 expected=2026-05-20 stale_days=1
# latest signal: date=2026-05-19 profile=balanced_rule_residual_v19 dry_run=true executed=false
# postclose status: due but only cached dry-run fallback completed
```

### Decision

- The `2026-05-20` blocker is provider coverage, not a single missing VIX leg:
  under Polygon/CBOE all three legs currently end at `2026-05-19`.
- The execution guard is behaving correctly: it preserves scheduled inference
  visibility while preventing paper broker mutation on stale data.
- Keep model optimization paused until the provider surface advances and a
  non-dry post-close paper run can be proven.

## 2026-05-21 Post-Close Full Success

Goal: finish the production proof that was blocking renewed SOXX-aware model
optimization: fresh data -> scheduled inference -> non-dry paper execution ->
dashboard/Vercel static bundle -> LaunchAgent health success.

### Root Cause

- A wrapper scheduler run at `2026-05-21T06:09:34+08:00` showed yfinance had
  SOXL, SOXX, and VIX rows through `2026-05-20`, but merge still failed:
  `SOXL=2026-05-11..2026-05-20(8)`,
  `SOXX=2026-05-11..2026-05-20(8)`,
  `VIX=2026-05-11..2026-05-20(8)`,
  `overlap_last_date=none`.
- The fetched provider frames could carry different `date` dtypes/timezones or
  intraday times, while `scripts/update_market_data.py` normalized dates only
  after merging. The merge had to normalize the provider date keys before the
  join.
- After the panel advanced, a separate data-surface check showed raw mirrors
  still ending `2026-05-19`; the production runner updated canonical panel and
  qlib, but did not call `sync_local_data_surfaces()`.

### Changes

- `scripts/update_market_data.py`
  - normalizes provider `date` columns to day-level naive timestamps before
    diagnostics and merge.
- `production/soxl_paper_trading.py`
  - calls `sync_local_data_surfaces()` after a successful production data update
    and before `refresh_qlib()`.
- Tests:
  - `test_update_master_csv_normalizes_provider_dates_before_merge`
  - `test_data_update_syncs_raw_surfaces_before_qlib_refresh`

### Execution Evidence

Manual wrapper run:

```bash
SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh --broker paper --no-notify
# yfinance source_ranges ... overlap_last_date=2026-05-20
# before=4073 after=4074 new_rows=1 last_date=2026-05-20
# data fresh: last_date=2026-05-20 expected=2026-05-20 stale_days=0
# as_of=2026-05-20
# order submitted=True status=filled
# [scheduler] all done
```

Post-run state:

- canonical panel: `4074` rows, ending `2026-05-20`
- raw SOXL/SOXX/VIX mirrors: `4074` rows, ending `2026-05-20`
- paper broker: `date=2026-05-20`, `shares=2800`,
  `position=0.4840856822`, `account_value=1001806.1044`
- summary: `runtime/soxl/paper_trading/2026-05-20/summary.json`,
  `dry_run=false`, `executed=true`
- paper dashboard CSV: `4` rows, latest row `2026-05-20`, action `SELL`
- dashboard static export: paper version ending `2026-05-20` with `4` rows

LaunchAgent proof:

```bash
launchctl kickstart -k "gui/$(id -u)/com.soxl.daily.scheduler"
# data fresh: last_date=2026-05-20 expected=2026-05-20 stale_days=0
# already executed for 2026-05-20 broker=paper; skipping duplicate order
# [scheduler] all done

production/check_scheduler_health.sh
# data: ok last_date=2026-05-20 expected=2026-05-20
# latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true
# dashboard_static_export ... paper_trading_rows=4 paper_trading_end=2026-05-20
# postclose status: success marker found
```

Dashboard visual proof:

- `runtime/production/card/postclose_full_success_v1/dashboard_visual_2026-05-21.png`
  - captured at `1440 x 1000`
  - non-white pixels: `99.35%`

Final gates:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_update_market_data.py tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 48 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=130 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 06:13:40 versions=25 missing=0 model_versions=90 latest=balanced_rule_residual_v22_soxx paper_trading_rows=4 paper_trading_end=2026-05-20 paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK

git diff --check
# clean
```

### Decision

- The production path is now proven through both the repo wrapper and the macOS
  LaunchAgent health surface.
- Same-day idempotency prevented duplicate paper execution when the LaunchAgent
  proof was kicked after the manual filled order.
- Model optimization can resume from the SOXX-aware / selective-release frontier
  after this commit; the production loop is no longer the blocker.

## 2026-05-21 v22/Post-Release-Cap Runtime Auto-Select Approval

Goal: resume model optimization after SOXX diagnostics by finishing the governed
v22/post-release-cap promotion path, while preserving the working paper loop.

### Baselines And Role

- Canonical baseline before this loop: `balanced_rule_residual_v19`
- Direct comparison baseline: `balanced_rule_residual_v22` with
  `execution_overlay.auto_select_allowed=false`
- Role: `replace`
- Changed layer: `selection` plus explicit `execution` overlay governance
- External reference: walk-forward model-selection evidence; use complete
  out-of-sample walk-forward windows rather than training `best_score`
- Target window: scheduled runtime auto-selection and paper dry-run on latest
  local data
- Accepted trade-off: v22 has worse `wf_2022` drawdown than v19 but remains
  inside gate and improves complete walk-forward mean score and recovery
  behavior
- Comparison-table row key: `v22_post_release_cap_runtime_auto_select_v1`

Closest history:

1. Success case: `selective_release_post_release_cap_v1` improved
   `wf_2023_2024` return and MaxDD while leaving `wf_2022` unchanged.
2. Failure case: `soxx_relative_weakness_walkforward_v1` had fixed-window gains
   but was absorbed in saved walk-forward actors.
3. Not a duplicate: this loop changes runtime model-selection evidence and
   explicit overlay approval; it does not tune SOXX cap thresholds or retrain
   the actor.

### Changes

- `src/soxl_quant/runtime.py`
  - added complete walk-forward runtime selection scoring.
  - uses `relative_outperformance`, Sharpe, and MaxDD with the profile's
    existing `selection_score` weights.
  - falls back to legacy `best_score` when complete walk-forward reports are
    missing.
- `runtime/model/card/balanced_rule_residual_v22/profile.yaml`
  - set `execution_overlay.auto_select_allowed=true`.
- `production/soxl_paper_trading.py`
  - dry-run after a same-day executed paper run now writes under
    `<date>/dry_run/` instead of overwriting root `summary.json`.
  - duplicate guard can recover from paper broker state plus the cumulative
    paper CSV if the root summary was already polluted.
- Registry:
  - `balanced_rule_residual_v22` is now `canonical`.
  - `balanced_rule_residual_v19` is retained as archived rollback actor.

### Evidence

Runtime selection scores:

| profile | score | source |
|---|---:|---|
| `balanced_rule_residual_v22` | `64.8811` | `walk_forward_mean` |
| `balanced_rule_residual_v19` | `46.1670` | `walk_forward_mean` |

Review cards:

- `runtime/model/card/balanced_rule_residual_v22/runtime_selection_review.json`
  - `decision=approved`
  - `runtime_auto_select_allowed=true`
  - `current_auto_select_profile=balanced_rule_residual_v22`
- `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json`
  - `runtime_auto_select_allowed=true`
  - `runtime_auto_select_reason=promotion review, profile flag, and model gate all allow auto-selection`

Default paper dry-run:

```bash
PYTHONPATH=src:. qlib_venv/bin/python production/soxl_paper_trading.py --skip-data-update --dry-run --no-notify
# [auto_select] family=balanced model=soxl_balanced_rule_residual_v22_actor.pt score=64.88 source=walk_forward_mean
# [paper] auto-selected profile=balanced_rule_residual_v22 model=soxl_balanced_rule_residual_v22_actor.pt
# [paper] as_of=2026-05-20
# [paper] signal action=0 target=50.00% price=$173.20
# [paper] risk_gate approved=True
# [paper] dry-run: skipped order placement
# [paper] output=/Users/rziiiii/.openclaw/workspace/runtime/soxl/paper_trading/2026-05-20/dry_run
```

Healthcheck:

```bash
production/check_scheduler_health.sh
# data: ok last_date=2026-05-20 expected=2026-05-20
# model: ok balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)
# latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true
# postclose status: success marker found
```

### Decision

- v22/post-release-cap is now the canonical scheduled selection path.
- Do not resume SOXX cap threshold tuning; the SOXX fixed-actor cap failed
  walk-forward absorption.
- Next natural proof is the next post-close non-dry paper run selecting v22 with
  fresh data.

## 2026-05-21 Dashboard Static Default To Paper Trading

Goal:

- Keep the Vercel/static dashboard aligned with the current simulated paper
  loop instead of opening on the newest offline post-train artifact.

Root cause:

- `scripts/build_backtest_dashboard.py::select_latest_version()` prioritized
  `post_train_review` over generalization snapshots, but did not treat
  `paper_trading` as the production/live default source.
- The regenerated static export therefore had
  `latest_version=balanced_rule_residual_v22_soxx`, even though
  `paper_trading` was present and ended on `2026-05-20`.

Changes:

- Added a regression test requiring `paper_trading` to win static dashboard
  default selection when exported.
- Added source-kind priority:
  `paper_trading > post_train_review > generalization_version`.
- Strengthened `scripts/check_dashboard_static_export.py` so an exported
  `paper_trading` version requires `latest_version=paper_trading`.
- Regenerated `dashboard.html` and `dashboard_data/`.

Evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py::test_select_latest_version_prefers_paper_trading_for_static_dashboard -q
# initially failed: expected paper_trading, got post_train

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_vercel_dashboard.py::test_dashboard_static_export_integrity_check_passes -q
# initially failed: stdout had latest=balanced_rule_residual_v22_soxx

PYTHONPATH=src:. qlib_venv/bin/python scripts/build_backtest_dashboard.py
# wrote dashboard.html, dashboard_data/index.json, dashboard_data/model_versions.json, 25 version payloads

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 06:51:30 versions=25 missing=0 model_versions=90 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 paper_trading

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py tests/test_vercel_dashboard.py -q
# 13 passed
```

Browser/render evidence:

- Browser plugin Node REPL tooling was not exposed in this session, and
  Computer Use was blocked on macOS permissions.
- Fallback headless Chrome loaded
  `http://127.0.0.1:8876/dashboard.html?codex_cdp=20260521`.
- Runtime DOM readout:
  `readyState=complete`, `latest=paper_trading`,
  `selectValue=paper_trading`, `selectedText="paper_trading (SOXL)"`,
  `optionCount=26`.
- Screenshot proof:
  `runtime/production/card/dashboard_static_default_v1/dashboard_static_default_2026-05-21.png`,
  `1440 x 1100`, coarse sample `15481 / 15840` non-white pixels.

Decision:

- Static/Vercel dashboard default is now the simulated paper-trading view.
- Historical SOXX/post-train snapshots remain available in the version selector
  but no longer override the production paper view on first load.
- The model-side next proof remains the next fresh post-close non-dry paper run
  selecting v22.

## 2026-05-21 v23/v24 Model Retention Audit

Goal:

- Resume model-side cleanup after the SOXX and dashboard loops by removing stale
  ambiguity around old `balanced_rule_residual_v23` / `v24` summaries.

Baseline and governance:

- Canonical baseline: `balanced_rule_residual_v22` with
  `selective_release_recovery_post_release_cap_v1` approved for scheduled
  runtime selection.
- Direct comparisons: summary-only `balanced_rule_residual_v23` and
  `balanced_rule_residual_v24`.
- Role: `diagnose` / cleanup.
- Changed layer: `selection` evidence and retention metadata.
- External reference: model-selection discipline from
  `docs/research_knowledge/model_selection.md`; fixed-split-only gains are not
  promotion evidence without actor retention and walk-forward validation.
- Target window: current model registry / active surface.
- Accepted trade-off: keep stale summaries searchable while removing them from
  active-candidate interpretation.
- Comparison-table row keys:
  `balanced_rule_residual_v23`, `balanced_rule_residual_v24`.

Closest history:

1. Success case: `v22_post_release_cap_runtime_auto_select_v1` promoted a model
   only after corrected walk-forward and runtime-selection evidence.
2. Failure case: `balanced_rule_residual_v22_soxx` improved some fixed-split
   recovery metrics but failed the gate and later SOXX cap walk-forward
   absorption.
3. Not a duplicate: this is not a threshold or reward tweak; it audits stale
   summary-only profiles and fixes registry/knowledge drift before the next
   real experiment.

Evidence:

```bash
qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py \
  --latest runtime/model/card/balanced_rule_residual_v23/soxl_balanced_rule_residual_v23_summary.json \
  --previous runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json \
  --latest-label balanced_rule_residual_v23 \
  --previous-label balanced_rule_residual_v22 \
  --output runtime/model/card/balanced_rule_residual_v23/v23_vs_v22_review.md
# Wrote runtime/model/card/balanced_rule_residual_v23/v23_vs_v22_review.md

qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py \
  --latest runtime/model/card/balanced_rule_residual_v24/soxl_balanced_rule_residual_v24_summary.json \
  --previous runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json \
  --latest-label balanced_rule_residual_v24 \
  --previous-label balanced_rule_residual_v22 \
  --output runtime/model/card/balanced_rule_residual_v24/v24_vs_v22_review.md
# Wrote runtime/model/card/balanced_rule_residual_v24/v24_vs_v22_review.md
```

Readout:

- `v23`: proxy return `493.23%` vs v22 `695.45%`; MaxDD improves to
  `51.20%`, but `test_2020` and `test_2023_2024` deteriorate and checkpoint
  selection peaks at iter `0`.
- `v24`: proxy return `758.59%` vs v22 `695.45%`, but proxy MaxDD is slightly
  worse, `test_2020` deteriorates, and there is no walk-forward evidence.
- Both summaries point to missing actor paths under `artifacts/models/`.
- Neither card has a retained `profile.yaml`.

Changes:

- Added `notes.md` and generated review markdown for v23/v24.
- Added registry rows with `status=dropped`, `keep_actor=false`,
  `keep_summary=true`.
- Added master table rows in `docs/experiment-landscape-review.md`.
- Corrected stale `runtime/model/knowledge.md` text that still described v22
  scheduler auto-selection as blocked.

Decision:

- v23/v24 are not active candidates.
- Do not attempt walk-forward validation for them because their actors are
  missing.
- Next model optimization should be a distinct governed mechanism, not a
  revival of these summary-only branches or another SOXX threshold tweak.

## 2026-05-21 v19_reg_v22 Artifact Gap Audit

Goal:

- Continue model-side cleanup by checking whether
  `balanced_rule_residual_v19_reg_v22` is a viable next candidate or another
  stale active-surface artifact.

Baseline and governance:

- Canonical baseline: `balanced_rule_residual_v22` with
  `selective_release_recovery_post_release_cap_v1` approved for scheduled
  runtime selection.
- Direct comparison: `balanced_rule_residual_v19_reg`.
- Role: `diagnose` / cleanup.
- Changed layer: `feature` + `network`.
- External reference: VIX term-structure feature family plus regularization
  discipline from `docs/research_knowledge/networks.md` and
  `docs/research_knowledge/model_selection.md`.
- Target window: current model registry / active surface.
- Accepted trade-off: preserve the old profile as searchable context while
  removing it from candidate interpretation.
- Comparison-table row key: `balanced_rule_residual_v19_reg_v22`.

Closest history:

1. Success case: `balanced_rule_residual_v19_reg` produced real walk-forward
   evidence, improving `wf_2022` while sacrificing `wf_2023_2024`.
2. Failure case: `balanced_rule_residual_v22_soxx` and direct release-feature
   injection showed fixed-window gains that did not survive walk-forward or
   execution absorption.
3. Not a duplicate: this loop does not tune thresholds or start PPO training;
   it fixes stale retention metadata before opening a fresh governed run.

Evidence:

```bash
rg -n "v19_reg_v22|balanced_rule_residual_v19_reg_v22" \
  runtime/harness/card/registry/profile.yaml runtime/model/knowledge.md \
  docs/experiment-landscape-review.md docs/active_workpad.md
# found only registry/profile references plus older knowledge claims

find runtime model experiments -name '*v19_reg_v22*' -maxdepth 5 2>/dev/null | sort
# runtime/model/card/balanced_rule_residual_v19_reg_v22

ls -la runtime/model/card/balanced_rule_residual_v19_reg_v22
# profile.yaml only before this audit
```

Decision:

- `balanced_rule_residual_v19_reg_v22` is profile-only in the current repo.
- The older fixed-split pass and "walk-forward started" note are not durable
  promotion evidence because no actor, summary JSON, or walk-forward output is
  present.
- Registry now marks it as `dropped`, with notes at
  `runtime/model/card/balanced_rule_residual_v19_reg_v22/notes.md`.
- Do not revive it unless a new governed experiment is opened with fresh
  retained artifacts and complete walk-forward validation.

## 2026-05-21 v22 Checkpoint Asset Audit

Goal:

- Check whether the canonical v22 actor can be improved through checkpoint
  re-selection before opening a new PPO retrain.

Baseline and governance:

- Canonical baseline: `balanced_rule_residual_v22` with
  `selective_release_recovery_post_release_cap_v1`.
- Direct comparison: existing v22 selection history and retained actor.
- Role: `diagnose`.
- Changed layer: `selection`.
- External reference: checkpoint-selection discipline from
  `docs/research_knowledge/model_selection.md`.
- Target window: existing v22 training summary and retained assets.
- Accepted trade-off: do not retrain or change runtime behavior in this loop.
- Comparison-table row key: `v22_checkpoint_asset_audit_v1`.

Evidence:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/checkpoint_review.py \
  --summary runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json \
  --output runtime/model/card/balanced_rule_residual_v22/v22_checkpoint_asset_review.md
# [done] wrote runtime/model/card/balanced_rule_residual_v22/v22_checkpoint_asset_review.md
```

Readout:

- v22 selection history captures only iter `0` and iter `80`, despite staged
  training with `stage_iters=20`.
- Iter `80` is the best-scored retained checkpoint: score `91.94` vs iter `0`
  score `89.54`.
- The current repo retains only
  `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt`.
- No iter `20`, `40`, or `60` actors are available for replay.

Decision:

- No no-retrain checkpoint switch is available for v22.
- Do not claim the model is under-selected; the asset gap prevents direct
  checkpoint rescue.
- If the next model run is approved, the harness should retain stage/top-N
  actors so future checkpoint reviews can compare actual actor paths, not just
  summary rows.

## 2026-05-21 Scheduler Runtime-Proof Visibility

Goal:

- Make the remaining production proof explicit in the health check: current
  model selection is v22, but latest non-dry paper execution is still v19.

Baseline and evidence:

- `production/check_scheduler_health.sh` before the patch printed both facts but
  did not classify their mismatch.
- Current live output before/after the patch:
  - model: `balanced_rule_residual_v22`
  - latest executed artifact:
    `runtime/soxl/paper_trading/2026-05-20/summary.json`
  - latest executed profile: `balanced_rule_residual_v19`

TDD evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_pending_runtime_profile_proof -q
# red before implementation, green after implementation

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 8 passed

production/check_scheduler_health.sh
# runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 selected_model=balanced_rule_residual_v22
```

Change:

- `production/check_scheduler_health.sh` now compares the current
  healthcheck-selected profile against the latest non-dry executed paper
  summary profile.
- It prints `runtime proof: pending ...` on mismatch and `runtime proof: ok ...`
  once the latest executed paper artifact matches the selected model.
- Evidence card:
  `runtime/production/card/v22_runtime_proof_pending_v1/notes.md`

Decision:

- This does not manufacture the missing proof; it makes the proof gap visible.
- The production goal remains active until the next fresh post-close non-dry
  paper run naturally writes a v22 executed artifact.

## 2026-05-21 Vercel Static Bundle Audit

Goal:

- Isolate the Vercel/dashboard evidence from the larger production workpad so
  goal completion can be audited directly.

Evidence:

```bash
cat vercel.json
# "/" rewrites to "/dashboard.html"
# no-cache headers are configured for "/", "/dashboard.html", and "/dashboard_data/:path*"

find dashboard_data/versioned -maxdepth 2 -type f | wc -l
# 25

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_vercel_dashboard.py -q
# 3 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 06:51:30 versions=25 missing=0 model_versions=90 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 paper_trading
```

Decision:

- Repo-owned Vercel static packaging is complete for the dashboard bundle:
  `vercel.json`, `dashboard.html`, `dashboard_data/index.json`,
  `dashboard_data/model_versions.json`, and 25 versioned SOXL payloads are
  present and validated.
- The deployed first-load view would be `paper_trading`.
- This does not prove a remote Vercel URL is live; it proves the repository
  bundle and routing config are ready for Vercel serving.
- Evidence card:
  `runtime/production/card/vercel_static_bundle_audit_v1/notes.md`

## 2026-05-21 Checkpoint Retention Harness Support

Goal:

- Remove the immediate model-optimization harness bottleneck exposed by the v22
  checkpoint asset audit: stage summaries existed, but intermediate actors were
  not replayable.

Design:

- Add default-off profile config:
  `checkpoint_retention.enabled`, `checkpoint_retention.keep_stage`, and
  `checkpoint_retention.keep_top_n`.
- Keep selection and persistence separated:
  - `src/soxl_quant/app/selection.py` returns retained checkpoint candidates.
  - `src/soxl_quant/app/artifacts.py` writes actor bundles from retained
    `state_dict`s.
  - `src/soxl_quant/app/train.py` stores summary `retained_checkpoints`.
- Do not change existing profile behavior unless an experiment explicitly opts
  into checkpoint retention.

TDD evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_checkpoint_retention.py -q
# red first on missing retention interfaces; green after implementation
# 6 passed
```

Implemented files:

- `src/soxl_quant/config.py`
- `src/soxl_quant/core/config.py`
- `src/soxl_quant/settings.py`
- `src/soxl_quant/app/selection.py`
- `src/soxl_quant/app/artifacts.py`
- `src/soxl_quant/app/train.py`
- `tests/test_checkpoint_retention.py`
- `scripts/check_harness.py`

Decision:

- This is harness support, not a promoted model change.
- The next model experiment can now opt in to stage/top-N actor retention and
  produce replayable checkpoint evidence.
- Evidence card:
  `runtime/harness/card/checkpoint_retention_v1/notes.md`

## 2026-05-21 Runtime-Proof Waiting-State Clarification

Goal:

- Clarify the current production state after v22 auto-select approval: the
  latest available data and scheduler are healthy, but the latest already
  executed paper date was filled by v19 before v22 became the selected runtime
  model.

Root cause:

- The live post-close run at `2026-05-21 06:12 CST` refreshed data through
  `2026-05-20` and executed/summarized the paper date with
  `balanced_rule_residual_v19`.
- v22 auto-select approval happened later in the repo history, so current
  health now selects `balanced_rule_residual_v22`.
- The paper runner correctly refuses a duplicate non-dry execution for the same
  signal date, so the missing v22 proof cannot be manufactured without
  corrupting same-day idempotency.

TDD evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_pending_runtime_profile_proof -q
# red first on missing detail line; green after implementation
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 44 passed

production/check_scheduler_health.sh
# data: ok last_date=2026-05-20 expected=2026-05-20
# model: ok balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)
# latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true
# runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 selected_model=balanced_rule_residual_v22
# runtime proof detail: latest executed date already matches expected data date; next non-dry-run execution must use selected model
```

Change:

- `production/check_scheduler_health.sh` now carries the paper health
  `expected_last_trading_day` into the runtime-proof section.
- When the latest executed date already equals expected data date but the
  executed profile differs from the current selected model, health reports that
  the proof waits for the next non-dry-run trading day instead of implying a
  data/scheduler failure.

Decision:

- Do not re-execute `2026-05-20` just to force a v22 artifact.
- Keep the production goal active until the next fresh post-close non-dry run
  writes a v22 executed summary naturally.
- Evidence card:
  `runtime/production/card/v22_runtime_proof_pending_v1/notes.md`

## 2026-05-21 Scheduler Next-Run Health Display

Goal:

- Remove ambiguity from `production/check_scheduler_health.sh` after the local
  post-close window has already passed.

Root cause:

- The health script printed a static `Next Scheduled Runs (today)` block.
- At `2026-05-21 08:00 CST`, the `05:35 Tue-Sat Asia/Shanghai` post-close run
  was already in the past, but the script still listed it under "today".

Change:

- The next-run section now labels each schedule from
  `SOXL_SCHEDULER_HEALTH_WEEKDAY` and `SOXL_SCHEDULER_HEALTH_HHMM`.
- Pre-open/post-close windows show `today` only when that local window is still
  ahead on an eligible day; otherwise they show `next eligible day`.
- Weekly walk-forward shows `today` only before Saturday `07:00`, otherwise
  `next Saturday`.

TDD evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_next_runs_skip_elapsed_postclose_window -q
# red first on static "today" output; green after implementation
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 49 passed

production/check_scheduler_health.sh
# --- Next Scheduled Runs ---
#   pre-open: today 21:00 Mon-Fri Asia/Shanghai — US pre-open dry-run + dashboard
#   post-close: next eligible day 05:35 Tue-Sat Asia/Shanghai — US post-close data + paper trading + dashboard
#   weekly: next Saturday 07:00 Sat Asia/Shanghai — weekly walk-forward after post-close refresh
```

Decision:

- This is an observability fix, not a schedule or execution change.
- The remaining completion proof is still the next fresh post-close non-dry
  paper artifact under v22.
- Evidence card:
  `runtime/production/card/scheduler_next_runs_v1/notes.md`

## 2026-05-21 v22 Continued-Retention Diagnostic

Goal:

- Test whether the canonical v22 actor was merely undertrained or under-selected
  now that checkpoint retention can save replayable stage actors.

Governance:

- Profile: `balanced_rule_residual_v22_continued_retention_v1`
- Direction: `model_selection`
- Role: `diagnose`
- Changed layer: `selection`
- Baseline: `balanced_rule_residual_v22`
- External reference: `docs/research_knowledge/model_selection.md`

Command:

```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v22_continued_retention_v1 --skip-data-refresh --seed 20260521
```

Evidence:

- Retained checkpoints: iter `0`, `10`, and `20`.
- Selection stayed at iter `0`: score `86.24`; iter `10` score `77.14`;
  iter `20` score `69.26`.
- `test_proxy` return improved from v22 `695.45%` to `1128.94%`, but MaxDD
  worsened from `54.52%` to `59.13%`.
- `test_2022` deteriorated from `-24.52% / MaxDD 32.71% / avg_pos 0.269` to
  `-51.24% / MaxDD 55.26% / avg_pos 0.321`, failing the fixed-split MaxDD
  limit by `0.26pp`.
- Promotion: blocked.

Harness fix:

- The run exposed summary JSON being persisted before main-backtest metric
  corrections.
- Fixed `src/soxl_quant/app/train.py` so `persist_summary_bundle()` runs after
  `backtest_profile` corrections and promotion rebuild.
- Added `tests/test_train_summary_persistence.py` and wired it into
  `scripts/check_harness.py`.

Decision:

- Marked registry status as `ablation`, not candidate.
- Do not walk-forward or runtime-bind this profile.
- Keep actor artifacts only as checkpoint-retention proof.
- Current production goal remains active: the missing proof is still the next
  fresh post-close non-dry paper execution under selected v22.

## 2026-05-21 Weekly Walk-Forward Scheduler Surface

Goal:

- Remove the mismatch where scheduler health showed a weekly walk-forward next
  run and `wf_weekly.log`, but only pre-open/post-close LaunchAgents existed.

Root cause:

- `production/check_scheduler_health.sh` printed a weekly next-run line.
- `production/crontab.txt` still had a legacy weekly command using
  `/home/rziiiii/dev/workspace` and dropped `balanced_trade_lock`.
- There was no repo-owned weekly LaunchAgent or wrapper for the current
  canonical runtime profile.

Change:

- Added `production/run_soxl_weekly_wf.sh`.
- Added `production/com.soxl.weekly.wf.plist` for Saturday `07:00`
  Asia/Shanghai, after post-close refresh.
- The wrapper refreshes corrected walk-forward evidence for
  `balanced_rule_residual_v22`, then reruns runtime-selection review with
  `--persist-model-promotion`.
- Updated health to check `com.soxl.weekly.wf` and `runtime/soxl/wf_weekly.err`.
- Installed and loaded the local LaunchAgent; health now reports weekly loaded
  and matching repo.

Evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py tests/test_scheduler_health.py -q
# 16 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.weekly.wf: loaded
# LaunchAgent com.soxl.weekly.wf plist: matches repo
```

Decision:

- Weekly walk-forward is now part of the scheduled production surface.
- The remaining full-goal proof is still the next fresh post-close non-dry
  paper execution under selected v22, plus the first Saturday weekly log once
  that schedule naturally fires.

## 2026-05-21 Vercel Remote Deployment Attempt

Goal:

- Move beyond local static-bundle readiness and make the Vercel deployment state
  auditable from the repo.

Current state:

- `vercel` CLI exists locally: `Vercel CLI 50.32.5`.
- The repository is not linked to a Vercel project: no `.vercel/project.json`.
- `vercel whoami` reports no existing credentials.
- Static dashboard export remains valid:
  `latest=paper_trading`, `paper_trading_rows=4`,
  `paper_trading_end=2026-05-20`, `missing=0`.

Change:

- Added `scripts/deploy_vercel_dashboard.py`.
- The script validates `dashboard.html`, `dashboard_data/`, and `vercel.json`
  before any deploy attempt.
- It supports non-interactive production deploy with either local Vercel login
  or `VERCEL_TOKEN`; token values are redacted from saved evidence.
- Without credentials it writes explicit blocked evidence instead of implying a
  remote deployment exists.

Evidence:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --output-dir runtime/production/card/vercel_remote_deploy_v1
# vercel_dashboard status=blocked_no_credentials

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_vercel_dashboard.py -q
# 6 passed
```

Decision:

- Vercel static packaging is ready, but remote deployment is not proven because
  this machine has no Vercel credentials.
- Next actionable Vercel step is to provide `VERCEL_TOKEN` or run
  `vercel login`, then rerun the same script with `--deploy`.

## 2026-05-21 v22 Next Execution Preflight

Goal:

- Confirm the next selected paper-trading path is v22 without violating
  same-date idempotency on the already executed `2026-05-20` signal date.

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker paper --dry-run --no-notify --skip-data-update
```

Evidence:

- Data freshness: `last_date=2026-05-20`, `expected=2026-05-20`,
  `stale_days=0`.
- Auto-selected profile: `balanced_rule_residual_v22`.
- Signal date: `2026-05-20`.
- Dry-run output path:
  `runtime/soxl/paper_trading/2026-05-20/dry_run`.
- Risk gate: approved.
- Step check: `change=0.016 limit=0.350`.
- Target/current position: `0.5000` / `0.4841`.
- Reconcile: `position aligned`, `action_required=none`.
- Root executed summary remains
  `runtime/soxl/paper_trading/2026-05-20/summary.json` with
  `profile=balanced_rule_residual_v19`, `dry_run=false`, `executed=true`.

Decision:

- The v22 paper path is preflighted and would pass the current risk gate.
- Do not force a duplicate non-dry execution for `2026-05-20`; preserve the
  v19 executed artifact and wait for the next fresh post-close signal date to
  produce the first non-dry v22 paper proof.
- Evidence card:
  `runtime/production/card/v22_next_execution_preflight_v1/notes.md`.

## 2026-05-21 Vercel Health Surface

Goal:

- Make remote Vercel deployment status visible in the same production health
  check that already reports scheduler, paper, and static-dashboard health.

Root cause:

- Static export readiness was visible through `check_scheduler_health.sh`.
- Remote deployment evidence existed in
  `runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`,
  but the operator had to inspect that card separately.

Change:

- `production/check_scheduler_health.sh` now prints a
  `Vercel Remote Deployment` section.
- The section reports status, auth state, deployment URL, static dashboard
  latest version, paper-trading end date, and the evidence path.

Evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_vercel_remote_deployment_status -q
# red first on missing health section, then green
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 10 passed

production/check_scheduler_health.sh
# --- Vercel Remote Deployment ---
# status: blocked_no_credentials
# auth: blocked_no_credentials
# deployment_url: none
# dashboard_latest=paper_trading paper_trading_end=2026-05-20
```

Decision:

- Remote Vercel deployment is still blocked by missing credentials, but the
  blocker is now part of the canonical production health output.
- Evidence card:
  `runtime/production/card/vercel_health_surface_v1/notes.md`.

## 2026-05-21 v19_reg + v22 Feature Retest

Goal:

- Run exactly one governed model optimization loop after the SOXX/Vercel/paper
  infrastructure work, without reviving stale profile-only evidence.

Setup:

- Canonical baseline: `balanced_rule_residual_v22`.
- Direct comparison: `balanced_rule_residual_v19_reg`.
- Profile: `balanced_rule_residual_v19_reg_v22_retest_v1`.
- Changed layer: `feature` + `network`.
- External reference: VIX term-structure feature family plus dropout/weight
  decay regularization.

Command:

```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_reg_v22_retest_v1 --skip-data-refresh --seed 20260521
```

Evidence:

- Summary:
  `runtime/model/card/balanced_rule_residual_v19_reg_v22_retest_v1/soxl_balanced_rule_residual_v19_reg_v22_retest_v1_summary.json`
- Review:
  `runtime/model/card/balanced_rule_residual_v19_reg_v22_retest_v1/v19_reg_v22_retest_review.md`
- Dashboard version:
  `dashboard_data/versioned/balanced_rule_residual_v19_reg_v22_retest_v1/SOXL.json`
- Health check after dashboard refresh:
  `dashboard_static_export generated_at=2026-05-21 09:40:42 versions=27
  missing=0 model_versions=93 latest=paper_trading`; Vercel remote deployment
  remains `blocked_no_credentials`; runtime proof remains pending until the
  next fresh non-dry paper execution selects v22.

Readout:

- Proxy: `1280.4%` return / `1.020` Sharpe / `59.6%` MaxDD.
- `test_2022`: `-54.1%` return / `58.3%` MaxDD / `0.381` avg position.
- Fixed-split promotion is blocked because `test_2022_max_dd=58.30% > 55%`
  and `test_2022_avg_pos=0.381 > 0.35`.
- Versus canonical v22, 2022 MaxDD worsens from `32.7%` to `58.3%`.
- Versus direct v19_reg, 2023-2024 return falls from `58.7%` to `44.5%`.

Decision:

- Marked the profile as `ablation`.
- Do not run walk-forward and do not bind runtime.
- Keep retained checkpoints only as replay evidence for this governed retest.
- The model-optimization readout is now narrower: v22 is not simply
  undertrained, and v19_reg + v22 feature fusion increases exposure rather than
  fixing bear-market risk.
