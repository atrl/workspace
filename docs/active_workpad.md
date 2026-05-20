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

Answer: no. The repo now has a model-side runtime selection sidecar, and it blocks automatic selection because the base v22 model sidecar has not completed walk-forward promotion and the profile overlay flag remains `auto_select_allowed=false`.

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
- Passing checks:
  - `model_fixed_split_eligible=true`
  - `overlay_review_candidate=true`
  - `overlay_runtime_profile_bound=true`
- Blocking checks:
  - `model_walk_forward_evaluated=false`
  - `model_promotion_eligible=false`
  - `profile_overlay_auto_select_allowed=false`
- Current auto-selected profile remains `balanced_rule_residual_v19`.

### Decision

- Do not turn on `balanced_rule_residual_v22.execution_overlay.auto_select_allowed`.
- Do not patch `soxl_balanced_rule_residual_v22_summary.json` into a promoted state.
- The post-release cap line is now fully governed as a blocked runtime-selection candidate.
- Next model work should be a separate narrow optimization experiment unless a future run first produces real walk-forward model-promotion evidence for v22/post-release-cap.

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
