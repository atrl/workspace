# PRODUCTION Knowledge

> 最后更新: 2026-05-21
> 该方向负责定时任务、数据刷新到推理闭环、paper/Futu 模拟盘、通知、dashboard 发布与运行证据。

## 当前结论

1. **生产闭环优先级高于继续盲目训练**：SOXX 数据已进入 canonical panel，下一步应先保证定时数据刷新、推理、风控、持久化和 dashboard 更新可重复。
2. **scheduler 失败根因已定位**：`soxl_daily_scheduler.py` 原先丢弃 child stderr，只看到 `paper trading failed with exit code 1`，所以无法识别 paper runner 内部的 `Too Many Requests` / `Rate limited` / `No overlapping dates for SOXL/SOXX/VIX` 并回退到 cached data。
3. **本地数据面已重新对齐**：canonical panel、raw SOXL/SOXX/VIX mirrors、qlib calendar 现在全部是 `4071` rows，结束于 `2026-05-15`。
4. **paper dashboard 接入已修复第一层**：dashboard builder 现在允许 `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv` 少于 10 行时也进入 version list，新模拟盘不会因为历史太短而完全不可见。
5. **paper CSV append 已修复第一层**：paper runner 现在用 post-execution broker snapshot 写 cumulative dashboard row，避免 `BUY` 但 `position=0.0` 的不可能状态。
6. **auto-select 不能再静默回退到未晋升 actor**：`best_model_for_family()` 现在只返回 promotion sidecar 通过的候选；没有 promoted candidate 时直接失败，避免 scheduler 选中 `balanced` 后又被 runtime gate 拒绝。
7. **冷启动 paper account 不再永久卡在 step gate**：paper mode 默认启用 step clipping，先按 `max_position_change=0.35` 向目标仓位移动，并把 cash/shares 持久化到 `runtime/soxl/paper_trading/paper_broker_state.json`。
8. **daily drawdown circuit-breaker 不再破坏 frozen decision contract**：触发时用 blocked copy 替换 risk-gate decision，不再原地修改 frozen dataclass。
9. **Vercel 静态发布路径已确定**：`dashboard.html` 与 `dashboard_data/` 作为静态 export 入库，`vercel.json` 将 `/` rewrite 到 `/dashboard.html`，并对 dashboard HTML/data 设置 `Cache-Control: public, max-age=0, must-revalidate`。
10. **auto-select 现在同时看 model gate 和 profile overlay gate**：`best_model_for_family()` 会跳过绑定 execution overlay 但 `auto_select_allowed=false` 的 profile。这样即使 `balanced_rule_residual_v22` 的 model sidecar 将来被标 promoted，scheduler 也不会绕过 post-release-cap 的 runtime approval。
11. **定时表已按 Asia/Shanghai 本地时区修正**：pre-open 改成 `21:00 Mon-Fri` cached-data `--dry-run`，post-close 完整数据刷新 + 模拟盘 + dashboard 改成 `05:35 Tue-Sat`，避免本地 `09:30/16:35` 误跑在美股收盘前。
12. **scheduler health 现在匹配 macOS LaunchAgent**：`production/check_scheduler_health.sh` 会在 Darwin 上检查 `launchctl list` 中的 `com.soxl.daily.scheduler`，并用 portable `stat/date` 读取日志，不再把 macOS 运行态显示成空白 cron 服务。
13. **macOS 定时推理路径已补齐**：新增 `com.soxl.preopen.scheduler`，用于 `21:00 Mon-Fri Asia/Shanghai` cached-data dry-run 推理 + dashboard；`com.soxl.daily.scheduler` 继续负责 `05:35 Tue-Sat` post-close 完整流程。两个 LaunchAgent 已同步到 `~/Library/LaunchAgents/` 并 loaded。
14. **数据源失败时不再用旧数据推进模拟盘**：post-close 数据刷新遇到 transient provider failure 时，scheduler fallback 会追加 `--skip-data-update --dry-run`，只保留 cached signal/dashboard 可见性，不落 paper order / broker state。
15. **scheduler health 现在直接暴露 paper 数据新鲜度**：`production/check_scheduler_health.sh` 会运行 paper healthcheck 并显示 `last_date`、`expected_last_trading_day`、`stale_days`、model 和 broker 状态。当前 `2026-05-21 02:48 CST` 读数已恢复为 `last_date=2026-05-19`、`expected=2026-05-19`、`stale_days=0`；到 `05:35 Tue-Sat Asia/Shanghai` 后才应该期待刚收盘的下一交易日。
16. **stale data 现在也是执行硬门槛**：paper 主流程在 qlib init / 模型选择 / broker 执行前检查 freshness。非 dry-run 遇到 stale data 直接退出；scheduler 将其转成 cached dry-run fallback，避免 stale 数据推进 paper broker state。
17. **freshness 期望日已按纽约收盘边界修正**：`_last_trading_day()` 使用 `America/New_York` 和 `17:30 ET` data-ready cutoff。`2026-05-21 02:23 CST` 期望 `2026-05-19`，`05:35 CST` 后才期望 `2026-05-20`。
18. **production data refresh 现在支持 provider failover**：paper runner 和 scheduler 都支持 `--data-provider {auto,yfinance,yahoo_api,polygon}`；默认 `auto` 先试 yfinance，再试 yahoo_api，最后试 Polygon + CBOE VIX。当前实测 yfinance 是 `No overlapping dates`，yahoo_api 是 `429 Too Many Requests`，Polygon + CBOE 能把数据更新到 `2026-05-19`。
19. **scheduler 成功日志现在保留 provider fallback stderr**：`production/soxl_daily_scheduler.py` 在 child step 成功时也会 replay stderr，因此 yfinance/yahoo_api 失败后由 polygon 接管的路径能出现在 LaunchAgent 日志里，不再只看到最终成功 stdout。
20. **paper full-loop 已能推进模拟盘状态**：`2026-05-21 03:02 CST` 手动运行 default paper broker 的 full scheduler，auto provider 经 yfinance/yahoo_api fallback 到 polygon，数据 fresh，风险门通过，paper BUY filled；broker state 更新到 `2026-05-18`、`3000` shares、`48.56%` position，reconcile 为 `position aligned`，dashboard paper version 增至 `3` 行并结束于 `2026-05-18`。
21. **scheduler health 现在能发现安装态 plist 漂移**：health 不只看 LaunchAgent label 是否 loaded，也会比较 `~/Library/LaunchAgents/com.soxl.*.plist` 与 repo `production/com.soxl.*.plist` 是否一致。当前读数为 preopen/daily 都 `matches repo`，所以自然 `05:35` proof 的前置配置已对齐。
22. **post-close proof 状态现在显式可见**：health 新增 `Post-close Run Status`。`05:35 Tue-Sat Asia/Shanghai` 前显示 `pending`；到点后如果没有 `[scheduler] all done` 会显示 `due but no success marker found`；有成功标记则显示 `success marker found`。
23. **Vercel dashboard 静态包现在可自检**：`scripts/check_dashboard_static_export.py` 会校验 `/`、`/dashboard.html`、`/dashboard_data/:path*` 的 no-cache header、`dashboard_data/index.json` 中所有 versioned payload 是否存在并匹配 row count，以及 `paper_trading` 是否仍在静态包中。`production/check_scheduler_health.sh` 现在直接显示该检查结果；当前读数为 `versions=25`、`missing=0`、`model_versions=90`、`paper_trading_rows=3`、`paper_trading_end=2026-05-18`。
24. **post-close success 现在要求本次 cutoff 后的新日志**：`production/check_scheduler_health.sh` 不再只看 `scheduler_postclose.log` 是否含 `[scheduler] all done`。它会比较日志 mtime 和当前日 `05:35` cutoff；旧/手动 success marker 在 due 后会显示 `due but no current success marker found`，避免污染自然 LaunchAgent proof。
25. **append-only scheduler log 现在按 run segment 判定成功**：`production/soxl_daily_scheduler.py` 会在每次启动时输出 `[scheduler] run started at ...`；health 若发现 run-start marker，只接受最新 run-start 之后的 `[scheduler] all done`。这样旧成功行 + 新失败追加不会被误判为当前 post-close 成功。
26. **paper signal 不再丢弃最新 closed bar**：`plan_manual_session()` 现在把 `as_of` 当作最新已收盘 bar 并使用 `<= as_of` 生成信号。当前 dry-run 证据：`last_date=2026-05-19`、`as_of=2026-05-19`、summary `signal.date=2026-05-19`、`dry_run=true`、`executed=false`。这修复了 scheduler/paper runner 用 fresh data 只生成前一日 signal 的模拟盘滞后。
27. **scheduler health 现在区分 inference artifact 和 paper dashboard row**：新增 `Latest Signal Artifact`，直接显示最新 `runtime/soxl/paper_trading/<date>/summary.json`。当前 live health 读数为 `date=2026-05-19`、`profile=balanced_rule_residual_v19`、`dry_run=true`、`executed=false`、`target=0.3999999985`、`risk_gate=approved`、`reconcile=manual_review`，而 dashboard paper row 仍结束于 `2026-05-18`。这使定时推理、模拟盘执行、dashboard 导出可以分层判断。

## 2026-05-20 验证记录

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_update_market_data.py tests/test_dashboard.py -q
# 13 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
# completed, as_of=2026-05-15, signal date=2026-05-14
```

Dry-run readout:

- initial auto-selected profile before fix: `balanced`
- initial risk gate: blocked by `walk-forward promotion gate not evaluated` and large cold-start step
- post-fix auto-selected profile: `balanced_rule_residual_v19`
- post-fix paper dry-run: risk gate approved with clipped cold-start target
- local non-dry paper run: filled a clipped BUY, wrote cumulative dashboard CSV from post-execution account, and persisted broker state
- follow-up dry-run: read persisted `33.5%` paper position and approved the next step toward the full target

Additional commands:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_dashboard.py tests/test_vercel_dashboard.py -q
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --smoke
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```

Final status:

- Focused regression set: `29 passed`.
- Scheduler smoke: completed cached-data dry-run inference, selected `balanced_rule_residual_v19`, approved risk gate, rebuilt dashboard static data with `25` versions.
- Harness check: `passed=71 failures=0`.
- Local static HTTP checks: `dashboard.html` and `dashboard_data/index.json` returned `200 OK`.

## 2026-05-21 Runtime Selection Guard

Additional regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_best_model_for_family_skips_overlay_profile_without_auto_select_approval -q
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from src.soxl_quant.runtime import auto_select_profile
print(auto_select_profile("balanced"))
PY
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
```

Readout:

- The regression first proved a promoted high-score v22 actor could beat v19 despite `execution_overlay.auto_select_allowed=false`.
- `best_model_for_family()` now skips that profile unless the profile overlay approval flag is explicit.
- Current live auto-select remains `balanced_rule_residual_v19`.
- Cached-data scheduler dry-run still selected `balanced_rule_residual_v19`, approved risk gate, and skipped order placement under `--dry-run`.

## 2026-05-21 Scheduler Timezone Guard

Production schedule review found that `production/crontab.txt` and the macOS
LaunchAgent were still using local `09:30` / `16:35` rows while this workspace is
operated in `Asia/Shanghai`. That could run the "post-close" data refresh before
the US close, making provider empty/stale-data failures much more likely.

Changes:

- `production/crontab.txt`: SOXL rows now use `21:00 Mon-Fri Asia/Shanghai` for
  cached-data pre-open `--dry-run` inference/dashboard, and `05:35 Tue-Sat
  Asia/Shanghai` for the full post-close data + paper trading + dashboard loop.
- `production/com.soxl.daily.scheduler.plist`: macOS LaunchAgent moved to the
  same `05:35 Tue-Sat` post-close full-pipeline window.
- `production/check_scheduler_health.sh` and `production/FUTU_SEMI_AUTO.md` now
  match the new schedule.
- `tests/test_production_schedule.py` guards against reverting to stale local
  `09:30` or `16:35` SOXL scheduler rows.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py -q
# 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_production_schedule.py tests/test_dashboard.py -q
# 20 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update
# completed cached-data dry-run, selected balanced_rule_residual_v19, rebuilt dashboard_data with 25 versions
```

Dashboard / Vercel visual evidence:

- Local HTTP returned `200 OK` for `dashboard.html` and
  `dashboard_data/index.json`.
- Playwright with local Chrome captured
  `runtime/production/card/scheduler_timezone_v1/dashboard_visual_2026-05-21.png`
  at `1440 x 1000`; pixel check showed `94.84%` non-white pixels.
- Computer Use screenshot remains blocked by pending macOS Accessibility /
  Screen Recording permissions, but the browser-rendered Chrome screenshot is
  now saved as repo evidence.

## 2026-05-21 Scheduler Health Observability

Root cause:

- The schedule had moved to macOS LaunchAgent, but
  `production/check_scheduler_health.sh` still inspected Linux cron.
- On this host the service section printed blank, so the operator could not see
  whether `com.soxl.daily.scheduler` was loaded.
- The log timestamp path also used GNU-only `ls --time-style`, which is not
  portable to macOS once logs exist.

Change:

- Added `tests/test_scheduler_health.py` with a fake `launchctl` and temp
  scheduler logs.
- `production/check_scheduler_health.sh` now detects Darwin, reports
  `LaunchAgent com.soxl.daily.scheduler: loaded/not loaded`, supports
  `SOXL_SCHEDULER_HEALTH_ROOT`, and formats log mtimes through portable
  `stat/date` helpers.

Verification:

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

Interpretation:

- The LaunchAgent is loaded on the current macOS host.
- The corrected schedule logs have not been produced yet, so the live
  post-close proof remains pending until the first `05:35 Tue-Sat
  Asia/Shanghai` run.

## 2026-05-21 macOS LaunchAgent Split

Root cause:

- `production/com.soxl.daily.scheduler.plist` was moved to the correct
  `05:35 Tue-Sat` post-close time, but still wrote legacy `scheduler.log` /
  `scheduler.err`.
- `production/check_scheduler_health.sh` expected `scheduler_postclose.log`, so
  the first live post-close run would not have shown up in the health output.
- The 21:00 pre-open cached inference/dashboard run existed in cron only; the
  macOS LaunchAgent path had no matching job.

Change:

- `production/com.soxl.daily.scheduler.plist` now writes
  `scheduler_postclose.log` / `scheduler_postclose.err`.
- Added `production/com.soxl.preopen.scheduler.plist` for
  `21:00 Mon-Fri Asia/Shanghai` dry-run + skip-data-update inference/dashboard.
- `production/check_scheduler_health.sh` now reports both
  `com.soxl.preopen.scheduler` and `com.soxl.daily.scheduler`, and lists both
  stdout/stderr log files.
- `production/FUTU_SEMI_AUTO.md` documents loading/unloading both LaunchAgents.

External state:

- Synced both plists to `~/Library/LaunchAgents/` and reloaded them.
- `launchctl list | grep 'com.soxl.*scheduler'` reported both labels with
  status `0`.

Verification:

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

Boundary:

- Pre-open scheduled inference is now installed and manually verified through
  LaunchAgent.
- Post-close online data refresh still needs the first real `05:35 Tue-Sat`
  run to prove whether data advances beyond `2026-05-15`.

## 2026-05-21 Cached Fallback Dry-Run Guard

Root cause:

- `production/soxl_daily_scheduler.py` caught transient data-source failures and
  retried with `--skip-data-update`.
- That retry did not force `--dry-run`, so a post-close data failure could still
  execute a paper order using stale cached data.

Change:

- Cached fallback retry now appends `--skip-data-update` and `--dry-run`.
- Retry description and success marker changed to `paper trading (cached
  dry-run)` / `[scheduler] all done (with cached dry-run)`.
- Warning notification now states cached data was used for dry-run only and no
  order was placed.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_retries_data_source_failure_as_cached_dry_run -q
# 1 passed after initially failing against the old retry path

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 22 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update
# completed cached-data dry-run, skipped order placement, rebuilt dashboard_data with 25 versions

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0
```

Operational rule:

- Fresh post-close data update succeeds -> full paper trading is allowed.
- Transient data update failure -> cached signal/dashboard only; paper broker
  state must not change.

## 2026-05-21 Paper Trading Freshness Health

Root cause:

- Scheduler health reported LaunchAgent load state and log markers, but did not
  show whether `production/soxl_paper_trading.py --healthcheck` considered the
  data panel fresh.
- That hid the distinction between "scheduler installed" and "post-close data
  refresh has actually advanced the panel".

Change:

- `production/soxl_paper_trading.py` healthcheck now reports structured data
  freshness with `last_date`, `expected_last_trading_day`, and `stale_days`.
- `production/check_scheduler_health.sh` now includes a `Paper Trading Health`
  section and parses the JSON payload even when the paper healthcheck exits
  non-zero because data is stale.
- Regression coverage lives in `tests/test_paper_trading.py` and
  `tests/test_scheduler_health.py`.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_freshness_status_reports_expected_date_and_lag -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_macos_launchagent_and_logs -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 23 passed

production/check_scheduler_health.sh
# data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

Readout:

- Current production blocker is data freshness: scheduler and broker are
  observable, but the data panel has not advanced beyond `2026-05-15`.
- The next proof must come from the real `05:35 Tue-Sat Asia/Shanghai`
  post-close run; if online refresh fails, the scheduler should remain cached
  dry-run only.

## 2026-05-21 Stale Execution Guard

Root cause:

- Healthcheck could report stale data, but the main paper-trading path did not
  enforce that status before qlib initialization and order planning.
- A data update that returned successfully but left the CSV stale could
  otherwise continue into non-dry-run paper execution.

Change:

- `production/soxl_paper_trading.py` checks `_csv_data_freshness_status()` after
  the update step and before `init_qlib()`.
- Non-dry-run stale data exits with a clear error. Dry-run stale data is allowed
  only for signal/dashboard visibility and prints a warning.
- `production/soxl_daily_scheduler.py` classifies stale post-update data as a
  cached dry-run fallback condition.
- Regression coverage:
  - `tests/test_paper_trading.py::test_main_blocks_non_dry_run_when_data_remains_stale`
  - `tests/test_daily_scheduler.py::test_scheduler_retries_stale_post_update_data_as_cached_dry_run`

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_blocks_non_dry_run_when_data_remains_stale -q
# 1 passed after initially failing by reaching qlib init

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_retries_stale_post_update_data_as_cached_dry_run -q
# 1 passed after initially exiting instead of retrying

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 25 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
# first pass blocked stale non-dry-run data
# retry completed as cached dry-run with skipped order placement
```

Operational rule:

- Fresh data -> full paper execution can proceed.
- Stale data or transient provider failure -> cached dry-run visibility only,
  no broker state mutation.

## 2026-05-21 Freshness Market-Close Boundary

Root cause:

- `_last_trading_day()` used the local `Asia/Shanghai` calendar date.
- At `2026-05-21 02:23 CST`, the `2026-05-20` US trading session was still open
  in New York, but healthcheck already expected `2026-05-20`.
- This created an over-strict stale read before the scheduled
  `05:35 Tue-Sat Asia/Shanghai` post-close update.

Change:

- `production/soxl_paper_trading.py` now computes expected data freshness in
  `America/New_York`.
- The current New York weekday becomes expected only at or after `17:30 ET`.
- Before that cutoff, the expected date is the previous weekday.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_last_trading_day_waits_for_new_york_post_close_cutoff tests/test_paper_trading.py::test_last_trading_day_uses_previous_friday_before_monday_us_close -q
# 2 passed after initially failing on the old function signature/local-date behavior

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 27 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --healthcheck --skip-data-update --no-notify
# data expected_last_trading_day=2026-05-19 stale_days=4 at 02:xx CST

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
# blocked stale non-dry-run data at expected=2026-05-19, then completed cached dry-run
```

Readout:

- Before the post-close window, health no longer expects the still-open US
  session.
- The live data-refresh proof is still the first real `05:35 Tue-Sat
  Asia/Shanghai` run.

## 2026-05-21 Provider Failover

Root cause:

- `scripts/update_market_data.py` had both `yfinance` and `yahoo_api`, but the
  production paper runner used only the default yfinance path.
- Current live probes still failed:
  - yfinance: `No overlapping dates for SOXL/SOXX/VIX`
  - yahoo_api: `429 Too Many Requests`

Change:

- `production/soxl_paper_trading.py` gained
  `--data-provider {auto,yfinance,yahoo_api}`.
- `auto` tries yfinance first and falls back to yahoo_api on retryable provider
  failures.
- `production/soxl_daily_scheduler.py` gained matching pass-through support.
- `production/FUTU_SEMI_AUTO.md` documents the provider mode.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_yahoo_api -q
# 1 passed after initially failing because helper did not exist

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_passes_data_provider_to_paper_runner -q
# 1 passed after initially failing because scheduler did not accept the flag

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 29 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-dashboard --data-provider auto
# yfinance failed, yahoo_api failed, scheduler completed cached dry-run without placing orders
```

Readout:

- The refresh path is more robust for the next live window, but provider
  availability remains the active external blocker.
- Cached dry-run fallback still preserves dashboard/signal visibility without
  changing paper broker state.

## 2026-05-21 Polygon + CBOE Data Recovery

Root cause:

- The Yahoo-family providers were still not sufficient for the production
  freshness gate: yfinance returned `No overlapping dates for SOXL/SOXX/VIX`,
  and Yahoo chart API returned HTTP `429 Too Many Requests`.
- Futu OpenD was not reachable locally, so the immediate recovery path needed to
  be a repo-owned market-data provider, not the broker gateway.
- Live probes showed Polygon returned SOXL/SOXX through `2026-05-19`, but
  Polygon VIX was not authorized. CBOE's public VIX history CSV returned VIX
  through `2026-05-19`.

Change:

- Added `polygon` provider to `scripts/update_market_data.py`.
- `polygon` uses Polygon aggregates for SOXL/SOXX and CBOE VIX history for VIX.
- Production `auto` provider order is now `yfinance -> yahoo_api -> polygon`.
- Scheduler success logs replay child stderr, preserving provider fallback
  evidence in LaunchAgent logs.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_update_market_data.py::test_update_master_csv_polygon_provider_uses_cboe_vix tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_polygon tests/test_daily_scheduler.py::test_scheduler_accepts_polygon_data_provider -q
# 3 passed after red

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_run_replays_child_stderr_on_success -q
# 1 passed after red

PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --provider polygon
# before=4071 after=4073 new_rows=2 last_date=2026-05-19
# raw SOXL/SOXX/VIX mirrors and qlib refreshed to 2026-05-19

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
# update_data_registry: OK symbols=6 panels=10 qlib=6

production/check_scheduler_health.sh
# data: ok last_date=2026-05-19 expected=2026-05-19

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --data-provider auto
# yfinance failed, yahoo_api failed, polygon succeeded
# as_of=2026-05-19, signal date=2026-05-18, risk_gate approved=True
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

Readout:

- The current production data freshness blocker is cleared for the expected
  completed US trading day at `2026-05-21 02:48 CST`.
- The next live proof remains the natural `05:35 Tue-Sat Asia/Shanghai`
  post-close LaunchAgent run after the next completed session.

## 2026-05-21 LaunchAgent Installed Plist Drift Guard

Root cause:

- `production/check_scheduler_health.sh` reported whether the LaunchAgent labels
  were loaded, but not whether the installed plist files matched the repo-owned
  plists.
- A stale installed plist could keep old schedule times, arguments, or log paths
  while the health check still showed `loaded`.

Change:

- Added `SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR` for testable installed-plist
  lookup.
- On Darwin, health now compares each installed plist with the repo plist and
  prints `matches repo`, `differs from repo`, `not installed`, or
  `repo file missing`.
- `tests/test_scheduler_health.py` now covers the drift readout and is
  compatible with `scripts/check_harness.py`.
- `scripts/check_harness.py` now includes scheduler-health tests.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 2 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py --test-file tests/test_scheduler_health.py
# passed=122 failures=0

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.preopen.scheduler plist: matches repo
# LaunchAgent com.soxl.daily.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler plist: matches repo
# scheduler_postclose.log: NOT FOUND
# data: ok last_date=2026-05-19 expected=2026-05-19
```

Readout:

- The natural post-close proof is still pending, but the external LaunchAgent
  installation now matches the repo schedule and log-path definitions.

## 2026-05-21 Post-close Due Status Guard

Root cause:

- Before the scheduled post-close time, a missing `scheduler_postclose.log` is
  expected.
- After `05:35 Tue-Sat Asia/Shanghai`, a missing post-close success marker is a
  production proof failure.
- The previous health output exposed the raw log state but did not classify the
  timing context.

Change:

- `production/check_scheduler_health.sh` now prints a `Post-close Run Status`
  section.
- It reports one of:
  - `postclose status: pending until 05:35 Tue-Sat Asia/Shanghai`
  - `postclose status: due but no success marker found`
  - `postclose status: success marker found`
- `SOXL_SCHEDULER_HEALTH_WEEKDAY` and `SOXL_SCHEDULER_HEALTH_HHMM` make the
  due/pending branch deterministic in tests.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 3 passed

production/check_scheduler_health.sh
# postclose status: pending until 05:35 Tue-Sat Asia/Shanghai
```

Readout:

- At `2026-05-21 03:54 CST`, the natural post-close proof is correctly
  classified as pending rather than missing.
- After `05:35`, the same script should be enough to distinguish completed,
  missing, or fallback post-close behavior.

## 2026-05-21 Paper Full-Loop Recovery

Baseline:

- Time: `2026-05-21 03:01 CST`, before the natural post-close LaunchAgent run.
- Healthcheck showed data fresh for the completed US session:
  `last_date=2026-05-19 expected=2026-05-19`.
- Paper broker state was still dated `2026-05-14` with `1800` shares and
  `33.53%` position.
- The cumulative paper dashboard CSV had `2` rows and ended at `2026-05-14`.

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --no-notify --data-provider auto
```

Readout:

- yfinance failed with `No overlapping dates for SOXL/SOXX/VIX`.
- yahoo_api failed with Yahoo HTTP `429` on `^VIX`.
- polygon succeeded, keeping `last_date=2026-05-19`.
- freshness gate passed with `stale_days=0`.
- model auto-select chose `balanced_rule_residual_v19`.
- signal date was `2026-05-18`, target `50.00%`, price `$151.75`.
- risk gate approved.
- default `paper` broker submitted and filled a BUY.
- dashboard rebuilt.

Post-loop state:

- `runtime/soxl/paper_trading/paper_broker_state.json`:
  `date=2026-05-18`, `shares=3000`, `position=0.4856047894`,
  `account_value=937490.7536`.
- `runtime/soxl/paper_trading/2026-05-18/reconcile.json`:
  `action_required=none`, `note=position aligned`.
- `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv`:
  now has `3` rows, latest row `2026-05-18`, action `BUY`,
  position `0.485605`.
- `dashboard_data/index.json`:
  `paper_trading` summary now ends at `2026-05-18` with `3` rows.
- Dashboard static verification after rebuild:
  `dashboard.html`, `dashboard_data/index.json`, and
  `dashboard_data/model_versions.json` all returned `200 OK`; Playwright/Chrome
  captured
  `runtime/production/card/paper_full_loop_v1/dashboard_visual_2026-05-21.png`
  at `1440 x 1000` with `99.35%` non-white pixels.

Decision:

- The simulated paper loop can now mutate broker state, reconcile within
  tolerance, append cumulative dashboard CSV, and rebuild dashboard under fresh
  data.
- Natural `05:35 Tue-Sat Asia/Shanghai` LaunchAgent proof is still pending
  because this manual run occurred before the scheduled post-close window.

## Open Production Tasks

- Run the first real `05:35 Tue-Sat Asia/Shanghai` post-close scheduler pass when the data provider is available; current healthcheck data status is fresh at `last_date=2026-05-19 expected=2026-05-19`, but the natural LaunchAgent log proof is still pending.
- After the production loop is stable in live schedule, resume SOXX-aware model optimization only from the non-duplicate frontier; the direct SOXX relative cap path is blocked by walk-forward absorption.

## 2026-05-21 Paper Same-Day Idempotency Guard

Context:

- The manual full paper loop submitted and filled a paper BUY for signal date
  `2026-05-18`.
- Before the next signal date, a rerun of the scheduler would plan the same
  `2026-05-18` signal again.
- Because the paper broker uses a step clip, repeated non-dry-run calls for the
  same signal date could otherwise inch the simulated account toward the same
  target multiple times.

Change:

- `production/soxl_paper_trading.py` now checks
  `runtime/soxl/paper_trading/<signal_date>/summary.json` before order
  placement.
- If the summary has the same signal date, same broker, `dry_run=false`, and
  `executed=true`, the paper path prints a duplicate-execution message and
  returns before placing another order, appending the dashboard CSV, or saving
  broker state.
- `tests/test_paper_trading.py` covers the duplicate-rerun path.

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_skips_duplicate_non_dry_run_for_existing_executed_summary -q
# 1 passed after red

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --no-notify --skip-data-update
# [paper] already executed for 2026-05-18 broker=paper; skipping duplicate order
# [scheduler] dashboard build
# [scheduler] all done
```

Post-rerun state:

- `paper_broker_state.json`: still `date=2026-05-18`, `shares=3000`,
  `position=0.4856047894`.
- `paper_trading_soxl_daily.csv`: still `3` rows, ending `2026-05-18`.
- `dashboard_data/index.json`: paper summary still
  `start=2026-03-31`, `end=2026-05-18`, `rows=3`.
- Final gates:
  - focused production/dashboard tests: `29 passed`
  - `scripts/check_harness.py`: `passed=112 failures=0`
  - `scripts/build_runtime_index.py --check`: `runtime index OK`
  - `git diff --check`: clean
  - healthcheck: LaunchAgents loaded, data/model/broker ok

## 2026-05-21 Natural Post-Close LaunchAgent Readout

Context:

- The first natural `05:35 Tue-Sat Asia/Shanghai` post-close LaunchAgent run
  fired at `2026-05-21T05:35:04+08:00`.
- It entered the full path (`broker=paper`, `dry_run=False`,
  `skip_data_update=False`, `data_provider=auto`) but could not obtain a fresh
  `2026-05-20` data row.

Readout:

- `yfinance` failed with `No overlapping dates for SOXL/SOXX/VIX`.
- `yahoo_api` failed with Yahoo `429 Too Many Requests` on `^VIX`.
- `polygon` initially failed under LaunchAgent because `POLYGON_API_KEY` was
  missing from the LaunchAgent environment.
- The scheduler fell back to cached dry-run and wrote
  `[scheduler] all done (with cached dry-run)`.
- After adding `production/run_soxl_scheduler.sh` and installing
  `~/.openclaw/soxl.env`, a manual
  `launchctl kickstart -k gui/$(id -u)/com.soxl.daily.scheduler` no longer hit
  the missing Polygon key. It still fell back because data remained stale:
  `last_date=2026-05-19 expected=2026-05-20 stale_days=1`.

Change:

- `production/check_scheduler_health.sh` now treats only exact
  `[scheduler] all done` as full post-close success and reports
  `[scheduler] all done (with cached dry-run)` separately.
- Both LaunchAgent plists now call `production/run_soxl_scheduler.sh`, which
  loads `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` before executing the
  scheduler.
- `production/FUTU_SEMI_AUTO.md` documents the private env file and reload
  sequence.

Decision:

- Current blocker is provider freshness for the `2026-05-20` US session, not
  LaunchAgent installation, promoted model selection, paper broker readiness, or
  dashboard static export.
- A cached dry-run keeps latest-signal/dashboard visibility but must not count
  as full paper execution proof.
- Keep model optimization paused until a post-close run completes fresh data
  refresh, non-dry paper execution, and dashboard rebuild.

Final gates:

- focused production/dashboard tests: `43 passed`
- `scripts/check_harness.py`: `passed=128 failures=0`
- `scripts/check_dashboard_static_export.py`: `generated_at=2026-05-21 05:52:40`,
  `versions=25`, `missing=0`, `paper_trading_rows=3`,
  `paper_trading_end=2026-05-18`
- `scripts/build_runtime_index.py --check`: `runtime index OK`
- `git diff --check`: clean
- healthcheck: LaunchAgents loaded, data stale at `last_date=2026-05-19`
  vs `expected=2026-05-20`, and post-close status is
  `due but only cached dry-run fallback completed`
