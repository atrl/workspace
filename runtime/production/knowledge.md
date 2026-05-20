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

## Open Production Tasks

- Run the first real `05:35 Tue-Sat Asia/Shanghai` post-close scheduler pass when the data provider is available; current healthcheck data status remains stale at `last_date=2026-05-15` until an online refresh extends the panel.
- After the production loop is stable in live schedule, resume SOXX-aware model optimization only from the non-duplicate frontier; the direct SOXX relative cap path is blocked by walk-forward absorption.
