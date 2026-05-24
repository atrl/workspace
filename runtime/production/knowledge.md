# PRODUCTION Knowledge

> 最后更新: 2026-05-24
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
11. **定时表已按 Asia/Shanghai 本地时区修正**：pre-open 为 `21:00 Mon-Fri` cached-data `--dry-run`，post-close 完整数据刷新 + 模拟盘 + dashboard 为 `06:15 Tue-Sat`，避免本地 `09:30/16:35` 误跑在美股收盘前，也避开 05:35 附近的供应商数据滞后窗口。
12. **scheduler health 现在匹配 macOS LaunchAgent**：`production/check_scheduler_health.sh` 会在 Darwin 上检查 `launchctl list` 中的 `com.soxl.daily.scheduler`，并用 portable `stat/date` 读取日志，不再把 macOS 运行态显示成空白 cron 服务。
13. **macOS 定时推理路径已补齐**：新增 `com.soxl.preopen.scheduler`，用于 `21:00 Mon-Fri Asia/Shanghai` cached-data dry-run 推理 + dashboard；`com.soxl.daily.scheduler` 负责 `06:15 Tue-Sat` post-close 完整流程。pre-open / daily / weekly LaunchAgent 已同步到 `~/Library/LaunchAgents/` 并 loaded。
14. **数据源失败时不再用旧数据推进模拟盘**：post-close 数据刷新遇到 transient provider failure 时，scheduler fallback 会追加 `--skip-data-update --dry-run`，只保留 cached signal/dashboard 可见性，不落 paper order / broker state。
15. **scheduler health 现在直接暴露 paper 数据新鲜度**：`production/check_scheduler_health.sh` 会运行 paper healthcheck 并显示 `last_date`、`expected_last_trading_day`、`stale_days`、model 和 broker 状态。post-close proof 现在按 `06:15 Tue-Sat Asia/Shanghai` 判定，避免 05:35 附近数据源尚未齐全时误报或反复 fallback。
16. **stale data 现在也是执行硬门槛**：paper 主流程在 qlib init / 模型选择 / broker 执行前检查 freshness。非 dry-run 遇到 stale data 直接退出；scheduler 将其转成 cached dry-run fallback，避免 stale 数据推进 paper broker state。
17. **freshness 期望日已按纽约收盘边界修正**：`_last_trading_day()` 使用 `America/New_York` 和 `17:30 ET` data-ready cutoff。`2026-05-21 02:23 CST` 期望 `2026-05-19`，`05:35 CST` 后才期望 `2026-05-20`。
18. **production data refresh 现在支持 provider failover**：paper runner 和 scheduler 都支持 `--data-provider {auto,yfinance,yahoo_api,polygon}`；默认 `auto` 先试 yfinance，再试 yahoo_api，最后试 Polygon + CBOE VIX。当前实测 yfinance 是 `No overlapping dates`，yahoo_api 是 `429 Too Many Requests`，Polygon + CBOE 能把数据更新到 `2026-05-19`。
19. **scheduler 成功日志现在保留 provider fallback stderr**：`production/soxl_daily_scheduler.py` 在 child step 成功时也会 replay stderr，因此 yfinance/yahoo_api 失败后由 polygon 接管的路径能出现在 LaunchAgent 日志里，不再只看到最终成功 stdout。
20. **paper full-loop 已能推进模拟盘状态**：`2026-05-21 03:02 CST` 手动运行 default paper broker 的 full scheduler，auto provider 经 yfinance/yahoo_api fallback 到 polygon，数据 fresh，风险门通过，paper BUY filled；broker state 更新到 `2026-05-18`、`3000` shares、`48.56%` position，reconcile 为 `position aligned`，dashboard paper version 增至 `3` 行并结束于 `2026-05-18`。
21. **scheduler health 现在能发现安装态 plist 漂移**：health 不只看 LaunchAgent label 是否 loaded，也会比较 `~/Library/LaunchAgents/com.soxl.*.plist` 与 repo `production/com.soxl.*.plist` 是否一致。当前读数为 preopen/daily/weekly 都 `matches repo`，所以自然 `06:15` proof 的前置配置已对齐。
22. **post-close proof 状态现在显式可见**：health 新增 `Post-close Run Status`。`06:15 Tue-Sat Asia/Shanghai` 前显示 `pending`；到点后如果没有 `[scheduler] all done` 会显示 `due but no success marker found`；有成功标记则显示 `success marker found`。
23. **Vercel dashboard 静态包现在可自检**：`scripts/check_dashboard_static_export.py` 会校验 `/`、`/dashboard.html`、`/dashboard_data/:path*` 的 no-cache header、`dashboard_data/index.json` 中所有 versioned payload 是否存在并匹配 row count，以及 `paper_trading` 是否仍在静态包中。`production/check_scheduler_health.sh` 现在直接显示该检查结果；当前读数为 `versions=25`、`missing=0`、`model_versions=90`、`paper_trading_rows=3`、`paper_trading_end=2026-05-18`。
24. **post-close success 现在要求本次 cutoff 后的新日志**：`production/check_scheduler_health.sh` 不再只看 `scheduler_postclose.log` 是否含 `[scheduler] all done`。它会比较日志 mtime 和当前日 `06:15` cutoff；旧/手动 success marker 在 due 后会显示 `due but no current success marker found`，避免污染自然 LaunchAgent proof。
25. **append-only scheduler log 现在按 run segment 判定成功**：`production/soxl_daily_scheduler.py` 会在每次启动时输出 `[scheduler] run started at ...`；health 若发现 run-start marker，只接受最新 run-start 之后的 `[scheduler] all done`。这样旧成功行 + 新失败追加不会被误判为当前 post-close 成功。
26. **paper signal 不再丢弃最新 closed bar**：`plan_manual_session()` 现在把 `as_of` 当作最新已收盘 bar 并使用 `<= as_of` 生成信号。当前 dry-run 证据：`last_date=2026-05-19`、`as_of=2026-05-19`、summary `signal.date=2026-05-19`、`dry_run=true`、`executed=false`。这修复了 scheduler/paper runner 用 fresh data 只生成前一日 signal 的模拟盘滞后。
27. **scheduler health 现在区分 inference artifact 和 paper dashboard row**：新增 `Latest Signal Artifact`，直接显示最新 `runtime/soxl/paper_trading/<date>/summary.json`。当前 live health 读数为 `date=2026-05-19`、`profile=balanced_rule_residual_v19`、`dry_run=true`、`executed=false`、`target=0.3999999985`、`risk_gate=approved`、`reconcile=manual_review`，而 dashboard paper row 仍结束于 `2026-05-18`。这使定时推理、模拟盘执行、dashboard 导出可以分层判断。
28. **v22 runtime proof 缺口现在归因到同日幂等保护**：当前 health 选择 `balanced_rule_residual_v22`，但最新非 dry-run paper artifact 是 `2026-05-20` 的 `balanced_rule_residual_v19`。因为该日期已经执行，paper runner 不应重复落单；health 现在会在 pending 行后补充 `latest executed date already matches expected data date; next non-dry-run execution must use selected model`，把剩余证明归因到等待下一交易日，而不是误判为数据/调度失败。
29. **Next Scheduled Runs 不再把已过窗口标成 today**：`production/check_scheduler_health.sh` 现在按 `HEALTH_WEEKDAY/HEALTH_HHMM` 判断 pre-open、post-close、weekly 窗口。当前 post-close 输出为 `next eligible day 06:15 Tue-Sat Asia/Shanghai`，避免把当天已过的 post-close 误读成下一次运行。
30. **pre-open scheduled dry-run 现在有独立 health 状态**：health 新增 `Pre-open Run Status`，按 `21:00 Mon-Fri Asia/Shanghai` 判断 pending/due/current，并在当前成功时要求 `schedule_slot=preopen dry_run=True skip_data_update=True` 以及 runtime composer 目标字段；这不新增最终 goal gate。
31. **静态 dashboard 包现在携带生产状态 JSON**：`scripts/deploy_vercel_dashboard.py` 会写 `dashboard_data/production_status.json`，只包含非 secret 的 Vercel/auth/project/static 状态；`scripts/check_dashboard_static_export.py` 会要求并报告 `production_status=<status>`。当前状态仍是 `blocked_no_credentials`。
32. **scheduled proof 现在要求触发来源**：scheduler run-start marker 会输出 `schedule_trigger`，cron/LaunchAgent 分别设置 `cron_*` / `launchagent_*`。post-close target proof 现在必须同时满足 `schedule_slot=postclose` 和 `schedule_trigger=launchagent_postclose|cron_postclose`，避免手动 replay 用同一个 slot 清掉自然定时 gate。
33. **pre-open 定时 composer 路径已自然触发成功**：`2026-05-21 21:00 CST` 的 `com.soxl.preopen.scheduler` 写入 `schedule_slot=preopen schedule_trigger=launchagent_preopen`，用 `v19reg_v22_budget_usage_bull_guard_v1` runtime composer 在 `2026-05-20` fresh data 上完成 dry-run 推理、dashboard rebuild、goal-gate export 和 Vercel evidence refresh。health 显示 `preopen target: ok ... runtime_composer=budget_usage`。
34. **dashboard 模型回放现在会随静态包刷新**：`scripts/refresh_dashboard_model_replays.py` 会在 dashboard build 前刷新 registry 中 `keep_actor=true`、未 dropped、actor/profile 可解析的模型日线 CSV，并写 `dashboard_data/model_replay_freshness.json`。当前本地数据最新日是 `2026-05-22`，qlib 曲线安全回测末端是 `2026-05-21`，10/10 个可回放模型已刷新到该末端；`paper_trading` 仍作为 Vercel 首屏最新信号视图，结束于 `2026-05-22`。
35. **dashboard 纸盘价格/交易展示已改为证据驱动**：`paper_trading` 静态视图现在丢弃没有非 dry-run summary 的旧 seed 行，并把 Buy&Hold NAV 重基准到首个展示账户值；终端 `HOLD` 不再被合成 `SELL`，零仓位 `BUY`/零状态 `SELL` 会被屏蔽。静态版本 JSON 现在保留 OHLC/MA180 等价格字段，页面已移除不可靠的历史模型对比和 Training Curve。
36. **Futu OpenD quote 数据探针已通过本地验证**：2026-05-23 启动本机 OpenD 后，`127.0.0.1:11111` 端口可达，初次 `OpenQuoteContext` 被图形验证阻断；完成本地 phone/UI 验证后，`get_capital_flow` / `get_capital_distribution` 已能获取 `US.SOXL` / `US.SOXX` 小单资金流并落地到 `data/raw/external/futu_*`。这只证明 quote/data path 可用，交易 broker readiness 仍需用 `production/check_futu_opend.py` 和模拟盘连接路径单独验证。
37. **2026-05-23 上线证明已缩小到外部 Vercel 配置**：自然 `06:15 Tue-Sat Asia/Shanghai` post-close LaunchAgent 已成功运行，`schedule_trigger=launchagent_postclose`，数据新鲜度 `last_date=2026-05-22 expected=2026-05-22`，当前 runtime composer 是 `v19reg_v22_budget_usage_bull_guard_v1`，ordinary paper proof 为 `ok selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=skipped`，dashboard 静态包 `versions=38 missing=0 latest=paper_trading paper_trading_end=2026-05-22`。唯一剩余 goal gate 是 `external_vercel_setup`：`VERCEL_TOKEN` 不在当前环境/launch env file，且项目未通过 `.vercel/project.json` 或 `VERCEL_ORG_ID`/`VERCEL_PROJECT_ID` 链接。
38. **IBKR Python 侧已安装并接入统一 broker 边界**：`ib-insync==0.9.86` 和 `eventkit==1.0.3` 已加入 `qlib_venv` / `requirements/qlib.txt`，repo-owned adapter 为 `src/soxl_quant/app/ibkr.py`，`production/check_ibkr_gateway.py` 可快速检测本机 TWS / IB Gateway。`soxl_paper_trading.py`、`soxl_daily_scheduler.py` 和 `composed_session.py` 都支持显式 `--broker ibkr`，并已通过 `scripts/check_harness.py`（203 checks）。当前 `127.0.0.1:7497` 不可达；Homebrew `trader-workstation` cask 在 install4j quiet installer 阶段挂起并被回滚，所以 IBKR readiness 仍取决于本机可靠安装/登录 TWS 或 IB Gateway 并启用 socket API。
39. **Futu 模拟盘账户与下单/撤单链路已实测通过**：`2026-05-24` 通过本机 OpenD 的 `OpenSecTradeContext(TrdMarket.US)` 读取 `TrdEnv.SIMULATE` 账户，模拟账户 cash/total_assets 为 `1000000.0`，`US.SOXL` 初始仓位为空；提交 `BUY 1` 限价单后 Futu 返回 `ret=OK`，随后 `modify_order(CANCEL)` 返回 `OK`，最终订单状态 `CANCELLED_ALL`、成交数量 `0`、仓位仍为空，复查活跃挂单数为 `0`。这证明 Futu 模拟交易通道已可用；下一步仍需等待自然 post-close LaunchAgent 产出 scheduled non-dry Futu proof。
40. **Futu scheduled simulation 已重置为 10000 notional 起步**：`2026-05-24` 新增 `--account-notional`，daily/preopen LaunchAgent 与 cron 均配置 `--broker futu --account-notional 10000`；本地旧 paper state 已归档到 `runtime/soxl/paper_trading/archive_reset_20260524_10000`，新的 active `paper_broker_state.json` 与 dashboard CSV 从 `2026-05-22` 的 `10000.0`、`0` 仓位开始。Futu dry-run 证据显示 `account_value=10000.0`、目标 `0.85` 被 cold-start step clip 到 `0.35`，`risk_gate.approved=true` 且未下单。Futu 后台模拟账户仍可能显示券商侧现金，但 SOXL 策略 sizing/dashboard 现在以 repo-side `10000` notional 为准。
41. **Vercel paper_trading 推理显示已从 row metadata 解耦**：`paper_trading` 当前推理模型是 `v19reg_v22_budget_usage_bull_guard_v1` runtime composer，slow actor 为 `balanced_rule_residual_v19_reg`，fast actor 为 `balanced_rule_residual_v22`。`10000` reset 后 ordinary paper CSV 只有 seed row，且 dashboard builder 不应把 dry-run summary 贴到普通 row 上；现在静态导出改为在 meta 中携带 `latest_signal_*`，直接指向 `runtime/soxl/paper_trading/2026-05-22/dry_run/futu/summary.json`，Vercel 首屏可显示 profile、dry_run、Futu broker、target `0.85` 与 capped target `0.35`。

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

## 2026-05-21 Provider Coverage Diagnostics

Context:

- A manual full wrapper scheduler attempt at `2026-05-21T06:01:51+08:00` still
  could not complete non-dry paper execution.
- It used the same LaunchAgent wrapper/env path:
  `SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh --broker paper --no-notify`.

Readout:

- `yfinance`: `No overlapping dates for SOXL/SOXX/VIX`.
- `yahoo_api`: Yahoo `429 Too Many Requests` on `^VIX`.
- `polygon`: fetched successfully, wrote/refreshed data surfaces, but produced
  `new_rows=0` and `last_date=2026-05-19`.
- The scheduler fell back to cached dry-run and did not mutate paper broker
  state.

Change:

- `scripts/update_market_data.py` now returns and prints per-leg
  `source_ranges` plus `overlap_last_date`.
- The regression in `tests/test_update_market_data.py` proves a lagging data leg
  is visible in the returned result.

Live diagnostic:

- Polygon/CBOE dry-run at `2026-05-21T06:04:42+08:00` reported:
  `SOXL=2026-05-11..2026-05-19(7)`,
  `SOXX=2026-05-11..2026-05-19(7)`,
  `VIX=2026-05-11..2026-05-19(7)`,
  `overlap_last_date=2026-05-19`.

Decision:

- The current freshness blocker is provider coverage for the latest expected
  trading day, not a single VIX-only gap.
- The next successful production proof must show `overlap_last_date` and panel
  `last_date` advancing to the expected closed US session before non-dry paper
  execution.

Final gates:

- focused data/production/dashboard tests: `46 passed`
- `scripts/check_harness.py`: `passed=129 failures=0`
- `scripts/check_dashboard_static_export.py`: `generated_at=2026-05-21 06:02:28`,
  `versions=25`, `missing=0`, `paper_trading_rows=3`,
  `paper_trading_end=2026-05-18`
- `scripts/build_runtime_index.py --check`: `runtime index OK`
- `git diff --check`: clean
- healthcheck: data stale at `last_date=2026-05-19` vs
  `expected=2026-05-20`; latest signal artifact remains dry-run
  `2026-05-19`; post-close status remains cached dry-run fallback only

## 2026-05-21 Post-Close Full Success

Context:

- The provider coverage diagnostic exposed a second bug: yfinance had SOXL,
  SOXX, and VIX rows through `2026-05-20`, but merge still failed because
  provider date keys were normalized only after merge.
- The production runner also refreshed canonical panel and qlib without syncing
  raw SOXL/SOXX/VIX mirror CSVs.

Change:

- `scripts/update_market_data.py` normalizes provider `date` columns before
  source-range diagnostics and merge.
- `production/soxl_paper_trading.py` syncs raw local surfaces after successful
  production data update and before qlib refresh.
- New regressions cover provider date normalization and production raw-surface
  sync.

Readout:

- Manual wrapper run:
  `SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh --broker paper --no-notify`
  advanced yfinance overlap to `2026-05-20`, added one canonical row, passed the
  freshness gate, selected `balanced_rule_residual_v19`, generated a
  `2026-05-20` signal, filled a non-dry paper order, and rebuilt dashboard data.
- LaunchAgent kickstart then ran the same path from
  `com.soxl.daily.scheduler`; same-day idempotency skipped the duplicate order
  and wrote exact `[scheduler] all done`.
- `production/check_scheduler_health.sh` reported data ok, latest signal
  `dry_run=false executed=true`, dashboard static export ok, and
  `postclose status: success marker found`.

Artifacts:

- canonical panel and raw mirrors: `4074` rows, ending `2026-05-20`
- paper broker state: `date=2026-05-20`, `shares=2800`,
  `position=0.4840856822`, `account_value=1001806.1044`
- paper summary: `runtime/soxl/paper_trading/2026-05-20/summary.json`,
  `dry_run=false`, `executed=true`
- paper dashboard CSV: `4` rows, latest row `2026-05-20`, action `SELL`
- Vercel/static dashboard bundle: paper version ending `2026-05-20` with
  `4` rows
- visual screenshot:
  `runtime/production/card/postclose_full_success_v1/dashboard_visual_2026-05-21.png`,
  `1440 x 1000`, `99.35%` non-white pixels

Final gates:

- focused data/production/dashboard tests: `48 passed`
- `scripts/check_harness.py`: `passed=130 failures=0`
- `scripts/check_dashboard_static_export.py`: `generated_at=2026-05-21 06:13:40`,
  `versions=25`, `missing=0`, `paper_trading_rows=4`,
  `paper_trading_end=2026-05-20`
- `scripts/build_runtime_index.py --check`: `runtime index OK`
- `git diff --check`: clean
- healthcheck: data ok at `last_date=2026-05-20 expected=2026-05-20`, latest
  signal `dry_run=false executed=true`, post-close status
  `success marker found`

Decision:

- Production is no longer blocked on data freshness, scheduled inference,
  dashboard static export, or simulated paper execution.
- The next work can resume SOXX-aware model optimization from the existing
  selective-release / execution-absorption frontier, using the normal experiment
  governance gate.

## 2026-05-21 v22 Auto-Select And Dry-Run Preservation

Context:

- After production proof, model optimization resumed through the governed
  v22/post-release-cap path.
- Simply approving the overlay flag was not enough because legacy auto-select
  still ranked promoted profiles by training `best_score`, where v19 `94.48`
  exceeded v22 `91.94`.

Change:

- Runtime auto-selection now uses a complete walk-forward report when present,
  scoring each window with the profile selection weights and relative
  outperformance as the return term; otherwise it falls back to legacy
  `best_score`.
- `balanced_rule_residual_v22` now has
  `execution_overlay.auto_select_allowed=true`.
- Dry-run output after an already executed paper day is redirected to
  `runtime/soxl/paper_trading/<date>/dry_run/` so the root executed
  `summary.json` remains the post-close proof.
- Duplicate non-dry-run protection now falls back to paper broker state plus the
  cumulative paper CSV if `summary.json` is accidentally polluted by a dry-run.

Readout:

- Live auto-select: `balanced_rule_residual_v22`
  (`soxl_balanced_rule_residual_v22_actor.pt`), score `64.88`,
  source `walk_forward_mean`.
- Retained v19 score: `46.17`, source `walk_forward_mean`.
- Default paper dry-run on `2026-05-20` selected v22/post-release-cap and wrote
  dry-run output under `runtime/soxl/paper_trading/2026-05-20/dry_run/`.
- `production/check_scheduler_health.sh` reports model health on v22 and keeps
  latest executed paper marker at `2026-05-20`, `dry_run=false`,
  `executed=true`.

Decision:

- Scheduler model selection is now aligned with corrected walk-forward evidence
  and explicit overlay governance.
- The next post-close non-dry paper run should naturally execute with v22 unless
  the promotion gate or overlay approval is revoked.

## 2026-05-21 Dashboard Static Default Guard

Context:

- The static/Vercel dashboard export included paper-trading data through
  `2026-05-20`, but `latest_version` still pointed to
  `balanced_rule_residual_v22_soxx`, a failed offline SOXX post-train snapshot.

Change:

- `scripts/build_backtest_dashboard.py` now ranks source kinds for static
  default selection as
  `paper_trading > post_train_review > generalization_version`.
- `scripts/check_dashboard_static_export.py` now fails if `paper_trading` is
  exported but `latest_version` is not `paper_trading`.
- `dashboard.html` and `dashboard_data/` were regenerated with
  `latest_version=paper_trading`.

Evidence:

- Static check:
  `latest=paper_trading`, `paper_trading_rows=4`,
  `paper_trading_end=2026-05-20`, `missing=0`.
- Focused tests:
  `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py tests/test_vercel_dashboard.py -q`
  returned `13 passed`.
- Headless Chrome runtime read:
  `selectValue=paper_trading`, `selectedText="paper_trading (SOXL)"`,
  `readyState=complete`.
- Render screenshot:
  `runtime/production/card/dashboard_static_default_v1/dashboard_static_default_2026-05-21.png`.

Decision:

- The production dashboard default should track the simulated paper loop.
- Offline experiment snapshots stay selectable but must not be the first-load
  default while `paper_trading` is present.

## 2026-05-21 Scheduler Runtime-Proof Visibility

The health check now explicitly reports whether the latest non-dry executed
paper artifact proves the current selected runtime model.

Reason:

- current model health selects `balanced_rule_residual_v22`
- latest executed paper artifact remains
  `runtime/soxl/paper_trading/2026-05-20/summary.json`
- that artifact was created before v22 auto-select approval and still has
  `profile=balanced_rule_residual_v19`

Change:

- `production/check_scheduler_health.sh` compares the latest executed summary
  profile with the healthcheck-selected model profile.
- On the current state it prints:
  `runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 selected_model=balanced_rule_residual_v22`
- Once the next fresh post-close non-dry paper run writes a v22 summary, the
  same line should switch to `runtime proof: ok ...`.

Evidence:

- `tests/test_scheduler_health.py::test_scheduler_health_reports_pending_runtime_profile_proof`
- `runtime/production/card/v22_runtime_proof_pending_v1/notes.md`

Decision:

- Scheduler, data, dashboard, and paper broker health are still ok.
- The remaining production proof is specifically the next fresh post-close
  non-dry paper execution under v22.

Update:

- The `2026-05-21 06:12 CST` post-close run already executed expected data date
  `2026-05-20` under v19 before v22 auto-select approval.
- Current health now selects v22 and still sees the executed v19 artifact, so
  the mismatch is expected until the next eligible non-dry paper date.
- `production/check_scheduler_health.sh` now prints an explicit detail line when
  `latest_executed_date == expected_last_trading_day`:
  `runtime proof detail: latest executed date already matches expected data date; next non-dry-run execution must use selected model`.
- Do not rerun the same signal date to manufacture v22 proof; preserve same-day
  idempotency and wait for the next fresh post-close execution.

## 2026-05-21 Vercel Static Bundle Audit

The repo-owned static dashboard bundle is Vercel-ready:

- `vercel.json` rewrites `/` to `/dashboard.html`
- no-cache headers are configured for `/`, `/dashboard.html`, and
  `/dashboard_data/:path*`
- `dashboard.html` is present at repo root
- `dashboard_data/index.json` and `dashboard_data/model_versions.json` are
  present
- `dashboard_data/versioned/` contains 25 SOXL payload files
- `scripts/check_dashboard_static_export.py` reports
  `latest=paper_trading`, `paper_trading_rows=4`, and
  `paper_trading_end=2026-05-20`

Evidence:

- `tests/test_vercel_dashboard.py`
- `scripts/check_dashboard_static_export.py`
- `runtime/production/card/vercel_static_bundle_audit_v1/notes.md`

Boundary:

- This proves repository static packaging and routing readiness.
- It does not prove a remote Vercel deployment is live, because Vercel project
  state and deployment URL are not stored in the repo.

## 2026-05-21 Scheduler Next-Run Health Display

`production/check_scheduler_health.sh` used to print a static
`Next Scheduled Runs (today)` block even after the `05:35 Tue-Sat
Asia/Shanghai` post-close window had already passed.

Change:

- Added next-run labels derived from `SOXL_SCHEDULER_HEALTH_WEEKDAY` and
  `SOXL_SCHEDULER_HEALTH_HHMM`.
- Eligible windows that are still ahead show `today`.
- Elapsed or ineligible pre-open/post-close windows show
  `next eligible day`.
- Weekly walk-forward shows `next Saturday` after Saturday `07:00`.

Evidence:

- `tests/test_scheduler_health.py::test_scheduler_health_next_runs_skip_elapsed_postclose_window`
- focused production/dashboard suite returned `49 passed`
- live health at `2026-05-21 08:00 CST` printed post-close as
  `next eligible day 05:35 Tue-Sat Asia/Shanghai`

Decision:

- This only improves operator observability.
- It does not change the LaunchAgent schedules or paper execution behavior.

## 2026-05-21 Weekly Walk-Forward Scheduler Surface

The previous health display mentioned a weekly walk-forward next run and checked
`runtime/soxl/wf_weekly.log`, but there was no repo-owned weekly LaunchAgent.
The only weekly command lived in `production/crontab.txt`, pointed at stale
`/home/rziiiii/dev/workspace`, and still ran dropped `balanced_trade_lock`.

Change:

- Added `production/run_soxl_weekly_wf.sh`.
- Added `production/com.soxl.weekly.wf.plist` for Saturday `07:00`
  Asia/Shanghai.
- Updated `production/check_scheduler_health.sh` to report
  `com.soxl.weekly.wf` load/plist status and `runtime/soxl/wf_weekly.err`.
- Updated `production/crontab.txt` to delegate weekly work to the wrapper.

The wrapper refreshes corrected walk-forward evidence for
`balanced_rule_residual_v22`, then reruns `scripts/review_runtime_selection.py`
with `--persist-model-promotion` using the existing post-release-cap overlay
review.

Evidence:

- `tests/test_production_schedule.py`
- `tests/test_scheduler_health.py`
- `runtime/production/card/weekly_walkforward_scheduler_v1/notes.md`
- live health now reports `LaunchAgent com.soxl.weekly.wf: loaded` and
  `plist: matches repo`

Decision:

- Weekly walk-forward is now a real scheduled production surface.
- The first `runtime/soxl/wf_weekly.log` will appear after the next Saturday
  run; until then the absence of that log is expected.

## 2026-05-21 Vercel Remote Deployment Attempt

The repo-owned static dashboard bundle is valid, but remote Vercel deployment is
not yet proven.

Evidence:

- Static export check:
  `latest=paper_trading`, `paper_trading_rows=4`,
  `paper_trading_end=2026-05-20`, `missing=0`.
- `vercel` CLI is installed locally as version `50.32.5`.
- `vercel whoami` reports no existing credentials.
- There is no `.vercel/project.json` link in the repo.
- Deployment check card:
  `runtime/production/card/vercel_remote_deploy_v1/notes.md`

Change:

- Added `scripts/deploy_vercel_dashboard.py`.
- The script validates static dashboard integrity first.
- It can deploy non-interactively with local Vercel login or `VERCEL_TOKEN`.
- It redacts token values from saved command evidence.
- In the current environment it writes `status=blocked_no_credentials` instead
  of claiming a remote deployment URL.

Operational next step:

```bash
VERCEL_TOKEN=... PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy
```

or run `vercel login`, then:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy
```

## 2026-05-21 v22 Next Execution Preflight

Current runtime auto-selection is already pointed at
`balanced_rule_residual_v22`, but the latest non-dry paper execution for
`2026-05-20` remains the earlier `balanced_rule_residual_v19` artifact. The
same signal date must not be executed twice just to produce a v22 proof.

Safe preflight command:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker paper --dry-run --no-notify --skip-data-update
```

Readout:

- data fresh: `last_date=2026-05-20`, `expected=2026-05-20`, `stale_days=0`
- auto-selected: `balanced_rule_residual_v22`
- signal date: `2026-05-20`
- risk gate: approved
- step check: `change=0.016 limit=0.350`
- target/current position: `0.5000` / `0.4841`
- dry-run path: `runtime/soxl/paper_trading/2026-05-20/dry_run`
- root executed summary remains the v19 non-dry proof for `2026-05-20`

Decision:

- The next live paper execution path is ready for v22 under current data,
  broker state, and risk gates.
- Completion proof still requires the next fresh post-close non-dry execution
  to write a root summary with `profile=balanced_rule_residual_v22`.
- Evidence card:
  `runtime/production/card/v22_next_execution_preflight_v1/notes.md`.

## 2026-05-21 Vercel Health Surface

`production/check_scheduler_health.sh` now includes remote Vercel deployment
status, sourced from
`runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`.

Live readout at `2026-05-21 09:02 CST`:

```text
--- Vercel Remote Deployment ---
  status: blocked_no_credentials
  auth: blocked_no_credentials
  deployment_url: none
  dashboard_latest=paper_trading paper_trading_end=2026-05-20
  evidence=runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json
```

Regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_vercel_remote_deployment_status -q
```

Decision:

- Static dashboard readiness and remote deployment status are now visible in one
  production health command.
- Remote Vercel deployment is still not proven because the current machine has
  no Vercel credentials.
- Evidence card:
  `runtime/production/card/vercel_health_surface_v1/notes.md`.

## 2026-05-21 Vercel Scheduled Deploy Integration

The Vercel deployment script is now part of the scheduled dashboard update path.

Change:

- `production/soxl_daily_scheduler.py` accepts `--vercel-deploy`.
- After dashboard rebuild, the scheduler runs
  `scripts/deploy_vercel_dashboard.py --deploy`.
- Return code `2` from the deploy script means `blocked_no_credentials` and is
  non-fatal by default, so paper trading and dashboard generation remain healthy
  while remote deployment is blocked.
- `--strict-vercel-deploy` can make that blocker fatal for operator runs.
- The pre-open and post-close LaunchAgent plists now pass `--vercel-deploy`.
- `production/crontab.txt` daily SOXL entries now use the current repo wrapper
  path instead of stale `/home/rziiiii/dev/workspace` commands.
- Local LaunchAgent copies were synced and reloaded.

Evidence:

- Card:
  `runtime/production/card/vercel_scheduled_deploy_v1/notes.md`
- Tests:
  `tests/test_daily_scheduler.py`
  `tests/test_production_schedule.py`
- Live smoke:
  `PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker paper --dry-run --no-notify --skip-data-update --vercel-deploy`

Readout:

- The live smoke selected `balanced_rule_residual_v22`, wrote the
  `2026-05-20/dry_run` output, rebuilt dashboard static files, attempted Vercel
  deploy, refreshed `vercel_remote_deploy_v1` evidence, and exited successfully.
- Vercel remains `blocked_no_credentials` with `deploy_requested=true`.
- Health now reports both scheduled dashboard windows as including
  `Vercel deploy attempt`, and installed LaunchAgent plists match the repo
  copies.

Decision:

- Scheduled dashboard updates are now Vercel-aware.
- Completion still requires scheduler-visible Vercel credentials and a
  successful remote deployment URL.

## 2026-05-21 Post-close 06:15 Provider Buffer

The original `05:35 Tue-Sat Asia/Shanghai` post-close window was structurally
after the NYSE close boundary, but live evidence showed provider lag:

- `05:35`, `05:44`, and `05:50` attempts could not produce a fresh
  `2026-05-20` SOXL/SOXX/VIX overlap and correctly fell back to cached dry-run.
- `06:12` produced fresh overlap through `2026-05-20`, refreshed qlib, and
  completed the post-close paper/dashboard path.

Change:

- Post-close full scheduler is now `06:15 Tue-Sat Asia/Shanghai` in cron,
  LaunchAgent plist, health output, and operator docs.
- Health uses `POSTCLOSE_HHMM=0615` / `POSTCLOSE_TIME_LABEL=06:15`, and a new
  test ensures `06:00` remains pending during the provider buffer.
- `production/run_soxl_scheduler.sh` prepends `/opt/homebrew/bin` and related
  binary directories to PATH so scheduled Vercel deploy attempts can find the
  local CLI from LaunchAgent.
- `scripts/build_backtest_dashboard.py` now preserves retained
  `dashboard_data/versioned/` exports in the rebuilt index when a dropped
  profile's source daily CSV has intentionally been removed.
- Installed `com.soxl.daily.scheduler` was synced and reloaded.

Evidence:

- Focused dashboard/scheduler/Vercel suite: `46 passed`.
- Static dashboard export: `versions=28 missing=0 model_versions=94
  latest=paper_trading`.
- LaunchAgent `print` shows `Hour=6 Minute=15 Weekday=2..6`, `state=not running`,
  and `last exit code = 0` after kickstart.
- Health at `2026-05-21 11:00 CST` reports daily plist `matches repo`,
  `postclose status: success marker found`, and next post-close as
  `next eligible day 06:15 Tue-Sat Asia/Shanghai`.
- Vercel evidence now records `vercel_cli=/opt/homebrew/bin/vercel`; remote
  deployment remains `blocked_no_credentials`.

Decision:

- Treat `06:15 Tue-Sat Asia/Shanghai` as the canonical post-close production
  schedule.
- Preserve retained dashboard evidence across scheduled rebuilds; do not require
  dropped profile source CSVs to stay active just to remain selectable.
- Do not force a same-day non-dry order for v22; the kickstart selected v22 but
  correctly skipped duplicate execution for `2026-05-20`.
- Completion still requires Vercel credentials and the next fresh non-dry paper
  date writing a root `balanced_rule_residual_v22` summary.

Evidence card:

- `runtime/production/card/scheduler_postclose_0615_v1/notes.md`

## 2026-05-21 Budget-Usage Runtime Composer Interface

The standard production entrypoints can now run a governed budget-usage
composer when explicitly requested:

- `production/soxl_paper_trading.py --runtime-composer budget_usage`
- `production/soxl_daily_scheduler.py --runtime-composer budget_usage`

The required operator-supplied fields are `--composer-profile`,
`--slow-profile`, and `--fast-profile`. The bull-participation guard is opt-in
through `--bull-participation-guard` and its window/threshold knobs.

Live smoke:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --dry-run --no-notify --skip-data-update --skip-dashboard \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard
```

Result: scheduler completed and wrote the composer dry-run through paper
trading.

Current gate:

- The candidate is not enabled in installed LaunchAgents.
- The explicit composer path now passes the runtime model gate through
  `runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.json`.
- Keep the production default on single selected `balanced_rule_residual_v22`
  unless an operator intentionally chooses the composer for non-dry paper
  execution on a fresh date.

Evidence card:

- `runtime/production/card/budget_usage_runtime_composer_v1/notes.md`
- `runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.md`

Verification:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_budget_usage_runtime_promotion_review.py tests/test_runtime.py::test_composer_promotion_gate_status_allows_approved_review_over_component_gate tests/test_runtime.py::test_composer_promotion_gate_status_blocks_component_model_mismatch tests/test_runtime.py::test_composer_promotion_gate_status_blocks_unapproved_review -q` ->
  `6 passed`.
- `PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py` ->
  `passed=167 failures=0`.
- `production/check_scheduler_health.sh` -> default data/model/broker/dashboard
  ok; runtime proof pending for selected v22; Vercel `blocked_no_credentials`.

## 2026-05-21 Paper Dry-Run Profile Scoping

The paper runner still must not duplicate same-day non-dry execution, but
same-date dry-run preflights for different profiles should not overwrite one
another. After the budget-usage composer dry-run, the older v22 preflight path
had been overwritten because both used
`runtime/soxl/paper_trading/2026-05-20/dry_run/`.

Change:

- `production/soxl_paper_trading.py` keeps `dry_run/` for the same profile, but
  writes colliding profile dry-runs under `dry_run/<profile>/`.
- Default v22 dry-run and scheduler dry-run now write
  `runtime/soxl/paper_trading/2026-05-20/dry_run/balanced_rule_residual_v22`.
- The composer dry-run remains at the shared `dry_run/summary.json`.

Evidence card:

- `runtime/production/card/paper_dry_run_profile_scope_v1/notes.md`

Verification:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py -q`
  -> `40 passed`.
- `PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py` ->
  `passed=167 failures=0`.
- `production/check_scheduler_health.sh` -> local scheduler/data/model/broker/dashboard
  ok; selected v22 non-dry proof still pending for the next fresh date; Vercel
  `blocked_no_credentials`.

Decision: this is evidence hygiene for simulated paper trading, not a substitute
for the pending selected-profile non-dry runtime proof.

## 2026-05-21 Paper Non-Dry Inference Proof Semantics

Issue: scheduler health required `executed=true` to prove the selected runtime
profile had run non-dry. In `production/soxl_paper_trading.py`, `executed`
means broker order submitted. A correct HOLD or below-lot-threshold selected
profile run would therefore be a false negative.

Change:

- non-dry broker attempts now persist `execution_result.json`
- `summary.json` includes `execution_result`
- `production/check_scheduler_health.sh` treats selected-profile,
  risk-approved, non-dry `execution_result.status=skipped` as runtime proof

Evidence:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_persists_skipped_execution_result_for_non_dry_hold tests/test_scheduler_health.py::test_scheduler_health_accepts_selected_profile_non_dry_skipped_order_as_runtime_proof -q`
  -> `2 passed`.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py -q`
  -> `42 passed`.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py -q`
  -> `62 passed`.
- `PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py` ->
  `passed=168 failures=0`.
- `production/check_scheduler_health.sh` -> local scheduler/data/model/broker/dashboard
  ok; selected v22 runtime proof still waits for the next fresh non-dry date;
  Vercel `blocked_no_credentials`.

Evidence card:

- `runtime/production/card/paper_non_dry_inference_proof_v1/notes.md`

Decision: the next fresh selected-profile non-dry paper run can prove scheduled
inference even if no order is submitted, as long as risk gate approves and the
broker result is explicitly `skipped`.

Harness note: `tests/test_runtime.py` now isolates the relative execution-overlay
state path under a temporary root, so local production state no longer changes
the expected default-state assertion.

## 2026-05-21 Weekly Scheduler Health Pending State

Issue: after installing the weekly walk-forward LaunchAgent,
`production/check_scheduler_health.sh` reported missing `runtime/soxl/wf_weekly`
log files as `NOT FOUND` before the first Saturday run had a chance to create
them.

Change:

- Missing `runtime/soxl/wf_weekly.log` and `runtime/soxl/wf_weekly.err` now
  print `pending (not created yet)`.
- The weekly completion marker now prints
  `wf_weekly: pending until first weekly run` when the weekly log is absent.
- Preopen and postclose missing logs still retain the generic `NOT FOUND`
  behavior.

Evidence:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_weekly_logs_pending_before_first_run -q`
  -> red before implementation, then `1 passed`.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q`
  -> `13 passed`.
- `production/check_scheduler_health.sh` -> weekly LaunchAgent loaded, plist
  matches repo, weekly logs pending/not-created, local data/model/broker/dashboard
  ok, selected v22 runtime proof still pending for the next fresh non-dry date,
  Vercel `blocked_no_credentials`.

Evidence card:

- `runtime/production/card/weekly_health_pending_v1/notes.md`

## 2026-05-21 Vercel Health Freshness

Issue: Vercel deployment evidence can lag behind the latest local static
dashboard rebuild. Health previously reported only the Vercel auth/deploy
status, so an operator could not tell whether the recorded Vercel attempt
covered the current `dashboard_data/index.json` bundle.

Baseline evidence:

- Local dashboard static export: `generated_at=2026-05-21 12:01:45`,
  `versions=35`.
- Existing `vercel_remote_deploy_v1` evidence before refresh:
  `static_export.generated_at=2026-05-21 11:07:30`, `versions=28`.
- Vercel auth was still `blocked_no_credentials`.

Change:

- `production/check_scheduler_health.sh` now compares the local
  `dashboard_data/index.json.generated_at` against
  `vercel_deployment_check.json.static_export.generated_at`.
- Health prints `freshness: stale ...` when local dashboard output is newer
  than Vercel evidence, and `freshness: current ...` when they match.
- The repo-owned Vercel deploy check was rerun with `--deploy`; it returned the
  expected missing-credentials code and refreshed `vercel_remote_deploy_v1`
  against the current dashboard bundle.

Evidence:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_stale_vercel_evidence_when_dashboard_is_newer -q`
  -> red before implementation, then `1 passed`.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q`
  -> `14 passed`.
- `PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy`
  -> expected `rc=2`, `status=blocked_no_credentials`, evidence refreshed.
- `production/check_scheduler_health.sh` -> Vercel
  `blocked_no_credentials`, freshness current for `2026-05-21 12:01:45`,
  local data/model/broker/dashboard ok, selected v22 runtime proof still
  pending for the next fresh non-dry date.

Evidence card:

- `runtime/production/card/vercel_health_freshness_v1/notes.md`

## 2026-05-21 Scheduler Stderr Signal Cleanup

Issue: `runtime/soxl/paper_trading/scheduler_postclose.err` contained both real
historical provider failures and the latest nonfatal Vercel
`blocked_no_credentials` continuation notice. The Vercel blocker is already
persisted in `vercel_remote_deploy_v1` and surfaced by health, so writing it to
stderr on every scheduled deploy attempt made `.err` less useful as a failure
surface.

Change:

- `_run_vercel_deploy()` still permits return code `2` when not strict.
- The nonfatal missing-credentials continuation notice now prints to stdout.
- Strict Vercel failures and normal scheduler errors still flow through stderr.

Evidence:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_treats_missing_vercel_credentials_as_nonfatal -q`
  -> red before implementation, then `1 passed`.
- `qlib_venv/bin/python -m py_compile production/soxl_daily_scheduler.py` -> ok.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_scheduler_health.py -q`
  -> `26 passed`.

Evidence card:

- `runtime/production/card/scheduler_stderr_signal_v1/notes.md`

## 2026-05-21 Selected v22 Duplicate-Skip Inference Proof

Issue: selected `balanced_rule_residual_v22` could run non-dry scheduled
inference on fresh data after a same-date root `balanced_rule_residual_v19`
execution, but the duplicate-order guard returned before persisting v22 evidence.
Health therefore kept reporting selected v22 runtime proof as pending even
though the scheduler had reached selected-profile inference and risk gating.

Change:

- Duplicate same-day non-dry attempts persist under
  `runtime/soxl/paper_trading/<date>/duplicate_skip/<profile>/`.
- The duplicate artifact records `dry_run=false`, `executed=false`,
  `execution_result.status=duplicate_skipped`, and the prior root execution
  under `duplicate_of`.
- Broker order placement, broker state save, and daily CSV append remain
  skipped.
- `production/check_scheduler_health.sh` scans selected-profile nested non-dry
  proof artifacts before declaring selected runtime proof pending.

Evidence:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_persists_duplicate_non_dry_inference_without_order tests/test_scheduler_health.py::test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof -q`
  -> red before implementation, then `2 passed`.
- `PYTHONPATH=src:. qlib_venv/bin/python production/soxl_paper_trading.py --no-notify --skip-data-update`
  -> selected `balanced_rule_residual_v22`, skipped duplicate order, wrote
  `runtime/soxl/paper_trading/2026-05-20/duplicate_skip/balanced_rule_residual_v22`.
- `production/check_scheduler_health.sh` -> selected v22 runtime proof ok with
  `execution=duplicate_skipped`.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py -q`
  -> `46 passed`.

Evidence card:

- `runtime/production/card/paper_duplicate_inference_proof_v1/notes.md`

## 2026-05-21 Vercel Scheduler Env Visibility

`production/check_scheduler_health.sh` now prints a `Scheduler Private Env`
section before log/status checks. It uses the same
`${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` convention as the
LaunchAgent wrappers, checks whether `VERCEL_TOKEN` is configured in either the
current environment or the launch env file, and never prints token values.

Live health on this host shows:

- launch env file present at `/Users/rziiiii/.openclaw/soxl.env`
- `VERCEL_TOKEN` missing in both current environment and launch env file
- Vercel deployment evidence still `blocked_no_credentials`
- selected `balanced_rule_residual_v22` runtime proof still ok via
  `duplicate_skipped`

Verification:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q`
  -> `17 passed`.
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_vercel_dashboard.py -q`
  -> `54 passed`.

Evidence card:

- `runtime/production/card/vercel_scheduler_env_visibility_v1/notes.md`

## 2026-05-21 Scheduler Env Runtime Composer

The LaunchAgent wrapper can now intentionally enable an approved runtime
composer from the private scheduler env file without editing plist arguments.

`production/run_soxl_scheduler.sh` still defaults to the existing single-profile
schedule. After loading `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}`, it
only appends composer flags when `SOXL_RUNTIME_COMPOSER` is set to a non-`none`
value and no explicit `--runtime-composer` arg was already passed.

Supported env fields:

- `SOXL_RUNTIME_COMPOSER`
- `SOXL_RUNTIME_COMPOSER_PROFILE`
- `SOXL_RUNTIME_SLOW_PROFILE`
- `SOXL_RUNTIME_FAST_PROFILE`
- `SOXL_RUNTIME_BULL_PARTICIPATION_GUARD`
- optional bull guard knobs prefixed with `SOXL_RUNTIME_BULL_`

Live smoke with a temporary env file ran:

```bash
SOXL_LAUNCH_ENV_FILE="$tmp_env" production/run_soxl_scheduler.sh \
  --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard
```

Result: scheduler selected `runtime composer=budget_usage`,
`profile=v19reg_v22_budget_usage_bull_guard_v1`, `as_of=2026-05-20`,
`risk_gate approved=True`, and wrote the dry-run artifact to
`runtime/soxl/paper_trading/2026-05-20/dry_run`.

Decision: keep default LaunchAgents on `balanced_rule_residual_v22`. To test the
composer on the next fresh non-dry paper date, add the `SOXL_RUNTIME_*` fields
to `/Users/rziiiii/.openclaw/soxl.env`; do not silently switch defaults.

Evidence card:

- `runtime/production/card/scheduler_env_runtime_composer_v1/notes.md`

## 2026-05-21 Scheduler Composer Env Health

`production/check_scheduler_health.sh` now reports runtime composer env state
inside `Scheduler Private Env`.

It reads non-secret `SOXL_RUNTIME_*` fields from either the current environment
or `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` without sourcing the env
file:

- unset / `none` -> `runtime composer: default single-profile path`
- missing required fields -> `runtime composer: incomplete ... missing=...`
- complete fields -> `runtime composer: enabled composer=... profile=... slow=... fast=... bull_guard=...`

Live health on this host now shows:

- launch env file present at `/Users/rziiiii/.openclaw/soxl.env`
- `VERCEL_TOKEN` missing
- `runtime composer: default single-profile path`
- selected `balanced_rule_residual_v22` runtime proof ok via
  `duplicate_skipped`

Evidence:

- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_launch_env_runtime_composer_configuration tests/test_scheduler_health.py::test_scheduler_health_reports_incomplete_launch_env_runtime_composer_configuration -q`
  -> red before implementation, then `2 passed`.
- `production/check_scheduler_health.sh` -> default single-profile composer
  state, local data/model/broker/dashboard ok, Vercel `blocked_no_credentials`.

Evidence card:

- `runtime/production/card/scheduler_composer_env_health_v1/notes.md`

## 2026-05-21 Scheduler Composer Env Enabled

The private scheduler env at `/Users/rziiiii/.openclaw/soxl.env` now enables
the approved `v19reg_v22_budget_usage_bull_guard_v1` composer for LaunchAgent
wrapper invocations:

- `SOXL_RUNTIME_COMPOSER=budget_usage`
- `SOXL_RUNTIME_COMPOSER_PROFILE=v19reg_v22_budget_usage_bull_guard_v1`
- `SOXL_RUNTIME_SLOW_PROFILE=balanced_rule_residual_v19_reg`
- `SOXL_RUNTIME_FAST_PROFILE=balanced_rule_residual_v22`
- `SOXL_RUNTIME_BULL_PARTICIPATION_GUARD=true`

`production/check_scheduler_health.sh` now reports:

```text
runtime composer: enabled composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22 bull_guard=true
```

The real-env wrapper smoke:

```bash
production/run_soxl_scheduler.sh --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard
```

selected `runtime composer=budget_usage`,
`profile=v19reg_v22_budget_usage_bull_guard_v1`, used cached `2026-05-20`
data, passed `risk_gate approved=True`, and wrote the dry-run artifact to
`runtime/soxl/paper_trading/2026-05-20/dry_run`.

Remaining production gates:

- next fresh-date non-dry paper proof for the composer
- scheduler-visible Vercel credentials
- first Saturday weekly walk-forward artifact

Evidence card:

- `runtime/production/card/scheduler_composer_env_enabled_v1/notes.md`

## 2026-05-21 Scheduler Composer Runtime Proof

`production/check_scheduler_health.sh` now uses the enabled
`SOXL_RUNTIME_*` composer env for both private-env reporting and paper-trading
healthcheck. Before this fix, health could show
`runtime composer: enabled` while still validating
`balanced_rule_residual_v22` and accepting the old v22 duplicate proof.

Live health now validates:

```text
model: ok v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)
```

The real-env non-dry wrapper run:

```bash
production/run_soxl_scheduler.sh --broker paper --no-notify --skip-data-update --skip-dashboard
```

selected `runtime composer=budget_usage`,
`profile=v19reg_v22_budget_usage_bull_guard_v1`, passed
`risk_gate approved=True`, and wrote a same-date duplicate-skip non-dry proof:

```text
runtime/soxl/paper_trading/2026-05-20/duplicate_skip/v19reg_v22_budget_usage_bull_guard_v1/summary.json
```

Current health reports:

```text
runtime proof: ok latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

This proves composer inference and risk gating on the current expected data
date without submitting a duplicate same-day paper order. The next new trading
date should create the first ordinary scheduled composer paper artifact.

Evidence card:

- `runtime/production/card/scheduler_composer_runtime_proof_v1/notes.md`

## 2026-05-21 Weekly Composer Review Wrapper

The weekly wrapper now follows the active runtime target when the private
scheduler env enables `SOXL_RUNTIME_COMPOSER=budget_usage`.

Before this change, `production/run_soxl_weekly_wf.sh` loaded the env file but
still defaulted to `balanced_rule_residual_v22` and the single-profile
`scripts/review_runtime_selection.py` path. That would make the weekly evidence
review a stale target after scheduler runtime switched to
`v19reg_v22_budget_usage_bull_guard_v1`.

Current behavior:

- complete `SOXL_RUNTIME_COMPOSER=budget_usage` env ->
  `scripts/review_budget_usage_runtime_promotion.py`
- no runtime composer env -> existing `balanced_rule_residual_v22`
  walk-forward + runtime-selection fallback
- `SOXL_WEEKLY_WF_ROOT` / `SOXL_WEEKLY_WF_PYTHON` are available for safe command
  capture tests

Manual real-env wrapper proof was appended to LaunchAgent-style logs:

```text
runtime/soxl/wf_weekly.log
[weekly-wf] run started ... mode=runtime_composer composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 ...
[budget_usage_runtime_promotion] wrote runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1
[weekly-wf] all done
```

Health now reports:

```text
wf_weekly: [weekly-wf] all done
```

The composer runtime promotion review remains approved with no blocking
reasons. Vercel credentials remain the remaining external deployment blocker.

Evidence card:

- `runtime/production/card/weekly_composer_review_v1/notes.md`

## 2026-05-21 Vercel Project Readiness

Vercel health now reports project-link readiness in addition to auth status.

`scripts/deploy_vercel_dashboard.py` records a non-secret `project.status` in
`runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`:

- `linked` when `.vercel/project.json` has project/org IDs
- `env_configured` when `VERCEL_PROJECT_ID` and `VERCEL_ORG_ID` are set
- `missing_project_link` when neither is present
- `invalid_project_link` when `.vercel/project.json` is malformed or incomplete

Current live evidence:

```text
status=blocked_no_credentials
auth=blocked_no_credentials
project=missing_project_link
deployment_url=null
dashboard_latest=paper_trading
paper_trading_end=2026-05-20
```

Health now prints:

```text
project: missing_project_link
```

The remote Vercel deployment blocker is therefore two-part:

1. scheduler-visible credentials: `VERCEL_TOKEN` or `vercel login`
2. project link/config: `vercel link --yes --team <team-id>
   --project <project-name-or-id>` or `VERCEL_ORG_ID` + `VERCEL_PROJECT_ID`

Evidence card:

- `runtime/production/card/vercel_project_readiness_v1/notes.md`

## 2026-05-21 Ordinary Paper Proof Health

Scheduler health now separates selected-profile runtime proof from ordinary
new-date paper proof.

The active composer currently has selected-profile non-dry proof through
same-date duplicate-skip idempotency:

```text
runtime proof: ok latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

That is not the same as the first ordinary scheduled post-close paper artifact
for the composer, so health now also prints:

```text
ordinary paper proof: pending latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
ordinary paper proof detail: duplicate-skip proves same-date idempotent inference only; next new-date scheduled non-dry run must produce executed or skipped order proof
```

Interpretation:

- `runtime proof` can be `ok` for `duplicate_skipped` because it proves
  inference, risk gating, and idempotent duplicate protection.
- `ordinary paper proof` is `ok` only for `executed=true` or fresh non-dry
  `execution=skipped`.
- The remaining simulated-paper gate is the next new-date scheduled post-close
  non-dry run under the enabled composer.

Evidence card:

- `runtime/production/card/ordinary_paper_proof_health_v1/notes.md`

## 2026-05-21 Thread Goal Completion Audit

The thread objective now has a requirement-by-requirement completion audit in
`runtime/production/card/thread_goal_completion_audit_v1/notes.md`.

Current readout:

- SOXX data integration is not the blocker: canonical SOXL/SOXX/VIX panel, raw
  mirrors, and qlib calendar all end at `2026-05-20`.
- Model optimization did resume after SOXX: direct SOXX actor / SOXX cap lines
  were rejected, and the governed continuation is
  `v19reg_v22_budget_usage_bull_guard_v1`.
- Dashboard local/static proof is current with `latest=paper_trading`,
  `missing=0`, and paper rows through `2026-05-20`.
- Scheduler env is enabled for the runtime composer and health validates that
  composer target.
- Simulated-paper runtime proof is ok through same-date duplicate-skip
  idempotency, but ordinary paper proof remains pending until the next new-date
  scheduled post-close non-dry run.
- Vercel remote deployment is not complete because credentials and project link
  are both missing.

The goal should remain active until at least these gates clear:

1. `ordinary paper proof: ok ...` for
   `v19reg_v22_budget_usage_bull_guard_v1`
2. Vercel remote evidence with credentials configured, project linked, and a
   deployment URL

Evidence card:

- `runtime/production/card/thread_goal_completion_audit_v1/notes.md`

## 2026-05-21 Post-close Proof Status

`production/check_scheduler_health.sh` now carries ordinary paper proof state
into the post-close section. This avoids a misleading split where
`Latest Signal Artifact` says ordinary paper proof is pending but
`Post-close Run Status` only says `success marker found`.

Current live readout:

```text
postclose status: success marker found
postclose proof: ordinary paper proof pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

Interpretation:

- The wrapper completed.
- Composer non-dry inference/risk/idempotency is proven.
- Ordinary simulated-paper proof still waits for the next new-date scheduled
  non-dry run to produce `executed=true` or fresh `execution=skipped`.

Evidence card:

- `runtime/production/card/postclose_proof_status_v1/notes.md`

## 2026-05-21 Vercel Project Env Visibility

Scheduler health now reports non-secret Vercel project env visibility in
`Scheduler Private Env`.

Current live readout:

```text
VERCEL_TOKEN: missing in current environment and launch env file
VERCEL_PROJECT: missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file
```

If an operator adds `VERCEL_PROJECT_ID` and `VERCEL_ORG_ID` to the LaunchAgent
env file, health will report `VERCEL_PROJECT: configured in launch env file`
without printing either ID value. This complements the remote deployment
evidence card, which still reports whether `.vercel/project.json` or env IDs
are accepted by the deploy script.

Evidence card:

- `runtime/production/card/vercel_project_env_visibility_v1/notes.md`

## 2026-05-21 Post-close Runtime Target Marker

The daily scheduler now prints runtime-composer target fields in its run-start
marker when the wrapper passes `--runtime-composer`.

The current real wrapper smoke:

```text
[scheduler] run started ... runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1 slow_profile=balanced_rule_residual_v19_reg fast_profile=balanced_rule_residual_v22 bull_guard=True
[paper] using runtime composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22
[scheduler] all done
```

Scheduler health now also audits the latest post-close success marker against
the enabled private-env composer target. Current live readout:

```text
postclose status: success marker found
postclose target: pending latest_success_marker lacks runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
postclose proof: ordinary paper proof pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

Interpretation:

- The latest scheduled post-close wrapper success predates the explicit runtime
  composer target marker.
- The next scheduled post-close run must show both `postclose target: ok ...`
  and `postclose proof: ordinary paper proof ok ...`.
- Until both lines clear, the simulated-paper part of the thread goal remains
  incomplete.

Evidence card:

- `runtime/production/card/postclose_runtime_target_marker_v1/notes.md`

## 2026-05-21 Vercel Launch Env Probe

`scripts/deploy_vercel_dashboard.py` now reads the scheduler private env file
for Vercel readiness checks. It defaults to
`${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` and only parses these keys:

- `VERCEL_TOKEN`
- `VERCEL_PROJECT_ID`
- `VERCEL_ORG_ID`

Current process env remains higher priority; the env file fills missing values.
Evidence records only env-file presence and variable names, never values.

Current live readiness:

```text
vercel_dashboard status=blocked_no_credentials output=/Users/rziiiii/.openclaw/workspace/runtime/production/card/vercel_remote_deploy_v1
```

Current health readout:

```text
status: blocked_no_credentials
auth: blocked_no_credentials
project: missing_project_link
env_file: present applied_keys=none
deployment_url: none
```

Interpretation:

- The readiness check now uses the same env layer as the scheduled wrapper.
- The remaining Vercel blocker is not script visibility anymore; the private
  env file simply does not contain Vercel token/project keys yet.
- Add `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, and `VERCEL_ORG_ID` to that env file,
  then rerun `scripts/deploy_vercel_dashboard.py` without `--deploy` before the
  next scheduled deploy attempt.

Evidence card:

- `runtime/production/card/vercel_launch_env_probe_v1/notes.md`

## 2026-05-21 Production Goal Gate Summary

`production/check_scheduler_health.sh` now prints a final
`Production Goal Gates` section that maps the active thread goal to explicit
gate states.

Current live summary:

```text
data: ok last_date=2026-05-20 expected=2026-05-20
model: ok v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)
broker: ok paper broker ready
dashboard static: ok dashboard_static_export ... latest=paper_trading ... paper_trading_end=2026-05-20
Vercel remote: blocked status=blocked_no_credentials auth=blocked_no_credentials project=missing_project_link deployment_url=none env_file_applied_keys=none
post-close composer target: pending latest_success_marker lacks runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
ordinary paper proof: pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Interpretation:

- Data/model/broker/dashboard static are currently good.
- Vercel remains externally blocked by missing scheduler-visible Vercel
  credentials/project IDs.
- The scheduled post-close run still has not produced current-composer target
  proof or ordinary paper proof.
- Future sessions can use the final `overall:` line as the first completion
  audit entry point, then inspect the relevant detailed section only for any
  issue keys listed there.

Evidence card:

- `runtime/production/card/production_goal_gate_summary_v1/notes.md`

## 2026-05-21 Scheduled Goal Gate Logging

The daily scheduler now accepts `--health-summary`. After a successful normal
run, and after the cached dry-run fallback success path, it prints:

```text
[scheduler] production goal gates
--- Production Goal Gates ---
...
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

`production/crontab.txt`, `production/com.soxl.daily.scheduler.plist`, and
`production/com.soxl.preopen.scheduler.plist` now pass this flag. The installed
daily and pre-open LaunchAgent plists were copied from repo and reloaded; health
reported both installed plists as matching repo.

The validation smoke used:

```text
production/run_soxl_scheduler.sh --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard --health-summary
```

It completed paper inference in dry-run mode, skipped order placement, and then
printed the goal-gate summary. This means the next real scheduled pre-open or
post-close log should carry its own thread-level completion audit.

Current live state is still:

```text
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Evidence card:

- `runtime/production/card/scheduled_goal_gate_logging_v1/notes.md`

## 2026-05-21 Scheduler Health Summary Noise

`_run_health_summary()` now suppresses stderr from
`production/check_scheduler_health.sh` when the health command exits 0. This
keeps scheduled `--health-summary` output focused on the final gate summary:

```text
[scheduler] all done
[scheduler] production goal gates
--- Production Goal Gates ---
...
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

If the health command exits non-zero, stderr is still replayed and the scheduler
prints the exit-code warning. The production completion state did not change;
the live pending gates remain:

```text
external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Evidence card:

- `runtime/production/card/scheduler_health_summary_noise_v1/notes.md`

## 2026-05-21 Operator Scheduler Docs Sync

`production/FUTU_SEMI_AUTO.md` now matches the current scheduler behavior:

- manual pre-open and post-close cron examples include `--vercel-deploy
  --health-summary`;
- the Vercel scheduler smoke command includes `--health-summary`;
- LaunchAgent install/uninstall examples use `launchctl bootstrap` /
  `launchctl bootout`;
- the private env-file setup shows commented placeholders for
  `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, and `VERCEL_ORG_ID`, so an operator sees
  the required keys without accidentally exporting fake `REPLACE_ME` values.

A regression test now guards the documented scheduler examples:

```text
tests/test_production_schedule.py::test_operator_doc_scheduler_examples_include_current_production_flags
```

Live production state remains unchanged:

```text
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Evidence card:

- `runtime/production/card/operator_scheduler_docs_sync_v1/notes.md`

## 2026-05-21 Scheduled Post-close Slot Marker

The daily scheduler now accepts `--schedule-slot {manual,preopen,postclose}`
and prints `schedule_slot=<value>` in the run-start marker.

Current repo-owned scheduled entrypoints pass explicit slots:

- pre-open crontab/LaunchAgent: `--schedule-slot preopen`
- post-close crontab/LaunchAgent: `--schedule-slot postclose`

`production/check_scheduler_health.sh` now requires
`schedule_slot=postclose` before the post-close composer target can be marked
OK. This prevents a manual run written to `scheduler_postclose.log` from
clearing the scheduled post-close target gate.

Live validation after reloading installed daily/pre-open LaunchAgents:

```text
LaunchAgent com.soxl.preopen.scheduler plist: matches repo
LaunchAgent com.soxl.daily.scheduler plist: matches repo
post-close composer target: pending latest_success_marker lacks schedule_slot=postclose
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Evidence card:

- `runtime/production/card/scheduled_postclose_slot_marker_v1/notes.md`

## 2026-05-21 Vercel Placeholder Guard

The Vercel readiness path now treats `REPLACE_ME` / `replace-me` placeholder
values as missing for:

- `VERCEL_TOKEN`
- `VERCEL_PROJECT_ID`
- `VERCEL_ORG_ID`

This is enforced in both:

- `scripts/deploy_vercel_dashboard.py`
- `production/check_scheduler_health.sh`

Current live checks after the change still report:

```text
VERCEL_TOKEN: missing in current environment and launch env file
VERCEL_PROJECT: missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

This prevents operator-doc placeholders from creating a false
`ready_for_deploy` or `configured` signal.

Evidence card:

- `runtime/production/card/vercel_placeholder_guard_v1/notes.md`

## 2026-05-21 Pre-open Run Status

`production/check_scheduler_health.sh` now has a `Pre-open Run Status` section
for the `21:00 Mon-Fri Asia/Shanghai` cached-data dry-run inference path.

The section reports whether the pre-open slot is pending, due with no success,
due with only stale success, or has a current success marker. A current success
also prints `preopen target` and requires `schedule_slot=preopen`,
`dry_run=True`, `skip_data_update=True`, plus the enabled runtime-composer
fields when `SOXL_RUNTIME_COMPOSER=budget_usage` is configured.

Current live readout before the 21:00 slot:

```text
preopen status: pending until 21:00 Mon-Fri Asia/Shanghai
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

This is an observability improvement only. The final production goal gates did
not change; Vercel setup, scheduled post-close target, and ordinary paper proof
remain the active blockers.

Evidence card:

- `runtime/production/card/preopen_run_status_v1/notes.md`

## 2026-05-21 Goal Gate Recheck

Continuation recheck after adding pre-open health:

```text
preopen status: pending until 21:00 Mon-Fri Asia/Shanghai
postclose target: pending latest_success_marker lacks schedule_slot=postclose
ordinary paper proof: pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
Vercel remote: blocked status=blocked_no_credentials auth=blocked_no_credentials project=missing_project_link deployment_url=none
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

The Vercel probe refreshed
`runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`
at `2026-05-21T08:52:06.384353+00:00`; it still has
`env_file_applied_keys=none` and no deployment URL.

No local code change can truthfully clear these remaining gates before either:

- Vercel token/project config is added for the scheduler user; or
- the next natural scheduled windows produce current pre-open/post-close
  evidence.

Evidence card:

- `runtime/production/card/goal_gate_recheck_20260521_v1/notes.md`

## 2026-05-21 Dashboard Production Status

The static dashboard bundle now includes a redacted production status artifact:

```text
dashboard_data/production_status.json
```

It is written by `scripts/deploy_vercel_dashboard.py` on every readiness/deploy
check and currently reports:

```text
status=blocked_no_credentials
auth_status=blocked_no_credentials
project_status=missing_project_link
deployment_url=null
static_export.latest=paper_trading
static_export.paper_trading_end=2026-05-20
env_file.present=true
env_file.applied_keys=[]
```

`scripts/check_dashboard_static_export.py` now treats this as a required static
artifact and prints `production_status=blocked_no_credentials` in the dashboard
static-export summary.

This moves the Vercel/dashboard surface forward, but does not clear the Vercel
remote gate; credentials and project link/env IDs are still missing.

Evidence card:

- `runtime/production/card/dashboard_production_status_v1/notes.md`

## 2026-05-21 Dashboard Production Status UI

The static dashboard now visibly renders the redacted production status export
from:

```text
dashboard_data/production_status.json
```

The first viewport shows:

```text
Production: blocked_no_credentials
Auth: blocked_no_credentials
Project: missing_project_link
Paper End: 2026-05-20
Deploy: none
```

The UI is generated by `scripts/build_backtest_dashboard.py`, committed in
`dashboard.html`, and gated by `tests/test_vercel_dashboard.py` so the static
page cannot drop the production-status loader unnoticed.

Desktop and mobile screenshots were saved under:

```text
runtime/production/card/dashboard_production_status_ui_v1/
```

This closes the local dashboard visibility gap only. Vercel remote readiness is
still blocked by missing credentials/project config, and production proof still
depends on the next natural scheduled post-close and non-dry paper windows.

Evidence card:

- `runtime/production/card/dashboard_production_status_ui_v1/notes.md`

## 2026-05-21 Scheduled Data Registry Refresh

While resuming the SOXX/model continuation loop, the actual data surfaces were
current at `2026-05-20`, but `data/datasets.yaml` still reported
`2026-05-19`. This was a scheduled-pipeline freshness gap: the paper runner
updated master/raw/qlib data but did not refresh the registry.

The scheduled data path now refreshes `data/datasets.yaml` after raw/qlib sync,
and `scripts/update_data_registry.py --check` now compares stored registry
metadata against actual files. `scripts/check_harness.py` includes that check by
default.

Current registry state after `scripts/update_market_data.py --local-only`:

```text
SOXL=2026-05-20 rows=4074
SOXX=2026-05-20 rows=4074
soxl_soxx_vix_full=2026-05-20 rows=4074
```

This improves scheduled data freshness evidence, but does not clear the Vercel
credential gate or natural scheduled paper-proof gates.

Evidence card:

- `runtime/data/card/data_registry_autorefresh_v1/notes.md`

## 2026-05-21 Data Registry Health Gate

`production/check_scheduler_health.sh` now surfaces registry freshness as its
own production check:

```text
Data Registry Health: update_data_registry: OK symbols=6 panels=10 qlib=6
Production Goal Gates: data registry: ok update_data_registry: OK symbols=6 panels=10 qlib=6
```

This closes the observability gap where paper data could be fresh while
`data/datasets.yaml` was stale. A failed or unavailable
`scripts/update_data_registry.py --check` now adds
`data_registry_freshness` to the overall pending issue list.

Live production state remains:

```text
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Evidence card:

- `runtime/production/card/data_registry_health_gate_v1/notes.md`

## 2026-05-21 Dashboard Goal Gates UI

The Vercel/static dashboard bundle now carries the scheduler production goal
gates as a separate redacted artifact:

```text
dashboard_data/production_goal_gates.json
```

`production/soxl_daily_scheduler.py --vercel-deploy` exports this artifact
after the dashboard build and before `scripts/deploy_vercel_dashboard.py
--deploy`, so the deployed static bundle can show the same blocker list as
`production/check_scheduler_health.sh`.

The dashboard production strip now renders:

```text
Production: blocked_no_credentials
Auth: blocked_no_credentials
Project: missing_project_link
Paper End: 2026-05-20
Deploy: none
Gates: pending
Blockers: external_vercel_setup, scheduled_postclose_target, scheduled_paper_proof
```

The static export checker reports `goal_gates=pending` when the artifact is
present. A live scheduler smoke run verified the order:

```text
dashboard build
production goal gates export
vercel dashboard deploy
```

This improves local/Vercel dashboard observability only. The remaining goal
blockers are unchanged: Vercel credentials/project link plus natural scheduled
post-close and ordinary paper proof.

Evidence card:

- `runtime/production/card/dashboard_goal_gates_ui_v1/notes.md`

## 2026-05-21 Post-close Slot Success Parser

`production/check_scheduler_health.sh` now scans
`scheduler_postclose.log` for the latest current-date successful
`schedule_slot=postclose` segment instead of relying only on the final
run-start marker. This prevents later manual/no-slot debug runs, including
incomplete appends, from masking an earlier scheduled post-close target proof
in the append-only log.

Current live state is still pending because the real post-close log has no
valid `schedule_slot=postclose` success segment yet:

```text
postclose status: due but current success marker lacks schedule_slot=postclose
postclose target: pending latest_success_marker lacks schedule_slot=postclose
ordinary paper proof: pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

The top-line post-close status now avoids saying `success marker found` unless
the successful segment is a scheduled `postclose` slot. A strict completion
without the slot marker remains useful diagnostic evidence, but it cannot clear
scheduled post-close proof.

Evidence card:

- `runtime/production/card/postclose_slot_success_parser_v1/notes.md`

## 2026-05-21 Scheduled Trigger Provenance Guard

`production/soxl_daily_scheduler.py` now writes non-secret trigger provenance in
the run-start marker:

```text
schedule_trigger=<manual|launchagent_preopen|launchagent_postclose|cron_preopen|cron_postclose>
```

Cron rows and macOS LaunchAgents set the scheduled trigger explicitly:

- pre-open cron: `SOXL_SCHEDULE_TRIGGER=cron_preopen`
- post-close cron: `SOXL_SCHEDULE_TRIGGER=cron_postclose`
- pre-open LaunchAgent: `SOXL_SCHEDULE_TRIGGER=launchagent_preopen`
- post-close LaunchAgent: `SOXL_SCHEDULE_TRIGGER=launchagent_postclose`

`production/check_scheduler_health.sh` now requires a scheduled post-close
trigger before accepting a post-close target proof. A manual run with
`--schedule-slot postclose` but no scheduled trigger remains pending:

```text
postclose target: pending latest_success_marker lacks schedule_trigger=postclose
post-close composer target: pending latest_success_marker lacks schedule_trigger=postclose
```

Live host state after syncing and reloading both scheduler LaunchAgents:

```text
LaunchAgent com.soxl.preopen.scheduler plist: matches repo
LaunchAgent com.soxl.daily.scheduler plist: matches repo
postclose status: due but current success marker lacks schedule_slot=postclose
postclose target: pending latest_success_marker lacks schedule_slot=postclose
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py tests/test_daily_scheduler.py \
  tests/test_production_schedule.py tests/test_vercel_dashboard.py -q
# 70 passed

plutil -lint production/com.soxl.daily.scheduler.plist production/com.soxl.preopen.scheduler.plist
# both OK

PYTHONPATH=. qlib_venv/bin/python scripts/export_production_goal_gates.py
# production_goal_gates status=pending issues=external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

This closes the manual-replay loophole only. The production goal remains pending
until the next natural scheduled post-close success and next-date scheduled paper
proof land, and Vercel still needs credentials/project setup.

Evidence card:

- `runtime/production/card/schedule_trigger_provenance_v1/notes.md`

## 2026-05-21 Pre-open Scheduled Composer Proof

The natural pre-open LaunchAgent window fired at
`2026-05-21T21:00:02+08:00` and produced a scheduled trigger-provenance marker:

```text
schedule_slot=preopen schedule_trigger=launchagent_preopen runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1 slow_profile=balanced_rule_residual_v19_reg fast_profile=balanced_rule_residual_v22 bull_guard=True
```

The run used fresh `2026-05-20` data, skipped data update as designed for
pre-open, approved the risk gate, and wrote the composer dry-run artifact:

```text
runtime/soxl/paper_trading/2026-05-20/dry_run/summary.json
```

Key dry-run readout:

- profile: `v19reg_v22_budget_usage_bull_guard_v1`
- target/current position: `0.48408568215209924` / `0.48408568215209924`
- risk gate: approved
- composer mode: `crash_rebuild`
- bull guard: enabled but not ready

Scheduler follow-through completed:

- rebuilt `dashboard.html`
- rebuilt `dashboard_data/index.json`
- exported `dashboard_data/production_goal_gates.json`
- refreshed `dashboard_data/production_status.json`
- attempted Vercel deploy and wrote refreshed blocked evidence under
  `runtime/production/card/vercel_remote_deploy_v1/`

Health after the run:

```text
preopen status: success marker found
preopen target: ok schedule_slot=preopen schedule_trigger=preopen dry_run=True skip_data_update=True runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
dashboard static: ok dashboard_static_export generated_at=2026-05-21 21:01:35 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 production_status=blocked_no_credentials goal_gates=pending paper_trading
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Boundary:

- This proves the pre-open timed inference/dashboard/Vercel-evidence refresh.
- It does not clear final production gates: Vercel still needs credentials and a
  project link, post-close still needs a scheduled `postclose` slot success, and
  ordinary paper proof still needs the next new-date scheduled non-dry run.
- For model optimization, the current evidence still favors proving the
  approved composer path on scheduled data before any new PPO branch; direct SOXX
  actor features, SOXX cap threshold tuning, and v22 continuation already have
  negative evidence.

Evidence card:

- `runtime/production/card/preopen_scheduled_composer_proof_v1/notes.md`

## 2026-05-21 Scheduled Marker Time Window Guard

The scheduler health gate now rejects scheduled proof markers whose run-start
timestamp is outside the expected schedule window, even when the marker includes
valid `schedule_slot` and `schedule_trigger` provenance.

This closes a remaining LaunchAgent replay loophole:

```text
schedule_slot=postclose schedule_trigger=launchagent_postclose
```

is not sufficient if the marker starts outside the `06:15` post-close window.
The same rule applies to the `21:00` pre-open window.

Default windows are configurable through:

- `SOXL_SCHEDULER_HEALTH_PREOPEN_WINDOW_MINUTES` (default 90)
- `SOXL_SCHEDULER_HEALTH_POSTCLOSE_WINDOW_MINUTES` (default 90)

Regression coverage:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q -k 'outside_schedule_window'
# red before fix: 2 failed
# green after fix: 2 passed, 28 deselected

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q
# 30 passed
```

Current live health remains correctly pending:

```text
preopen status: success marker found
preopen target: ok schedule_slot=preopen schedule_trigger=preopen dry_run=True skip_data_update=True runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
postclose status: due but current success marker lacks schedule_slot=postclose
postclose target: pending latest_success_marker lacks schedule_slot=postclose
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Boundary:

- This hardens proof integrity only; it does not clear production gates.
- Remaining blockers are unchanged: external Vercel setup, the next natural
  scheduled post-close success, and next-date scheduled non-dry paper proof.
- Model-side optimization should still wait for scheduled composer proof unless
  a new experiment changes checkpoint selection/scoring or another distinct
  mechanism. Direct SOXX actor features, SOXX cap threshold tuning, and v22
  continuation already have negative or duplicate evidence.

Evidence card:

- `runtime/production/card/scheduled_marker_time_window_v1/notes.md`

## 2026-05-21 Paper Trading Dashboard Metadata

The static dashboard now enriches `paper_trading` rows from non-dry per-date
paper summaries so the simulated-trading view can show which runtime profile
produced the latest proof.

Before this change, health could find:

```text
runtime/soxl/paper_trading/2026-05-20/duplicate_skip/v19reg_v22_budget_usage_bull_guard_v1/summary.json
```

but `dashboard_data/versioned/paper_trading/SOXL.json` only had curve fields and
could not identify the composer profile or `duplicate_skipped` status.

Current behavior:

- non-dry paper summaries attach `profile`, `execution_status`,
  `target_position`, `reconcile_action`, and `summary_path` to dashboard rows
- `duplicate_skip/<profile>/summary.json` outranks the top-level executed
  summary for the same date
- dry-run summaries are ignored for ordinary paper rows, so pre-open dry-run
  evidence is not confused with non-dry paper proof

Current dashboard payload evidence:

```text
2026-05-18 balanced_rule_residual_v19 executed False
2026-05-20 v19reg_v22_budget_usage_bull_guard_v1 duplicate_skipped False
```

Rendered/local browser evidence:

```text
latest=paper_trading
paper_profile=v19reg_v22_budget_usage_bull_guard_v1
paper_execution_status=duplicate_skipped
paper_target_position=0.48408568215209924
info_ids={'profile': True, 'execution': True, 'target': True, 'chart': True}
production_strip=PRODUCTION | blocked_no_credentials | AUTH | blocked_no_credentials | PROJECT | missing_project_link | PAPER END | 2026-05-20 | DEPLOY | none | GATES | pending | BLOCKERS | external_vercel_setup, scheduled_postclose_target, scheduled_paper_proof
```

Boundary:

- This fixes dashboard and paper observability only.
- It does not clear ordinary paper proof because the selected runtime profile is
  still represented by same-date `duplicate_skipped` evidence.
- The next natural scheduled post-close run must still produce next-date
  executed or skipped non-dry evidence.

Evidence card:

- `runtime/production/card/paper_trading_dashboard_metadata_v1/notes.md`

## 2026-05-22 Dashboard Paper Transparency And Vercel Gate Refresh

The local static dashboard now makes the current paper-trading provenance
visible in the production strip:

```text
paper_model=v19reg_v22_budget_usage_bull_guard_v1
paper_execution=filled
paper_source=2026-05-21 · target 77.5% · runtime/soxl/paper_trading/2026-05-21/summary.json
paper_end=2026-05-21
```

The dashboard JSON detail payload also includes paper `meta` fields, so future
sessions can inspect `dashboard_data/versioned/paper_trading/SOXL.json` without
chasing the summary path manually.

Follow-up: direct `file://` opens cannot load `dashboard_data/*.json` because
browser fetch blocks local file URLs. `dashboard.html` now falls back to the
embedded paper-trading payload for the production strip, so paper model,
execution status, source summary, and paper end remain visible even when JSON
fetches fail.

The Vercel remote gate is still blocked externally:

```text
status=blocked_no_credentials
auth=blocked_no_credentials
project=missing_project_link
deployment_url=none
overall=pending external_vercel_setup
```

This is not a static-export/data-generation failure. Local dashboard export and
paper data are current through 2026-05-21; the remote site cannot update until a
Vercel token/login and project link are available.

Scheduler nuance:

- `production/soxl_daily_scheduler.py` now emits `[scheduler] core done` before
  goal-gate export and Vercel deploy attempts.
- `production/check_scheduler_health.sh` accepts `core done` as an in-run proof
  marker for those exports, while preserving schedule-slot/window validation.

Verification snapshot:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py tests/test_scheduler_health.py \
  tests/test_dashboard.py tests/test_vercel_dashboard.py -q
# 76 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-22 14:41:54 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=5 paper_trading_end=2026-05-21 production_status=blocked_no_credentials goal_gates=pending paper_trading

production/check_scheduler_health.sh
# overall: pending external_vercel_setup
```

Rendered evidence:

- `runtime/production/card/dashboard_paper_transparency_v1/dashboard_verified.png`
- `runtime/production/card/dashboard_paper_transparency_v1/file_fallback_verified.png`
- `runtime/production/card/dashboard_paper_transparency_v1/notes.md`

## 2026-05-22 Futu Simulated Trading Src Adapter And Startup

Futu simulated-trading ownership is now centered under `src/`:

- `src/soxl_quant/app/futu.py` owns `FutuBroker`, `check_opend_port`, SDK
  connection timeout handling, and US `OpenSecTradeContext` construction.
- `production/futu_broker.py` remains only as a compatibility shim.
- Production Futu entrypoints import from `src.soxl_quant.app.futu`.

Current startup state:

```text
OpenD reachable: 127.0.0.1:11111
data.ok=true last_date=2026-05-21 expected=2026-05-21 stale_days=0
model.ok=true v19reg_v22_budget_usage_bull_guard_v1
broker.ok=false Futu OpenD SDK connection timed out
OpenD SDK log=需要图形验证码
```

The blocker is OpenD graphical verification/login, not local data freshness or
runtime model selection. No Futu order submission was attempted.

Next action: complete the Futu OpenD graphical verification/login/API
questionnaire in the UI, rerun the Futu healthcheck, and only then run a
non-dry Futu simulated-trading submission.

Evidence card:

- `runtime/production/card/futu_sim_src_startup_v1/notes.md`

## 2026-05-24 Futu Scheduled Broker Configuration

Scheduled simulated trading is now configured for Futu:

- `production/com.soxl.daily.scheduler.plist`: `--broker futu`
- `production/com.soxl.preopen.scheduler.plist`: `--broker futu`
- installed `~/Library/LaunchAgents/com.soxl.daily.scheduler.plist`: `--broker futu`
- installed `~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist`: `--broker futu`

`production/check_scheduler_health.sh` now reads the installed daily
LaunchAgent broker and runs `production/soxl_paper_trading.py --healthcheck`
with the same broker. The current live check reports:

```text
scheduled broker: futu
broker target: futu
data: ok last_date=2026-05-22 expected=2026-05-22
model: ok v19reg_v22_budget_usage_bull_guard_v1
broker: ok futu quote ok
```

A manual Futu dry-run through the scheduler completed without order placement
and wrote to:

```text
runtime/soxl/paper_trading/2026-05-22/dry_run/futu
```

IBKR remains an explicit/manual broker option only; it is not the scheduled
simulated-trading broker. The remaining proof gap is the next natural
post-close LaunchAgent run producing fresh scheduled non-dry Futu evidence.

Evidence card:

- `runtime/production/card/futu_scheduled_broker_v1/notes.md`

## 2026-05-24 Futu Simulated Account Order Probe

Direct Futu simulated-trading probe:

- OpenD was reachable at `127.0.0.1:11111`.
- `OpenSecTradeContext(filter_trdmarket=TrdMarket.US)` connected through the
  local OpenD instance.
- `accinfo_query(trd_env=SIMULATE)` read the simulated account:
  `cash=1000000.0`, `total_assets=1000000.0`, `currency=N/A`.
- `position_list_query(code=US.SOXL, trd_env=SIMULATE)` was empty before the
  test order.
- Submitted a simulated `US.SOXL` `BUY 1` normal limit order at `181.03` when
  the snapshot last price was `190.56`.
- Futu returned `place_order ret=OK`, then `modify_order(CANCEL) ret=OK`.
- Final order status was `CANCELLED_ALL`, `dealt_qty=0`, `dealt_avg_price=0`.
- Position after the probe was still empty.
- A follow-up active-order query returned `active_order_count=0`.

Decision:

- Futu simulated account access and order submit/cancel are now proven.
- The probe intentionally did not leave an active order or position.
- IBKR is unrelated to this path; scheduled simulated trading remains Futu.

Evidence card:

- `runtime/production/card/futu_sim_order_probe_v1/notes.md`

## 2026-05-24 Futu 10000 Simulation Reset

The scheduled Futu simulated-trading path now starts from a repo-side notional
account value of `10000`:

- `production/soxl_paper_trading.py` accepts `--account-notional`.
- `production/soxl_daily_scheduler.py` forwards `--account-notional`.
- `production/com.soxl.daily.scheduler.plist`,
  `production/com.soxl.preopen.scheduler.plist`, and `production/crontab.txt`
  include `--broker futu --account-notional 10000`.
- Installed `~/Library/LaunchAgents/com.soxl.daily.scheduler.plist` and
  `~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist` were synced and
  reloaded; both report `account_notional=10000 broker=futu`.
- `src/soxl_quant/app/futu.py` uses the notional snapshot for share sizing and
  dashboard/account reporting when configured.
- Explicit Futu notional simulation uses the existing cold-start step clip:
  model target `0.8500000003` from zero was capped to `0.35`, with
  `risk_gate.approved=true` in dry-run.
- Old generated state was archived under
  `runtime/soxl/paper_trading/archive_reset_20260524_10000`.
- Active dashboard export now has `paper_trading_rows=1`,
  `paper_trading_end=2026-05-22`, and the row starts with
  `strategy_nav=10000.0`, `buyhold_nav=10000.0`, `position=0.0`.

Decision:

- Futu remains the scheduled simulated broker.
- IBKR remains unrelated to this path.
- The Futu backend cash balance is no longer the strategy sizing base when
  `--account-notional 10000` is present.

Evidence card:

- `runtime/production/card/futu_10000_reset_v1/notes.md`

## 2026-05-24 Dashboard Paper Signal Metadata

The `paper_trading` static export now exposes latest signal metadata separately
from ordinary paper CSV row metadata:

- current profile: `v19reg_v22_budget_usage_bull_guard_v1`
- slow actor: `balanced_rule_residual_v19_reg`
- fast actor: `balanced_rule_residual_v22`
- broker: `futu`
- dry-run summary:
  `runtime/soxl/paper_trading/2026-05-22/dry_run/futu/summary.json`
- account value: `10000.0`
- target position: `0.8500000003`
- risk-gate capped target: `0.35`

Decision:

- Dry-run summaries remain excluded from ordinary paper row enrichment.
- `dashboard_data/versioned/paper_trading/SOXL.json` carries
  `meta.latest_signal_*` for the production strip and Vercel static view.
- External Vercel deployment can still be blocked by missing credentials/project
  linkage, but the repo static export now contains the inference metadata.

Evidence card:

- `runtime/production/card/dashboard_paper_signal_meta_v1/notes.md`

## 2026-05-24 Broker Intent Clarification

Current routing intent:

- Futu remains the scheduled simulated-trading broker.
- IBKR is the intended live broker, but must remain an explicit production path
  until local TWS / IB Gateway readiness and socket connectivity pass
  `production/check_ibkr_gateway.py`.
- Do not silently switch live execution to Futu. Treat Futu paper proof and IBKR
  live readiness as separate production gates.
