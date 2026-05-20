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

## Open Production Tasks

- Visual browser screenshot verification is still pending because macOS Computer Use permissions were not granted during this pass; file/HTTP checks passed.
- After committing the production loop, resume SOXX-aware model optimization from the existing frontier.
