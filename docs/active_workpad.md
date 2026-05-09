---
date: '2026-05-09'
title: 'Data Surface Sync + Bottleneck Reset'
kind: workpad
status: active
direction: data
---

# Data Surface Sync + Bottleneck Reset

## Goal

1. Refresh the repo-owned local data surfaces from the canonical SOXL/SOXX/VIX panel.
2. Re-ground the next optimization loop on the current frontier instead of the stale macro branch.

## Scope

This pass is a `data` / `harness` / `execution` diagnosis loop, not a new training run.

## Baselines And Role

- Canonical baseline: `balanced_rule_residual_v19`
- Direct comparison baseline: `risk_budget_usage_v1`
- Role: `diagnose`
- Target window: local canonical panel through `2026-04-27`
- Accepted trade-off: no live internet refresh inside the current sandboxed automation environment
- Changed layer: `data`
- External reference: Yahoo Finance transport path (`yfinance` vs direct chart API) as the current update-source boundary
- Comparison-table row key: `data_surface_sync_2026_05_09`

## Closest Historical Cases

1. Closest success case: dashboard freshness fix that extended `TEST_SEG` and re-pointed consumers to `data/processed/panels/soxl_soxx_vix_full.csv`
2. Closest failure case: the stale raw-data path that kept `data/raw/symbols/SOXL.csv` at `2026-04-01` while the canonical panel already reached `2026-04-27`
3. Why this loop is not a duplicate:
   It does not introduce a new feature/model mechanism. It closes the harness gap between canonical processed data and downstream raw/qlib consumers, and it documents the real blocker for live refresh in this automation environment.

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

## Bottleneck Reset

Current project bottleneck is **not**:

- another macro feature injection
- another looser mode envelope
- another more-conservative risk-budget ablation
- another “maybe training not enough” guess

Current project bottleneck **is**:

- **re-entry / recovery timing after de-risking**

Evidence trail:

- `risk_budget_usage_v1` has excellent 2022 defense but still recovers too slowly versus SOXL buy-and-hold
- `balanced_rule_residual_v19` / `v22` remain defensively acceptable but also lag 2023-2024 upside
- `balanced_macro_v1` improved recovery but worsened bear-market exposure, so it does not solve the real trade-off
- `runtime/execution/knowledge.md` already has `显式 release/re-entry overlay` as the next valid execution line

## Next Action

1. Keep local data sync on the canonical panel path; do not treat stale `raw/` dates as the primary blocker anymore.
2. Open the next narrow optimization loop around **release/re-entry overlay / faster recovery timing**, using `balanced_rule_residual_v19` and `risk_budget_usage_v1` as comparison anchors.
3. Avoid more macro-input or more-conservative budget branches unless they introduce a genuinely new recovery mechanism with external justification.

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
