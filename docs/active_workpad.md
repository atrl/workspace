---
date: '2026-04-28'
title: 'Macro Feature Pack Evaluation — balanced_macro_v1 vs balanced (same seed)'
kind: experiment
status: active
direction: model
---

# Macro Feature Pack Evaluation

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
