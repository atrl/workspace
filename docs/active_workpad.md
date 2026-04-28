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

## 实验档案

- `balanced_macro_v1` profile: `runtime/model/card/balanced_macro_v1/profile.yaml`
- `balanced` baseline (seed 42): `runtime/model/card/balanced/soxl_balanced_summary.json`
- `balanced_macro_v1` summary: `runtime/model/card/balanced_macro_v1/soxl_balanced_macro_v1_summary.json`
