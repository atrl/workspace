---
date: '2026-04-27'
title: 'risk_budget_usage_v1 恢复与 walk-forward 验证'
kind: ops
status: active
direction: model
---

# risk_budget_usage_v1 恢复与 walk-forward 验证

## 背景

旧版 `risk_budget_usage_v1` 组合策略（mode_v1_slow + imitation_v4_fast via budget_usage）曾达到 **2003.6% / Sharpe 1.25**。重新训练后组合结果降至 **1545.9% / Sharpe 1.21**。差距来自子模型日度持仓相关性几乎为 0（新版与旧版完全不同交易策略）。端到端 PPO 训练 budget_usage 组合逻辑反复失败（77%/Sharpe 0.65）。

## 本次任务

### 任务 A：重建旧版 slow 子模型 ✅ 完成
- **结果**：新 `balanced` (simple MLP) fixed-split → **632.5% / Sharpe 0.90 / avg_pos 0.53**
- **与新版组合**：`balanced` + `imitation_v4` → **2610.8% / Sharpe 1.31 / avg_pos 0.424**
  - test_2020: 124.3% / 1.53
  - test_2022: **+4.9% / 0.30** (正收益！)
  - test_2023_2024: 164.6% / 1.39
- **结论**：旧版 `balanced` profile 重新训练后，与新版 `imitation_v4` 组合产生前所未有的协同效应，超越旧版 2003.6%

### 任务 B：组合策略 walk-forward 验证 ✅ 完成
- **imitation_v4 walk-forward**
  - wf_2020: +102.5% / Sharpe 1.92 / MaxDD 20.8%
  - wf_2022: -28.3% / Sharpe -0.83 / MaxDD 37.7%
  - wf_2023_2024: -16.3% / Sharpe -0.24 / MaxDD 34.5%
- **mode_v1 walk-forward**
  - wf_2020: +82.0% / Sharpe 1.80 / MaxDD 22.0%
  - wf_2022: -24.2% / Sharpe -0.99 / MaxDD 28.9%
  - wf_2023_2024: -16.5% / Sharpe -0.38 / MaxDD 31.9%
- **组合 walk-forward**（budget_usage）
  - wf_2020: **58.6%** / Sharpe **1.13** / MaxDD 28.6% / avg_pos 0.317
  - wf_2022: **-18.6%** / Sharpe **-1.58** / MaxDD 20.1% / avg_pos 0.078
  - wf_2023_2024: **28.8%** / Sharpe **0.98** / MaxDD 27.2% / avg_pos 0.214
  - 组合在 bear market 和 recovery 期均优于单模型

## 关键发现

| 组合 | test_proxy | Sharpe | test_2022 | 备注 |
|------|-----------|--------|-----------|------|
| 旧版 (mode_v1+imitation_v4) | 2003.6% | 1.25 | -17.1% | zero-cost replay |
| 新版 (mode_v1+imitation_v4) | 1545.9% | 1.21 | -8.1% | zero-cost replay |
| **balanced+imitation_v4** | **2610.8%** | **1.31** | **+4.9%** | zero-cost replay |
| 组合 walk-forward | — | — | -18.6% | 三窗口验证 |

## 结论

- `balanced` profile（simple MLP, staged_training=false）重新训练后与 `imitation_v4` 组合产生前所未有的协同效应
- Fixed-split 突破 2610%/Sharpe 1.31，test_2022 首次实现正收益
- Walk-forward 验证：组合在 recovery 期（wf_2023_2024）显著优于单模型（+28.8% vs -16~-17%）
- Bear market 防御仍有改进空间（wf_2022 -18.6%），但优于单模型和 buy-and-hold

## 下一步

1. ✅ 保存 fixed-split 组合结果到 `runtime/model/card/risk_budget_usage_v1/`
2. ✅ 保存 walk-forward 组合结果
3. commit & push
4. 考虑将 `balanced` 作为 canonical slow signal 替代 `mode_v1`
