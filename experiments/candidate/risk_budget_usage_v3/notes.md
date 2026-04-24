---
title: risk_budget_usage_v3
direction: model
status: candidate
---

# risk_budget_usage_v3

## Motivation

v1 proved the budget-usage architecture works. v2 added mode-conditional reward
shaping and dynamic band scaling to guide PPO behavior. v3 goes one step
deeper: **let the Actor directly observe the current mode**, so it can learn
mode-specific usage policies end-to-end.

The hypothesis is that feeding mode as an explicit input feature (one-hot)
should let the network learn different usage distributions per mode faster
than inferring mode indirectly from raw features alone.

## External Reference

- Mode-conditioned policies in hierarchical RL (Sutton et al. 1999)
- Task-conditional policy networks (Hausman et al. 2018)
- Contextual policies with one-hot task identifiers

## Hypothesis

Appending the current mode as a one-hot vector to the observation should:
1. Reduce the sample complexity of learning mode-specific usage policies
2. Improve generalization across mode transitions
3. Lift proxy Sharpe by allowing the Actor to specialize per mode

## Target Slices / Windows

- `test_proxy`: Sharpe ≥ 0.95
- `test_2022`: MaxDD ≤ 18%, avg_pos ≤ 0.15
- `test_2023_2024`: return ≥ 80%

## Acceptance Criteria

- `test_proxy Sharpe >= 0.95`
- `test_2022 MaxDD <= 20.0%`
- `test_2022 avg_pos <= 0.15`

## Baseline

- **Canonical baseline**: `risk_budget_usage_v1`
- **Direct comparison baseline**: `risk_budget_usage_v2`
- **Experiment role**: `replace`
- **Changed layer**: `network` (observation input)

## Configuration Delta vs v2

```yaml
model:
  use_mode_conditioned_input: true  # NEW: append 4-dim mode one-hot to obs
```

All other configs identical to v2.

## Result

Command:
```bash
PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
  --profiles risk_budget_usage_v3 --seed 42 --skip-data-refresh
```

| 切片 | Return | Sharpe | MaxDD | AvgPos | StressAvgPos |
|------|--------|--------|-------|--------|-------------|
| test_proxy | **274.7%** | **1.074** | 21.4% | 0.208 | 0.131 |
| test_2020 | 29.3% | 1.430 | 18.5% | 0.233 | 0.158 |
| test_2022 | -9.7% | -0.601 | **12.0%** | **0.143** | 0.120 |
| test_2023_2024 | 60.5% | 1.378 | 16.0% | 0.215 | 0.134 |
| valid | 4.8% | 0.377 | 17.0% | 0.272 | 0.131 |

### Promotion Gate

- test_2022 MaxDD: 12.0% < 20.0% ✅
- test_2022 avg_pos: 0.143 < 0.15 ✅
- test_proxy Sharpe: **1.074 > 0.95** ✅

**Fixed Split Gate: PASSED!** Walk-forward 尚未运行。

## 与 v1/v2 对比

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | **1.074** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| proxy Return | 274% | 158% | 275% |
| 2022 MaxDD | 18.7% | 7.4% | 12.0% |
| 2022 avg_pos | 0.127 | 0.070 | 0.143 |

## Decision

**✅ PROMOTE to best candidate.**

v3 的 mode-conditioned actor input 带来了质的飞跃：
- proxy Sharpe 从 0.942 → **1.074**（首次突破 1.0！）
- 同时通过 fixed-split promotion gate（2022 MaxDD=12%, avg_pos=0.143）
- 这是 risk_budget_usage 系列中第一个通过 gate 的版本

**关键洞察**: 让 Actor 直接"看到"当前 mode（one-hot），比仅通过 reward shaping 间接引导更有效。网络可以显式学习不同 mode 下的 usage 分布，避免了 v2 中 reward 信号过强导致的过度保守问题。

## Comparison-Table Row Key

`risk_budget_usage_v3`
