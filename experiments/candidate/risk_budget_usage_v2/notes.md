---
title: risk_budget_usage_v2
direction: model
status: candidate
---

# risk_budget_usage_v2

## Motivation

`risk_budget_usage_v1` proved that the budget-usage architecture can pass the
test_2022 promotion gate (MaxDD=18.7%, avg_pos=0.127), but its proxy Sharpe
(0.942) leaves room for improvement. The v1 PPO policy discovered a conservative
budget-usage strategy rather than the replay-scale performance.

This experiment introduces two complementary extensions:

1. **Mode-Conditional Reward Shaping**: encode the semantic meaning of each mode
directly into the reward function, so the actor learns faster what behavior is
expected in each risk regime.
2. **Dynamic Band Scaling**: compress/expand mode bands based on realized
volatility, so the risk budget adapts to market conditions in real time.

## External Reference

- Hierarchical RL with temporally abstract options (Sutton, Precup, Singh 1999)
- Volatility-targeting position sizing (CBOE, Risk Parity literature)
- Mode-conditional policy gradients (Mode-Connected RL, arXiv)

## Hypothesis

Reward shaping that reinforces mode semantics (flat_wait = cash, crash_rebuild
= bottom-fishing, trend_hold = trend alignment) should guide PPO to discover
less conservative policies. Dynamic band scaling should further improve risk
adjustment by tightening bands in high-vol regimes and loosening them in calm
periods.

Combined, these two mechanisms should lift proxy Sharpe above 0.95 without
degrading the 2022 stress performance.

## Target Slices / Windows

- `test_2022`: maintain or improve MaxDD ≤ 18.7%, avg_pos ≤ 0.15
- `test_proxy`: Sharpe ≥ 0.95 (up from 0.942)
- `test_2023_2024`: return ≥ 80% (do not degrade restart participation)
- `test_2020`: return ≥ 38% (do not degrade trend participation)

## Acceptance Criteria

- `test_2022 MaxDD <= 20.0%`
- `test_2022 avg_pos <= 0.15`
- `test_proxy Sharpe >= 0.95`
- `test_2023_2024 return >= 80.0%`
- `test_2020 return >= 38.0%`

## Baseline

- **Canonical baseline**: `risk_budget_usage_v1`
- **Direct comparison baseline**: `risk_budget_usage_v1`
- **Experiment role**: `replace` (if successful, becomes new best candidate)
- **Changed layer**: `reward` + `execution` (mode router)

## Configuration Delta vs v1

### Mode-Conditional Reward

```yaml
reward:
  mode_conditional_reward_enabled: true
  flat_wait_pos_penalty: 0.15        # penalize ANY position in flat_wait
  crash_rebuild_bonus: 0.20          # bonus for increasing position in crash_rebuild
  trend_hold_alignment_multiplier: 1.5  # amplify trend alignment in trend_hold
  defensive_hold_neutral_bonus: 0.05    # small stability bonus
```

### Dynamic Band Scaling

```yaml
mode_router:
  use_dynamic_bands: true
  dynamic_band_vol_baseline: 0.50
  dynamic_band_vol_sensitivity: 0.30
  dynamic_band_min_scale: 0.60
  dynamic_band_max_scale: 1.20
```

## Result

Command:
```bash
PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
  --profiles risk_budget_usage_v2 --seed 42 --skip-data-refresh
```

| 切片 | Return | Sharpe | MaxDD | AvgPos | StressAvgPos |
|------|--------|--------|-------|--------|-------------|
| test_proxy | 158.3% | **0.884** | 20.6% | 0.180 | 0.070 |
| test_2020 | 27.7% | 1.515 | 16.7% | 0.210 | 0.097 |
| test_2022 | -3.2% | -0.272 | **7.4%** | **0.070** | 0.035 |
| test_2023_2024 | 40.1% | 1.227 | 15.4% | 0.179 | 0.098 |
| valid | 12.7% | 0.795 | 17.1% | 0.255 | 0.095 |

### Promotion Gate

- test_2022 MaxDD: 7.4% < 20.0% ✅
- test_2022 avg_pos: 0.070 < 0.15 ✅
- test_proxy Sharpe: 0.884 < 0.95 ❌ (未达标)

## Decision

**Keep as candidate, parameter tuning needed.**

v2 的 mode-conditional reward + dynamic band scaling 确实让策略更保守：
- ✅ 2022 MaxDD 从 18.7% → 7.4%（显著改善）
- ✅ 2022 avg_pos 从 0.127 → 0.070（更保守）
- ❌ proxy Sharpe 从 0.942 → 0.884（未达 0.95 目标）
- ❌ 2020/2023-2024 回报下降（过度保守）

**诊断**: `flat_wait_pos_penalty=0.15` 可能太强，导致 Actor 在 flat_wait 模式下过度空仓，错失重启参与机会。下一步应降低该惩罚或提高 `crash_rebuild_bonus`。

## Comparison-Table Row Key

`risk_budget_usage_v2`
