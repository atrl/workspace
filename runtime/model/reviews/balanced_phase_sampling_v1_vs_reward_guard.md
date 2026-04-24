# Post-Train Backtest Review: soxl_balanced_phase_sampling_v1_summary vs soxl_balanced_reward_guard_summary

## Summary
- Test proxy return: 523.36% -> 306.89% (-216.47) [worse]
- Test proxy maxDD: 62.27% -> 63.94% (+1.67) [worse]
- Validation score: 83.32 -> 80.84 (-2.48)
- Promotion eligible: False -> False

## Window Review
- test_2020: bad
  return 137.91% -> 121.83% | sharpe 2.31 -> 2.19 | maxDD 21.10% -> 22.21% | avg_pos 0.609 -> 0.619
  evidence: return worsened (-16.08); sharpe worsened (-0.12); max_dd worsened (+1.11)
- test_2022: bad
  return -36.00% -> -40.67% | sharpe -0.72 -> -1.22 | maxDD 47.36% -> 47.50% | avg_pos 0.412 -> 0.364
  evidence: return worsened (-4.67); sharpe worsened (-0.50); max_dd worsened (+0.14)
- test_2023_2024: bad
  return 50.23% -> 41.67% | sharpe 0.71 -> 0.64 | maxDD 39.85% -> 41.66% | avg_pos 0.572 -> 0.572
  evidence: return worsened (-8.56); sharpe worsened (-0.07); max_dd worsened (+1.81)

## Mechanism Hypotheses
- 最新 run 的最佳 checkpoint 出现在 iter 20，之后分数回落到 iter 80，应优先考虑 early stopping 或更稳健的 checkpoint 选择。

## Gate Checks
- test_2022_avg_pos: False (0.41 / limit 0.35) -> False (0.36 / limit 0.35)
- test_2022_max_dd: False (47.36 / limit 35.00) -> False (47.50 / limit 35.00)
- test_proxy_sharpe: True (0.83 / limit 0.75) -> False (0.71 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。

## Inputs
- latest: `artifacts/models/soxl_balanced_phase_sampling_v1_summary.json`
- previous: `artifacts/models/soxl_balanced_reward_guard_summary.json`
