# Post-Train Backtest Review: soxl_balanced_phase_sampling_v1_summary vs soxl_balanced_summary

## Summary
- Test proxy return: 468.90% -> 306.89% (-162.01) [worse]
- Test proxy maxDD: 50.18% -> 63.94% (+13.76) [worse]
- Validation score: 61.42 -> 80.84 (+19.42)
- Promotion eligible: True -> False

## Window Review
- test_2020: mixed
  return 96.96% -> 121.83% | sharpe 2.16 -> 2.19 | maxDD 21.12% -> 22.21% | avg_pos 0.518 -> 0.619
  evidence: return improved (+24.87); sharpe improved (+0.03); max_dd worsened (+1.09)
- test_2022: bad
  return -25.82% -> -40.67% | sharpe -0.88 -> -1.22 | maxDD 33.04% -> 47.50% | avg_pos 0.283 -> 0.364
  evidence: return worsened (-14.85); sharpe worsened (-0.34); max_dd worsened (+14.47)
- test_2023_2024: bad
  return 43.52% -> 41.67% | sharpe 0.69 -> 0.64 | maxDD 33.87% -> 41.66% | avg_pos 0.498 -> 0.572
  evidence: return worsened (-1.85); sharpe worsened (-0.04); max_dd worsened (+7.79)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。
- 最新 run 的最佳 checkpoint 出现在 iter 20，之后分数回落到 iter 80，应优先考虑 early stopping 或更稳健的 checkpoint 选择。

## Gate Checks
- test_2022_avg_pos: True (0.28 / limit 0.35) -> False (0.36 / limit 0.35)
- test_2022_max_dd: True (33.04 / limit 35.00) -> False (47.50 / limit 35.00)
- test_proxy_sharpe: True (0.86 / limit 0.75) -> False (0.71 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。
- validation 变强但 OOS 变弱，优先复核 checkpoint 选择和 early stopping，而不是继续加特征。

## Inputs
- latest: `artifacts/models/soxl_balanced_phase_sampling_v1_summary.json`
- previous: `artifacts/models/soxl_balanced_summary.json`
