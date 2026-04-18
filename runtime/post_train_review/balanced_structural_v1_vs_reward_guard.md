# Post-Train Backtest Review: soxl_balanced_structural_v1_summary vs soxl_balanced_reward_guard_summary

## Summary
- Test proxy return: 523.36% -> 306.97% (-216.39) [worse]
- Test proxy maxDD: 62.27% -> 53.66% (-8.61) [better]
- Validation score: 83.32 -> 75.67 (-7.65)
- Promotion eligible: False -> False

## Window Review
- test_2020: mixed
  return 137.91% -> 101.26% | sharpe 2.31 -> 2.35 | maxDD 21.10% -> 19.98% | avg_pos 0.609 -> 0.472
  evidence: return worsened (-36.65); sharpe improved (+0.03); max_dd improved (-1.12)
- test_2022: mixed
  return -36.00% -> -31.27% | sharpe -0.72 -> -0.90 | maxDD 47.36% -> 39.86% | avg_pos 0.412 -> 0.338
  evidence: return improved (+4.73); sharpe worsened (-0.18); max_dd improved (-7.50)
- test_2023_2024: mixed
  return 50.23% -> 45.53% | sharpe 0.71 -> 0.71 | maxDD 39.85% -> 33.47% | avg_pos 0.572 -> 0.477
  evidence: return worsened (-4.70); sharpe worsened (-0.00); max_dd improved (-6.38)

## Mechanism Hypotheses
- `2022` 改善伴随平均仓位下降，说明风险控制更积极，防守侧机制更有效。
- `2023-2024` 收益下滑且平均仓位下降，说明新模型可能在顺风阶段过度保守。
- 总体回报下降但回撤收敛，说明模型风格向保守漂移，需判断是否值得。

## Gate Checks
- test_2022_avg_pos: False (0.41 / limit 0.35) -> True (0.34 / limit 0.35)
- test_2022_max_dd: False (47.36 / limit 35.00) -> False (39.86 / limit 35.00)
- test_proxy_sharpe: True (0.83 / limit 0.75) -> False (0.74 / limit 0.75)

## Recommended Next Actions
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。

## Inputs
- latest: `artifacts/models/soxl_balanced_structural_v1_summary.json`
- previous: `artifacts/models/soxl_balanced_reward_guard_summary.json`
