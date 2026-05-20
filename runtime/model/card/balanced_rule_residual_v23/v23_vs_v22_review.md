# Post-Train Backtest Review: balanced_rule_residual_v23 vs balanced_rule_residual_v22

## Summary
- Test proxy return: 695.45% -> 493.23% (-202.23) [worse]
- Test proxy maxDD: 54.52% -> 51.20% (-3.32) [better]
- Validation score: 91.94 -> 81.59 (-10.35)
- Promotion eligible: True -> False

## Window Review
- test_2020: mixed
  return 115.60% -> 91.61% | sharpe 2.74 -> 2.48 | maxDD 16.22% -> 16.22% | avg_pos 0.415 -> 0.376
  evidence: return worsened (-24.00); sharpe worsened (-0.26); max_dd improved (-0.00)
- test_2022: good
  return -24.52% -> -21.88% | sharpe -0.91 -> -0.86 | maxDD 32.71% -> 30.23% | avg_pos 0.269 -> 0.250
  evidence: return improved (+2.64); sharpe improved (+0.05); max_dd improved (-2.48)
- test_2023_2024: mixed
  return 37.85% -> 34.25% | sharpe 0.66 -> 0.64 | maxDD 32.43% -> 29.39% | avg_pos 0.447 -> 0.421
  evidence: return worsened (-3.60); sharpe worsened (-0.02); max_dd improved (-3.03)

## Mechanism Hypotheses
- `2022` 改善伴随平均仓位下降，说明风险控制更积极，防守侧机制更有效。
- `2023-2024` 收益下滑且平均仓位下降，说明新模型可能在顺风阶段过度保守。
- 总体回报下降但回撤收敛，说明模型风格向保守漂移，需判断是否值得。
- 最新 run 的最佳 checkpoint 出现在 iter 0，之后分数回落到 iter 80，应优先考虑 early stopping 或更稳健的 checkpoint 选择。

## Gate Checks
- test_2022_avg_pos: True (0.27 / limit 0.35) -> True (0.25 / limit 0.35)
- test_2022_max_dd: True (32.71 / limit 55.00) -> True (30.23 / limit 35.00)
- test_proxy_sharpe: True (0.94 / limit 0.50) -> True (0.89 / limit 0.75)

## Recommended Next Actions
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。

## Inputs
- latest: `runtime/model/card/balanced_rule_residual_v23/soxl_balanced_rule_residual_v23_summary.json`
- previous: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json`
