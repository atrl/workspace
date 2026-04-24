# Post-Train Backtest Review: soxl_balanced_stress_select_v2_summary vs soxl_balanced_stress_select_summary

## Summary
- Test proxy return: 424.33% -> 462.28% (+37.95) [better]
- Test proxy maxDD: 63.09% -> 63.28% (+0.19) [worse]
- Validation score: 81.65 -> 71.11 (-10.54)
- Promotion eligible: False -> False

## Window Review
- test_2020: bad
  return 150.32% -> 128.18% | sharpe 2.43 -> 2.37 | maxDD 19.71% -> 22.27% | avg_pos 0.600 -> 0.568
  evidence: return worsened (-22.14); sharpe worsened (-0.07); max_dd worsened (+2.56)
- test_2022: good
  return -39.36% -> -34.68% | sharpe -0.95 -> -0.77 | maxDD 47.67% -> 46.28% | avg_pos 0.404 -> 0.398
  evidence: return improved (+4.68); sharpe improved (+0.18); max_dd improved (-1.39)
- test_2023_2024: mixed
  return 53.14% -> 52.02% | sharpe 0.74 -> 0.73 | maxDD 39.92% -> 37.21% | avg_pos 0.550 -> 0.548
  evidence: return worsened (-1.13); sharpe worsened (-0.00); max_dd improved (-2.71)

## Mechanism Hypotheses
- `2022` 改善伴随平均仓位下降，说明风险控制更积极，防守侧机制更有效。
- `2023-2024` 收益下滑且平均仓位下降，说明新模型可能在顺风阶段过度保守。
- 总体回报上升但总回撤加深，属于典型的收益换风险型改进，需要确认是否越过 promotion 风险边界。

## Gate Checks
- test_2022_avg_pos: False (0.40 / limit 0.35) -> False (0.40 / limit 0.35)
- test_2022_max_dd: False (47.67 / limit 35.00) -> False (46.28 / limit 35.00)
- test_proxy_sharpe: True (0.79 / limit 0.75) -> True (0.82 / limit 0.75)

## Recommended Next Actions
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。

## Inputs
- latest: `artifacts/models/soxl_balanced_stress_select_v2_summary.json`
- previous: `artifacts/models/soxl_balanced_stress_select_summary.json`
