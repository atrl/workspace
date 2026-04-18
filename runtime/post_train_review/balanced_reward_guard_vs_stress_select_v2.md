# Post-Train Backtest Review: soxl_balanced_reward_guard_summary vs soxl_balanced_stress_select_v2_summary

## Summary
- Test proxy return: 462.28% -> 523.36% (+61.07) [better]
- Test proxy maxDD: 63.28% -> 62.27% (-1.01) [better]
- Validation score: 71.11 -> 83.32 (+12.21)
- Promotion eligible: False -> False

## Window Review
- test_2020: mixed
  return 128.18% -> 137.91% | sharpe 2.37 -> 2.31 | maxDD 22.27% -> 21.10% | avg_pos 0.568 -> 0.609
  evidence: return improved (+9.73); sharpe worsened (-0.05); max_dd improved (-1.17)
- test_2022: mixed
  return -34.68% -> -36.00% | sharpe -0.77 -> -0.72 | maxDD 46.28% -> 47.36% | avg_pos 0.398 -> 0.412
  evidence: return worsened (-1.32); sharpe improved (+0.05); max_dd worsened (+1.08)
- test_2023_2024: bad
  return 52.02% -> 50.23% | sharpe 0.73 -> 0.71 | maxDD 37.21% -> 39.85% | avg_pos 0.548 -> 0.572
  evidence: return worsened (-1.78); sharpe worsened (-0.02); max_dd worsened (+2.64)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。

## Gate Checks
- test_2022_avg_pos: False (0.40 / limit 0.35) -> False (0.41 / limit 0.35)
- test_2022_max_dd: False (46.28 / limit 35.00) -> False (47.36 / limit 35.00)
- test_proxy_sharpe: True (0.82 / limit 0.75) -> True (0.83 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。

## Inputs
- latest: `artifacts/models/soxl_balanced_reward_guard_summary.json`
- previous: `artifacts/models/soxl_balanced_stress_select_v2_summary.json`
