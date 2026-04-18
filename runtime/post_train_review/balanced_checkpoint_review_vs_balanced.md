# Post-Train Backtest Review: soxl_balanced_checkpoint_review_summary vs soxl_balanced_summary

## Summary
- Test proxy return: 468.90% -> 371.50% (-97.40) [worse]
- Test proxy maxDD: 50.18% -> 60.26% (+10.08) [worse]
- Validation score: 61.42 -> 80.60 (+19.18)
- Promotion eligible: True -> False

## Window Review
- test_2020: mixed
  return 96.96% -> 114.65% | sharpe 2.16 -> 2.27 | maxDD 21.12% -> 21.29% | avg_pos 0.518 -> 0.560
  evidence: return improved (+17.69); sharpe improved (+0.10); max_dd worsened (+0.17)
- test_2022: bad
  return -25.82% -> -39.54% | sharpe -0.88 -> -1.09 | maxDD 33.04% -> 46.30% | avg_pos 0.283 -> 0.372
  evidence: return worsened (-13.72); sharpe worsened (-0.22); max_dd worsened (+13.26)
- test_2023_2024: bad
  return 43.52% -> 39.59% | sharpe 0.69 -> 0.64 | maxDD 33.87% -> 38.65% | avg_pos 0.498 -> 0.535
  evidence: return worsened (-3.92); sharpe worsened (-0.05); max_dd worsened (+4.78)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。
- 最新 run 的最佳 checkpoint 出现在 iter 40，之后分数回落到 iter 80，应优先考虑 early stopping 或更稳健的 checkpoint 选择。

## Gate Checks
- test_2022_avg_pos: True (0.28 / limit 0.35) -> False (0.37 / limit 0.35)
- test_2022_max_dd: True (33.04 / limit 35.00) -> False (46.30 / limit 35.00)
- test_proxy_sharpe: True (0.86 / limit 0.75) -> True (0.76 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。
- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。
- validation 变强但 OOS 变弱，优先复核 checkpoint 选择和 early stopping，而不是继续加特征。

## Inputs
- latest: `artifacts/models/soxl_balanced_checkpoint_review_summary.json`
- previous: `artifacts/models/soxl_balanced_summary.json`
