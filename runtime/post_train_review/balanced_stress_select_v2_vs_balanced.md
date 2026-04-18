# Post-Train Backtest Review: soxl_balanced_stress_select_v2_summary vs soxl_balanced_summary

## Summary
- Test proxy return: 468.90% -> 462.28% (-6.61) [worse]
- Test proxy maxDD: 50.18% -> 63.28% (+13.10) [worse]
- Validation score: 61.42 -> 71.11 (+9.70)
- Promotion eligible: True -> False

## Window Review
- test_2020: mixed
  return 96.96% -> 128.18% | sharpe 2.16 -> 2.37 | maxDD 21.12% -> 22.27% | avg_pos 0.518 -> 0.568
  evidence: return improved (+31.22); sharpe improved (+0.21); max_dd worsened (+1.15)
- test_2022: mixed
  return -25.82% -> -34.68% | sharpe -0.88 -> -0.77 | maxDD 33.04% -> 46.28% | avg_pos 0.283 -> 0.398
  evidence: return worsened (-8.86); sharpe improved (+0.11); max_dd worsened (+13.25)
- test_2023_2024: mixed
  return 43.52% -> 52.02% | sharpe 0.69 -> 0.73 | maxDD 33.87% -> 37.21% | avg_pos 0.498 -> 0.548
  evidence: return improved (+8.50); sharpe improved (+0.05); max_dd worsened (+3.33)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。
- `2023-2024` 收益改善伴随更高持仓，说明新模型更愿意保留趋势暴露。

## Gate Checks
- test_2022_avg_pos: True (0.28 / limit 0.35) -> False (0.40 / limit 0.35)
- test_2022_max_dd: True (33.04 / limit 35.00) -> False (46.28 / limit 35.00)
- test_proxy_sharpe: True (0.86 / limit 0.75) -> True (0.82 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。
- validation 变强但 OOS 变弱，优先复核 checkpoint 选择和 early stopping，而不是继续加特征。

## Inputs
- latest: `artifacts/models/soxl_balanced_stress_select_v2_summary.json`
- previous: `artifacts/models/soxl_balanced_summary.json`
