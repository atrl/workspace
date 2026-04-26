# Post-Train Backtest Review: soxl_balanced_structural_v1_summary vs soxl_balanced_summary

## Summary
- Test proxy return: 468.90% -> 306.97% (-161.93) [worse]
- Test proxy maxDD: 50.18% -> 53.66% (+3.48) [worse]
- Validation score: 61.42 -> 75.67 (+14.25)
- Promotion eligible: True -> False

## Window Review
- test_2020: good
  return 96.96% -> 101.26% | sharpe 2.16 -> 2.35 | maxDD 21.12% -> 19.98% | avg_pos 0.518 -> 0.472
  evidence: return improved (+4.30); sharpe improved (+0.19); max_dd improved (-1.14)
- test_2022: bad
  return -25.82% -> -31.27% | sharpe -0.88 -> -0.90 | maxDD 33.04% -> 39.86% | avg_pos 0.283 -> 0.338
  evidence: return worsened (-5.45); sharpe worsened (-0.03); max_dd worsened (+6.82)
- test_2023_2024: good
  return 43.52% -> 45.53% | sharpe 0.69 -> 0.71 | maxDD 33.87% -> 33.47% | avg_pos 0.498 -> 0.477
  evidence: return improved (+2.02); sharpe improved (+0.02); max_dd improved (-0.40)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。

## Gate Checks
- test_2022_avg_pos: True (0.28 / limit 0.35) -> True (0.34 / limit 0.35)
- test_2022_max_dd: True (33.04 / limit 35.00) -> False (39.86 / limit 35.00)
- test_proxy_sharpe: True (0.86 / limit 0.75) -> False (0.74 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。
- validation 变强但 OOS 变弱，优先复核 checkpoint 选择和 early stopping，而不是继续加特征。

## Inputs
- latest: `artifacts/models/soxl_balanced_structural_v1_summary.json`
- previous: `artifacts/models/soxl_balanced_summary.json`
