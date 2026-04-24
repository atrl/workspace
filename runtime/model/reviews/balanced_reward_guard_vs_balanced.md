# Post-Train Backtest Review: soxl_balanced_reward_guard_summary vs soxl_balanced_summary

## Summary
- Test proxy return: 468.90% -> 523.36% (+54.46) [better]
- Test proxy maxDD: 50.18% -> 62.27% (+12.09) [worse]
- Validation score: 61.42 -> 83.32 (+21.90)
- Promotion eligible: True -> False

## Window Review
- test_2020: good
  return 96.96% -> 137.91% | sharpe 2.16 -> 2.31 | maxDD 21.12% -> 21.10% | avg_pos 0.518 -> 0.609
  evidence: return improved (+40.95); sharpe improved (+0.15); max_dd improved (-0.02)
- test_2022: mixed
  return -25.82% -> -36.00% | sharpe -0.88 -> -0.72 | maxDD 33.04% -> 47.36% | avg_pos 0.283 -> 0.412
  evidence: return worsened (-10.18); sharpe improved (+0.16); max_dd worsened (+14.32)
- test_2023_2024: mixed
  return 43.52% -> 50.23% | sharpe 0.69 -> 0.71 | maxDD 33.87% -> 39.85% | avg_pos 0.498 -> 0.572
  evidence: return improved (+6.72); sharpe improved (+0.02); max_dd worsened (+5.98)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。
- `2023-2024` 收益改善伴随更高持仓，说明新模型更愿意保留趋势暴露。
- 总体回报上升但总回撤加深，属于典型的收益换风险型改进，需要确认是否越过 promotion 风险边界。

## Gate Checks
- test_2022_avg_pos: True (0.28 / limit 0.35) -> False (0.41 / limit 0.35)
- test_2022_max_dd: True (33.04 / limit 35.00) -> False (47.36 / limit 35.00)
- test_proxy_sharpe: True (0.86 / limit 0.75) -> True (0.83 / limit 0.75)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。

## Inputs
- latest: `artifacts/models/soxl_balanced_reward_guard_summary.json`
- previous: `artifacts/models/soxl_balanced_summary.json`
