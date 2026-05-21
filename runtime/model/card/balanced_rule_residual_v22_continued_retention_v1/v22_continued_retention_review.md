# Post-Train Backtest Review: soxl_balanced_rule_residual_v22_continued_retention_v1_summary vs soxl_balanced_rule_residual_v22_summary

## Summary
- Test proxy return: 695.45% -> 1128.94% (+433.48) [better]
- Test proxy maxDD: 54.52% -> 59.13% (+4.61) [worse]
- Validation score: 91.94 -> 86.24 (-5.70)
- Promotion eligible: True -> False

## Window Review
- test_2020: bad
  return 115.60% -> 108.37% | sharpe 2.74 -> 2.35 | maxDD 16.22% -> 18.58% | avg_pos 0.415 -> 0.505
  evidence: return worsened (-7.23); sharpe worsened (-0.40); max_dd worsened (+2.36)
- test_2022: bad
  return -24.52% -> -51.24% | sharpe -0.91 -> -1.33 | maxDD 32.71% -> 55.26% | avg_pos 0.269 -> 0.321
  evidence: return worsened (-26.71); sharpe worsened (-0.41); max_dd worsened (+22.54)
- test_2023_2024: mixed
  return 37.85% -> 60.79% | sharpe 0.66 -> 0.80 | maxDD 32.43% -> 35.94% | avg_pos 0.447 -> 0.513
  evidence: return improved (+22.94); sharpe improved (+0.14); max_dd worsened (+3.51)

## Mechanism Hypotheses
- `2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。
- `2023-2024` 收益改善伴随更高持仓，说明新模型更愿意保留趋势暴露。
- 总体回报上升但总回撤加深，属于典型的收益换风险型改进，需要确认是否越过 promotion 风险边界。
- 最新 run 的最佳 checkpoint 出现在 iter 0，之后分数回落到 iter 20，应优先考虑 early stopping 或更稳健的 checkpoint 选择。

## Gate Checks
- test_2022_avg_pos: True (0.27 / limit 0.35) -> True (0.32 / limit 0.35)
- test_2022_max_dd: True (32.71 / limit 55.00) -> False (55.26 / limit 55.00)
- test_proxy_sharpe: True (0.94 / limit 0.50) -> True (1.03 / limit 0.50)

## Recommended Next Actions
- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。

## Inputs
- latest: `runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/soxl_balanced_rule_residual_v22_continued_retention_v1_summary.json`
- previous: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json`
