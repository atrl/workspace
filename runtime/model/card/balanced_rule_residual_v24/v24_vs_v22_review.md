# Post-Train Backtest Review: balanced_rule_residual_v24 vs balanced_rule_residual_v22

## Summary
- Test proxy return: 695.45% -> 758.59% (+63.14) [better]
- Test proxy maxDD: 54.52% -> 54.58% (+0.06) [worse]
- Validation score: 91.94 -> 91.43 (-0.51)
- Promotion eligible: True -> False

## Window Review
- test_2020: bad
  return 115.60% -> 97.48% | sharpe 2.74 -> 2.53 | maxDD 16.22% -> 17.76% | avg_pos 0.415 -> 0.401
  evidence: return worsened (-18.12); sharpe worsened (-0.22); max_dd worsened (+1.54)
- test_2022: good
  return -24.52% -> -22.24% | sharpe -0.91 -> -0.86 | maxDD 32.71% -> 30.67% | avg_pos 0.269 -> 0.257
  evidence: return improved (+2.28); sharpe improved (+0.06); max_dd improved (-2.04)
- test_2023_2024: good
  return 37.85% -> 38.23% | sharpe 0.66 -> 0.66 | maxDD 32.43% -> 31.58% | avg_pos 0.447 -> 0.446
  evidence: return improved (+0.38); sharpe improved (+0.00); max_dd improved (-0.85)

## Mechanism Hypotheses
- `2022` 改善伴随平均仓位下降，说明风险控制更积极，防守侧机制更有效。
- 总体回报上升但总回撤加深，属于典型的收益换风险型改进，需要确认是否越过 promotion 风险边界。

## Gate Checks
- test_2022_avg_pos: True (0.27 / limit 0.35) -> True (0.26 / limit 0.35)
- test_2022_max_dd: True (32.71 / limit 55.00) -> True (30.67 / limit 35.00)
- test_proxy_sharpe: True (0.94 / limit 0.50) -> True (0.96 / limit 0.75)

## Recommended Next Actions
- 需要更多证据后再决定下一步；建议补充 walk-forward、交易明细或 feature summary。

## Inputs
- latest: `runtime/model/card/balanced_rule_residual_v24/soxl_balanced_rule_residual_v24_summary.json`
- previous: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json`
