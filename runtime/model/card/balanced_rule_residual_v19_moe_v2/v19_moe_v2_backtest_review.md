# Post-Train Backtest Review: soxl_balanced_rule_residual_v19_moe_v2_summary vs soxl_balanced_rule_residual_v19_summary

## Summary
- Test proxy return: 830.01% -> 690.97% (-139.04) [worse]
- Test proxy maxDD: 51.47% -> 49.01% (-2.46) [better]
- Validation score: 94.48 -> 75.75 (-18.73)
- Promotion eligible: True -> False

Decision: keep `balanced_rule_residual_v19_moe_v2` as an ablation and do not
run walk-forward. It passes the fixed-split hard gate, but the stress-window
regression versus v19 and the weaker proxy Sharpe versus the first MoE run do
not justify expanding this branch.

## Window Review
- test_2020: bad
  return 123.93% -> 97.13% | sharpe 2.40 -> 2.19 | maxDD 22.26% -> 22.27% | avg_pos 0.560 -> 0.512
  evidence: return worsened (-26.81); sharpe worsened (-0.21); max_dd worsened (+0.00)
- test_2022: bad
  return -26.24% -> -42.67% | sharpe -0.86 -> -1.49 | maxDD 36.42% -> 45.15% | avg_pos 0.292 -> 0.288
  evidence: return worsened (-16.43); sharpe worsened (-0.63); max_dd worsened (+8.73)
- test_2023_2024: good
  return 53.23% -> 60.04% | sharpe 0.73 -> 0.83 | maxDD 38.19% -> 32.78% | avg_pos 0.557 -> 0.472
  evidence: return improved (+6.80); sharpe improved (+0.10); max_dd improved (-5.41)

## Mechanism Hypotheses
- 总体回报下降但回撤收敛，说明模型风格向保守漂移，需判断是否值得。
- 最新 run 的最佳 checkpoint 出现在 iter 30，之后分数回落到 iter 80，应优先考虑 early stopping 或更稳健的 checkpoint 选择。

## Gate Checks
- test_2022_avg_pos: True (0.29 / limit 0.35) -> True (0.29 / limit 0.35)
- test_2022_max_dd: True (36.42 / limit 55.00) -> True (45.15 / limit 55.00)
- test_proxy_sharpe: True (0.98 / limit 0.50) -> True (0.96 / limit 0.50)

## Recommended Next Actions
- Do not continue with MoE LR/load-balance/stage-length tuning alone; this run
  already improved optimizer stability but worsened 2022 regime allocation.
- If model optimization continues, use a distinct mechanism and declare how it
  avoids the MoE stress-window regression.
- Production/dashboard work remains blocked only by external Vercel
  credentials and the next fresh non-dry paper signal date.

## Inputs
- latest: `runtime/model/card/balanced_rule_residual_v19_moe_v2/soxl_balanced_rule_residual_v19_moe_v2_summary.json`
- previous: `runtime/model/card/balanced_rule_residual_v19/soxl_balanced_rule_residual_v19_summary.json`
