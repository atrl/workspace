# v19reglight_v22_budget_usage_bull_guard_ablation_v1 Walk-Forward Review

## Summary

The lighter-regularization slow path does not improve the approved
`v19reg_v22_budget_usage_bull_guard_v1` composer. With comparable saved
walk-forward actors, it ties `wf_2020`, weakens `wf_2022` return/sharpe despite
lower MaxDD, and materially regresses `wf_2023_2024`.

This remains a dropped ablation. Do not bind it to runtime selection.

## Results

| window | approved composer | light slow-path ablation | delta vs approved | decision |
| --- | ---: | ---: | ---: | --- |
| `wf_2020` | `98.27% / 1.456 / 26.14% / pos 0.399 / trades 25` | `98.27% / 1.456 / 26.14% / pos 0.399 / trades 25` | return `+0.00pp`, MaxDD `+0.00pp` | no improvement |
| `wf_2022` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | `-23.22% / -1.231 / 24.77% / pos 0.114 / trades 22` | return `-2.53pp`, MaxDD `-1.88pp` | lower drawdown but weaker return/sharpe |
| `wf_2023_2024` | `58.19% / 1.326 / 30.18% / pos 0.337 / trades 29` | `33.34% / 0.873 / 36.91% / pos 0.359 / trades 24` | return `-24.86pp`, MaxDD `+6.73pp` | fails recovery trade-off |

## Mechanism Readout

- Final inputs were exported with `--tag wf_2020`, `--tag wf_2022`, and
  `--tag wf_2023_2024` so the slow path uses the saved
  `balanced_rule_residual_v19_reg_light_wf_*` actors.
- Guard rule is unchanged from the approved composer: `price > MA50`, `20d`
  momentum `>= 3%`, and 50d rolling-high drawdown no worse than `8%`.
- Guard days are unchanged by the slow-path swap: `113` in `wf_2020`, `8` in
  `wf_2022`, and `49` in `wf_2023_2024`.
- The `wf_2023_2024` light slow actor has high raw exposure (`0.554` avg
  position, `39.80%` MaxDD). The two-speed mapping does not preserve enough of
  its upside and still widens composed drawdown versus the approved composer.

## Decision

Drop this ablation. Keep the card as evidence that the slow path should remain
`balanced_rule_residual_v19_reg` for the approved budget-usage composer.

The next non-production model loop should introduce a distinct mechanism rather
than another slow-path regularization swap.

## Evidence

- `walkforward/inputs/wf_2020/balanced_rule_residual_v19_reg_light_input.csv`
- `walkforward/inputs/wf_2022/balanced_rule_residual_v19_reg_light_input.csv`
- `walkforward/inputs/wf_2023_2024/balanced_rule_residual_v19_reg_light_input.csv`
- `walkforward/wf_2020/v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2020.json`
- `walkforward/wf_2022/v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2022.json`
- `walkforward/wf_2023_2024/v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2023_2024.json`
