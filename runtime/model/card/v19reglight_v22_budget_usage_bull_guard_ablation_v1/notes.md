# v19reglight_v22_budget_usage_bull_guard_ablation_v1

## Gate

- Canonical baseline: `v19reg_v22_budget_usage_bull_guard_v1`.
- Direct comparison baseline: same budget-usage composer with slow path
  `balanced_rule_residual_v19_reg` and fast path `balanced_rule_residual_v22`.
- Role: `ablate`.
- Target window: saved actor walk-forward windows `wf_2020`, `wf_2022`, and
  `wf_2023_2024`.
- Accepted trade-off: only keep if the lighter-regularization slow path improves
  bull participation without worsening the 2022 defense or the 2023-2024
  recovery drawdown/return trade-off versus the approved composer.
- Changed layer: `model_selection`.
- External reference: regularization-frontier model-selection evidence in the
  existing residual actor family; no retraining or SOXX feature injection.
- Hypothesis: `balanced_rule_residual_v19_reg_light` can serve as the slow
  budget-mode path, preserving its stronger `wf_2020` participation while the
  v22 fast path and bull guard keep the composer inside the current risk budget.
- Acceptance criteria: walk-forward composition must beat or match
  `v19reg_v22_budget_usage_bull_guard_v1` on `wf_2022` defense and avoid a
  material `wf_2023_2024` drawdown/return regression; any bull-window gain is
  secondary to those checks.
- Expected trade-off: better `wf_2020` return, higher recovery exposure risk.
- Failure mode: the light slow actor has a known weak `wf_2023_2024` profile,
  so the composer may inherit excessive recovery drawdown or return giveback.
- Comparison-table row key:
  `v19reglight_v22_budget_usage_bull_guard_ablation_v1`.

## Historical Contrast

1. Closest success:
   `v19reg_v22_budget_usage_bull_guard_v1` improved the no-retrain composer by
   adding a price-confirmed bull participation guard while preserving the 2022
   defensive result.
2. Closest failure:
   `balanced_rule_residual_v19_reg_light` is strong in `wf_2020` but failed to
   improve bear/recovery generalization by itself, especially `wf_2023_2024`.
3. Not a duplicate:
   this loop does not retrain or tune the regularization actor; it tests whether
   the existing two-speed composer can use the light actor only as the slow-mode
   selector while v22 remains the fast usage path.

## Planned Evidence

- Export missing `balanced_rule_residual_v19_reg_light` walk-forward daily paths
  from the saved actor.
- Compose those paths with the existing saved `balanced_rule_residual_v22`
  walk-forward inputs using the same bull guard as the approved composer.
- Compare every walk-forward window against
  `v19reg_v22_budget_usage_bull_guard_v1`.

## Result

Final inputs were exported with the saved walk-forward actor tags
`wf_2020`, `wf_2022`, and `wf_2023_2024`. An earlier base-actor export was
discarded and overwritten because it was not comparable to the approved
walk-forward composer evidence.

| window | approved composer | light slow-path ablation | delta vs approved | decision |
| --- | ---: | ---: | ---: | --- |
| `wf_2020` | `98.27% / 1.456 / 26.14% / pos 0.399 / trades 25` | `98.27% / 1.456 / 26.14% / pos 0.399 / trades 25` | return `+0.00pp`, MaxDD `+0.00pp` | no improvement |
| `wf_2022` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | `-23.22% / -1.231 / 24.77% / pos 0.114 / trades 22` | return `-2.53pp`, MaxDD `-1.88pp` | weaker return/sharpe |
| `wf_2023_2024` | `58.19% / 1.326 / 30.18% / pos 0.337 / trades 29` | `33.34% / 0.873 / 36.91% / pos 0.359 / trades 24` | return `-24.86pp`, MaxDD `+6.73pp` | fails recovery trade-off |

Decision: drop the ablation. The approved composer should keep
`balanced_rule_residual_v19_reg` as the slow path. Do not bind this card to
runtime selection.
