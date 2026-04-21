# balanced_rule_residual_mode_v1 Expectations

## Target Slices

- fixed split `test_2022`
- fixed split `test_2023_2024`
- full `test_proxy`

## Acceptance Criteria

- complete one end-to-end training run without harness breakage
- produce summary + backtest artifacts
- reduce `test_2022` exposure or DD relative to `balanced_rule_residual_v19`
- avoid a catastrophic collapse in `test_2023_2024`

## Expected Trade-Off

Some proxy return giveback is acceptable if the branch buys materially better
stress behavior through persistent modes rather than through churn-heavy
execution routing.

## Failure Mode

The mode router may simply freeze the actor into low-value conservative bands,
causing broad upside loss without a compensating `2022` improvement.

## Comparison Row Key

- `balanced_rule_residual_mode_v1`

## Outcome

- harness breakage: none
- acceptance readout:
  - end-to-end training: pass
  - summary + backtest artifacts: pass
  - `test_2022` exposure / DD reduction vs `v19`: pass
  - avoid catastrophic `test_2023_2024` collapse: pass
- verdict:
  - mechanism validated as a real new direction
  - not yet a replacement because validation and `test_2020` upside are still too weak
