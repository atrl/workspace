# balanced_rule_residual_mode_v2 Expectations

## Target Slices

- validation
- fixed split `test_2020`
- fixed split `test_2022`
- fixed split `test_2023_2024`
- full `test_proxy`

## Acceptance Criteria

- complete one end-to-end training run without harness breakage
- produce summary + backtest artifacts
- improve validation return or `test_2020` return relative to `balanced_rule_residual_mode_v1`
- keep `test_2022` max drawdown materially below `balanced_rule_residual_v19`
- avoid a material deterioration of `test_proxy` Sharpe or max drawdown versus `balanced_rule_residual_mode_v1`

## Expected Trade-Off

Some giveback of the extreme defensiveness from `mode_v1` is acceptable. The
goal is not to beat `mode_v1` on every risk axis; the goal is to recover bull
participation while preserving most of the hierarchy-driven risk improvement.

## Failure Mode

Loosening favorable modes may simply collapse back toward `v19` behavior,
recovering bull returns while giving back the `2022` and proxy drawdown gains
that justified the hierarchy in the first place.

## Comparison Row Key

- `balanced_rule_residual_mode_v2`

## Outcome

- harness breakage: none
- acceptance readout:
  - end-to-end training: pass
  - summary + backtest artifacts: pass
  - improve validation or `test_2020` vs `mode_v1`: fail
  - keep `test_2022` materially below `v19`: pass
  - avoid material `test_proxy` deterioration vs `mode_v1`: fail
- verdict:
  - drop this parameterization
  - keep the profile-configurable mode extension path
