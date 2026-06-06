# macro_extreme_risk_retrain_v1

## Experiment Governance

- Baselines:
  - canonical runtime baseline: `v19reg_v22_budget_usage_bull_guard_v1`
  - direct actor baseline: `balanced_rule_residual_v22`
  - direct macro failure baseline: `balanced_macro_v1`
- Role: `diagnose`
- Target windows: fixed split `test_2020`, `test_2022`, `test_2023_2024`, `test_proxy`; walk-forward `wf_2020`, `wf_2022`, `wf_2023_2024` only after fixed-split gate evidence.
- Accepted trade-off: tolerate small 2020/2023-2024 giveback only if `test_2022_avg_pos <= 0.35` and 2022 MaxDD does not worsen versus `balanced_rule_residual_v22`.
- Changed layer: `feature`
- External references: FRED series observations API, Cboe volatility historical data.
- Hypothesis: a narrow, lagged macro-volatility pack can identify extreme-day preconditions better than broad raw macro injection and avoid the `balanced_macro_v1` 2022 exposure failure.
- Acceptance criteria: feature gate shows non-redundant predictive signal; fixed-split retrain preserves 2022 defense; walk-forward improves or preserves all standard windows before promotion.
- Expected trade-off: less broad macro noise than `balanced_macro_v1`, but added observation dimension may still slow learning.
- Failure mode: macro stress variables are stale or redundant with `regime_risk` / `vix_percentile_252`, causing higher 2022 exposure or no incremental signal.
- Comparison-table row key: `macro_extreme_risk_retrain_v1`

## Closest Historical Cases

- Closest success: `v19reg_v22_budget_usage_bull_guard_v1` improved walk-forward daily paths by changing risk-budget composition while preserving 2022 defense.
- Closest failure: `balanced_macro_v1` broad macro feature injection improved 2020 and 2023-2024 but worsened 2022 and failed the average-position gate.
- Difference: this run narrows macro inputs, lags all outputs by one day, includes VIX-term and macro-volatility interaction state, and does not reuse raw credit-spread level.

## Current Status

- `macro_extreme_risk_v1` feature pack exists and is tested.
- This card's notebook is the primary run-local evidence and conclusion surface.
