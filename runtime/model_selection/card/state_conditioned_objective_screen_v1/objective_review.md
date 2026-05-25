# State-Conditioned Objective Screen

## Decision

- decision: `no_objective_pass_candidate`
- selected_candidate: `state_conditioned_rebound_risk_router_v1`
- next_mechanism: `train_state_conditioned_multi_objective`

## Ranking

| candidate | pass | loss | wf pass | cluster pass | rel loss | dd loss | part loss | worst wf | worst cluster |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| state_conditioned_rebound_risk_router_v1 | False | 303.65 | 1/4 | False | 209.20 | 54.45 | 40.00 | wf_2020 | cluster_2025_high_vol_rebound |
| recovery_reentry_conditional_vol_target_v1 | False | 585.64 | 0/4 | False | 211.88 | 356.92 | 16.84 | strict_oos | cluster_2026_high_vol_rebound |
| recovery_reentry_cppi_floor_v1 | False | 649.71 | 0/4 | False | 228.33 | 404.54 | 16.84 | wf_2023_2024 | cluster_2026_high_vol_rebound |
| current_coverage_residual_guard_refresh_v1 | False | 726.53 | 0/4 | False | 358.71 | 350.98 | 16.84 | strict_oos | cluster_2026_high_vol_rebound |
| recovery_reentry_equity_drawdown_guard_v1 | False | 806.48 | 0/4 | False | 267.80 | 520.31 | 18.37 | wf_2022 | cluster_2026_high_vol_rebound |
| recovery_reentry_anchor_confirm_v1 | False | 827.47 | 0/4 | False | 202.36 | 608.27 | 16.84 | strict_oos | cluster_2026_high_vol_rebound |
| current_coverage_rebound_downtrend_router_v1 | False | 881.90 | 0/4 | False | 390.52 | 487.56 | 3.83 | strict_oos | cluster_2025_high_vol_rebound |
| state_conditioned_actor_selector_v1 | False | 1204.09 | 0/4 | False | 231.73 | 942.23 | 30.13 | wf_2022 | cluster_2026_high_vol_rebound |

## Readout

- A passing row is a promotion-review input, not an automatic runtime bind.
- If no row passes, treat the selected row as a diagnostic target and keep no runtime promotion.
- Required terms should be implemented in training-time reward/checkpoint selection before another path-level threshold search.
