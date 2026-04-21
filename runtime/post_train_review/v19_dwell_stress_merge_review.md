# v19 Dwell-Stress Merge Review

- Profile: `balanced_rule_residual_v19`
- Model: `soxl_balanced_rule_residual_v19_actor.pt`

| variant | fixed gate | proxy ret | proxy sharpe | proxy maxdd | proxy trades | 2022 ret | 2022 maxdd | 2022 avg_pos | 2023-24 ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regime_route_v1 | pass | 659.1% | 0.924 | 48.85% | 464 | -23.9% | 32.61% | 0.272 | 61.8% |
| dwell_hold_v2 | pass | 626.8% | 0.883 | 50.56% | 121 | -26.4% | 34.73% | 0.285 | 56.9% |
| base | pass | 567.4% | 0.874 | 49.28% | 332 | -24.5% | 34.67% | 0.279 | 46.1% |
| dwell_stress_merge_v1 | pass | 451.7% | 0.810 | 60.60% | 358 | -24.6% | 31.90% | 0.267 | 56.0% |

## Readout

- This review tests whether a single stress-only asymmetry layer can improve `dwell_hold_v2` without reopening the broad overlay search space.
- Ranking: regime_route_v1, dwell_hold_v2, base, dwell_stress_merge_v1
