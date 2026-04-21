# v19 Overlay Promotion Review

- Profile: `balanced_rule_residual_v19`
- Model: `soxl_balanced_rule_residual_v19_actor.pt`

| variant | fixed gate | proxy ret | proxy sharpe | proxy maxdd | 2022 ret | 2022 maxdd | 2022 avg_pos | 2023-24 ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dwell_hold_v2 | pass | 1061.2% | 1.064 | 44.42% | -26.4% | 34.73% | 0.285 | 56.9% |
| dwell_hold_v1 | pass | 892.5% | 1.011 | 49.63% | -26.2% | 34.38% | 0.292 | 49.7% |
| regime_route_v1 | pass | 808.0% | 0.994 | 49.66% | -23.9% | 32.61% | 0.272 | 61.8% |
| stress_release_route_v1 | pass | 808.0% | 0.994 | 49.66% | -23.9% | 32.61% | 0.272 | 61.8% |
| base | pass | 774.1% | 0.968 | 51.79% | -24.5% | 34.67% | 0.279 | 46.1% |
| stress_asym_hold_v1 | pass | 682.6% | 0.954 | 48.17% | -23.8% | 32.47% | 0.273 | 50.9% |
| fast_reentry | pass | 692.9% | 0.949 | 51.06% | -22.2% | 31.22% | 0.272 | 59.3% |
| cta_defensive | pass | 607.8% | 0.907 | 51.53% | -23.8% | 32.47% | 0.273 | 53.5% |
| cta_trend_hold | pass | 584.9% | 0.891 | 48.18% | -26.5% | 34.21% | 0.288 | 43.2% |

## Readout

- Ranking: dwell_hold_v2, dwell_hold_v1, regime_route_v1, stress_release_route_v1, base, stress_asym_hold_v1, fast_reentry, cta_defensive, cta_trend_hold
- `dwell_hold_v2` is the question of interest because it was the strongest SOXL-specific trade-compression overlay in the prior cross-ETF diagnostic.