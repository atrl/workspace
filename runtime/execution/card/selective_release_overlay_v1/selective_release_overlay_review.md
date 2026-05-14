# Selective Release Overlay Review

- Profile: `balanced_rule_residual_v22`
- Actor: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt`
- Dataset feature packs for overlay columns: `['vix_term', 'release_signals_v1']`
- Role: fixed-actor counterfactual before any retrain

## Results

| variant | window | return | sharpe | max_dd | avg_pos | trades | delta_return | delta_max_dd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | test_proxy | 1132.19% | 1.043 | 56.48% | 0.5226 | 172 | +0.00pp | +0.00pp |
| baseline | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| baseline | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| baseline | test_2023_2024 | 57.12% | 0.719 | 34.87% | 0.5402 | 50 | +0.00pp | +0.00pp |
| selective_release_cicc_v1 | test_proxy | 1253.47% | 1.067 | 59.22% | 0.5260 | 169 | +121.27pp | +2.74pp |
| selective_release_cicc_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_cicc_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_cicc_v1 | test_2023_2024 | 58.74% | 0.730 | 34.41% | 0.5418 | 50 | +1.61pp | -0.46pp |
| selective_release_tight_v1 | test_proxy | 1253.82% | 1.074 | 54.34% | 0.5226 | 170 | +121.62pp | -2.14pp |
| selective_release_tight_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_tight_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_tight_v1 | test_2023_2024 | 58.47% | 0.728 | 34.41% | 0.5411 | 50 | +1.35pp | -0.46pp |
| selective_release_recovery_v1 | test_proxy | 1415.10% | 1.105 | 54.30% | 0.5230 | 171 | +282.91pp | -2.19pp |
| selective_release_recovery_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_v1 | test_2023_2024 | 82.10% | 0.877 | 36.95% | 0.5510 | 50 | +24.98pp | +2.08pp |

## Readout

- This is a fixed-actor overlay separation test. It does not create a new PPO profile.
- A promotion-worthy release overlay should improve `test_2023_2024` without widening `test_2022` drawdown or lifting `test_2022` average exposure materially.
- If no row clears that trade-off, the release trio remains useful as diagnostic evidence but is not ready for runtime routing.

## Decision

`selective_release_recovery_v1` is the retained counterfactual:

- `test_2023_2024` improves from `57.12% / 0.719 / 34.87%` to `82.10% / 0.877 / 36.95%`.
- `test_2022` is unchanged at `-30.97% / -0.764 / 40.64% / avg_pos 0.3304`.
- `test_proxy` improves from `1132.19% / 1.043 / 56.48%` to `1415.10% / 1.105 / 54.30%`.
- Trade count stays essentially flat (`172 -> 171` on `test_proxy`, `50 -> 50` on `2023-2024`).

This separates the bottleneck: the release factors are useful when gated in
execution, but direct retraining on the ungated release pack was not the right
next step. Do not open another PPO branch until this overlay is checked on
walk-forward actors and encoded as a governed execution profile or runtime
overlay.
