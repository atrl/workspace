# Selective Release Overlay Review

- Profile: `balanced_rule_residual_v22`
- Actor: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt`
- Dataset feature packs for overlay columns: `['vix_term', 'release_signals_v1']`
- Role: fixed-actor counterfactual before any retrain

## Results

| variant | window | return | sharpe | max_dd | avg_pos | trades | delta_return | delta_max_dd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | test_proxy | 1511.97% | 1.112 | 56.48% | 0.5242 | 174 | +0.00pp | +0.00pp |
| baseline | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| baseline | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| baseline | test_2023_2024 | 57.12% | 0.719 | 34.87% | 0.5402 | 50 | +0.00pp | +0.00pp |
| selective_release_cicc_v1 | test_proxy | 1697.25% | 1.138 | 59.22% | 0.5280 | 172 | +185.28pp | +2.74pp |
| selective_release_cicc_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_cicc_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_cicc_v1 | test_2023_2024 | 58.74% | 0.730 | 34.41% | 0.5418 | 50 | +1.61pp | -0.46pp |
| selective_release_tight_v1 | test_proxy | 1670.72% | 1.142 | 54.34% | 0.5241 | 172 | +158.74pp | -2.14pp |
| selective_release_tight_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_tight_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_tight_v1 | test_2023_2024 | 58.47% | 0.728 | 34.41% | 0.5411 | 50 | +1.35pp | -0.46pp |
| selective_release_recovery_v1 | test_proxy | 1882.20% | 1.172 | 54.30% | 0.5246 | 173 | +370.23pp | -2.19pp |
| selective_release_recovery_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_v1 | test_2023_2024 | 82.10% | 0.877 | 36.95% | 0.5510 | 50 | +24.98pp | +2.08pp |
| selective_release_recovery_threshold_bypass_v1 | test_proxy | 1882.19% | 1.176 | 54.37% | 0.5234 | 175 | +370.22pp | -2.11pp |
| selective_release_recovery_threshold_bypass_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_threshold_bypass_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_threshold_bypass_v1 | test_2023_2024 | 82.10% | 0.877 | 36.95% | 0.5510 | 50 | +24.98pp | +2.08pp |
| selective_release_recovery_target_lift_v1 | test_proxy | 1878.62% | 1.171 | 54.37% | 0.5266 | 174 | +366.65pp | -2.11pp |
| selective_release_recovery_target_lift_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_target_lift_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_target_lift_v1 | test_2023_2024 | 81.41% | 0.872 | 37.04% | 0.5508 | 51 | +24.29pp | +2.17pp |
| selective_release_recovery_trade_lock_bypass_v1 | test_proxy | 1770.65% | 1.165 | 54.37% | 0.5234 | 212 | +258.68pp | -2.12pp |
| selective_release_recovery_trade_lock_bypass_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_bypass_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_bypass_v1 | test_2023_2024 | 61.84% | 0.741 | 38.65% | 0.5540 | 67 | +4.72pp | +3.78pp |
| selective_release_recovery_trade_lock_short_v1 | test_proxy | 1655.03% | 1.145 | 54.34% | 0.5183 | 192 | +143.06pp | -2.14pp |
| selective_release_recovery_trade_lock_short_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_short_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_short_v1 | test_2023_2024 | 43.20% | 0.618 | 36.24% | 0.5353 | 53 | -13.93pp | +1.37pp |
| selective_release_recovery_trade_lock_guarded_short_v1 | test_proxy | 1511.97% | 1.112 | 56.48% | 0.5242 | 174 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_guarded_short_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_guarded_short_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_trade_lock_guarded_short_v1 | test_2023_2024 | 57.12% | 0.719 | 34.87% | 0.5402 | 50 | +0.00pp | +0.00pp |
| selective_release_recovery_post_release_cap_v1 | test_proxy | 1511.97% | 1.112 | 56.48% | 0.5242 | 174 | +0.00pp | +0.00pp |
| selective_release_recovery_post_release_cap_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | +0.00pp | +0.00pp |
| selective_release_recovery_post_release_cap_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | +0.00pp | +0.00pp |
| selective_release_recovery_post_release_cap_v1 | test_2023_2024 | 57.12% | 0.719 | 34.87% | 0.5402 | 50 | +0.00pp | +0.00pp |

## Readout

- This is a fixed-actor overlay separation test. It does not create a new PPO profile.
- A promotion-worthy release overlay should improve `test_2023_2024` without widening `test_2022` drawdown or lifting `test_2022` average exposure materially.
- If no row clears that trade-off, the release trio remains useful as diagnostic evidence but is not ready for runtime routing.
