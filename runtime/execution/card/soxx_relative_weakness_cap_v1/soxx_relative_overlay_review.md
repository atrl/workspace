# SOXX Relative Weakness Cap Review

- Profile: `balanced_rule_residual_v22`
- Actor: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt`
- Dataset feature packs for overlay columns: `['vix_term', 'soxx_signals', 'transition_signals']`
- Diagnostic variant: `soxx_relative_weakness_cap_v1`
- Decision: `candidate`
- Role: fixed-actor SOXX execution diagnostic before any retrain

## Governance

- Canonical baseline: `balanced_rule_residual_v22` fixed actor.
- Direct comparison baseline: failed direct actor feature injection `balanced_rule_residual_v22_soxx`.
- Changed layer: `execution`.
- External reference: SOXX underlying ETF dataset plus prior `SIGNED_TE_REVERSION_10` factor card.
- Acceptance: preserve or improve `test_2022` avg position / max drawdown while avoiding a large `test_2023_2024` return give-up.

## Results

| variant | window | return | sharpe | max_dd | avg_pos | trades | eligible_days | cap_changed_days | eligible_order_days | delta_return | delta_max_dd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | test_proxy | 1511.97% | 1.112 | 56.48% | 0.5242 | 174 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| baseline | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| baseline | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| baseline | test_2023_2024 | 57.12% | 0.719 | 34.87% | 0.5402 | 50 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| soxx_relative_weakness_cap_v1 | test_proxy | 1621.98% | 1.138 | 54.73% | 0.5162 | 173 | 186 | 77 | 19 | +110.01pp | -1.75pp |
| soxx_relative_weakness_cap_v1 | test_2020 | 131.13% | 2.160 | 20.92% | 0.5356 | 27 | 15 | 12 | 0 | +0.00pp | +0.00pp |
| soxx_relative_weakness_cap_v1 | test_2022 | -30.97% | -0.764 | 40.64% | 0.3304 | 21 | 44 | 0 | 5 | +0.00pp | +0.00pp |
| soxx_relative_weakness_cap_v1 | test_2023_2024 | 72.61% | 0.817 | 36.89% | 0.5447 | 50 | 40 | 19 | 5 | +15.48pp | +2.02pp |

## Decision Checks

| check | passed |
|---|---:|
| `bear_avg_pos_preserved` | `true` |
| `bear_dd_preserved` | `true` |
| `recovery_giveup_ok` | `true` |
| `cap_has_effect` | `true` |

## Readout

- `test_2022` delta avg_pos: `+0.000000`.
- `test_2022` delta max_dd: `+0.0000pp`.
- `test_2023_2024` delta return: `+15.4842pp`.
- This diagnostic does not create a new PPO profile and is not a runtime promotion approval.
