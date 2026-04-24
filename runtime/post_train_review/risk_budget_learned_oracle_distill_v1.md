# risk_budget_learned_oracle_distill_v1 Oracle Distillation

No-promotion diagnostic trained from a replay daily path.

This actor is an upper-bound capacity probe only. It uses replay labels from held-out dates and must not be promoted.

## Dataset

- replay_daily: `runtime/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv`
- samples: `1508`
- label_distribution: `{'0': 238, '1': 1138, '2': 48, '3': 52, '4': 32}`

## BC Stats

- loss: `1.123583`
- accuracy: `0.781250`
- steps: `1600`

## Fixed-Split Probe

| slice | return / sharpe / maxdd / avg_pos |
| --- | --- |
| test_2020 | 72.5% / Sharpe 3.162 / MaxDD 8.0% / avg_pos 0.286 |
| test_2022 | -11.7% / Sharpe -1.054 / MaxDD 13.2% / avg_pos 0.105 |
| test_2023_2024 | 78.6% / Sharpe 1.130 / MaxDD 24.9% / avg_pos 0.336 |
| test_proxy | 412.9% / Sharpe 1.042 / MaxDD 28.7% / avg_pos 0.296 |

## Decision

no-promotion diagnostic only; labels come from held-out replay daily path

Actor bundle: `artifacts/models/soxl_risk_budget_learned_oracle_distill_v1_oracle_actor.pt`
