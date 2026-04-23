# risk_budget_oracle_distill_v1 Oracle Distillation

No-promotion diagnostic trained from the fixed replay daily path.

This actor is an upper-bound capacity probe only. It uses replay labels from held-out dates and must not be promoted.

## Dataset

- replay_daily: `runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv`
- samples: `1508`
- label_distribution: `{'0': 127, '1': 436, '2': 383, '3': 423, '4': 139}`

## BC Stats

- loss: `1.195009`
- accuracy: `0.710938`
- steps: `1600`

## Fixed-Split Probe

| slice | return / sharpe / maxdd / avg_pos |
| --- | --- |
| test_2020 | 35.7% / Sharpe 1.322 / MaxDD 25.9% / avg_pos 0.320 |
| test_2022 | -5.5% / Sharpe -0.350 / MaxDD 11.0% / avg_pos 0.117 |
| test_2023_2024 | 58.5% / Sharpe 1.180 / MaxDD 19.3% / avg_pos 0.271 |
| test_proxy | 363.8% / Sharpe 1.090 / MaxDD 23.7% / avg_pos 0.264 |

## Decision

no-promotion diagnostic only; labels come from held-out replay daily path

Actor bundle: `artifacts/models/soxl_risk_budget_oracle_distill_v1_oracle_actor.pt`
