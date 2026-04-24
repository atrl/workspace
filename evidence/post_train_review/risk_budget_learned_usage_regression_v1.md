# risk_budget_learned_usage_regression_v1 Usage Regression Probe

No-promotion diagnostic trained from a replay daily path with a continuous usage regression head.

## Dataset

- replay_daily: `evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv`
- samples: `1508`
- target_mean: `0.247283`
- target_std: `0.176626`

## Train Stats

- loss: `0.002299`
- mae: `0.035062`
- steps: `2000`

## Fixed-Split Probe

| slice | return / sharpe / maxdd / avg_pos |
| --- | --- |
| test_2020 | 50.2% / Sharpe 2.069 / MaxDD 19.9% / avg_pos 0.287 |
| test_2022 | -15.1% / Sharpe -1.436 / MaxDD 17.6% / avg_pos 0.102 |
| test_2023_2024 | 63.0% / Sharpe 0.978 / MaxDD 25.7% / avg_pos 0.332 |
| test_proxy | 222.8% / Sharpe 0.772 / MaxDD 34.7% / avg_pos 0.303 |

## Decision

no-promotion diagnostic only; continuous usage regression from held-out replay daily path
