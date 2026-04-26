# risk_budget_bc_warmstart_v1 BC pretrain

Budget-usage behavior cloning from a retained fast expert actor on the train split only.

## Dataset

- expert_profile: `balanced_rule_residual_imitation_v4`
- expert_model: `artifacts/models/soxl_balanced_rule_residual_imitation_v4_actor.pt`
- samples: `1904`
- label_distribution: `{'0': 0, '1': 422, '2': 710, '3': 772}`

## BC Stats

- loss: `0.950989`
- accuracy: `0.960938`
- steps: `1200`

## Fixed-Split Probe

| slice | return / sharpe / maxdd / avg_pos |
| --- | --- |
| test_2020 | 36.0% / Sharpe 1.493 / MaxDD 20.5% / avg_pos 0.296 |
| test_2022 | -5.0% / Sharpe -0.345 / MaxDD 10.7% / avg_pos 0.108 |
| test_2023_2024 | 49.0% / Sharpe 1.010 / MaxDD 20.1% / avg_pos 0.273 |
| test_proxy | 250.7% / Sharpe 0.911 / MaxDD 27.3% / avg_pos 0.266 |

Actor bundle: `/Users/rziiiii/.openclaw/workspace/.worktrees/risk-budget-bc-warmstart-v1/artifacts/models/soxl_risk_budget_bc_warmstart_v1_bc_actor.pt`
