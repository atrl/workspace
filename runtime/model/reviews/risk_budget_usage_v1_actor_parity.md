# risk_budget_usage_v1 Actor-Backed Parity

- model: `artifacts/models/soxl_balanced_rule_residual_imitation_v4_actor.pt`
- model profile: `balanced_rule_residual_imitation_v4`
- position mapping: `budget_usage`

| window | return / sharpe / maxdd / avg_pos / trades |
| --- | --- |
| test_2020 | 19.0% / 1.189 / 17.6% / pos 0.158 / trades 0 |
| test_2022 | -0.8% / -0.083 / 4.6% / pos 0.043 / trades 0 |
| test_2023_2024 | 23.5% / 0.899 / 16.7% / pos 0.144 / trades 0 |
| test_proxy | 77.5% / 0.647 / 18.9% / pos 0.141 / trades 0 |
| qlib_backtest | 148.9% / 0.839 / 22.2% / trades 206 |

## Decision

Actor-backed parity does not reproduce the fixed-replay upside.

Reinterpreting the regenerated `imitation_v4` actor as budget usage makes the
policy too conservative: `test_proxy` falls to `77.5% / Sharpe 0.647` with
average position `0.141`. Training a native `risk_budget_usage_v1` actor fixes
the participation collapse only partially: fixed-split risk gates pass, but
`test_proxy` reaches only `274.4% / Sharpe 0.942`, far below the replay result.

Keep the runtime mapping and regression test. Do not promote either actor-backed
variant. The next model optimization should train the actor with an explicit
budget-usage imitation target or BC warm start from the replay path, not rely on
PPO discovering the usage scale from sparse portfolio reward.
