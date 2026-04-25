# risk_budget_usage_v1 Full Review

**Date**: 2026-04-20
**Profile**: `risk_budget_usage_v1`
**Direction**: model / execution overlay
**Baseline**: `balanced_rule_residual_imitation_v4`

## Hypothesis

Treat the fast path as a **budget usage ratio**, not as a direct target position.
The slow mode layer owns the allowed risk band, and the fast path only decides
how much of that band to use.

## Replay Results (Two-Speed Composition)

Command:
```bash
PYTHONPATH=. qlib_venv/bin/python scripts/replay_two_speed_mode_budget.py \
  --slow-daily runtime/model/reviews/balanced_rule_residual_mode_v1_soxl_daily.csv \
  --fast-daily runtime/model/reviews/balanced_rule_residual_imitation_soxl_daily.csv \
  --composition budget_usage --profile-name risk_budget_usage_v1
```

| Slice | Return | Sharpe | MaxDD | avg_pos |
|-------|--------|--------|-------|---------|
| test_2020 | 239.0% | 2.285 | 21.7% | 0.490 |
| test_2022 | -17.1% | -0.358 | 25.6% | 0.214 |
| test_2023_2024 | 127.8% | 1.278 | 32.1% | 0.389 |
| test_proxy | 2003.6% | 1.248 | 45.0% | — |

## Actor-Backed Parity

Regenerated `imitation_v4` actor under `budget_usage` mapping:

| Slice | Return | Sharpe | MaxDD | avg_pos |
|-------|--------|--------|-------|---------|
| test_2020 | 19.0% | 1.189 | 17.6% | 0.158 |
| test_2022 | -0.8% | -0.083 | 4.6% | 0.043 |
| test_2023_2024 | 23.5% | 0.899 | 16.7% | 0.144 |
| test_proxy | 77.5% | 0.647 | 18.9% | 0.141 |
| qlib backtest | 148.9% | 0.839 | 22.2% | — |

Native PPO actor (`risk_budget_usage_v1` profile):

| Slice | Return | Sharpe | MaxDD | avg_pos |
|-------|--------|--------|-------|---------|
| test_2020 | 40.9% | 1.609 | 23.7% | 0.274 |
| test_2022 | -17.3% | -1.213 | 18.7% | 0.127 |
| test_2023_2024 | 82.8% | 1.440 | 16.3% | 0.295 |
| test_proxy | 274.4% | 0.942 | 32.5% | 0.269 |
| qlib backtest | 335.6% | 0.88 | 40.4% | — |

## Decision

**Keep as candidate.** Budget-usage architecture is valid but PPO discovers a
conservative policy. Replay-scale performance is not reproduced by RL training.

## Next Action

Supervised budget-usage warm start from replay path before PPO.

## Links

- Experiment card: `experiments/candidate/risk_budget_usage_v1/notes.md`
- Landscape: `docs/experiment-landscape-review.md`
