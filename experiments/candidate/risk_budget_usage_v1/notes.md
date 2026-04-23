# risk_budget_usage_v1

## Motivation

The previous `two_speed_mode_budget_v1` replay proved that combining the slow
`mode_v1` risk anchor with the fast `imitation_v4` participation path improved
proxy quality, but it still failed the strict `test_2022 MaxDD <= 30%` gate.

This experiment tests the next narrower mechanism: treat the fast path as a
**budget usage ratio**, not as a direct target position. The slow mode layer owns
the allowed risk band, and the fast path only decides how much of that band to
use.

## External Reference

- Hierarchical / temporally abstract trading control: `TradeR`, `EarnHFT`,
  `Deep Stock Trading`.
- Risk-governor / volatility-targeting reference class: hard risk budget first,
  dynamic participation second.

## Hypothesis

Mapping `imitation_v4` into mode-owned risk bands should keep the upside
participation signal while preventing the additive/slack composition from
reopening the 2022 crisis drawdown.

## Target Slices / Windows

- `test_2022`: pass strict MaxDD and average-position gates.
- `test_2020`: recover trend participation relative to `mode_v1`.
- `test_2023_2024`: preserve restart participation.
- `test_proxy`: improve Sharpe and drawdown versus both source paths.

## Acceptance Criteria

- `test_2022 MaxDD <= 30.0%`.
- `test_2022 avg_pos <= 0.35`.
- `test_proxy Sharpe >= 0.90`.
- `test_2023_2024 return >= 46.8%`.
- proxy MaxDD improves by at least 2 points versus `imitation_v4_fast`.

## Result

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/replay_two_speed_mode_budget.py \
  --slow-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv \
  --fast-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv \
  --composition budget_usage \
  --profile-name risk_budget_usage_v1 \
  --output-json runtime/post_train_review/risk_budget_usage_v1.json \
  --output-md runtime/post_train_review/risk_budget_usage_v1.md \
  --output-daily runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv
```

Key metrics:

- `test_2020`: `239.0% / Sharpe 2.285 / MaxDD 21.7% / avg_pos 0.490`.
- `test_2022`: `-17.1% / Sharpe -0.358 / MaxDD 25.6% / avg_pos 0.214`.
- `test_2023_2024`: `127.8% / Sharpe 1.278 / MaxDD 32.1% / avg_pos 0.389`.
- `test_proxy`: `2003.6% / Sharpe 1.248 / MaxDD 45.0% / trades 188`.

## Decision

Keep as a candidate fixed-replay mechanism. Do not promote it yet: the result is
from daily-path replay, not a full runtime/qlib backtest or actor-trained
profile.

## Runtime/Profile Parity Wiring

The candidate now has a repo-owned profile at
`experiments/candidate/risk_budget_usage_v1/profile.yaml`.

- `model.position_mapping: budget_usage` means the actor output is interpreted
  as a usage ratio inside the current mode-owned band.
- For hybrid actors, the usage grid is derived from `delta_levels`, so the
  existing 5-action `imitation_v4` actor maps to `0%, 25%, 50%, 75%, 100%`
  band usage instead of accidentally using the default 11-action grid.
- `mode_router` encodes the replay bands:
  `flat_wait=0.00..0.25`, `defensive_hold=0.10..0.45`,
  `crash_rebuild=0.25..0.65`, `trend_hold=0.45..0.95`.
- Actor-backed parity is still blocked until the retained
  `balanced_rule_residual_imitation_v4` actor checkpoint is restored; the
  current artifact inventory retains the summary but not the actor bundle.

## Failure Mode To Check Next

This could be a replay-composition artifact if the source daily paths encode
slightly different execution timing than runtime signal generation. The next
step must implement the same budget-usage composition in the runtime/backtest
path and compare against this replay.

## Comparison-Table Row Key

`risk_budget_usage_v1`
