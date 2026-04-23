# two_speed_mode_budget_v1

## Motivation

The current frontier shows a split bottleneck:

- `balanced_rule_residual_mode_v1` controls drawdown well, but can suppress upside.
- `balanced_rule_residual_imitation_v4` restores participation, but misses the strict `test_2022 MaxDD <= 30%` gate.

This diagnostic tested whether a fixed-signal two-speed composition could keep
the slow mode risk anchor while allowing fast participation only inside
per-mode slack.

## External Reference

- Hierarchical / temporally abstract trading control, as already used for the
  retained mode-hierarchy line (`TradeR`, `EarnHFT`, `Deep Stock Trading`).
- Internal open-source-style reference: `soxl_backtest` slower risk gating and
  panic/risk budget mechanics.

## Hypothesis

Use `mode_v1` as the slow risk-budget layer and `imitation_v4` as the fast
participation layer. Cap the fast position by mode-specific slack over the slow
position, then filter low-value micro rebalances.

## Target Slices / Windows

- `test_2022`: reduce path drawdown without increasing average position.
- `test_2023_2024`: preserve or improve restart participation.
- `test_proxy`: improve Sharpe and MaxDD versus the fast participation path.

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
  --fast-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv
```

Key metrics:

- `test_2022`: `-16.5% / Sharpe -0.224 / MaxDD 33.9% / avg_pos 0.289`.
- `test_2023_2024`: `156.7% / Sharpe 1.278 / MaxDD 31.4%`.
- `test_proxy`: `1398.3% / Sharpe 1.100 / MaxDD 51.6% / trades 171`.

## Decision

Drop as a promotable candidate because it still fails the strict
`test_2022 MaxDD <= 30%` gate. Keep the replay tool and report artifacts because
the result proves the bottleneck more precisely: upside participation can be
restored without global exposure drift, but SOXL 2022 path drawdown still needs a
stronger crisis-state cap or a non-SOXL hedge rather than another mild slack
composition.

## Comparison-Table Row Key

`two_speed_mode_budget_v1`
