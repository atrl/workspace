# balanced_kelly_dynamic_v5 — Kelly Criterion Dynamic Position Control

## Motivation

v2/v4 models collapse to a fixed ~15-25% position in 2022 because they lack a
"signal quality / timing goodness" concept. The alpha reward (`step_return -
rule_return`) incentivizes being below rulebase during crashes. With no
information about recent statistical edge, the degenerate optimal is minimum
allowed exposure.

This experiment introduces **rolling Kelly criterion features** into the
observation space and a **Kelly tracking penalty** into the reward function to
encourage dynamic position sizing based on recent 21-day win-rate, odds, and
Sharpe ratio.

## External Reference

- Kelly Criterion: J. L. Kelly Jr., "A New Interpretation of Information Rate",
  Bell System Technical Journal, 1956.
- Application to position sizing: Thorp, E. O. "The Kelly Criterion in Blackjack,
  Sports Betting, and the Stock Market." Handbook of Asset and Liability Management
  (2006).

## Hypothesis

If the policy can observe recent 21-day statistical edge (Kelly-optimal
position, win rate, odds, Sharpe), it will learn to:
1. Increase position when recent history shows positive edge (bull markets,
   recovery phases).
2. Decrease position when recent history shows negative edge (crash phases).
3. Avoid the degenerate "always minimum exposure" solution.

## Baseline & Comparison

- **Canonical baseline**: rulebase (688.6% / 0.824 / 69.88%)
- **Direct comparison baseline**: v4 (balanced_rule_residual_imitation_v4)
  - test_2022 MaxDD: 31.3%
  - test_2023_2024 Return: 51.8%
  - test_2020 Return: 111.6%

## Experiment Role

`diagnose` — targeting the fixed low-position collapse.

## Changed Layer

- `feature`: Added `kelly` feature pack (`kelly_position_21`, `win_rate_21`,
  `odds_21`, `sharpe_21`)
- `reward`: Added `kelly_tracking_penalty` (0.30) with vol attenuation

## Target Improvement Window

- test_2022: Maintain MaxDD < 30% while increasing average position
  (currently ~20% fixed; target ~30-40% dynamic).
- test_2023_2024: Significantly improve return (currently 51.8%; target > 80%).

## Accepted Trade-off

- test_2022 MaxDD may increase slightly (up to 33%) if accompanied by
  significantly better walk-forward performance.
- Turnover may increase due to dynamic sizing.

## Failure Mode

- Kelly features are ignored (model still collapses to fixed low position).
- Kelly tracking creates excessive churn in volatile regimes.
- Model overfits to 21-day rolling statistics and misses regime transitions.

## Comparison Table Row Key

`balanced_kelly_dynamic_v5`
