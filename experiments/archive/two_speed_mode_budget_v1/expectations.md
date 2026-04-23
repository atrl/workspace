# Expectations: two_speed_mode_budget_v1

## Expected Win

The two-speed composition should improve proxy Sharpe and drawdown versus the
fast `imitation_v4` path while preserving most of its restart participation.

## Expected Trade-Off

Some `test_2020` upside may be sacrificed because the slow risk anchor limits
fast participation during ambiguous mode states.

## Failure Mode

If `test_2022` MaxDD remains above 30%, the issue is not just broad participation
drift. It means the single-asset SOXL path still needs stricter crisis-state risk
budgeting or a different protection mechanism.
