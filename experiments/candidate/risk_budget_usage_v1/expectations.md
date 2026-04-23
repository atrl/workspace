# Expectations: risk_budget_usage_v1

## Expected Win

Using `imitation_v4` as a budget usage ratio should restore trend participation
without letting the fast path exceed mode-owned crisis caps.

## Expected Trade-Off

The strategy may still lag buy-and-hold during violent rebounds because
`flat_wait` and `defensive_hold` deliberately keep low risk budgets.

## Failure Mode

If full runtime/backtest validation is much worse than replay, the replay is
overstating performance because it composes already-executed daily paths rather
than actor decisions inside the simulator.
