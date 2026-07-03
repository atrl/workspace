# Execution Readiness

- decision: research_reject_open2open_goal_gate
- execution_timing: close_t_signal_next_open_to_following_open
- return_definition: target_position(t) * (open(t+2) / open(t+1) - 1) minus turnover cost
- latest_signal: 2026-07-02 close signal maps to the next available open; realized PnL needs the following open, so same-day 7/2 open-to-close loss remains an intraday risk validation item.
- required_before_production: minute-level adverse-excursion check, next-open fill/slippage check, and after-hours liquidity check.
- not_advice: 不构成买卖建议。
