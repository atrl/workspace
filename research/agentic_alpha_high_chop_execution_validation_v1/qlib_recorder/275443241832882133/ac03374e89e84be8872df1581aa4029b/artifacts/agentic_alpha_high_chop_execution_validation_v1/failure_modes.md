# Failure Modes

- close_to_close_execution_gap: target_position(t) is evaluated on t to t+1 close-to-close return; live use requires a timestamped signal before the chosen close/after-hours/next-open execution cutoff.
- recent_near_cash_not_proof: 2026-06-26 through 2026-07-02 is near cash, so it avoids the July 2 drawdown but does not prove fills for high-turnover full-cash/full-long flips.
- minute_history_depth: local Futu 1m coverage starts in 2024, so execution validation cannot close the 2019-2026 promotion gate without a longer intraday vendor.
