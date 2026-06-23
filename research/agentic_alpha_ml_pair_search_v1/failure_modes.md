# Failure Modes

- walk_forward_crash_return_ensemble: high-return candidates still carry crash-level drawdowns; drawdown-gated candidates reduce exposure so much that return stays far below the production baseline.
- soxl_soxx_pair_hedge: SOXX hedges reduce drawdown only when net SOXL exposure is small; return then fails the baseline gate. Higher-return pair candidates remain exposed to the same semiconductor drawdown regime.
- next_action: use a genuinely new crash-risk signal source or longer intraday/order-flow history; daily SOXL/SOXX/VIX features and SOXX hedges do not close the full objective.
