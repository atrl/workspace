# Failure Modes
- high-return macro_credit_rate_gate candidates still carry SOXL-like crash drawdowns.
- drawdown-gated macro_composite_throttle candidates reduce risk but miss the production baseline return gate.
- trend_drawdown_stop is a control branch and did not create a drawdown-safe high-return candidate.
