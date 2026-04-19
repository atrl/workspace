# Expectations

- Must prove that residual-over-rule wiring is consistent across train, validation, and backtest.
- Must stay cheap and smoke-scale; this profile is for structure validation, not promotion claims.
- The first success criterion is behavioral:
  - actor actions only move around the rule-based base position
  - small residual churn is penalized more than in canonical `smoke`
- If the smoke run completes and emits a coherent summary, the next step can escalate to a balanced-family quick profile.
