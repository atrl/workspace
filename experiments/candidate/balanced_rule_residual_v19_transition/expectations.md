# Expectations: balanced_rule_residual_v19_transition

## Closest historical success
- `balanced_rule_residual_v19`: seed=12345, proxy `632.4% / Sharpe 0.91 / MaxDD 50.1%`

## Closest historical failure
- `balanced_rule_residual_v19_academic`: academic signals showed no improvement in 5-iter smoke test
- `balanced_rule_residual_minimal_v1`: stripping features caused -140pp return loss

## Why this is not a duplicate
- First test of transition-state / stress-release features in SOXL RL context
- Previous feature additions were academic (VRP, skewness) or rulebase (panic score, RSI6)
- Transition signals target a fundamentally different regime: "when to buy the dip" vs "how bad is the dip"

## External reference
- VIX divergence / put-call exhaustion patterns from options market microstructure
- Spring compression concept from classical technical analysis (Wyckoff methodology)
- Gap-fill statistics from intraday auction theory
