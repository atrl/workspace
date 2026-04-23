# Expectations: balanced_rule_residual_v19_academic

## Closest historical success
- `balanced_rule_residual_v19`: seed=12345, proxy `632.4% / Sharpe 0.91 / MaxDD 50.1%`

## Closest historical failure
- `balanced_rule_residual_v16` without kelly/rulebase: ~400-450% return range
- `balanced_rule_residual_minimal_v1`: stripping features/reward terms caused -140pp return loss

## Why this is not a duplicate
- First test of academic microstructure signals (Amihud, intraday reversal, volatility drag) in SOXL RL context
- Previous feature additions were rulebase-derived (panic score, RSI6 composites) or VIX term structure
- Academic signals target liquidity stress and intraday patterns, not regime state

## External reference
- Amihud (2002): "Illiquidity and stock returns"
- Cheng & Madhavan (2009): "The Dynamics of Leveraged and Inverse ETFs"
