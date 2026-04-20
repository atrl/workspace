# balanced_rule_residual_v5 — Notes

## Hypothesis
v4's flat penalty (0.75) was too weak — only ~1.3% of typical episode return. A 4x increase to 3.0 (~5% of episode return) combined with a longer trade lock (12 days) should force behavioral trade compression.

## Changes from v4
- `flat_trade_penalty`: 0.75 → 3.0 (4x increase)
- `trade_lock_days`: 10 → 12
- All other parameters identical to v4

## Training Observations
- Training stable across all 4 stages (80 iters)
- Rewards: -350 to -450 (much more negative than v4's -250 to -370, confirming penalty is applied)
- Entropy: declined from 1.55 → 1.52 across stages
- No divergence

## Validation Checkpoints
- @20: return=51.8% sharpe=1.36 maxdd=33.0% score=75.04
- @40: return=67.2% sharpe=1.61 maxdd=34.1% score=95.40 (best)
- @60: return=46.3% sharpe=1.26 maxdd=34.6% score=67.32
- @80: return=61.6% sharpe=1.55 maxdd=30.3% score=89.00

## Test Results
- **return=780.7%**
- **sharpe=0.93**
- **maxdd=55.9%**
- **trades=326**

## Result: FAILED — Trade Compression Still Not Working

Despite a 4x stronger flat penalty (3.0) and longer lock (12 days), trade count went from **327 → 326** — essentially no change. The penalty IS being applied (rewards are ~30% more negative), but it's not changing behavior.

## Key Insight
The flat per-trade penalty is being absorbed by the value function as a "cost of doing business." The alpha rewards (alpha_scale=2.0, Kelly-alpha, return scaling=1.15) are so large that a 3.0 penalty per trade is irrelevant relative to the upside from capturing market moves.

## Comparison History
| Version | Penalty | Lock | Return | Sharpe | MaxDD | Trades |
|---------|---------|------|--------|--------|-------|--------|
| v2 | N/A | N/A | 521.3% | 0.86 | 50.4% | 316 |
| v3 | 0.50 | 8 | 569.7% | 0.86 | 53.6% | 314 |
| v4 | 0.75 | 10 | 730.8% | 0.90 | 55.7% | 327 |
| v5 | 3.00 | 12 | 780.7% | 0.93 | 55.9% | 326 |

Trade count has remained essentially flat (314→327→326) across all penalty attempts.

## Next Steps / Hypotheses
1. **Structural mismatch**: The backtest may count trades differently than the reward penalty fires. Need to investigate turnover threshold vs backtest trade counting.
2. **Penalty too small relative to alpha**: Even 3.0 is dwarfed by per-timestep alpha rewards. Need penalty proportional to returns or a hard trade budget.
3. **Try a trade budget approach**: Instead of per-trade penalty, penalize heavily when cumulative trades exceed a threshold (e.g., 150).
4. **Reduce alpha_scale**: Make alpha rewards smaller so the penalty has relative impact.
5. **Progressive penalty**: First 50 trades are "free," then penalty escalates dramatically.

## Decision
- **Status**: candidate (failed trade compression, blocked from promotion)
- **Baseline**: balanced_rule_residual_v2
- **Actor bundle**: Retain for now, but likely will be dropped after next experiment attempt
