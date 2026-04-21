# balanced_rule_residual_v20 — Penalty Optimization (3.0)

## Hypothesis

v16/v19 with penalty=2.0 achieves ~635% return with MaxDD ~50.5%.
Historical data (v12/v13) showed 2.0-4.0 produced identical results.
But that was with a different config (no GRU, different window).

Testing penalty=3.0 with the proven GRU+window=5 config to see if:
1. Drawdown can be pushed below 45% (from ~50.5%)
2. Return remains in the ~600% range

## Changes from v19

- `episode_drawdown_penalty`: 2.0 → 3.0
- Same seed: 12345 (proven good seed)
- All else identical to v16/v19

## Results

### CLI Backtest
- **Return**: 551.8%
- **Sharpe**: 0.86
- **MaxDD**: 51.7%
- **Trades**: 319

### test_proxy
- **Return**: 623.2%
- **Sharpe**: 0.90
- **MaxDD**: 53.8%

### Slice Breakdown
- test_2020: return=110.1% sharpe=2.74 max_dd=16.2%
- test_2022: return=-23.4% sharpe=-0.90 max_dd=31.7% avg_pos=0.26
- test_2023_2024: return=42.6% sharpe=0.68 max_dd=32.7%

### Promotion Gate
- **fixed_split**: PASSED ✓ (2022 maxDD 31.7% < 35%, avg_pos 0.26 < 0.35, proxy Sharpe 0.90 > 0.75)
- **walk_forward**: not evaluated

### Selection History
- iter 0:  score=67.42  return=45.0%  sharpe=1.31  maxDD=31.2%
- iter 20: score=64.76  return=43.0%  sharpe=1.27  maxDD=29.9%
- iter 40: score=92.26  return=62.2%  sharpe=1.67  maxDD=27.2%
- iter 60: score=98.96  return=67.4%  sharpe=1.75  maxDD=28.3% ← BEST
- iter 80: score=72.14  return=47.3%  sharpe=1.42  maxDD=28.9%

## Analysis

### Penalty=3.0 is WORSE than 2.0 across the board

| Metric | v19 (penalty=2.0) | v20 (penalty=3.0) | Δ |
|--------|-------------------|-------------------|---|
| CLI Return | 635.1% | 551.8% | **-83pp** |
| CLI Sharpe | 0.90 | 0.86 | -0.04 |
| CLI MaxDD | 51.0% | 51.7% | **+0.7pp** |
| Proxy Return | 885.5% | 623.2% | **-262pp** |
| Proxy Sharpe | 0.99 | 0.90 | -0.09 |
| Trades | 336 | 319 | -17 |

**Critical finding**: penalty=3.0 is **strictly dominated** by penalty=2.0.
- Return drops significantly
- Drawdown actually gets slightly WORSE (not better!)
- The policy becomes less active (fewer trades) but not more effective

This contradicts the v12/v13 finding that 2.0-4.0 were "identical".
The difference: v12/v13 had no GRU and window=3. With GRU+window=5,
the policy is more sensitive to penalty strength. The temporal model
learns to exploit the penalty structure, and 3.0 pushes it into a
suboptimal basin.

### Selection Pattern Shift
Best checkpoint moved from iter 20 (v19) to iter 60 (v20).
Higher penalty needs more training to converge — the policy takes
longer to find a stable basin under stronger constraints.

### 7th Consecutive fixed_split Pass
v14-v20 all pass. The drawdown discipline is deeply embedded now.

## Lessons

1. **penalty=2.0 is the sweet spot** for GRU+window=5. Going higher hurts returns
   without improving drawdown.
2. **v12/v13's "2.0-4.0 identical" finding does NOT generalize** to the GRU config.
3. **Best checkpoint timing depends on penalty strength**: 2.0 converges by iter 20,
   3.0 needs iter 60.
4. **The ~50% MaxDD floor is structural** — no amount of terminal penalty tweaking
   has broken below it. To go lower, need different mechanisms (position_cap,
   regime-specific behavior, etc.).

## Next Actions

- Revert to penalty=2.0 (v19 config is the current best).
- Try increasing network capacity: `gru_hidden_size: 128` (currently 64).
  The temporal model might be bottlenecked by representation capacity.
- OR try fine-tuning observation_window: test 4 or 6 days.
