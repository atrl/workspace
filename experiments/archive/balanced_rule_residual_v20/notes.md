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

_(Trimmed by prune_experiments.py)_
