# balanced_kelly_attention_v10 — Notes

## Motivation

After v5-v9b all failed with the same pattern (valid score degrades from ~84 to 63-78,
selection falls back to iter 0), the hypothesis was that the problem was not Kelly
features per se but the **combination of architecture and reward**. v10 tests a
fundamentally re-architected approach:
- GRU → WindowSelfAttentionNet (explicit temporal feature weighting)
- Complex 12-term reward → simplified 5-term reward
- 5 episodes → ~15 episodes via stride=100
- 20 iters → 40 iters with lower lr

## Hypothesis

A clean-slate Attention network with simplified reward and more data can learn to
use Kelly features effectively and break out of the "fixed low-position collapse"
that plagued v5-v9b.

## External Reference

- Self-attention for time series: Vaswani et al. "Attention Is All You Need"
- Kelly criterion in position sizing: Thorp, "The Kelly Criterion in Blackjack,
  Sports Betting, and the Stock Market"

## Target Slices / Windows

- test_2022: improve MaxDD from v4's 31.26% to <30%, return should be positive
- test_2023_2024: improve return from v4's 51.76% to >60%
- Overall proxy: maintain or improve v4's 688.6% return with <35% MaxDD

## Acceptance Criteria

- [ ] valid score >= 86 (v4 baseline)
- [ ] test_2022 MaxDD < 30%
- [ ] test_2022 return > 0%
- [ ] test_2023_2024 return > 50%
- [ ] proxy MaxDD < 35%

## Expected Trade-off

- Higher training time due to Attention network
- May need more iters to converge due to clean-slate initialization
- Risk: simplified reward may remove necessary protective guardrails

## Results

```
valid:  return=58.8% sharpe=1.57 max_dd=30.3% score=86.53 (iter 20, best)
        return=54.2% sharpe=1.47 max_dd=30.5% score=79.99 (iter 40, final)

test_2020: return=97.3% sharpe=2.37 max_dd=19.3%
test_2022: return=-24.2% sharpe=-0.92 max_dd=32.6%
test_2023_2024: return=43.0% sharpe=0.70 max_dd=32.6%

proxy: return=553.3% sharpe=0.85 max_dd=53.2% trades=325
```

## Key Observations

1. **Valid score transiently exceeded v4** (86.53 at iter 20 vs v4's 86.31) — this is
   the first time any post-v4 training produced a valid improvement.
2. **Valid improvement did not generalize**: test_2022 return=-24.2%, proxy MaxDD=53.2%.
3. **Training rewards consistently negative** (-187 to -267), suggesting reward
   misalignment or a hard-to-beat baseline.
4. **Model degraded after iter 20**: valid score fell from 86.53 to 79.99 by iter 40.

## Failure Mode

Valid-test divergence. The model learned to optimize the valid slice but lost
generalization to test slices, especially the 2022 bear market. The valid slice
likely lacks sufficient 2022-like stress examples.

## Comparison Table Row

| profile | category | baseline | status | key delta | valid ret/sh/dd | 2020 ret/dd | 2022 ret/dd/pos | 2023-24 ret/dd | proxy ret/sh/dd | gate |
|---------|----------|----------|--------|-----------|-----------------|-------------|-----------------|----------------|-----------------|------|
| balanced_kelly_attention_v10 | architecture | balanced_rule_residual_imitation_v4 | dropped | Attention + Kelly + simplified reward + stride=100 | 58.8 / 1.57 / 30.3 | 97.3 / 19.3 | -24.19 / 32.55 / 0.26 | 43.03 / 32.60 | 553.3 / 0.85 / 53.2 | blocked |

## Keep / Drop / Next Action

- **Keep**: summary, memory note, experiment card (validates that architecture swap
  alone does not solve the problem)
- **Drop**: actor bundle (failed promotion, no walk-forward)
- **Next action**: Address valid-test divergence before further architecture/reward
  experiments. Options:
  1. Expand valid slice to include more 2022-like stress
  2. Freeze v4 actor and only tune execution overlays
  3. Behavioral cloning from v4 instead of PPO training
