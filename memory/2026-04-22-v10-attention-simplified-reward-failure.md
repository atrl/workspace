---
date: '2026-04-22'
title: "v10 Attention + Simplified Reward + Kelly \u2014 Failure Report"
kind: failure
profiles:
- balanced_kelly_attention_v10
status: dropped
tags:
- balanced_kelly_attention_v10
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
- stress-regime
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. **Profile**: `balanced_kelly_attention_v10`\
  \ **Hypothesis**: A fundamentally re-architected approach would break the \"fixed\
  \ low-position collapse\": Network: GRU \u2192 WindowSelfAttentionNe"
---

# v10 Attention + Simplified Reward + Kelly — Failure Report

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Experiment

- **Profile**: `balanced_kelly_attention_v10`
- **Hypothesis**: A fundamentally re-architected approach would break the "fixed low-position collapse":
  - Network: GRU → WindowSelfAttentionNet (explicit temporal feature weighting)
  - Reward: Simplified 5-term (alpha + tx cost + episodic DD + excess pos + inactivity)
  - Data: episode_len=500, stride=100 (≈15 episodes vs 5 previously)
  - Training: 40 iters, lr=0.0001 (vs v4's 20 iters, 0.0003)
  - Features: Kelly pack (`kelly_position_21`, `win_rate_21`, `odds_21`, `sharpe_21`)

## Results

```
[balanced_kelly_attention_v10]
return=553.29%
sharpe=0.849
max_dd=53.21%
trades=325

test_2020: return=97.32% sharpe=2.373 max_dd=19.27%
test_2022: return=-24.19% sharpe=-0.918 max_dd=32.55%
test_2023_2024: return=43.03% sharpe=0.699 max_dd=32.60%
```

**promotion: blocked**

## Comparison vs v4 Baseline

| Metric | v4 | v10 | Δ |
|--------|-----|------|---|
| return | 688.6% | 553.3% | -135.3pp |
| sharpe | 0.824 | 0.849 | +0.025 |
| max_dd | 31.26% | 53.21% | +21.95pp |
| test_2022 | (positive) | -24.19% | 🔴 |
| test_2023_2024 | 51.76% | 43.03% | -8.73pp |

## Selection History (Key Finding!)

Unlike v5-v9b where iter 0 was always best, v10 showed a different pattern:

| Iter | Score | Return | Sharpe | MaxDD |
|------|-------|--------|--------|-------|
| 0 | 77.28 | 49.9% | 1.53 | 26.89% |
| 20 | **86.53** | 58.8% | 1.57 | 30.35% | ← **BEST**
| 40 | 79.99 | 54.2% | 1.47 | 30.54% |

**v10's best valid score (86.53 at iter 20) actually exceeds v4's (86.31)!**

However, the test slices do not reflect this valid improvement:
- test_2022 MaxDD = 32.55% (fails 30% gate)
- test_2022 return = -24.19% (v4 was positive)

This indicates **valid-test divergence**: the model learned to optimize the valid slice
but lost generalization to the test slices, especially the 2022 bear market.

## Diagnosis

1. **Training rewards are consistently negative** (-187 to -267 across iters), suggesting
   the reward function is anti-correlated with good portfolio outcomes or the baseline
   (rulebase) is hard to beat.
2. **Valid score improved to iter 20 then degraded** — this is the first experiment since
   v4 where training produced a meaningful valid improvement, but it was not durable.
3. **Test_2022 failure persists**: Even with Attention architecture and Kelly features,
   the model cannot defend against the 2022 bear market. The valid slice likely lacks
   sufficient 2022-like stress examples for the model to learn robust defense.
4. **MaxDD on full test = 53.2%** vs v4's 31.3% — the model is taking more risk despite
   episodic drawdown penalties.

## Root Cause Assessment

The v10 result changes the diagnosis slightly:

- **v5-v9b**: PPO training could not improve valid score at all (always degraded from iter 0)
- **v10**: PPO training *can* improve valid score temporarily (iter 20 > iter 0), but
  the improvement does not generalize to test slices and is not durable (degrades by iter 40)

This suggests the issue is not purely "PPO destroys v4's local optimum" but rather:
1. The valid slice is too small or not representative of test stress
2. The model overfits to valid patterns within 20 iters, then forgets them by iter 40
3. The reward function may still be misaligned — high valid score does not translate to
   good test metrics

## Conclusion

v10 is **not a success** but it is **not a pure repeat** of v5-v9b either. The Attention
network + simplified reward + increased data allowed transient valid improvement,
but the fundamental problem remains: **training produces policies that score well on
valid but fail on test_2022 and have much higher overall drawdown.**

Next directions to consider:
1. **Early stopping at iter 20** — the best valid checkpoint; test if it performs better
   than the iter-40 checkpoint (it should, based on valid score)
2. **Valid slice expansion** — include more 2022-like stress in valid to reduce
   valid-test divergence
3. **Freeze v4 actor entirely** — use it as a fixed signal generator and only tune
   execution/risk overlays (position caps, hard floors, delta bands)
4. **Reward-test correlation audit** — verify whether higher training reward actually
   correlates with better test metrics; if not, the reward is misaligned
5. **Behavioral cloning from v4** — train a new network to imitate v4's actions without
   PPO fine-tuning, then see if PPO can improve from that starting point

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
