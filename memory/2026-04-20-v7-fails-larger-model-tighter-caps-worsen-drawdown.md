---
date: '2026-04-20'
title: 'v7 Post-Mortem: [256,128] + Stronger Penalties + Tighter Caps'
kind: review
profiles:
- balanced_rule_residual_v2
- balanced_rule_residual_v7
status: kept
tags:
- balanced_rule_residual_v2
- balanced_rule_residual_v7
- eval-2022
- kept
- review
artifacts: []
workpad: memory/index.json
summary: 'Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20
  Profile: `balanced_rule_residual_v7` Baseline comparison: `balanced_rule_residual_v2`
  (best drawdown so far), `v5` (best return) | Metric | v2 (baseline) | v5 | v7 |'
---

# v7 Post-Mortem: [256,128] + Stronger Penalties + Tighter Caps

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-20
Profile: `balanced_rule_residual_v7`
Baseline comparison: `balanced_rule_residual_v2` (best drawdown so far), `v5` (best return)

## Results

| Metric | v2 (baseline) | v5 | v7 |
|--------|--------------|-----|-----|
| **Test Return** | 521.3% | 780.7% | 548.0% |
| **Test Sharpe** | 0.86 | 0.93 | 0.85 |
| **Test MaxDD** | **50.4%** | 55.9% | **58.4%** |
| **Trades** | 316 | 326 | 328 |
| Valid Best Score | 104.61 (@40) | 95.40 (@40) | 99.44 (@20) |

### Slice Breakdown

| Slice | v2 | v5 | v7 |
|-------|-----|-----|-----|
| test_2020 ret / maxdd | +84.5% / 15.9% | +102.0% / 19.0% | +106.4% / 17.1% |
| test_2022 ret / maxdd | **-16.2% / 23.6%** | -24.7% / 32.7% | -25.2% / 33.9% |
| test_2023_24 ret / maxdd | +43.1% / 29.2% | +55.1% / 32.8% | +60.5% / 31.7% |
| test_2022 avg_pos | **0.18** | 0.27 | 0.27 |

## Key Findings

### 1. Larger Model [256,128] → Faster Overfitting
- v7 valid score @20 = 99.44, then crashes to 71.88 @60.
- v2 valid score @40 = 104.61, stays healthy across stages.
- **Conclusion**: [256,128] has too much capacity for 59-dim features + 10 episodes/iter. Overfits training episodes by stage 2.

### 2. Tighter Position Caps Did NOT Help Drawdown
- v7: start=0.08, full=0.22 (vs v2: 0.10, 0.30)
- v7 test maxdd = 58.4% > v2's 50.4%
- Tighter caps limited upside in bull markets but did not protect downside in bears.
- **Why**: `drawdown_position_cap` uses **market drawdown** (`cum_drop_30`), not portfolio drawdown. Strategy learned to work around the cap.

### 3. Hard Floor = 0.40 Prevents Effective Bear Market De-Risking
- v2 hard_floor = 0.20 → 2022 avg_pos = 0.18
- v5/v7 hard_floor = 0.40 → 2022 avg_pos = 0.27
- **18% vs 27% average position in 2022 explains most of the return gap**:
  - v2: -16.2% in 2022
  - v5/v7: -24~25% in 2022
- Higher hard_floor forces strategy to stay exposed during crashes.

### 4. Stronger Step-Level Penalties Are Insufficient
- v7 drawdown_penalty = 0.50 (vs v2: 0.28, v5: 0.28)
- v7 downside_penalty = 0.35 (vs v2: 0.20)
- Despite 2x penalties, drawdown worsened.
- **Reason**: Step-level penalty is small relative to alpha reward. Agent learns to "absorb" the penalty when chasing alpha.

### 5. Valid Score Is NOT a Reliable Predictor of Test Drawdown
- v7 valid maxdd @best = 25.1%, but test maxdd = 58.4%
- v2 valid maxdd @best = 26.0%, test maxdd = 50.4%
- Valid period (~252d) is too short to cover bear regimes. Low valid maxdd does not guarantee low test maxdd.

## Root Cause Summary

The drawdown problem is fundamentally about **cross-period cumulative drawdown**, not intra-year drawdown:
- test_2022 maxdd = 33.9% (passes 35% gate)
- test_proxy (full cycle) maxdd = 60.4% (fails)
- The 60% drawdown is cumulative across 2022→2023 boundary, not a single-year event.

Primary driver: **strategy holds too much exposure during bear markets** due to hard_floor=0.40.

## Next Direction: v8

Revert to v2's successful architecture and enhance drawdown control via reward:
1. **hard_floor = 0.20** — restore v2 flexibility
2. **hidden_sizes = [128, 64]** — avoid overfitting
3. **position_cap = v2 settings** — looser, proven caps
4. **Much higher drawdown_penalty** (0.75) — make step-level penalty bite
5. **Higher downside + risk_budget penalties** — reinforce
6. Keep alpha_scale = 3.0 (v2 level) — higher sensitivity to negative alpha forces faster exits

If v8 still fails to break 50% maxdd, next step is **code-level reward redesign**: episode-level max drawdown penalty or portfolio-path-dependent cap.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
