---
date: '2026-04-20'
title: 'v8 Results: Passes Fixed-Split Promotion Gate, But Penalties Degrade Learning'
kind: failure
profiles:
- balanced_rule_residual_v2
- balanced_rule_residual_v8
status: kept
tags:
- balanced_rule_residual_v2
- balanced_rule_residual_v8
- checkpoint-selection
- eval-2022
- failure
- kept
- walk-forward
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20
  Profile: `balanced_rule_residual_v8` Baseline: `balanced_rule_residual_v2` (best
  drawdown), `v7` (previous attempt) | Metric | v2 | v5 | v7 | v8 | |--------|-----|---'
---

# v8 Results: Passes Fixed-Split Promotion Gate, But Penalties Degrade Learning

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-20
Profile: `balanced_rule_residual_v8`
Baseline: `balanced_rule_residual_v2` (best drawdown), `v7` (previous attempt)

## Results

| Metric | v2 | v5 | v7 | v8 |
|--------|-----|-----|-----|-----|
| Test Return | 521.3% | 780.7% | 548.0% | **539.3%** |
| Test Sharpe | 0.86 | 0.93 | 0.85 | 0.85 |
| Test MaxDD | **50.4%** | 55.9% | 58.4% | 53.4% |
| test_2022 maxdd | 23.6% | 32.7% | 33.9% | **31.3%** |
| test_2022 avg_pos | **0.18** | 0.27 | 0.27 | 0.266 |
| test_2022 return | -16.2% | -24.7% | -25.2% | -22.8% |
| Valid Best Score | **104.61** | 95.40 | 99.44 | **89.54** |
| Trades | 316 | 326 | 328 | 318 |

## Milestone: First Experiment to Pass All Fixed-Split Promotion Criteria

v8 is the **first experiment to pass all fixed-split promotion gate checks**:
- test_2022_max_dd=31.31% ≤ 35.0% ✓
- test_2022_avg_pos=0.266 ≤ 0.35 ✓
- test_proxy_sharpe=0.883 ≥ 0.75 ✓

Promotion blocked only because walk-forward evaluation was not run.

## Key Findings

### 1. Stronger Penalties Improved 2022 Drawdown vs v7
- v8 test_2022_max_dd = 31.3% vs v7's 33.9% (-2.6pp)
- Reverting to v2 architecture (hard_floor=0.20, [128,64]) + stronger penalties was the right direction

### 2. But Valid Score Degraded Severely
- v8 valid best score = 89.54 (lowest of all experiments)
- v2 = 104.61, v5 = 95.40, v7 = 99.44
- **The 2.7x stronger drawdown penalty (0.75) and 2x downside/risk penalties are overwhelming the alpha signal**
- Selection history shows score peaked at iter 0 (89.54) and oscillated between 68-89 across stages

### 3. Overall MaxDD Still Above v2
- v8 test MaxDD = 53.4% vs v2's 50.4%
- Step-level penalties alone are insufficient for cross-period cumulative drawdown control
- The 53.4% drawdown accumulates across 2022→2023 boundary, not within a single year

### 4. 2022 Avg_pos Remained Elevated vs v2
- v8 test_2022_avg_pos = 0.266 vs v2's 0.18
- Despite same hard_floor=0.20, v8 did not de-risk as effectively as v2
- v8 2022 return = -22.8% vs v2's -16.2%

## Root Cause

The penalty magnitudes in v8 are too high relative to the alpha reward scale. The agent is being punished so heavily for risk-taking that it cannot learn effective alpha-generating behavior. The valid score degradation proves this.

## Next Direction: v9

Revert penalties to moderate levels and lower hard_floor for deeper de-risking:
1. **drawdown_penalty: 0.45** (down from 0.75)
2. **downside_penalty: 0.20** (revert to v2)
3. **risk_budget_penalty: 0.20** (revert to v2)
4. **high_exposure_penalty: 0.08** (revert to v2)
5. **hard_floor: 0.10** (lower than v2's 0.20) — allow deeper de-risking

If v9 still fails to break 50% MaxDD, the v7 post-mortem recommendation applies: **code-level reward redesign** (episode-level max drawdown penalty).

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
