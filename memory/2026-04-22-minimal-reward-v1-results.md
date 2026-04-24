---
date: '2026-04-22'
title: "Minimal Reward v1 Results \u2014 Hypothesis Rejected"
kind: failure
profiles:
- balanced_rule_residual_minimal_v1
status: reference
tags:
- balanced_rule_residual_minimal_v1
- checkpoint-selection
- eval-2022
- execution-overlay
- failure
- reference
- stress-regime
artifacts: []
workpad: memory/index.json
summary: "Minimal reward (5 terms) + regime budget in training produced 492.3%/0.85/48.8%\
  \ \u2014 identical to v12/v13 baseline. Training degraded from iter 0 (random init\
  \ was best). The \"extra\" reward terms (activity_bonus, profit_taking, panic_buy,\
  \ opportunity_cost) contribute ~140pp of return. Participation signals are necessary,\
  \ not noise. Budget overlay works on fixed actors, not during training."
---

# Minimal Reward v1 Results — Hypothesis Rejected

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Experiment

- **Profile**: `balanced_rule_residual_minimal_v1`
- **Seed**: 20260420
- **Hypothesis**: Strip reward from ~20 terms to 5 terms + integrate regime-conditional budget into training
- **Command**: `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_minimal_v1 --skip-data-refresh --seed 20260420`

## Results

| Metric | v16 baseline | minimal_v1 | Delta |
|--------|-------------|-----------|-------|
| proxy return | **632.4%** | **492.3%** | **-140pp** |
| proxy sharpe | **0.91** | 0.85 | -0.06 |
| proxy MaxDD | 50.1% | **48.8%** | -1.3pp |
| test_2020 ret/dd | 102.8% / 16.7% | 107.6% / 17.3% | similar |
| test_2022 ret/dd | -23.6% / 32.7% | -22.8% / 31.3% | similar |
| test_2023_24 ret/dd | **57.9%** / 33.6% | **40.2%** / 31.1% | **-17.7pp** |
| trades | 334 | 313 | -21 |

## Critical Finding: Training Degraded from Iter 0

```
selection_history:
- iter 0:  56.4% return / 1.65 sharpe / 26.6% MaxDD  ← BEST
- iter 20: 56.4% return / 1.47 sharpe / 32.6% MaxDD
- iter 40: 46.8% return / 1.33 sharpe / 30.2% MaxDD
- iter 60: 52.7% return / 1.47 sharpe / 31.1% MaxDD
- iter 80: 36.2% return / 1.12 sharpe / 33.5% MaxDD  ← WORST
```

The model became MORE conservative as training progressed. Random initialization outperformed 80 iterations of training.

## Root Cause Analysis

1. **Minimal reward removed ALL participation signals**
   - No activity_bonus, no opportunity_cost, no trend_alignment, no profit_taking, no panic_buy
   - Model had no "when to deploy capital" signal

2. **Regime budget IN training ≠ budget overlay on fixed actor**
   - Fixed overlay (v19 + rt0.6): +5pt DD improvement, ~0 return cost
   - Training with budget: model adapts around constraint, never learns aggressive behavior
   - Result: +1.3pt DD improvement, -140pp return cost

3. **The "extra" terms are not noise — they are return drivers**
   - v16 has ~17 active reward terms vs minimal_v1's 5
   - The difference is exactly 140pp of return
   - Key return drivers: activity_bonus, profit_taking, panic_buy, opportunity_cost

4. **Cash alone cannot reduce drawdown**
   - Removing forced-trading incentives made model overly conservative
   - test_2023_24 collapsed 17.7pp because model missed post-stress melt-ups
   - The problem is not "too many parameters" but "missing the right parameters"

## Correct Major Adjustment Direction

Not reward minimalization. Instead:
1. **Keep essential participation signals** (trend_alignment, opportunity_cost)
2. **Remove micro-management terms** (stale_position, trapped_loser, micro_rebalance, flat_trade)
3. **Use structural constraints for DD control** (mode hierarchy, budget overlay on fixed actor)
4. **Do NOT integrate budget into training reward** — it kills return

## Decision

- `balanced_rule_residual_minimal_v1`: **dropped**
- Minimal reward hypothesis: **rejected**
- Next: selective reward simplification (moderate reward) or seed sweep on v16

## Retrieval tags

- minimal-reward-failure
- participation-signal
- regime-budget-in-training
- training-degradation
- reward-complexity

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
