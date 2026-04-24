---
date: '2026-04-15'
title: VIX Ablation Experiment Results
kind: decision
profiles:
- balanced
- moe
status: superseded
tags:
- balanced
- decision
- eval-2022
- moe
- stress-regime
- superseded
- walk-forward
artifacts: []
workpad: memory/index.json
summary: 'Legacy memory note normalized into the repo-wide schema. Date: 2026-04-15
  Seed: 20260411 | Profile: balanced | Iters: 20 | Architecture: MoE **Baseline**:
  `balanced` + `--moe` (VIX features fully enabled) **Ablation**: `balanced` + `--moe`'
---

# VIX Ablation Experiment Results

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-15
Seed: 20260411 | Profile: balanced | Iters: 20 | Architecture: MoE

## Experiment Design
- **Baseline**: `balanced` + `--moe` (VIX features fully enabled)
- **Ablation**: `balanced` + `--moe` + `--ablate-vix` (VIX columns zeroed, `regime_risk` replaced by `regime_risk_no_vix`)

## Aggregate Results

| Variant | Valid Return | Valid Sharpe | Valid MaxDD | Valid Score | Test Return | Test Sharpe | Test MaxDD | Trades |
|---------|-------------|--------------|-------------|-------------|-------------|-------------|------------|--------|
| **Baseline (VIX on)** | 44.0% | 1.54 | 23.7% | 71.88 | 329.9% | 0.77 | 50.6% | 965 |
| **Ablation (VIX off)** | 45.1% | 1.56 | 24.0% | **73.35** | **367.7%** | **0.80** | 52.6% | 978 |

## Slice Breakdown

| Window | Metric | Baseline (VIX on) | Ablation (VIX off) | Delta |
|--------|--------|-------------------|--------------------|-------|
| **2020** | Return | 91.03% | 93.30% | +2.27pp |
| | Sharpe | 2.30 | 2.33 | +0.03 |
| | MaxDD | 17.06% | 16.83% | -0.23pp |
| | AvgPos | 0.458 | 0.458 | ~0 |
| **2022** | Return | -26.81% | -26.33% | +0.48pp |
| | Sharpe | -0.93 | -0.81 | +0.12 |
| | MaxDD | 33.48% | 35.70% | **+2.22pp** |
| | AvgPos | 0.280 | 0.301 | +0.021 |
| **2023-2024** | Return | 46.51% | 49.95% | +3.44pp |
| | Sharpe | 0.74 | 0.75 | +0.01 |
| | MaxDD | 30.03% | 30.57% | +0.54pp |
| | AvgPos | 0.462 | 0.473 | +0.011 |

## Interpretation

1. **Overall, the VIX-ablated model wins on raw returns and Sharpe**, with test return ~38pp higher.
2. **However, the 2022 bear market reveals VIX's protective value**: MaxDD increased by +2.2pp when VIX was removed (33.5% → 35.7%). VIX appears to help the model *slightly* compress drawdowns in the most severe stress window.
3. **2020 and 2023-2024 ablation is strictly better**: the price-only stress index (`vol_stress_score`) captures enough regime information to outperform the VIX-inclusive variant in both the COVID crash rebound and the recent tech rally.
4. **Conclusion**: Removing VIX is **not catastrophic**—in fact it is largely competitive—but it does trade a small amount of 2022 drawdown control for higher upside in other regimes.

## Recommendation

- **Keep both paths alive in the harness**: do not hard-delete VIX. Instead treat `--ablate-vix` as a permanent toggle.
- **For future production deployment**, run walk-forward validation on *both* variants and select the one that clears the promotion gate (especially the `test_2022_max_dd <= 35%` criterion).
- **Next step**: run walk-forward on both variants to see if the 2022 drawdown difference persists across multiple train/valid/test windows.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
