---
date: '2026-04-29'
title: 'RegimeMoE Network — First Full SOXL Training'
kind: experiment
status: active
direction: network
---

# balanced_rule_residual_v19_moe

## Baselines

- **Canonical baseline**: `balanced_rule_residual_v19` (GRU-64 + MLP [128,64])
- **Direct comparison baseline**: `balanced_rule_residual_v19` (same reward, position cap, execution, seed)

## Role

`ablate` — network architecture change only; everything else held constant vs v19.

## Target Window

Fixed-split evaluation (test_proxy, test_2020, test_2022, test_2023_2024).
Walk-forward to follow if fixed-split passes gate.

## Changed Layer

`network` — replace GRU-64 + MLP [128,64] with RegimeMoENet (3 experts + top-1 routing + shared expert + load-balance loss).

## External Reference

- `src/soxl_quant/policy/models/moe.py`: `RegimeMoENet`, `SoxlPPOPolicy`
- `runtime/network/knowledge.md`: "MoE 未实际实验"
- `docs/research_knowledge/networks.md`: "MoE additions do not automatically repair drawdown"

## Hypothesis

Different market regimes (bull / bear / neutral) require qualitatively different policy behaviors. A Mixture-of-Experts architecture with regime-aware routing can learn regime-specialized sub-policies, improving generalization to the 2022 bear market compared to a single shared GRU backbone.

## Acceptance Criteria

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| test_proxy Sharpe | ≥ 0.50 | promotion gate |
| test_2022 MaxDD | < 55% | promotion gate |
| test_2022 avg_pos | < 0.35 | promotion gate |
| vs v19 proxy Sharpe | not worse by > 0.15 | direct comparison |
| vs v19 test_2022 Sharpe | improvement | aspirational |

## Expected Trade-off

- **Possible upside**: Better bear-market behavior if an expert specializes in defensive actions.
- **Expected downside**: Higher training variance due to routing instability; load-balance loss adds auxiliary objective that may conflict with task reward.
- **Capacity risk**: MoE uses simpler per-expert networks (Linear+ReLU) vs GRU+MLP; may underfit on bull-market patterns.

## Failure Mode

1. **Routing collapse**: Gate always picks one expert → equivalent to single MLP.
2. **No regime specialization**: Experts learn similar weights → no benefit over shared backbone.
3. **Walk-forward degradation**: As seen with mode-conditioned input (v3), architecture that improves fixed-split can fail walk-forward.

## Results — Run 1 (2026-04-29, seed=42)

**Status**: Training completed successfully. Fixed-split eligible, but weaker than v19 baseline.

### Training Dynamics
- **Best checkpoint**: iter 0 (score=81.92) — model degraded with continued training
- **Final checkpoint**: iter 80 (score=64.37)
- **Degradation pattern**: Score dropped from 81.9 → 80.3 → 72.8 → 77.6 → 64.4 across stages

### Slice Metrics

| Slice | Return | Sharpe | MaxDD | AvgPos | B&H Return |
|-------|--------|--------|-------|--------|------------|
| test_2020 | +101.1% | 2.22 | 20.1% | 0.52 | +355.7% |
| test_2022 | -25.4% | -0.64 | 32.6% | 0.31 | -76.6% |
| test_2023_2024 | +37.2% | 0.61 | 36.7% | 0.50 | +55.7% |
| **test_proxy** | **+857.8%** | **1.00** | **53.5%** | **0.49** | **+1800.5%** |

### Promotion Gate
- Fixed-split: **PASS** (test_2022 max_dd=32.6% < 55%, avg_pos=0.31 < 0.35, proxy_sharpe=1.00 > 0.50)
- Walk-forward: Not evaluated

### vs v19 Baseline (approximate)

| Metric | v19_moe (this) | v19 (baseline) | Δ |
|--------|----------------|----------------|---|
| test_proxy Sharpe | 1.00 | ~1.15 | -0.15 |
| test_2022 Return | -25.4% | ~-18% | -7.4pp |
| test_2023_2024 Return | +37.2% | ~+48% | -10.8pp |
| test_2020 Return | +101.1% | ~+95% | +6.1pp |

### Diagnosis

1. **Overtraining degradation**: Best score at iter 0 suggests LR=3e-4 may be too high for MoE, or staged training with 20 iters/stage is too long without intermediate selection.
2. **No regime specialization evidence**: The load-balance loss coefficient (0.01) may be too weak, or the gate may be converging to a single dominant expert.
3. **2022 underperformance**: -25.4% in test_2022 vs -18% for v19 suggests the MoE experts did not effectively learn bear-market defense.
4. **2023-2024 underperformance**: +37.2% vs ~+48% for v19 — the simpler per-expert networks may underfit bull-market patterns.

### Next Actions

1. **Reduce LR**: Try lr=1e-4 or 5e-5 to reduce overtraining degradation.
2. **Strengthen load-balance loss**: Increase from 0.01 to 0.05 to force expert diversity.
3. **Add intermediate selection**: Evaluate every 5-10 iters instead of every 20.
4. **Inspect gate routing**: Check if gate is actually using all 3 experts or collapsing to one.
5. **Run walk-forward**: Even though fixed-split is weaker, run walk-forward for completeness.

## Comparison Table Row Key

`balanced_rule_residual_v19_moe` vs `balanced_rule_residual_v19`
