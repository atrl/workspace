---
date: '2026-04-29'
title: 'RegimeMoE v2 — Tuned Hyperparameters'
kind: experiment
status: complete
direction: network
---

# balanced_rule_residual_v19_moe_v2

## Baselines

- **Canonical baseline**: `balanced_rule_residual_v19` (GRU-64 + MLP [128,64])
- **Direct comparison baseline**: `balanced_rule_residual_v19_moe` (first MoE run, lr=3e-4, lb_coef=0.01, stage_iters=20)

## Role

`ablate` — network architecture + hyperparameter tuning. Based on v19_moe Run 1 diagnosis.

## Target Window

Fixed-split evaluation (test_proxy, test_2020, test_2022, test_2023_2024).
Walk-forward to follow if fixed-split passes gate.

## Changed Layer

`network` / `training` — MoE architecture with adjusted LR, load-balance loss, and selection frequency.

## Hypothesis

v19_moe Run 1 suffered from overtraining degradation (best score at iter 0) and possible routing collapse. Reducing LR, strengthening load-balance loss, and evaluating more frequently should stabilize training and improve regime specialization.

## Tuning Changes vs v19_moe

| Parameter | v19_moe (Run 1) | v19_moe_v2 (this) | Rationale |
|-----------|-----------------|-------------------|-----------|
| lr | 0.0003 | **0.0001** | Reduce overtraining degradation |
| moe_lb_coef | 0.01 | **0.05** | Force expert diversity, prevent routing collapse |
| stage_iters | 20 | **10** | More frequent intermediate selection |

All other parameters identical to v19_moe.

## Results

### Selection History

| iter | score | return | sharpe | max_dd |
|------|-------|--------|--------|--------|
| 0 | **81.85** | 53.9% | 1.56 | 26.9% |
| 10 | 75.42 | 47.4% | 1.55 | 25.0% |
| 20 | 71.63 | 45.5% | 1.46 | 25.1% |
| 30 | 69.50 | 44.8% | 1.39 | 26.7% |
| 40 | 78.84 | 50.9% | 1.55 | 25.7% |
| 50 | 78.35 | 50.4% | 1.55 | 25.6% |
| 60 | 78.81 | 50.7% | 1.56 | 25.6% |
| 70 | 78.59 | 50.5% | 1.56 | 25.6% |
| 80 | 78.59 | 50.5% | 1.56 | 25.6% |

**Best checkpoint**: iter 0 (warm-start initialization). Training does not improve over BC initialization.

### Fixed-Split Backtest

| Slice | Return | Sharpe | MaxDD | Avg Pos |
|-------|--------|--------|-------|---------|
| test_proxy | 642.4% | **0.919** | 55.1% | 0.484 |
| test_2020 | 103.8% | **2.260** | 20.1% | 0.518 |
| test_2022 | -25.0% | -0.637 | **31.7%** | 0.303 |
| test_2023_2024 | 43.1% | **0.664** | 35.9% | 0.492 |

### Gate Checks

- test_proxy Sharpe ≥ 0.50: **PASS** (0.919)
- test_2022 MaxDD < 55%: **PASS** (31.7%)
- test_2022 avg_pos < 0.35: **PASS** (0.303)
- Fixed-split eligible: **TRUE**
- Walk-forward: **pending**

## Diagnosis

### Overtraining Degradation: Partially Mitigated but Not Solved

- v19_moe (Run 1): score dropped 81.92 → 64.37 (-17.55 points over 80 iters)
- v19_moe_v2 (this): score dropped 81.85 → 78.59 (-3.26 points over 80 iters)

The tuning changes **significantly reduced degradation** (5.4× less score loss), but the fundamental problem remains: **best checkpoint is still iter 0**. The warm-start BC initialization is stronger than any PPO-trained checkpoint.

### Possible Root Causes

1. **Reward misalignment**: PPO reward shaping may be pushing the policy away from the high-Sharpe behavior captured by the BC warm-start.
2. **Value function collapse**: Critic may be underfitting or overfitting, causing poor advantage estimates.
3. **MoE routing noise**: Even with lb_coef=0.05, gate logits noise during training may destabilize the policy.
4. **Small batch / short episodes**: 10 episodes × 239 steps = 2,390 steps per iter. With batch_size=64, repeat=10, this is ~37 minibatch updates per iter. May be too many updates on too little data.

### Comparison vs v19_moe Run 1

| Metric | v19_moe | v19_moe_v2 | Delta |
|--------|---------|------------|-------|
| Best score | 81.92 | 81.85 | -0.07 |
| Final score (iter 80) | 64.37 | 78.59 | +14.22 |
| test_proxy Sharpe | 1.005 | 0.919 | -0.086 |
| test_2022 MaxDD | 32.57% | 31.74% | -0.83pp |
| test_2023_2024 return | 37.20% | 43.09% | +5.89pp |

## Next Steps

1. Run walk-forward validation (command below)
2. If walk-forward passes, compare vs v19 baseline for promotion decision
3. If promotion fails, consider:
   - Increasing `episode_per_iter` to reduce update-to-data ratio
   - Trying SAC instead of PPO (different optimization landscape)
   - Ablation: disable MoE and verify if overtraining persists with base MLP

## Commands

Training:
```bash
qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_moe_v2 --seed 42
```

Walk-forward (after training):
```bash
qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_moe_v2 --walk-forward --seed 42
```
