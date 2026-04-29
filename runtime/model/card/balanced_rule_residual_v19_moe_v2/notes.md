---
date: '2026-04-29'
title: 'RegimeMoE v2 — Tuned Hyperparameters'
kind: experiment
status: active
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

## Acceptance Criteria

Same as v19_moe:
- test_proxy Sharpe ≥ 0.50
- test_2022 MaxDD < 55%
- test_2022 avg_pos < 0.35
- vs v19 proxy Sharpe not worse by > 0.15

## Commands

Training:
```bash
qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_moe_v2 --seed 42
```

Walk-forward (after training):
```bash
qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_moe_v2 --walk-forward --seed 42
```
