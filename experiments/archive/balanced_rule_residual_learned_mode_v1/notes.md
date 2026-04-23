---
date: '2026-04-22'
title: balanced_rule_residual_learned_mode_v1
kind: experiment
status: candidate
baseline: balanced_rule_residual_v19
comparison_baseline: balanced_rule_residual_mode_v1
role: replace
target_window: test_proxy, test_2020, test_2022, test_2023_2024
accepted_trade_off: proxy return may drop vs v19, but proxy MaxDD should improve vs mode_v1
changed_layer: model / selection
external_reference: Jaderberg et al. 2016 UNREAL (auxiliary task RL)
---

# balanced_rule_residual_learned_mode_v1

## Hypothesis

The rule-based mode classifier in `mode_v1` is the single most effective drawdown
mechanism found so far (proxy MaxDD 32.0%), but it is brittle and kills bull-capture
(valid return 36.2% vs v19's 56.6%). A learned classifier, trained with hindsight-
optimal labels and an auxiliary forward-drawdown prediction task, can make smoother,
more accurate regime transitions — preserving the drawdown gains while recovering
return.

## Mechanism

1. Train a small MLP (`ModeClassifierNet`, 11 → 32 → 16 → 4) on historical SOXL bars.
2. Labels are hindsight-optimal modes derived from forward-window return/drawdown analysis.
3. Auxiliary task: predict forward 10-day max drawdown.
4. Replace `classify_rule_mode` with `classify_learned_mode` in `modes.py`.
5. Train a new residual-over-rule actor under the learned mode router.

## Evidence from fixed-actor overlay

Running v19 actor with different mode routers (no retraining):

| Router | proxy return | proxy Sharpe | proxy MaxDD | 2022 MaxDD | 2023-24 return |
|---|---|---|---|---|---|
| None | 673.3% | 1.01 | 46.8% | 31.1% | 43.6% |
| Rule (mode_v1) | 34.5% | 1.04 | 6.3% | 6.5% | 3.3% |
| **Learned** | **333.0%** | **0.99** | **29.7%** | **12.2%** | **43.7%** |

Key insight: learned router is **dramatically less conservative** than rule router on a
fixed actor. It achieves sub-30% MaxDD without crushing return. Rule router on a fixed
actor is pathologically conservative because v19 actor was never trained under it.

## Expected outcome after training

If the actor learns to operate inside learned-router constraints, the hope is:
- proxy MaxDD 28-32% (maintaining mode_v1-level drawdown control)
- proxy return 450-600% (recovering a large portion of the v19-mode_v1 gap)
- test_2020 and test_2023_2024 return should improve vs mode_v1

## Commands

```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py \
    --profiles balanced_rule_residual_learned_mode_v1 \
    --skip-data-refresh --seed 12345
```

## Acceptance criteria

- proxy MaxDD < 35% (better than v19's 54.8%)
- proxy return > 400% (better than mode_v1's 343.6%)
- fixed-split gates pass
- test_2022 MaxDD < 25% (better than mode_v1's 16.5% is nice but not required)
