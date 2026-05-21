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

## External Reference

- Mixture-of-Experts sparse routing with auxiliary load-balancing loss, used
  here as a regime-specialization diagnostic rather than a larger generic
  network.
- Closest local basis:
  `runtime/model/card/balanced_rule_residual_v19_moe/notes.md`, where the first
  RegimeMoE run degraded after the warm-start checkpoint.

## Hypothesis

v19_moe Run 1 suffered from overtraining degradation (best score at iter 0) and possible routing collapse. Reducing LR, strengthening load-balance loss, and evaluating more frequently should stabilize training and improve regime specialization.

## Tuning Changes vs v19_moe

| Parameter | v19_moe (Run 1) | v19_moe_v2 (this) | Rationale |
|-----------|-----------------|-------------------|-----------|
| lr | 0.0003 | **0.0001** | Reduce overtraining degradation |
| moe_lb_coef | 0.01 | **0.05** | Force expert diversity, prevent routing collapse |
| stage_iters | 20 | **10** | More frequent intermediate selection |

All other parameters identical to v19_moe.

## 2026-05-21 Verified Rerun

Command:

```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_moe_v2 --skip-data-refresh --seed 42
```

Evidence:

- Summary:
  `runtime/model/card/balanced_rule_residual_v19_moe_v2/soxl_balanced_rule_residual_v19_moe_v2_summary.json`
- Review:
  `runtime/model/card/balanced_rule_residual_v19_moe_v2/v19_moe_v2_backtest_review.md`
- Dashboard payload:
  `dashboard_data/versioned/balanced_rule_residual_v19_moe_v2/SOXL.json`

### Selection History

| iter | score | return | sharpe | max_dd |
|------|-------|--------|--------|--------|
| 0 | 68.70 | 44.5% | 1.37 | 27.0% |
| 10 | 68.70 | 44.0% | 1.40 | 26.7% |
| 20 | 66.31 | 43.0% | 1.34 | 29.3% |
| 30 | **75.75** | 48.4% | 1.52 | 24.9% |
| 40 | 72.13 | 45.9% | 1.46 | 25.0% |
| 50 | 70.45 | 44.9% | 1.43 | 25.4% |
| 60 | 70.47 | 44.2% | 1.47 | 25.4% |
| 70 | 68.30 | 42.5% | 1.44 | 25.0% |
| 80 | 68.47 | 42.6% | 1.44 | 25.0% |

Best checkpoint is iter 30, then later staged checkpoints decay. The lower
learning rate and stronger load-balance loss avoid the first MoE run's severe
iter-80 collapse, but they do not create a better policy than the v19 family
baseline.

### Fixed-Split Backtest

| Slice | Return | Sharpe | MaxDD | Avg Pos |
|-------|--------|--------|-------|---------|
| test_proxy | 691.0% | 0.963 | 49.0% | 0.462 |
| test_2020 | 97.1% | 2.191 | 22.3% | 0.512 |
| test_2022 | -42.7% | -1.486 | 45.1% | 0.288 |
| test_2023_2024 | 60.0% | 0.831 | 32.8% | 0.472 |

### Gate Checks

- Fixed-split gate: pass.
- Overall promotion: blocked because walk-forward is missing and the fixed-split
  comparison is not strong enough to justify paying that cost.
- Actor retention: deleted after review because registry says
  `keep_actor=false`; summary/review/dashboard payload are the durable proof.

## Diagnosis

### Comparison vs `balanced_rule_residual_v19`

- Proxy return falls from `830.0%` to `691.0%`; proxy Sharpe is nearly flat
  (`0.976 -> 0.963`) and MaxDD improves by `2.5pp`.
- `test_2020` weakens: return `123.9% -> 97.1%`, Sharpe `2.40 -> 2.19`.
- `test_2022` is the blocking regression: return `-26.2% -> -42.7%`,
  Sharpe `-0.86 -> -1.49`, MaxDD `36.4% -> 45.1%`.
- `test_2023_2024` improves: return `53.2% -> 60.0%`, Sharpe `0.73 -> 0.83`,
  MaxDD `38.2% -> 32.8%`.

### Comparison vs `balanced_rule_residual_v19_moe`

- The first MoE run had proxy Sharpe around `1.005`; this rerun is lower at
  `0.963`.
- The first MoE run had `test_2022` MaxDD around `32.6%`; this rerun worsens
  to `45.1%`.
- The main improvement is recovery: `test_2023_2024` return rises from about
  `37.2%` to `60.0%`.

### Mechanism Readout

The tuned MoE is not simply overtraining now: iter 30 beats iter 0 on the
validation score. The remaining issue is regime allocation. It sacrifices bull
capture and 2022 defense while improving the mixed recovery window. That is a
different failure than the first MoE run, but still not a promotable mechanism.

## Decision

- Keep `balanced_rule_residual_v19_moe_v2` as an `ablation`.
- Do not run walk-forward for this branch.
- Do not bind runtime or dashboard selection to this model.
- Do not continue MoE by only sweeping LR / load-balance / stage length; the
  problem is now slice-level regime allocation, not just optimizer stability.

## Next Steps

1. Prefer production/dashboard completion gates until Vercel credentials or the
   next paper signal date unlock the remaining operational proof.
2. If continuing model optimization, use a distinct mechanism such as
   actor/runtime counterfactuals or reward/selection diagnostics. Do not open
   another MoE hyperparameter-only profile.

## Commands

Training:
```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_moe_v2 --skip-data-refresh --seed 42
```

Walk-forward was intentionally not run after the verified rerun because the
fixed-split comparison did not beat the direct MoE baseline or v19 on the
stress window.
