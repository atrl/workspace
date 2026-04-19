---
date: 2026-04-19
title: 2026-04-19 balanced residual-over-rule full run
kind: decision
profiles:
- balanced
- balanced_rule_residual_v1
- balanced_rule_residual_v1_detailed_trades
- balanced_rule_residual_v1_diagnosis
- balanced_rule_residual_v1_trade_quality_audit
- smoke
- soxl
status: kept
tags:
- balanced
- balanced_rule_residual_v1
- balanced_rule_residual_v1_detailed_trades
- balanced_rule_residual_v1_diagnosis
- balanced_rule_residual_v1_trade_quality_audit
- checkpoint-selection
- decision
- eval-2022
- kept
- smoke
- soxl
- walk-forward
artifacts:
- runtime/post_train_review/balanced_rule_residual_v1_detailed_trades.csv
- runtime/post_train_review/balanced_rule_residual_v1_diagnosis/phase1_phase2_combined.json
- runtime/post_train_review/balanced_rule_residual_v1_trade_quality_audit.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Full `balanced_rule_residual_v1` training completed and selected the `@60`
  checkpoint over `@80`. The branch materially improved trade-quality versus the old
  negative-Kelly diagnosis, including positive `micro_<5%`, `small_5_15%`, and `mid_3_10d`
  buckets, while keeping `2022` drawdown/avg-pos controlled and lifting proxy Sharpe
  to `0.96`. It is strong enough to keep, but still needs walk-forward evidence before
  any promotion claim.
---

# 2026-04-19 balanced residual-over-rule full run

## Context

The user asked for a full training run rather than another smoke-only or method-only answer.

## Evidence

- Candidate experiment package:
  - `experiments/candidate/balanced_rule_residual_v1/`
- Registry row added:
  - `configs/soxl/profile_registry.yaml`
- Full training command:
  - `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v1 --skip-data-refresh --seed 20260419`
- Post-train evidence chain:
  - `runtime/post_train_review/balanced_rule_residual_v1_detailed_trades.csv`
  - `runtime/post_train_review/balanced_rule_residual_v1_diagnosis/phase1_phase2_combined.json`
  - `runtime/post_train_review/balanced_rule_residual_v1_trade_quality_audit.md`

## Results

- Staged checkpoint selection:
  - `@20 = 50.2% / Sharpe 1.47 / MaxDD 27.7%`
  - `@40 = 59.1% / Sharpe 1.61 / MaxDD 29.3%`
  - `@60 = 61.8% / Sharpe 1.69 / MaxDD 26.9%`
  - `@80 = 54.9% / Sharpe 1.59 / MaxDD 26.2%`
  - selected checkpoint:
    - `@60`
- Full summary:
  - `valid = 61.80% / Sharpe 1.689 / MaxDD 26.90%`
  - `test_2022 = -18.30% / Sharpe -0.950 / MaxDD 24.82% / avg_pos 0.195`
  - `test_2023_2024 = 52.14% / Sharpe 0.838 / MaxDD 27.31% / avg_pos 0.400`
  - `test_proxy = 672.33% / Sharpe 0.960 / MaxDD 46.42% / avg_pos 0.483`
  - fixed-split promotion gate:
    - `pass`
  - walk-forward gate:
    - `missing`
- Trade-quality audit:
  - `mid_3_10d = Kelly 0.3210 / Expectancy 4.33%`
  - `small_5_15% = Kelly 0.1794 / Expectancy 2.32%`
  - `micro_<5% = 48 trades / Kelly 0.2921 / Expectancy 4.49%`
  - total trade actions `318`
  - rough friction estimate `31.8%`

## Decision

- Keep the residual-over-rule balanced branch.
- Keep staged checkpoint selection; `@80` was worse than `@60`.
- Do not promote yet; walk-forward evidence is still missing.

## Next action

- Run walk-forward on `balanced_rule_residual_v1` before any replacement discussion.
- If the next loop is execution-side only, target friction reduction rather than expectancy repair because the core trade-quality buckets are already positive.
