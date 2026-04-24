---
date: '2026-04-22'
title: 2026-04-22 balanced_rule_residual_mode_v1 training result
kind: review
profiles:
- balanced_rule_residual_mode_v1
- balanced_rule_residual_v19
status: reference
tags:
- balanced_rule_residual_mode_v1
- balanced_rule_residual_v19
- eval-2022
- execution-overlay
- reference
- review
- stress-regime
artifacts:
- artifacts/models/soxl_balanced_rule_residual_mode_v1_summary.json
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: First full train of the mode-hierarchy scaffold over the retained `balanced_rule_residual_v19`
  family. This was the first attempt to add persistent rule-routed modes directly
  into train/eval/backtest/runtime instead of doing another overlay
---

# 2026-04-22 balanced_rule_residual_mode_v1 training result

## Context

First full train of the mode-hierarchy scaffold over the retained
`balanced_rule_residual_v19` family. This was the first attempt to add
persistent rule-routed modes directly into train/eval/backtest/runtime instead
of doing another overlay-only parameter sweep.

## Evidence

- command:
  - `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_mode_v1 --skip-data-refresh --seed 12345`
- summary artifact:
  - `artifacts/models/soxl_balanced_rule_residual_mode_v1_summary.json`

## Main result

- valid: `36.2% / Sharpe 1.17 / MaxDD 26.1%`
- test_2020: `64.3% / MaxDD 10.6%`
- test_2022: `-13.1% / MaxDD 16.5% / avg_pos 0.135`
- test_2023_2024: `43.0% / MaxDD 20.1%`
- test_proxy: `886.1% / Sharpe 1.09 / MaxDD 42.3% / avg_pos 0.450`

## Comparison vs v19

- big bull-capture giveback:
  - valid return `56.6% -> 36.2%`
  - `test_2020` return `109.4% -> 64.3%`
- strong stress / restart improvement:
  - `test_2022` MaxDD `31.1% -> 16.5%`
  - `test_2022` avg_pos `0.261 -> 0.135`
  - `test_2023_2024` MaxDD `37.3% -> 20.1%`
- proxy quality improves on all three major axes:
  - return `735.1% -> 886.1%`
  - Sharpe `0.95 -> 1.09`
  - MaxDD `54.8% -> 42.3%`

## Decision

- keep the mode hierarchy scaffold
- keep `balanced_rule_residual_mode_v1` as a candidate, not a replacement
- next loop should recover bull participation inside favorable modes instead of
  removing the mode mechanism

## Retrieval tags

- mode hierarchy
- temporal abstraction
- balanced_rule_residual_mode_v1
- v19 comparison
