---
date: '2026-04-22'
title: 2026-04-22 balanced_rule_residual_mode_v2 result
kind: review
profiles:
- balanced_rule_residual_mode_v1
- balanced_rule_residual_mode_v2
status: dropped
tags:
- balanced_rule_residual_mode_v1
- balanced_rule_residual_mode_v2
- dropped
- eval-2022
- review
artifacts:
- artifacts/models/soxl_balanced_rule_residual_mode_v2_summary.json
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: 'Second mode-hierarchy run after `balanced_rule_residual_mode_v1`. This loop
  had two goals: move mode behavior out of hard-coded helpers and into profile config
  test whether looser favorable modes could recover upside without losing the hier'
---

# 2026-04-22 balanced_rule_residual_mode_v2 result

## Context

Second mode-hierarchy run after `balanced_rule_residual_mode_v1`. This loop had
two goals:

- move mode behavior out of hard-coded helpers and into profile config
- test whether looser favorable modes could recover upside without losing the
  hierarchy-driven risk gains

## Evidence

- command:
  - `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_mode_v2 --skip-data-refresh --seed 12345`
- summary artifact:
  - `artifacts/models/soxl_balanced_rule_residual_mode_v2_summary.json`

## Main result

- valid: `35.8% / Sharpe 1.18 / MaxDD 26.1%`
- test_2020: `42.6% / MaxDD 22.9%`
- test_2022: `-3.6% / MaxDD 13.5% / avg_pos 0.128`
- test_2023_2024: `47.2% / MaxDD 28.7%`
- test_proxy: `693.8% / Sharpe 1.01 / MaxDD 50.4% / avg_pos 0.450`

## Comparison vs mode_v1

- `2022` improved further
- `2023_2024` return improved but DD worsened materially
- `2020` got much worse on both return and DD
- full proxy gave back a large part of `mode_v1`'s gain:
  - return `886.1% -> 693.8%`
  - Sharpe `1.09 -> 1.01`
  - MaxDD `42.3% -> 50.4%`

## Decision

- drop `balanced_rule_residual_mode_v2`
- keep the new config-owned mode behavior extension points
- keep `balanced_rule_residual_mode_v1` as the retained hierarchy frontier

## Retrieval tags

- mode hierarchy
- balanced_rule_residual_mode_v2
- dropped
- config extensibility
