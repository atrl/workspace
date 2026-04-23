---
date: '2026-04-22'
title: "Multi-Asset Joint Training with Refined_v3 Reward \u2014 Failure Analysis"
kind: failure
profiles:
- balanced_rule_residual_refined_v3
- soxl
status: reference
tags:
- balanced_rule_residual_refined_v3
- eval-2022
- execution-overlay
- failure
- reference
- soxl
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Multi-asset joint training with refined_v3 reward produced 388.6% return
  vs 637.8% single-asset (-249pp). The refined reward was tuned for SOXL's specific
  volatility profile and does not generalize to TQQQ/UPRO/SPXL/TECL. All test slices
  degraded. Multi-asset with a single shared actor and simple one-hot embedding is
  not viable with asset-specific reward tuning.
---

# Multi-Asset Joint Training with Refined_v3 Reward — Failure Analysis

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Experiment

- **Profile**: `balanced_rule_residual_refined_v3`
- **Flag**: `--multi-asset`
- **Assets**: SOXL, TQQQ, UPRO, SPXL, TECL (5 ETFs, 67 pooled episodes)
- **Seed**: 12345
- **Command**: `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_refined_v3 --multi-asset --skip-data-refresh --seed 12345`

## Results

| Metric | Single-Asset refined_v3 | Multi-Asset refined_v3 | Delta |
|--------|------------------------|------------------------|-------|
| proxy return | 637.8% | **388.6%** | -249pp |
| proxy sharpe | 0.89 | **0.78** | -0.11 |
| proxy MaxDD | 54.9% | **49.7%** | -5.2pp |
| test_2020 return | ~164% | **122.5%** | -42pp |
| test_2022 return | ~-16% | **-24.0%** | -8pp |
| test_2023_24 return | ~63% | **47.6%** | -15pp |
| trades | ~325 | **326** | similar |

## Root Cause Analysis

1. **Reward overfitting to SOXL volatility profile**
   - refined_v3 parameters (hard_floor=0.15, opportunity_cost=0.05, trend_alignment=0.30, drawdown_penalty=0.35) were tuned specifically for SOXL's 3x-leveraged semiconductor behavior
   - TQQQ (tech) and UPRO/SPXL (broad market) have different volatility clustering, gap behavior, and trend persistence
   - The model learns to be conservative across ALL assets to avoid SOXL-style drawdowns, killing returns on assets with different risk/reward profiles

2. **One-hot embedding is insufficient cross-asset signal**
   - Current architecture adds only a 5-dim one-hot vector to the observation
   - No asset-specific network heads, no sector embeddings, no volatility-scale normalization
   - The shared actor cannot learn distinct policies for different asset characteristics

3. **Previous multi-asset with simpler reward performed better**
   - 2026-04-14 multi-asset with P0 baseline reward: 514% return / 58.3% MaxDD
   - Current multi-asset with refined_v3: 388.6% return / 49.7% MaxDD
   - A simpler, more generic reward generalized BETTER than a SOXL-tuned one

## Comparison with Historical Multi-Asset

| Run | Reward | proxy return | proxy MaxDD |
|-----|--------|-------------|------------|
| 2026-04-14 P0 baseline | generic | 514.0% | 58.3% |
| 2026-04-22 refined_v3 | SOXL-tuned | 388.6% | 49.7% |

## Decision

- **Drop** multi-asset joint training with refined rewards
- **Keep** single-asset refined_v3 as the better variant
- If multi-asset is revisited, requirements:
  1. Asset-specific reward parameters OR generic reward
  2. Asset-specific network heads or sector embeddings
  3. Volatility-scale normalization per asset
  4. Much larger network capacity

## Next Actions

1. Integrate regime-conditional budget overlay (rt0.6_b0.3) into production runtime
2. Seed sweep on v16 (GRU, window=5, penalty=2.0) to close gap to v19 return
3. Do NOT run more multi-asset experiments without architectural changes

## Retrieval tags

- multi-asset-failure
- reward-overfitting
- cross-asset-generalization
- refined_v3
- soxl-only

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
