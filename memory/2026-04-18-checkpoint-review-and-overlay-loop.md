---
date: 2026-04-18
title: 2026-04-18 checkpoint review and overlay loop
kind: review
profiles:
- balanced
- balanced_checkpoint_review
- balanced_checkpoint_review_vs_balanced
- balanced_position_cap_counterfactual
- balanced_position_cap_counterfactual_ranking
- balanced_reward_guard
- balanced_stress_select
- balanced_stress_select_v2
- soxl
status: reference
tags:
- balanced
- balanced_checkpoint_review
- balanced_checkpoint_review_vs_balanced
- balanced_position_cap_counterfactual
- balanced_position_cap_counterfactual_ranking
- balanced_reward_guard
- balanced_stress_select
- balanced_stress_select_v2
- checkpoint-selection
- eval-2022
- execution-overlay
- reference
- review
- soxl
- stress-regime
- transition-features
artifacts:
- artifacts/models/soxl_balanced_actor.pt
- artifacts/models/soxl_balanced_checkpoint_review_summary.json
- artifacts/models/soxl_balanced_summary.json
- runtime/model/reviews/balanced_checkpoint_review.md
- runtime/model/reviews/balanced_checkpoint_review_vs_balanced.md
- runtime/model/reviews/balanced_position_cap_counterfactual.json
- runtime/model/reviews/balanced_position_cap_counterfactual_ranking.md
workpad: memory/index.json
summary: Staged checkpoint review proved the best validation checkpoint appears at
  iter 40 rather than iter 80, but the selected actor still failed fixed-split 2022
  gates. Fixed-actor position-cap sweeps improved 2022 avg_pos only marginally and
  never passed the full gate. A follow-up single-layer retrain that raised `stress_avg_pos_penalty`
  from `10` to `25` also failed, with `test_2022 maxDD` still at `46.28%` and `avg_pos`
  at `0.398`. A reward-only `balanced_reward_guard` retrain also failed, worsening
  `test_2022` to `47.36% maxDD / 0.412 avg_pos`, so the next loop should stop tuning
  selector penalties and reward coefficients and instead change the stress set or
  episode sampling itself.
---

# 2026-04-18 checkpoint review and overlay loop

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Goal

Compress the search space before starting another broad retrain by answering two questions:

1. is the current bottleneck partly just checkpoint-selection bias?
2. can fixed-actor execution / cap overlays already recover the `2022` gate?

## Commands

- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_checkpoint_review --skip-data-refresh --seed 20260411`
- `qlib_venv/bin/python scripts/checkpoint_review.py --summary artifacts/models/soxl_balanced_checkpoint_review_summary.json --output runtime/model/reviews/balanced_checkpoint_review.md`
- `qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py --latest artifacts/models/soxl_balanced_checkpoint_review_summary.json --previous artifacts/models/soxl_balanced_summary.json --output runtime/model/reviews/balanced_checkpoint_review_vs_balanced.md`
- `qlib_venv/bin/python scripts/position_cap_counterfactual.py --profile balanced --model artifacts/models/soxl_balanced_actor.pt --variants-json configs/soxl/position_cap_overlay_variants.json --output runtime/model/reviews/balanced_position_cap_counterfactual.json`
- `qlib_venv/bin/python scripts/rank_counterfactual_variants.py --input runtime/model/reviews/balanced_position_cap_counterfactual.json --output runtime/model/reviews/balanced_position_cap_counterfactual_ranking.md`

## Evidence

- Checkpoint review:
  - best validation checkpoint moved to `iter 40`
  - score path: `68.08 -> 76.99 -> 80.60 -> 79.02 -> 74.86`
  - selected checkpoint fixed-split:
    - `test_2022 = -39.54% / maxDD 46.30% / avg_pos 0.372`
    - `test_proxy Sharpe = 0.765`
  - result:
    - checkpoint choice matters
    - but it does not rescue `2022`

- Fixed-actor cap sweep:
  - best stress-side reduction on `2022 avg_pos` was only to `0.353-0.354`
  - best `2022 max_dd` stayed at `43.55%`
  - best proxy Sharpe variant (`bear_floor35_regime070`) still failed both `2022` checks
  - result:
    - overlays are a partial lever, not a sufficient fix

- Stronger selection-side stress penalty retrain:
  - `balanced_stress_select_v2` changed only `selection_score.stress_avg_pos_penalty`:
    - `10.0 -> 25.0`
  - selection path:
    - `59.03 -> 65.20 -> 67.03 -> 56.78 -> 71.11`
  - selected checkpoint still came from `iter 80`
  - fixed-split:
    - `test_2022 = -34.68% / maxDD 46.28% / avg_pos 0.398`
    - `test_proxy Sharpe = 0.817`
  - comparison:
    - slightly better than `balanced_stress_select`
    - still clearly worse than canonical `balanced`
  - result:
    - selection-only penalty tuning did not recover the `2022` gate
    - stress exposure measured on the current validation slice is too weak as the main selector

- Reward-only guard retrain:
  - `balanced_reward_guard` changed only reward coefficients while keeping cap and selector at canonical `balanced`
  - validation:
    - `55.81% / Sharpe 1.552 / MaxDD 29.36% / avg_pos 0.604`
  - fixed-split:
    - `test_2022 = -36.00% / maxDD 47.36% / avg_pos 0.412`
    - `test_proxy Sharpe = 0.830`
  - comparison:
    - versus canonical `balanced`, both `2022 maxDD` and `2022 avg_pos` worsened materially
    - versus `balanced_stress_select_v2`, `2022` also worsened slightly
  - result:
    - stronger reward coefficients alone did not recover crisis de-risking
    - the model still learned higher stress exposure despite a harsher objective

- Harness fix:
  - non-staged `selection_history` was misleading because it duplicated final metrics at both `iter 0` and `iter max_iters`
  - patched `src/soxl_quant/cli.py` so future non-staged runs now record:
    - true pre-train validation metrics at `iter 0`
    - true post-train metrics at `iter max_iters`
    - baseline weights restored if the untrained actor outscored the trained actor

## Keep / Drop

- Keep:
  - staged checkpoint review as a default diagnostic
  - fixed-actor overlay sweeps as a cheap first pass before feature churn
- Drop:
  - immediate return to transition-feature retrains
  - multi-layer sweeps that change feature, reward, cap, and selection together
  - further incremental tuning of `stress_avg_pos_penalty`
  - further reward-coefficient micro-tuning as the primary next lever

## Next action

If retraining continues, do not spend the next loop on another selector-penalty or reward-coefficient sweep. Change the stress set or episode construction itself and judge it first on:

- `test_2022 max_dd`
- `test_2022 avg_pos`
- `test_proxy sharpe`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
