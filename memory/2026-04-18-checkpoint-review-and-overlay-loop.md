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
- runtime/post_train_review/balanced_checkpoint_review.md
- runtime/post_train_review/balanced_checkpoint_review_vs_balanced.md
- runtime/post_train_review/balanced_position_cap_counterfactual.json
- runtime/post_train_review/balanced_position_cap_counterfactual_ranking.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Staged checkpoint review proved the best validation checkpoint appears at
  iter 40 rather than iter 80, but the selected actor still failed fixed-split 2022
  gates. Fixed-actor position-cap sweeps improved 2022 avg_pos only marginally and
  never passed the full gate. A follow-up single-layer retrain that raised `stress_avg_pos_penalty`
  from `10` to `25` also failed, with `test_2022 maxDD` still at `46.28%` and `avg_pos`
  at `0.398`, so the next loop should stop tuning selector penalties and instead change
  the optimization target or stress set.
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
- `qlib_venv/bin/python scripts/checkpoint_review.py --summary artifacts/models/soxl_balanced_checkpoint_review_summary.json --output runtime/post_train_review/balanced_checkpoint_review.md`
- `qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py --latest artifacts/models/soxl_balanced_checkpoint_review_summary.json --previous artifacts/models/soxl_balanced_summary.json --output runtime/post_train_review/balanced_checkpoint_review_vs_balanced.md`
- `qlib_venv/bin/python scripts/position_cap_counterfactual.py --profile balanced --model artifacts/models/soxl_balanced_actor.pt --variants-json configs/soxl/position_cap_overlay_variants.json --output runtime/post_train_review/balanced_position_cap_counterfactual.json`
- `qlib_venv/bin/python scripts/rank_counterfactual_variants.py --input runtime/post_train_review/balanced_position_cap_counterfactual.json --output runtime/post_train_review/balanced_position_cap_counterfactual_ranking.md`

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

## Keep / Drop

- Keep:
  - staged checkpoint review as a default diagnostic
  - fixed-actor overlay sweeps as a cheap first pass before feature churn
- Drop:
  - immediate return to transition-feature retrains
  - multi-layer sweeps that change feature, reward, cap, and selection together
  - further incremental tuning of `stress_avg_pos_penalty`

## Next action

If retraining continues, do not spend the next loop on another selector-penalty sweep. Change the optimization target itself and judge it first on:

- `test_2022 max_dd`
- `test_2022 avg_pos`
- `test_proxy sharpe`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
