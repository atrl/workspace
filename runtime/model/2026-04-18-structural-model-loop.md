---
date: 2026-04-18
title: 2026-04-18 structural model loop
kind: failure
profiles:
- balanced
- balanced_reward_guard
- balanced_structural_v1
- balanced_structural_v1_vs_balanced
- balanced_structural_v1_vs_reward_guard
- moe
status: dropped
tags:
- balanced
- balanced_reward_guard
- balanced_structural_v1
- balanced_structural_v1_vs_balanced
- balanced_structural_v1_vs_reward_guard
- checkpoint-selection
- dropped
- eval-2022
- failure
- moe
- stress-regime
artifacts:
- runtime/model/card/soxl_balanced_reward_guard_summary.json
- runtime/model/card/soxl_balanced_structural_v1_summary.json
- runtime/model/card/soxl_balanced_summary.json
- runtime/post_train_review/balanced_structural_v1.md
- runtime/post_train_review/balanced_structural_v1_vs_balanced.md
- runtime/post_train_review/balanced_structural_v1_vs_reward_guard.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: A structural retrain with longer observation context, custom action levels,
  hard regime routing, and regime-stratified stress sampling improved stress exposure
  versus the failed reward-only branch, but it still failed promotion. `test_2022
  avg_pos` passed at `0.338`, yet `test_2022 maxDD` remained too high at `39.86%`
  and `test_proxy Sharpe` fell to `0.739`. Keep the new harness support for model/sampling
  profiles, but drop `balanced_structural_v1` as a production candidate and ablate
  one structural lever at a time if model-side work continues.
---

# 2026-04-18 structural model loop

## Context

After the checkpoint-review, selector-only, and reward-only loops all failed, the next hypothesis was that the bottleneck had shifted from coefficients to model structure. This loop tested that directly.

## What changed

- Added profile-local `model` config:
  - `observation_window`
  - custom `action_levels`
  - regime-routed MoE actor controls
- Added profile-local `sampling` config:
  - regime-stratified stress oversampling
  - persistent-stress thresholds and repeat counts
- Added persistent-stress state features:
  - `drawdown_streak_63`
  - `below_ma63_days`
  - `rolling_new_low_ratio_63`
  - `negative_gap_frequency_21`
- Added `balanced_structural_v1`

## Commands

- `qlib_venv/bin/python scripts/check_harness.py`
- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_structural_v1 --skip-data-refresh --seed 20260411`
- `qlib_venv/bin/python scripts/checkpoint_review.py --summary runtime/model/card/soxl_balanced_structural_v1_summary.json --output runtime/post_train_review/balanced_structural_v1.md`
- `qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py --latest runtime/model/card/soxl_balanced_structural_v1_summary.json --previous runtime/model/card/soxl_balanced_summary.json --output runtime/post_train_review/balanced_structural_v1_vs_balanced.md`
- `qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py --latest runtime/model/card/soxl_balanced_structural_v1_summary.json --previous runtime/model/card/soxl_balanced_reward_guard_summary.json --output runtime/post_train_review/balanced_structural_v1_vs_reward_guard.md`

## Evidence

- Harness validation:
  - `scripts/check_harness.py => passed=29 failures=0`
- Structural profile settings:
  - `observation_window = 10`
  - hard regime routing with `3` experts and `top_k = 1`
  - action grid compressed to `9` hand-picked levels
  - regime-stratified sampling enabled
- Training behavior:
  - feature count increased to `48`
  - episode count increased to `24`
  - best validation checkpoint was again the final stage:
    - `63.82 -> 54.62 -> 55.92 -> 64.94 -> 75.67`
- Fixed split:
  - `test_2020 = 101.26% / Sharpe 2.35 / MaxDD 19.98%`
  - `test_2022 = -31.27% / Sharpe -0.903 / MaxDD 39.86% / avg_pos 0.338`
  - `test_2023_2024 = 45.53% / Sharpe 0.708 / MaxDD 33.47%`
  - `test_proxy = 306.97% / Sharpe 0.739 / MaxDD 53.66%`
- Gate status:
  - passed:
    - `test_2022_avg_pos`
  - failed:
    - `test_2022_max_dd`
    - `test_proxy_sharpe`

## Interpretation

- The model-side changes were directionally useful for stress exposure control:
  - versus `balanced_reward_guard`, `2022 avg_pos` improved from `0.412` to `0.338`
  - versus `balanced_reward_guard`, `2022 maxDD` improved from `47.36%` to `39.86%`
- But they were still not good enough:
  - versus canonical `balanced`, `2022 maxDD` worsened from `33.04%` to `39.86%`
  - versus canonical `balanced`, proxy Sharpe fell from `0.858` to `0.739`
- Conclusion:
  - "模型问题" is real in the sense that selector/reward tuning was not sufficient
  - but replacing the base actor with a larger, harder-routed, longer-context structure is too blunt and degrades overall efficiency

## Keep / Drop

- Keep:
  - model/sampling profile support in the harness
  - regime-stratified episode construction as an available lever
  - structural ablation tests
- Drop:
  - `balanced_structural_v1` as a default or promotable branch
  - multi-lever structural bundles as the next optimization unit

## Next action

If model-side experimentation continues, ablate one structural lever at a time and score it first on `test_2022 maxDD`, `test_2022 avg_pos`, and `test_proxy Sharpe`.
