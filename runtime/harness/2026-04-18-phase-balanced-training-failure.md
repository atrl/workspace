---
date: 2026-04-18
title: 2026-04-18 phase-balanced training failure
kind: failure
profiles:
- balanced
- balanced_phase_sampling_v1
- balanced_phase_sampling_v1_vs_balanced
- balanced_phase_sampling_v1_vs_reward_guard
status: dropped
tags:
- balanced
- balanced_phase_sampling_v1
- balanced_phase_sampling_v1_vs_balanced
- balanced_phase_sampling_v1_vs_reward_guard
- checkpoint-selection
- dropped
- eval-2022
- failure
- stress-regime
artifacts:
- artifacts/models/soxl_balanced_phase_sampling_v1_summary.json
- artifacts/models/soxl_balanced_reward_guard_summary.json
- artifacts/models/soxl_balanced_summary.json
- runtime/post_train_review/balanced_phase_sampling_v1.md
- runtime/post_train_review/balanced_phase_sampling_v1_vs_balanced.md
- runtime/post_train_review/balanced_phase_sampling_v1_vs_reward_guard.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: A training-procedure experiment widened episode coverage with overlapping
  windows plus phase-balanced oversampling for stress, rebound, and bull-trend regimes.
  The broader sampling materially changed training (`36` episodes, best checkpoint
  at `iter 20`), but it did not repair post-2022 excess. `2025` still lagged buy-and-hold
  by `-10.88%`, `2023-2024` degraded on both return and drawdown, and `2022` broke
  badly (`47.50%` maxDD, `0.364` avg_pos). Keep the harness support, but drop the
  branch and do not assume broader bull sampling alone fixes the style gap.
---

# 2026-04-18 phase-balanced training failure

## Context

The user called out that the problem is not only `2022`: excess in `2023`, `2024`, and `2025` has also been weak. This loop tested whether the training procedure itself was too stress-heavy and under-covering bull / restart regimes.

## What changed

- Added rolling overlapping episodes via `episode_stride`
- Added phase-balanced sampling for:
  - stress windows
  - rebound / restart windows
  - bull-trend windows
- Kept canonical `balanced` reward, cap, and actor structure fixed

## Commands

- `qlib_venv/bin/python scripts/check_harness.py`
- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_phase_sampling_v1 --skip-data-refresh --seed 20260411`
- `qlib_venv/bin/python scripts/checkpoint_review.py --summary artifacts/models/soxl_balanced_phase_sampling_v1_summary.json --output runtime/post_train_review/balanced_phase_sampling_v1.md`
- `qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py --latest artifacts/models/soxl_balanced_phase_sampling_v1_summary.json --previous artifacts/models/soxl_balanced_summary.json --output runtime/post_train_review/balanced_phase_sampling_v1_vs_balanced.md`
- `qlib_venv/bin/python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py --latest artifacts/models/soxl_balanced_phase_sampling_v1_summary.json --previous artifacts/models/soxl_balanced_reward_guard_summary.json --output runtime/post_train_review/balanced_phase_sampling_v1_vs_reward_guard.md`
- `qlib_venv/bin/python scripts/diagnosis_phase1_phase2.py`

## Evidence

- Training coverage changed materially:
  - initial state collection size increased to `36`
  - best validation checkpoint moved to `iter 20`
- Fixed split:
  - `test_2022 = -40.67% / maxDD 47.50% / avg_pos 0.364`
  - `test_2023_2024 = 41.67% / maxDD 41.66% / avg_pos 0.572`
  - `test_proxy = 306.89% / Sharpe 0.705 / maxDD 63.94%`
- Annual diagnosis from the latest replay:
  - `2025 = 35.98%` vs `46.86%` buy-and-hold, excess `-10.88%`
  - worst quarters still include:
    - `2025-Q2 = -35.59%`
    - `2025-Q3 = -22.45%`

## Interpretation

- The training procedure was indeed too narrow before this loop.
- But broadening the sample mix alone did not solve the post-2022 style gap.
- The actor mostly learned a higher-exposure style:
  - `2020` improved
  - `2022` got much worse
  - `2023-2024` did not improve enough to justify the extra exposure
  - `2025` remained meaningfully behind buy-and-hold

## Keep / Drop

- Keep:
  - phase-balanced sampling controls
  - rolling episode support
  - annual quarter diagnosis for `2025`-style questions
- Drop:
  - `balanced_phase_sampling_v1` as a production candidate
  - the idea that wider bull/rebound coverage by itself is the missing fix

## Next action

If training-method changes continue, combine broader coverage with stronger checkpoint discipline around the early `iter 20` peak and a mechanism that preserves trend core exposure without letting full-window average position drift upward everywhere.
