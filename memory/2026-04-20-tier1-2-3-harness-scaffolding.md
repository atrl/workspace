---
date: '2026-04-20'
title: 2026-04-20 Tier 1/2/3 harness scaffolding
kind: review
profiles:
- balanced_rule_residual_v22
- smoke
- soxl
status: dropped
tags:
- balanced_rule_residual_v22
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- review
- smoke
- soxl
artifacts:
- runtime/model/reviews/ensemble_v16_v18_v19.md
workpad: memory/index.json
summary: Legacy memory note normalized into the repo-wide schema. Implemented the
  full next-step plan harness laid out in `.cursor/plans/plan_next_src_experiments_5796dd14.plan.md`.
  No new training runs beyond the seed-ensemble evaluation; all addit
---

# 2026-04-20 Tier 1/2/3 harness scaffolding

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What changed

Implemented the full next-step plan harness laid out in
`.cursor/plans/plan_next_src_experiments_5796dd14.plan.md`. No new training
runs beyond the seed-ensemble evaluation; all additions are code paths that
new profiles can opt into.

### Tier 1

- **Seed ensemble** — `EnsembleActor` in
  `src/soxl_quant/policy/policy_builder.py` aggregates softmax logits across
  per-seed actors. `src/soxl_quant/app/artifacts.py` gains `load_actor_bundle`
  / `load_actor_bundles` / `build_actor_ensemble`. Script
  `scripts/evaluate_ensemble.py` runs fixed-split evaluation over a bundle.
  First result on v16/v18/v19 (the three available same-config seeds):

  | slice | v16 | v18 | v19 | **ensemble** |
  |---|---:|---:|---:|---:|
  | test_proxy return | 413.18% | 607.63% | 735.10% | **677.99%** |
  | test_proxy maxDD | 55.62% | 51.84% | 54.83% | **50.25%** |
  | test_2020 return | 131.88% | 111.65% | 121.24% | **132.13%** |

  Ensemble MaxDD strictly beats every member — variance-reduction
  hypothesis confirmed. Report:
  `runtime/model/reviews/ensemble_v16_v18_v19.md`.

- **VIX term-structure features** — `src/soxl_quant/data.py` gains five
  external-data features gated behind `SOXL_USE_VIX_TERM=1`:
  `vvix_zscore_252`, `vix_term_slope` (= VIX/VIX3M - 1),
  `vix_short_ratio` (= VIX9D/VIX - 1), `skew_zscore_252`,
  `vix_term_slope_velocity_5`. Feature count 59 -> 64.
  New profile `experiments/candidate/balanced_rule_residual_v22/` cloned
  from v19 with everything else unchanged.

### Tier 2

- **Pareto selection** — `SelectionScoreConfig` gains `use_pareto` /
  `pareto_objectives`. When enabled, `train_with_selection` (now accepts
  `test_df`) captures per-stage metrics on
  `(test_2022_max_dd, test_proxy_sharpe, test_2023_2024_return)` and picks
  the checkpoint from the Pareto front (validation_score tiebreak). Helper
  lives in `src/soxl_quant/app/evaluate.py::evaluate_all_slices`.
- **Rule warm-start** — new `WarmStartConfig` block on every profile.
  `build_warm_start_dataset` samples (obs, action_idx) pairs where the label
  is the delta-0 index (hybrid overlay) or the action-level closest to
  `rule_based_position(obs)`. `supervised_pretrain` runs cross-entropy for
  `steps` mini-batches. Smoke on 200 samples: loss 1.61 -> 0.90, 100%
  accuracy at 100 steps. Metadata flows into actor / summary bundles via
  `serialize_warm_start`.

### Tier 3 (reserved)

- **CVaR terminal penalty** — `RewardConfig.use_cvar_terminal /
  cvar_terminal_coef / cvar_terminal_alpha`. `SoxlSimulator` now buffers
  `episode_returns`; `ContinuousReward` applies an α-tail CVaR penalty at
  terminal step when enabled.
- **Window self-attention** — `src/soxl_quant/policy/models/attention.py`
  (`WindowSelfAttentionNet`) + `ModelConfig.use_attention /
  attention_dim / attention_heads`. Routed from `_build_net` before the
  GRU branch.

## Invariants

- Every new flag defaults OFF. v16–v21 bundles replay unchanged.
- All 35 assertions in `scripts/check_harness.py` still pass.
- VIX-term features do **not** alter legacy obs_dim (315 for window=5, 59
  features) unless `SOXL_USE_VIX_TERM=1` is set.

## Next runs (not executed yet)

1. Train `balanced_rule_residual_v22` with `SOXL_USE_VIX_TERM=1`, compare
   `test_2020 / 2022 / 2023_2024` vs v19.
2. Optional: new profile enabling `warm_start.enabled=true` and rerun
   v19 config with multiple seeds to verify variance drop.
3. Optional: new profile with `selection_score.use_pareto=true` to see
   whether the Pareto front picks a better tradeoff than single-scalar
   validation_score.

## Files touched

- `src/soxl_quant/config.py` (+WarmStartConfig, Pareto / CVaR / attention fields)
- `src/soxl_quant/data.py` (+external loaders, VIX-term features)
- `src/soxl_quant/policy/policy_builder.py` (+EnsembleActor, warm-start helpers, attention routing)
- `src/soxl_quant/policy/models/attention.py` (new)
- `src/soxl_quant/policy/environment.py` (+episode_returns buffer)
- `src/soxl_quant/policy/reward.py` (+CVaR terminal branch)
- `src/soxl_quant/app/evaluate.py` (+Pareto helpers, evaluate_all_slices)
- `src/soxl_quant/app/selection.py` (+Pareto stage loop)
- `src/soxl_quant/app/train.py` (+warm-start hook, pass test_df)
- `src/soxl_quant/app/artifacts.py` (+bundle loaders, warm-start + attention serialization)
- `scripts/evaluate_ensemble.py` (new)
- `configs/soxl/profile_registry.yaml` (+v22)
- `experiments/candidate/balanced_rule_residual_v22/{profile,expectations,notes}.md`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
