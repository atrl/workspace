---
date: '2026-04-15'
title: MoE + Signal Diversification + Reward Optimization
kind: decision
profiles:
- balanced
- balanced_moe
- moe
- smoke
- smoke_moe
status: superseded
tags:
- balanced
- balanced_moe
- checkpoint-selection
- decision
- moe
- smoke
- smoke_moe
- stress-regime
- superseded
- walk-forward
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. Date: 2026-04-15\
  \ Execute the approved \"Approach A \u2013 Surgical Upgrade\" plan to: 1. Upgrade\
  \ `RegimeMoENet` with noisy Top-1 sparse routing, load-balancing loss, and a shared\
  \ expert. 2."
---

# MoE + Signal Diversification + Reward Optimization

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-15

## Goal
Execute the approved "Approach A – Surgical Upgrade" plan to:
1. Upgrade `RegimeMoENet` with noisy Top-1 sparse routing, load-balancing loss, and a shared expert.
2. Reduce hard dependency on VIX by adding price-only stress proxies (reusing concepts from `soxl_backtest/`).
3. Optimize `ContinuousReward` with potential-based reward shaping (PBRS) and regime-gated activity bonus.

## Evidence

### 1. MoE upgrade (`src/soxl_quant/rl.py`)
- **Noisy Top-1 gating**: Gaussian noise (σ=0.1) injected into gate logits during training; deterministic at inference.
- **Load-balancing loss**: Standard Switch-Transformer auxiliary loss `α * N * Σ(f_i * P_i)` wired into a custom `SoxlPPOPolicy.learn()`.
- **Shared expert**: A 16-dim shared MLP is always active alongside the Top-1 routed expert, preventing catastrophic forgetting.
- Smoke-test validation (`--smoke --moe`) completed successfully:
  - `[train:smoke_moe] seed=20260411 episodes=4 iters=2 epi/iter=2`
  - `[valid:smoke@2] return=75.7% sharpe=1.56 maxdd=34.7% score=102.85`
  - `[test:smoke] return=743.3% sharpe=0.84 maxdd=71.6% trades=812`

### 2. VIX-free signal layer (`src/soxl_quant/data.py`)
Added 6 new price/volume-only features:
- `rsi_6` — fast-reversion gauge (ported from `soxl_backtest/indicators/indicator_calculator.py`)
- `realized_kurtosis_21` — tail-risk proxy (arXiv 2510.03236)
- `jump_variation_5` — HAR-style jump variation from daily returns
- `relative_volume_zscore` — volume-shock concept from `EnhancedPanicCalculator`
- `vol_stress_score` — composite stress index (realized-vol z-score + drawdown + intraday range + volume shock + kurtosis), normalized to percentile scale
- `regime_risk_no_vix` — clone of `regime_risk` with VIX percentile replaced by `vol_stress_score`

These features increase `FEATURE_COLUMNS` from 38 → 44. The smoke test confirms the pipeline builds the 44-feature matrix without errors.

### 3. Reward optimization (`src/soxl_quant/rl.py`)
- **PBRS**: Potential function `Φ = -drawdown * γ_pbrs`. The step reward adds `γ_pbrs * (drawdown_{t-1} - drawdown_t)`. This is a proper potential difference and therefore preserves the optimal policy (Ng et al. 1999) while accelerating drawdown-sensitive learning.
- **Regime-gated activity bonus**: When `gated_activity_bonus=True`, turnover is only rewarded if `regime_delta > threshold` or `realized_vol > threshold`. This reduces whipsaw costs in calm markets.
- Configurable via `RewardConfig`:
  - `pbrs_coef` (default 0.0)
  - `gated_activity_bonus` (default False)
  - `activity_gate_vol_threshold` (default 0.55)
  - `activity_gate_regime_delta_threshold` (default 0.08)
- `balanced.yaml` enables both (`pbrs_coef=0.15`, `gated_activity_bonus=true`); `aggressive.yaml` keeps them off to preserve existing behavior.

### 4. Tests (`tests/test_reward.py`)
Added 3 new test cases (all passing):
- `test_pbrs_rewards_improving_drawdown` — verifies PBRS gives positive bump when drawdown improves
- `test_pbrs_zero_when_drawdown_unchanged` — verifies PBRS delta is zero when drawdown is flat
- `test_gated_activity_bonus_zero_in_calm_regime` — verifies activity bonus is gated correctly

## Next Steps
1. Run a full `balanced` + `--moe` training (non-smoke) to establish a new baseline with the upgraded architecture.
2. Run a VIX-ablation experiment: zero-out VIX columns and substitute `regime_risk_no_vix` to quantify the price-only signal quality.
3. Compare walk-forward gates against the previous `balanced_moe` checkpoint (if any) to ensure we did not regress.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
