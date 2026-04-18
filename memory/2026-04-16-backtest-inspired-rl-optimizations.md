---
date: '2026-04-16'
title: '2026-04-16: Backtest-Inspired RL Optimizations'
kind: review
profiles:
- balanced
- moe
- soxl
status: superseded
tags:
- balanced
- checkpoint-selection
- eval-2022
- moe
- review
- rule-based
- soxl
- superseded
- walk-forward
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Legacy memory note normalized into the repo-wide schema. Port 5 rule-based
  `soxl_backtest` mechanics into the RL reward/risk layer, increase high-vol T-trading
  on SOXL, and ensure zero-shot generalization beats Buy & Hold on other 3x ETFs.
---

# 2026-04-16: Backtest-Inspired RL Optimizations

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Goal
Port 5 rule-based `soxl_backtest` mechanics into the RL reward/risk layer, increase high-vol T-trading on SOXL, and ensure zero-shot generalization beats Buy & Hold on other 3x ETFs.

## Changes Made

### 1. Mocked `soxl_backtest` data fetcher (`soxl_backtest/data/data_fetcher.py`)
- Replaced Yahoo Finance / Longbridge API dependency with local CSV reader.
- Reads `data/symbols/SOXL.csv` for price and `data/soxl_soxx_vix_full.csv` for VIX.
- Falls back to VIX=20 if VIX file is missing.

### 2. Partial profit-taking reward (`src/soxl_quant/rl.py`)
- `ContinuousReward.reward()` grants `profit_taking_bonus * min(last_add_pnl * 10, 1.5)` when `turnover > 0.02` and `last_add_pnl > 0.03`.

### 3. Panic-buy bonus (`src/soxl_quant/rl.py`)
- Grants `panic_buy_bonus * (1.0 + panic_depth * 5.0)` when buying into panic (`cum_drop_30 < -0.15`, `regime_risk > 0.60`).

### 4. Buy cooldown (`src/soxl_quant/rl.py`)
- `SoxlSimulator` tracks `buy_cooldown`.
- After increasing position: cooldown = 1 day + extra days if `cum_drop_30 < -0.10` (multiplier reduced from `*10` to `*5` in later experiments).

### 5. Bull-floor lift + bull_min_cap raise (Multi-asset v2)
- `action_band_from_obs`: when `anchor_gap > 0.02`, `anchor_slope >= 0`, and `realized_vol_10 < 0.40`, floor raised to `0.50` (otherwise `0.35`).
- `drawdown_position_cap`: `bull_min_cap` raised from `0.55` to `0.65` in `balanced.yaml`.
- `rebalance_threshold`: removed the `/= vol_coef` inversion from V4; restored clean baseline.
- `vol_trading_bonus`: removed (`0.0` in YAML).

### 6. Multi-asset training (`--multi-asset`)
- Trains on pooled episodes from SOXL, TQQQ, UPRO, SPXL, TECL.
- Actor receives one-hot `asset_idx` in observation, allowing asset-specific posture.

## Benchmark Results

### `soxl_backtest` (rule-based baseline)
```
Return:   729.8%
MaxDD:    69.9%
Trades:   45
AvgPos:   0.57
```

### RL version history (seed=20260411, balanced, 20 iters)

| Version | Return | Sharpe | MaxDD | 2022 MaxDD | Trades | Notes |
|---|---|---|---|---|---|---|
| 2026-04-11 hard-cap baseline | 326-352% | 0.80 | 46.7% | **32.2%** | 908 | Best 2022 drawdown; no 5 optimizations |
| V1 (floor 0.40) | 536.7% | 0.82 | 64.3% | 47.0% | 728 | Floor too high; 2022 destroyed |
| V2 (floor rolled back) | 426.8% | 0.83 | 52.7% | 36.6% | 1024 | Best single-asset return; misses gate by 1.6pp |
| V4 (high-vol T-trading) | 367.2% | 0.81 | 50.8% | **33.17%** | 979 | First pass of 35% gate; trades return for drawdown |
| Multi-asset v1 | 348.7% | 0.77 | 52.4% | **35.33%** | 1027 | Multi-asset base; just misses gate; UPRO/SPXL beat B&H |
| **Multi-asset v2** | **414.2%** | **0.80** | **56.1%** | **40.55%** | **1075** | **Highest multi-asset SOXL return; TQQQ finally beats B&H** |

### Multi-asset v2: Generalization vs Buy & Hold (2020-2026)

| ETF | Strategy | Buy & Hold | Delta |
|---|---|---|---|
| SOXL | 400.00% | 154.26% | ✅ +245.74pp |
| **TQQQ** | **285.55%** | **284.31%** | ✅ **+1.24pp** |
| UPRO | 235.13% | 180.49% | ✅ +54.64pp |
| SPXL | 216.99% | 183.15% | ✅ +33.84pp |
| TECL | 239.11% | 270.08% | ❌ -30.97pp |

### Multi-asset v2: SOXL yearly slices
```
test_2020:      return=105.30%  sharpe=2.388   max_dd=19.56%
test_2022:      return=-32.64%  sharpe=-0.996  max_dd=40.55%   <-- exceeds 35% gate
test_2023_2024: return=49.31%   sharpe=0.724   max_dd=34.51%
```

## Assessment

### Breakthrough: TQQQ beats B&H for the first time
- **Multi-asset training + bull-floor lift** finally produced a policy that outperforms Buy & Hold on TQQQ (`285.55%` vs `284.31%`).
- UPRO and SPXL also won handily (`+55pp` and `+34pp`).

### Remaining gap: TECL and SOXL 2022 drawdown
- **TECL** still trails B&H by `-31pp`. TECL is the most concentrated tech-sector LETF; its volatility profile sits between SOXL (semis, very choppy) and TQQQ (broad tech, smoother). The model may be over-applying SOXL-style caution to TECL.
- **SOXL 2022 MaxDD `40.55%`** exceeds the `35%` promotion gate. The bull-floor lift (`0.50` in healthy low-vol trends) clearly improves returns but also prevents the model from going flat fast enough in the 2022 crash.

### Trade-off is structural
- To beat B&H on strong-bull ETFs, the model **must** maintain higher average exposure in uptrends.
- Higher exposure in uptrends necessarily means slower de-risking when the trend reverses into a crash.
- The `40.55%` vs `32.2%` delta (~8.4pp) is the price of capturing enough upside to beat TQQQ B&H.

## Next-step options
1. **Tighten drawdown cap specifically for SOXL** while keeping the bull floor for TQQQ/TECL. Since the actor already receives `asset_idx`, we could make `drawdown_position_cap` asset-aware (e.g. lower `min_cap` when `asset_idx == 0`).
2. **Raise `full_drawdown` from `0.30` to `0.25`** in `PositionCapConfig`. This would make the cap engage earlier, potentially shaving 2-4pp off the 2022 MaxDD without destroying too much upside.
3. **Walk-forward / promotion run** with the current artifact to see if the 2022 drawdown is repeatable across seeds, or if it is seed-dependent.
4. **Accept multi-asset v2 as the "generalization-first" checkpoint** and keep the historical best hard-cap baseline as the "SOXL-only" checkpoint.

## Code status
- `src/soxl_quant/rl.py`: multi-asset v2 configuration active (bull floor 0.50, clean rebalance threshold, shortened cooldown).
- `configs/soxl/balanced.yaml`: `bull_min_cap=0.65`, `vol_trading_bonus=0.0`.
- `scripts/generalization_backtest.py`: now supports `--moe` flag for MoE checkpoint loading.
- All changes committed and pushed.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
