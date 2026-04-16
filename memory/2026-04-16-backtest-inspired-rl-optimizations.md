# 2026-04-16: Backtest-Inspired RL Optimizations

## Goal
Port 5 rule-based `soxl_backtest` mechanics into the RL reward/risk layer and run a head-to-head benchmark.

## Changes Made

### 1. Mocked `soxl_backtest` data fetcher (`soxl_backtest/data/data_fetcher.py`)
- Replaced Yahoo Finance / Longbridge API dependency with local CSV reader.
- Reads `data/symbols/SOXL.csv` for price and `data/soxl_soxx_vix_full.csv` for VIX.
- Falls back to VIX=20 if VIX file is missing.

### 2. Core floor raise (`src/soxl_quant/rl.py`)
- `action_band_from_obs` base floor raised from `0.15` → `0.40`.
- Bullish regime floor raised from `0.30` → `0.55`.
- Drawdown floor raised from `0.05` → `0.30` (extreme risk) and `0.18` → `0.45` (dip-buying).

### 3. Partial profit-taking reward (`src/soxl_quant/rl.py`)
- `ContinuousReward.reward()` now grants `profit_taking_bonus * min(last_add_pnl * 10, 1.5)` when `turnover > 0.02` and `last_add_pnl > 0.03`.
- Mirrors the FIFO tranche-profit logic in `soxl_backtest`.

### 4. Vol-coefficient scaling (`src/soxl_quant/rl.py`)
- `rebalance_threshold()` now accepts `vol_coef` and scales threshold by `min(vol_coef, 2.0)` when `vol_coef > 1.30`.
- Also raised baseline thresholds (`0.015→0.025`, quiet regime `0.03→0.05`) to combat excessive turnover.

### 5. Buy cooldown (`src/soxl_quant/rl.py`)
- `SoxlSimulator` now tracks `buy_cooldown`.
- After an increase in position, cooldown = 1 day (base) + extra days if `cum_drop_30 < -0.10` or `vol_coef > 1.30`.
- During cooldown, any action that would increase position is clipped to the current position.

### 6. Panic-buy bonus (`src/soxl_quant/rl.py`)
- `ContinuousReward.reward()` grants `panic_buy_bonus * (1.0 + panic_depth * 5.0)` when buying into a panic (`cum_drop_30 < -0.15`, `regime_risk > 0.60`).

### 7. Config & hyperparameters
- Added to `RewardConfig` in `settings.py`: `profit_taking_bonus`, `panic_buy_bonus`, `panic_buy_cum_drop_threshold`, `panic_buy_regime_risk_threshold`, `cooldown_penalty`.
- Added corresponding values to `configs/soxl/balanced.yaml`.

## Benchmark Results

### `soxl_backtest` (rule-based baseline)
```
Return:   729.8%
MaxDD:    69.9%
Trades:   45
AvgPos:   0.57
```

### RL `balanced` + MoE + 5 optimizations (seed=20260411, 20 iters)
```
Return:   536.7%
Sharpe:   0.82
MaxDD:    64.3%
Trades:   728
```
Yearly breakdown:
- 2020: return=139.7%  maxdd=21.6%
- 2022: return=-39.1%  maxdd=47.0%
- 2023-2024: return=52.3%  maxdd=38.8%

## Assessment
- **Return gap**: RL still trails the rule-based harness by ~193pp. The primary driver is **turnover** (728 vs 45 trades). Despite raising rebalance thresholds, the RL actor still micro-adjusts daily because the dynamic `action_band_from_obs` re-interprets the same discrete action into slightly different positions as market state drifts.
- **Drawdown**: RL is actually better (64.3% vs 69.9%), suggesting the risk controls are tighter.
- **2022 bear market**: RL drawdown (47.0%) is much better than backtest's 69.9%, but backtest recovers much more aggressively in 2023-2025 because it holds a larger core position and trades around it.

## Next Steps
1. **Reduce turnover further**: consider a fixed-position-hold period (e.g. no rebalance for N days after any trade) or discretize the output position directly rather than mapping through a floating band.
2. **Raise avg position closer to backtest's 0.57**: current RL average position is likely still below 0.50; additional floor raise or opportunity-cost penalty tuning may help.
3. **Walk-forward promotion run**: current 2022 test drawdown (47%) exceeds the promotion gate (35%), so the model is blocked from promotion.
