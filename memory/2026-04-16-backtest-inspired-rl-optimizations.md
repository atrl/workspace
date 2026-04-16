# 2026-04-16: Backtest-Inspired RL Optimizations

## Goal
Port 5 rule-based `soxl_backtest` mechanics into the RL reward/risk layer and run a head-to-head benchmark.

## Changes Made

### 1. Mocked `soxl_backtest` data fetcher (`soxl_backtest/data/data_fetcher.py`)
- Replaced Yahoo Finance / Longbridge API dependency with local CSV reader.
- Reads `data/symbols/SOXL.csv` for price and `data/soxl_soxx_vix_full.csv` for VIX.
- Falls back to VIX=20 if VIX file is missing.

### 2. Core floor raise (`src/soxl_quant/rl.py`) — **rolled back**
- Initial attempt: raised `action_band_from_obs` base floor `0.15 -> 0.40`.
- Result: **2022 MaxDD exploded to 47%** (vs historical best 32.17%).
- **Rolled back to original floor `0.15`**; hard `drawdown_position_cap` (min_cap=0.40) remains the primary drawdown control.

### 3. Partial profit-taking reward (`src/soxl_quant/rl.py`)
- `ContinuousReward.reward()` grants `profit_taking_bonus * min(last_add_pnl * 10, 1.5)` when `turnover > 0.02` and `last_add_pnl > 0.03`.
- Mirrors the FIFO tranche-profit logic in `soxl_backtest`.

### 4. Vol-coefficient scaling (`src/soxl_quant/rl.py`)
- `rebalance_threshold()` baseline restored to original `0.015 / 0.03 / 0.008`.
- Added `vol_coef` widening: when `vol_coef > 1.30`, `threshold *= min(vol_coef, 2.0)`.
- Effect: **high-vol regimes require larger state changes to trigger trades** (fewer trades in chaos), low-vol regimes remain sensitive.

### 5. Buy cooldown (`src/soxl_quant/rl.py`)
- `SoxlSimulator` tracks `buy_cooldown`.
- After increasing position: cooldown = 1 day + extra days if `cum_drop_30 < -0.10` or `vol_coef > 1.30`.
- Prevents rapid-fire dip-buying in falling markets.

### 6. Panic-buy bonus (`src/soxl_quant/rl.py`)
- Grants `panic_buy_bonus * (1.0 + panic_depth * 5.0)` when buying into panic (`cum_drop_30 < -0.15`, `regime_risk > 0.60`).
- Configured in `balanced.yaml` at `0.35`.

### 7. Config & hyperparameters
- Added to `RewardConfig` in `settings.py`: `profit_taking_bonus`, `panic_buy_bonus`, `panic_buy_cum_drop_threshold`, `panic_buy_regime_risk_threshold`, `cooldown_penalty`.
- Updated `configs/soxl/balanced.yaml`.

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
| **2026-04-11 hard-cap baseline** | **326-352%** | **0.80** | **46.7%** | **32.2%** | **908** | Best drawdown control ever; no 5 optimizations |
| V1 (floor 0.40 + wide threshold) | 536.7% | 0.82 | 64.3% | 47.0% | 728 | High floor destroyed 2022 drawdown |
| **V2 (rolled-back floor 0.15)** | **426.8%** | **0.83** | **52.7%** | **36.6%** | **1024** | **Best compromise so far**: retains 5 optimizations + better returns than historical best, but 2022 drawdown still +4.4pp above the 32.2% baseline |
| V3 (weaker panic-buy 0.15) | 390.2% | 0.80 | 52.5% | 36.5% | 1032 | Lowering panic-buy bonus hurt returns without improving drawdown |

### V2 detailed slices (current best 5-optimization configuration)
```
test_2020:      return=94.50%   sharpe=2.190   max_dd=17.98%
test_2022:      return=-29.57%  sharpe=-0.915  max_dd=36.61%
test_2023_2024: return=39.04%   sharpe=0.652   max_dd=34.21%
```

## Assessment
- **Floor raise was the fatal mistake** for drawdown. Raising `action_band_from_obs` floor from `0.15` to `0.40` prevented the model from going flat in 2022, pushing 2022 MaxDD from ~36% to 47%. It has been rolled back.
- **V2 is the sweet spot among 5-optimization variants**: higher return (+100pp) and Sharpe (+0.03) than the historical best hard-cap baseline, at the cost of only +6pp MaxDD and +4.4pp 2022 MaxDD.
- **Panic-buy bonus tuning is ineffective for drawdown**: Cutting the bonus from `0.35 -> 0.15` (V3) barely moved 2022 MaxDD (`36.61% -> 36.52%`) while costing ~37pp return. The bonus is likely helping capture rebound alpha more than it is hurting crash protection.
- **Promotion gate**: 2022 MaxDD 36.6% still exceeds the 35% gate, so the model remains blocked from promotion.

## Root-cause: why can’t we match the 32.2% 2022 MaxDD baseline?
The historical best (2026-04-11) achieved 2022 MaxDD 32.2% with an average 2022 position of **0.273**. In V2, 2022 average position is likely higher because:
1. `profit_taking_bonus` encourages selling on rebounds, but the model also re-buys later (higher trade count 1024 vs 908).
2. `panic_buy_bonus` and `cooldown` together create a "buy the dip" rhythm that keeps net exposure slightly elevated relative to the pure hard-cap baseline.

## Options to close the 4.4pp gap to 35%
| Option | Expected effect | Risk |
|---|---|---|
| Tighten `drawdown_position_cap` curve (lower `min_cap` to 0.35 or raise `full_drawdown` to 0.25) | Lower 2022 MaxDD | May also suppress 2020/2023-2024 upside |
| Remove panic-buy bonus entirely | Small drawdown improvement, but V3 showed negligible effect | Return loss |
| Shorten/strengthen cooldown | Reduce dip-buying frequency | May miss V-shaped rebounds |
| Keep current V2 code and train longer / with different seed | May naturally discover lower-exposure policy | Unpredictable |

## Current code status
- `src/soxl_quant/rl.py`: floor rolled back to `0.15`, threshold baseline restored to original `0.015/0.03/0.008`, vol_coef widening retained, profit-taking and panic-buy bonuses active, cooldown active.
- `configs/soxl/balanced.yaml`: `profit_taking_bonus=0.25`, `panic_buy_bonus=0.35`.
- All changes committed and pushed.
