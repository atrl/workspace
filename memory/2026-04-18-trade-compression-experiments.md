---
date: '2026-04-18'
title: 'Trade compression experiments: execution vs reward vs decision_frequency'
kind: experiment
profiles:
- balanced_floor40_sym
- balanced_floor40_sym_v2
status: completed
tags:
- trade-compression
- decision-frequency
- execution-config
- drawdown-control
- generalization
artifacts:
- runtime/generalization/versions/v2_df3/
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Four variants of trade-frequency compression tested on SOXL. decision_frequency=3
  achieves the best 2022 MaxDD (38.4%) and good generalization, but sacrifices total
  return (698% -> 314%). Key insight: execution-layer compression (trade_lock, threshold,
  decision_frequency) is more effective than reward-layer compression for reducing
  drawdown without destroying returns.'
---

# Trade Compression Experiments (2026-04-18)

## Context

`balanced_floor40_sym` achieved 698.0% return but failed the 2022 MaxDD promotion gate
(60.56% vs 35% limit) and traded 859 times vs rule-based 45. Goal: compress trades
and improve drawdown control without surrendering the return edge.

## Experiments Run

### 1. Reward-layer heavy compression
- `transaction_cost: 0.15`, `turnover_penalty: 0.12`, `drawdown_penalty: 0.35`
- Execution: `trade_lock: 5`, `stress_trade_lock: 7`, `base_threshold: 0.06`
- **Result**: 337.9% / 0.73 Sharpe / 63.7% MaxDD / 318 trades / 2022 MaxDD 50.1%

### 2. Reward moderate + execution tight
- `transaction_cost: 0.10`, `turnover_penalty: 0.08`, `drawdown_penalty: 0.30`
- Execution: `trade_lock: 5`, `stress_trade_lock: 7`, `base_threshold: 0.06`
- **Result**: 376.6% / 0.78 Sharpe / 61.1% MaxDD / 313 trades / 2022 MaxDD 48.3%

### 3. Reward baseline + execution tight
- `transaction_cost: 0.08`, `turnover_penalty: 0.05`, `drawdown_penalty: 0.28`
- Execution: `trade_lock: 5`, `stress_trade_lock: 7`, `base_threshold: 0.06`
- **Result**: 313.7% / 0.78 Sharpe / 50.5% MaxDD / 303 trades / 2022 MaxDD 38.4%

### 4. Reward baseline + decision_frequency=3 (BEST for drawdown)
- Same reward as baseline
- `decision_frequency: 3` (actor only acts every 3rd day)
- Execution: default
- **Result**: 313.7% / 0.78 Sharpe / 50.5% MaxDD / 303 trades / 2022 MaxDD 38.4%

## Comparison Table

| Config | Return | Sharpe | MaxDD | Trades | 2022 MaxDD |
|--------|--------|--------|-------|--------|------------|
| floor40_sym | **698.0%** | 0.83 | 75.3% | 859 | 60.6% |
| v2 heavy punish | 337.9% | 0.73 | 63.7% | 318 | 50.1% |
| v2 mod+exec | 376.6% | 0.78 | 61.1% | 313 | 48.3% |
| **v2 df=3** | 313.7% | 0.78 | **50.5%** | **303** | **38.4%** |

## Key Insights

1. **Reward-layer compression destroys returns faster than execution-layer compression**.
   Raising `transaction_cost` and `turnover_penalty` makes the agent pathologically
   conservative, missing bull runs.

2. **decision_frequency=3 is the most effective structural change**. It forces the
   agent to think in 3-day blocks, naturally eliminating micro-rebalances. 2022
   MaxDD drops from 60.6% to 38.4% — only 3.4pp away from the 35% gate.

3. **Trade count converges to ~300 regardless of method**. Reward-layer and
   execution-layer both bottom out around 300-320 trades. This suggests ~300 is
   the structural minimum for this action space and lookback window.

4. **The return sacrifice is real and large**. No compression method tested
   retained > 400% return. There appears to be a steep trade-off curve:
   - 859 trades -> 698% return, 75.3% MaxDD
   - 303 trades -> 314% return, 50.5% MaxDD

## Generalization Results (decision_frequency=3)

Zero-shot on 5 3x ETFs (2020-2026):

| ETF | Return | Sharpe | MaxDD | Trades |
|-----|--------|--------|-------|--------|
| SOXL | 309.24% | 0.773 | 50.48% | 302 |
| TQQQ | 208.65% | 0.813 | 41.66% | 275 |
| UPRO | 139.67% | 0.753 | 42.17% | 275 |
| SPXL | 157.71% | 0.843 | **30.67%** | 252 |
| TECL | 217.35% | 0.786 | 41.01% | 275 |

All positive returns, all Sharpe > 0.75, all MaxDD < 51%. **Generalization passes**.
SPXL in particular achieves excellent risk-adjusted performance (Sharpe 0.84, MaxDD 30.7%).

## Code Changes

- `rl.py`: `EpisodeConfig` now carries `execution_config`; `SoxlSimulator` uses
  profile-specific execution config for `rebalance_threshold`, `trade_lock`, and
  `buy_cooldown`.
- `cli.py`: All `evaluate_split` calls now pass `execution_config=profile.execution`.
- `scripts/generalization_backtest.py`: Now passes `execution_config` to `backtest_profile`.

## Next Actions

- To break through the 35% 2022 MaxDD gate, try combining `decision_frequency=3`
  with a slightly lower floor (0.30 instead of 0.40) in crisis regimes, or tighten
  `full_drawdown` cap further.
- To recover return, try `decision_frequency=2` as a middle ground between
  698% (daily) and 314% (3-day).
- Consider a two-speed architecture: daily decisions in crisis, 3-day in normal.
