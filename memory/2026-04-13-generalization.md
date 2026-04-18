---
date: '2026-04-13'
title: 2026-04-13 Generalization Backtest Report
kind: review
profiles:
- soxl
status: dropped
tags:
- checkpoint-selection
- dropped
- eval-2022
- review
- soxl
artifacts:
- runtime/generalization/comparison.png
- runtime/generalization/qlib_stores/
- runtime/generalization/report.txt
- runtime/generalization/results.json
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Legacy memory note normalized into the repo-wide schema. Test whether the
  SOXL-trained PPO actor (`soxl_balanced_actor.pt`) generalizes to other 3x leveraged
  ETFs without retraining. | ETF | Underlying | Sector | |-----|-----------|--------
---

# 2026-04-13 Generalization Backtest Report

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Goal
Test whether the SOXL-trained PPO actor (`soxl_balanced_actor.pt`) generalizes to other 3x leveraged ETFs without retraining.

## ETF Universe
| ETF | Underlying | Sector |
|-----|-----------|--------|
| SOXL | SOXX | Semiconductor |
| TQQQ | QQQ | Nasdaq-100 |
| UPRO | SPY | S&P 500 |
| SPXL | SPY | S&P 500 |
| TECL | XLK | Technology |

## Method
1. Fetched daily OHLCV for each ETF + underlying + VIX via `yfinance`.
2. Refactored `src/soxl_quant/data.py` and `src/soxl_quant/backtest.py` to be symbol-agnostic (parameterized ETF symbol, underlying symbol, and leverage ratio).
3. Built per-ETF qlib binary stores under `runtime/generalization/qlib_stores/`.
4. Ran zero-shot backtests using the existing `soxl_balanced_actor.pt` checkpoint.
5. Test window: `2020-01-02` to `2026-03-31` (1,569 trading days).

## Results

| ETF   | Return  | Sharpe | Max DD | Trades |
|-------|---------|--------|--------|--------|
| SOXL  | 374.85% | 0.810  | 51.48% | 891    |
| TQQQ  | 230.51% | 0.821  | 43.99% | 562    |
| UPRO  | 176.53% | 0.863  | 35.22% | 409    |
| SPXL  | 172.53% | 0.855  | 35.42% | 401    |
| TECL  | 223.01% | 0.761  | 42.02% | 624    |

## Baseline Sanity Check
- Existing SOXL backtest: return=368.68%, sharpe=0.805, max_dd=51.57%, trades=894
- Our SOXL run: return=374.85%, sharpe=0.810, max_dd=51.48%, trades=891
- **Match is excellent**; minor differences come from yfinance adjusted-close vs prior data source.

## Interpretation
1. **Positive returns across all 5 ETFs** over a 6+ year period that included the 2022 bear market and 2020 COVID crash.
2. **Sharpe ratios are all > 0.75**, suggesting the position-sizing and regime-risk logic transfers reasonably well.
3. **Drawdowns are lower on broad-index LETFs (UPRO/SPXL ~35%) than on sector LETFs (SOXL ~51%)**, which makes intuitive sense: the model's risk budget reacts to VIX and drawdown signals, and broad indices are less volatile than semiconductors.
4. **Trade counts vary significantly**: SOXL trades most frequently (891), while UPRO/SPXL trade less (~400). This suggests the model's panic-buy logic triggers more often in the more volatile SOXL path.

## Caveats
- The actor was trained exclusively on SOXL/SOXX features; no cross-ETF training or fine-tuning was performed.
- Tracking-error and decay features assume a 3x leverage ratio, which happens to match all test ETFs, but sector-specific volatility patterns differ.
- yfinance auto-adjusted prices were used; splits/dividends are handled implicitly.

## Artifacts
- Data: `data/generalization/*_with_underlying_vix.csv`
- Results: `runtime/generalization/results.json`
- Report: `runtime/generalization/report.txt`
- Chart: `runtime/generalization/comparison.png`

## Conclusion
The model exhibits **non-trivial zero-shot generalization**. It does not blow up on unseen 3x ETFs and produces positive risk-adjusted returns across the board. However, **absolute returns are highest on the training domain (SOXL)**, indicating there is still domain-specific value in the SOXL training distribution. A next step could be to train a single multi-ETF policy or ensemble to see if performance on TQQQ/UPRO/SPXL/TECL can be improved.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
