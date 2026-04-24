---
title: balanced_rule_residual_v19_academic
direction: model
status: candidate
---

# balanced_rule_residual_v19_academic

## Motivation

Test whether academic-literature-derived microstructure signals add value to the v19 feature stack.

Phase 1 (IC + backtest + correlation screening) identified 3 surviving candidates:
- `ACAD_AMIHUD_ZSCORE`: Amihud illiquidity z-score (1d/5d/10d IC pass, standalone PF=1.21, orthogonal)
- `ACAD_INTRADAY_REVERSAL_21`: Intraday momentum vs reversal (strong negative IC=-0.36, orthogonal)
- `ACAD_VOLATILITY_DRAG`: LETF volatility drag proxy (1d/10d IC pass, PF=1.09)

All other tested academic factors were dropped due to high correlation with existing features:
- `ACAD_DOWN_UP_RATIO` → ρ(mom_21) = -0.89
- `ACAD_VOL_SCALED_MOMENTUM` → ρ(sharpe_proxy_21) = +1.00
- `ACAD_DOWN_RUNS` → ρ(loss_streak_21) = +1.00

## Hypothesis

RL can learn nonlinear interactions between core regime features and microstructure signals (Amihud, intraday reversal) that are invisible to linear IC/backtest screens.

## Mechanism delta vs v19

- Feature pack: add `academic_signals` (7 features: 3 passing + 4 weak but orthogonal)
- All other hyperparameters identical to v19

## Acceptance criteria

- proxy return >= 500%
- proxy MaxDD <= 52%
- test_2022 MaxDD <= 35%
- Must beat v19 on at least 2 of: proxy return, proxy Sharpe, test_2022 MaxDD

## Expected trade-off

- If academic signals are useful: +20-50pp proxy return, tighter drawdown
- If academic signals are noise: no improvement or slight degradation due to expanded obs dim

## Failure mode

- No improvement over v19 → academic_signals pack is not valuable for SOXL RL
- Training instability → obs dim expansion from 56 → 63 hurts GRU learning
