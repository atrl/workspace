# Multi-Asset Joint Training: balanced_rule_residual_refined_v3

## Context

Single-asset refined_v3 achieved the best proxy return in the refined series
(637.8%) but still fell short of v19 baseline (735.1%). Promotion blocked.

The reward-only direction is now exhausted. Next lever: multi-asset joint
training to enrich regime pattern learning.

## Previous multi-asset results (2026-04-14, P0 baseline)

| ETF   | Single-Asset | Multi-Asset | Delta |
|-------|-------------|-------------|-------|
| SOXL  | 371.6% / 49.9% DD | 514.4% / 58.3% DD | +143pp ret, +8.4pp DD |
| TQQQ  | 224.3% / 43.6% DD | 298.9% / 52.9% DD | +75pp ret, +9.3pp DD |
| UPRO  | 169.6% / 35.1% DD | 227.1% / 43.6% DD | +58pp ret, +8.5pp DD |
| SPXL  | 167.3% / 35.4% DD | 224.2% / 44.0% DD | +57pp ret, +8.6pp DD |
| TECL  | 220.2% / 41.5% DD | 306.0% / 50.7% DD | +86pp ret, +9.2pp DD |

Previous multi-asset improved return significantly but DD worsened by ~8-9pp.
Hypothesis: refined_v3's stronger risk guardrails (drawdown_penalty=0.35,
risk_budget=0.30, episode_drawdown=2.5) will prevent the DD expansion seen
with the P0 baseline.

## Experiment Design

- Profile: `balanced_rule_residual_refined_v3`
- Flag: `--multi-asset`
- Pool: SOXL, TQQQ, UPRO, SPXL, TECL
- Asset embedding: 5-dim one-hot
- Validation/Test: SOXL-only (preserves comparability)
- Seed: 12345

## Success Criteria

| metric | target |
|--------|--------|
| SOXL proxy return | > 650% (improve over single-asset 637.8%) |
| SOXL proxy maxdd | < 56% (prevent previous +8pp DD expansion) |
| SOXL proxy sharpe | > 0.85 |
| Cross-asset maxdd | < 55% for all ETFs |

## Failure Criteria

- SOXL proxy DD > 58% (worse than previous multi-asset 58.3%)
- SOXL proxy return < 600% (no improvement over single-asset)
