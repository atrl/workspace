# balanced_rule_residual_minimal_v1

## Motivation

Address three fundamental critiques of the current harness:
1. **Cash doesn't reduce drawdown** — removing forced-trading incentives (activity_bonus) alone made models too conservative without improving DD
2. **80+ hard-coded reward parameters** — conflicting gradients, overfitting, brittle behavior
3. **No generalization testing** — multi-asset with shared actor failed, walk-forward missing

## Hypothesis

A truly minimal reward (5 terms, ~10 parameters) + regime-conditional budget (integrated into training, not just inference overlay) can:
- Reduce gradient conflicts and improve training stability
- Make cash a naturally learned risk-management tool
- Match or exceed v16's performance with far less complexity

## Mechanism Delta vs v16

| Layer | v16 | minimal_v1 |
|-------|-----|-----------|
| Reward terms | ~20 active | 5 active |
| Reward params | ~30 | ~10 |
| Drawdown control | episode_penalty + step_penalties + risk_budget + high_exposure + stale + trapped + ... | episode_penalty + regime_budget |
| Activity pressure | activity_bonus=1.25 + inactivity=0.80 | **none** |
| Trend participation | opportunity_cost + trend_alignment | **none** (let network learn) |
| Crisis response | panic_buy + profit_taking + downside_penalty | **none** (budget handles it) |

## External Reference

- mode_v1 proved structural constraint > reward complexity (42.3% MaxDD with same reward as v19)
- rt0.6 conditional budget proved +5pt DD improvement at zero return cost on fixed v19 actor
- v16 proved GRU + window=5 + penalty=2.0 is the best risk-adjusted config (632.4% / 0.91 / 50.1%)

## Target Slices / Windows

- proxy full (primary)
- test_2020, test_2022, test_2023_24
- walk-forward (after fixed_split pass)

## Acceptance Criteria

- proxy return >= 500% (above refined_v2, below v19 — acceptable if DD improves)
- proxy MaxDD <= 48% (beat v16's 50.1%)
- test_2022 MaxDD <= 35% (fixed_split gate)
- test_2022 avg_pos <= 0.35
- seed robustness: <=150pp swing vs another seed

## Expected Trade-off

- Lower return than v19 because we remove opportunistic bonuses (panic_buy, profit_taking)
- Better drawdown because regime_budget is learned during training, not just enforced at inference
- More stable across seeds because fewer conflicting gradients

## Failure Mode

- Model learns to permanently hold cash → return collapses below 400%
- Regime_budget is too blunt → model oscillates around threshold
- Episode_penalty alone insufficient → MaxDD > 50%

## Comparison Table Row Key

`minimal_v1` — minimal reward + regime budget
