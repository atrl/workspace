# Walk-Forward Validation: balanced_rule_residual_v19

## Goal
Validate whether `balanced_rule_residual_v19` — the strongest fixed-split candidate — generalizes under walk-forward conditions.

## Motivation
- `risk_budget_usage_v3` passed fixed-split gate but collapsed in walk-forward (wf_2022 MaxDD 59%, wf_2023_2024 Sharpe 0.22)
- This proves **fixed-split metrics are not sufficient** for promotion
- `v19` has never been walk-forward validated; this is the largest knowledge gap in the repo
- If `v19` fails walk-forward, the entire residual family narrative must be rewritten

## Closest Success
- `balanced_rule_residual_v19` fixed-split gate pass (proxy Sharpe ~0.97, 2022 MaxDD ~15%)
- `balanced` canonical baseline (stable, but never walk-forward validated either)

## Closest Failure
- `risk_budget_usage_v3` walk-forward collapse
- `balanced_rule_residual_mode_v2` envelope loosening degraded other windows

## Experiment Role
`diagnose` — validate generalization of the best fixed-split candidate before declaring any new mechanism

## Target Windows
- wf_2020: train 2010-03→2018-12, valid 2019, test 2020
- wf_2022: train 2010-03→2020-12, valid 2021, test 2022
- wf_2023_2024: train 2010-03→2022-12, valid 2023, test 2024

## Acceptance Criteria
- wf_2022 MaxDD < 35.0% (v19 promotion gate)
- wf_2022 avg_pos < 0.35
- wf_2023_2024 Sharpe > 0.0
- No proxy-to-backtest gap > 0.15 Sharpe points

## Accepted Trade-Off
- Sharpe may be lower than fixed-split (proxy ~0.97)
- Trade count may be lower in stress windows

## Changed Layer
`model_selection` / `validation` — no new network/reward/execution changes

## Baseline
- Direct comparison: `balanced` canonical (ran walk-forward in parallel)
- Canonical: `balanced`

## Commands
1. Walk-forward `balanced` (has actor, no retrain needed):
   ```
   qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced --walk-forward --seed 42
   ```
2. Walk-forward `balanced_rule_residual_v19` (needs retrain):
   ```
   qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19 --walk-forward --seed 42
   ```

## Evidence Path
- Results: `runtime/soxl/wf_workers/balanced_rule_residual_v19-wf_*-seed42.json`
- Review: `runtime/model/reviews/v19_walk_forward_review.md`

## Results

### wf_2020 (Bull)
- balanced: test Sharpe 2.30, return +125.8%, MaxDD 21.9%
- v19: test Sharpe 2.35, return +114.9%, MaxDD 20.4%

### wf_2022 (Bear)
- balanced: test Sharpe **-0.91**, return **-25.8%**, MaxDD 33.2%
- v19: test Sharpe **-0.80**, return **-25.2%**, MaxDD 35.2%

### Promotion Gates
- balanced: ❌ FAIL (Sharpe -0.91 < 0.75)
- v19: ❌ FAIL (MaxDD 35.2% > 35%, Sharpe -0.80 < 0.75)

## Conclusion

**Both models fail walk-forward on 2022 bear market.** This is a family-level failure, not v19-specific. The canonical `balanced` is equally broken.

The fundamental problem: training data (2010-2021) lacks prolonged secular bear markets. Models overfit to bull patterns and collapse during sustained drawdowns.

## Decision
- `balanced_rule_residual_v19`: **drop** — walk-forward failure confirmed
- Next: solve bear-market generalization before any new residual-family experiments

## Next Action
- Add bear-market-aware training (synthetic episodes, longer history, or regime-specific sampling)
- Consider credit spread / yield curve features as leading bear indicators
- Re-evaluate reward function for prolonged stress periods
