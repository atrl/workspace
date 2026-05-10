# balanced_rule_residual_v22_release_v1

**Date**: 2026-05-10
**Role**: `ablate`
**Changed layer**: `feature`

## Baselines

- Canonical baseline: `balanced_rule_residual_v19`
- Direct comparison baseline: `balanced_rule_residual_v22`

## External References

1. `RB_COOLDOWN_PROXY` — rulebase-derived cooldown / restart timing proxy
2. `TRANS_VIX_DIVERGENCE` — transition-state stress-release signal
3. `CICC_RETURN_VOLUME_SYNC` — CICC low-frequency price-volume timing adaptation

## Closest Historical Cases

1. Closest success case:
   `dwell_hold_v2` showed low-churn recovery improvement in fixed-actor overlay tests, and `balanced_rule_residual_v22` preserved acceptable fixed-split defense.
2. Closest failure case:
   `balanced_rule_residual_v19_transition` broad transition retrain and `balanced_macro_v1` broad macro injection both improved recovery narratives while reopening fixed-split risk or hurting proxy quality.
3. Why this is not a duplicate:
   This profile does not reactivate the full transition or macro packs. It isolates three retained release / re-entry signals that already survived standalone factor screening and are mutually complementary.

## Hypothesis

Adding a narrow release pack to the retained v22 residual stack can improve
`test_2023_2024` recovery timing without repeating the prior failure mode of
lifting `test_2022` exposure or broadening proxy drawdown.

## Acceptance Criteria

- Keep `test_2022 avg_pos <= 0.35`
- Keep `test_2022 maxdd <= 55%`
- Keep `test_proxy sharpe >= 0.50`
- Improve `test_2023_2024` return or Sharpe over `balanced_rule_residual_v22`

## Expected Trade-Off

Small `test_2020` giveback is acceptable if recovery improves materially and
the fixed-split defense profile remains intact.

## Failure Mode

Validation or proxy metrics improve while `test_2022 avg_pos`, `test_2022 maxdd`,
or proxy max drawdown drift upward in the same direction as the prior broad
transition / macro failures.

## Planned Command

```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v22_release_v1 --skip-data-refresh --seed 20260411
```

## Outcome

- Training completed successfully after the terminal-date replay fix in `src/soxl_quant/backtest.py`.
- Validation stayed strong (`best valid = 61.0% / 1.662 / 27.84%`), so the mechanism is not invisible to PPO.
- Corrected replay backtests showed the real trade-off:

| window | `balanced_rule_residual_v22` | `balanced_rule_residual_v22_release_v1` |
| --- | --- | --- |
| `test_proxy` | `1099.12% / 0.976 / 53.21% / 0.4984` | `1350.25% / 1.010 / 53.26% / 0.5204` |
| `test_2022` | `-41.75% / -1.335 / 43.63% / 0.2413` | `-46.34% / -1.480 / 48.45% / 0.2518` |
| `test_2023_2024` | `180.24% / 1.336 / 31.91% / 0.4855` | `234.34% / 1.479 / 32.80% / 0.5089` |

## Decision

- Keep as **ablation evidence**, not a replacement candidate.
- The release trio improves recovery, but direct RL injection turns it into broader `risk-on` pressure rather than a selective release classifier.
- Next valid line is a gated low-churn release overlay or fixed-actor counterfactual, not a broader release-pack retrain.

## Review

- Detailed replay review:
  `runtime/model/card/balanced_rule_residual_v22_release_v1/v22_release_v1_backtest_review.md`
