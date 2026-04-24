# v19 Dwell Overlay Diagnostic

Date: 2026-04-21

## Goal

Test whether execution-side dwell / throttle constraints can create the temporal
abstraction effect that `balanced_rule_residual_v24` failed to deliver through
retraining.

Base actor:

- `artifacts/models/soxl_balanced_rule_residual_v19_actor.pt`

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/execution_overlay_generalization.py \
  --profile balanced_rule_residual_v19 \
  --model artifacts/models/soxl_balanced_rule_residual_v19_actor.pt \
  --variants-json evidence/post_train_review/v19_dwell_overlay_variants.json \
  --output evidence/post_train_review/v19_dwell_overlay_diagnostic.json
```

## SOXL Result

| variant | proxy return | sharpe | maxdd | trades | bear_2022 maxdd |
| --- | --- | --- | --- | --- | --- |
| base | 412.7% | 0.837 | 48.79% | 313 | 31.32 |
| dwell_hold_v1 | 390.3% | 0.813 | 47.35% | 176 | 32.33 |
| dwell_hold_v2 | 418.4% | 0.822 | 48.30% | 135 | 31.59 |
| stress_asym_hold_v1 | 435.5% | 0.866 | 46.72% | 372 | 30.87 |

## Main Conclusion

- `dwell_hold_v2` is the key result.
- It cuts SOXL trades from `313 -> 135` while keeping return and drawdown roughly
  intact versus base.
- That means execution-side persistence can recover the trade-compression effect
  much more cleanly than `v24` retraining did.

## Secondary Readout

- `dwell_hold_v1` is the more conservative generalized compression variant.
  It improved `TQQQ` and `SPXL`, but gave back too much SOXL return.
- `stress_asym_hold_v1` improved SOXL return and drawdown, but churn expanded to
  `372` trades, so it is not a persistence solution.

## Decision

- Treat `dwell_hold_v2` as the next cheap SOXL-specific overlay candidate.
- Do not open a new temporal-compression training branch yet.
- First compare `dwell_hold_v2` against the existing fixed-actor overlay set and
  promotion-style fixed-split checks.
