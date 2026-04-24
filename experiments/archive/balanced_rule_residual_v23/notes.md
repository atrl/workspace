# balanced_rule_residual_v23 notes

## Motivation

The current mainline family already has a strong base policy in
`balanced_rule_residual_v19`, but the repo-level landscape review still shows a
repeatable weakness on crash-release and restart windows. The next attempt is a
narrow feature-only probe, not a reward or cap retune.

## Baseline

- Canonical baseline: `balanced`
- Direct comparison baseline: `balanced_rule_residual_v19`

## Mechanism Delta

Add a 6-column transition-state pack, enabled only with
`SOXL_USE_TRANSITION_V23=1`:

- `vix_zscore_delta_5`
- `downside_ratio_10_delta_5`
- `days_above_stress_threshold_21`
- `trend_slope_10`
- `stress_then_relief_flag`
- `trend_restart_flag`

No reward, position-cap, execution, checkpoint-selection, or sampling changes.

## Experiment Card

- Role: `replace`
- Changed layer: `feature`
- Target windows:
  - `2020-04`
  - `2022-07`
  - `2022-11`
  - `2023-01`
  - full `test_2022`
- Expected trade-off:
  - allow flat to slightly weaker `test_2023_2024` return if `test_2022`
    discipline remains intact
- Failure mode:
  - the new features improve local release windows by lifting broad `2022`
    exposure, repeating `balanced_transition_state` / `balanced_stress_release_delta`
- Comparison row key:
  - `balanced_rule_residual_v23`

## Evidence Links

- Landscape anchor:
  - `docs/experiment-landscape-review.md`
- Historical feature recall:
  - `evidence/post_train_review/historical_balanced_feature_recall.md`
- Result memory note:
  - `memory/2026-04-21-v23-transition-state-probe.md`
- Pairwise review target:
  - `evidence/post_train_review/balanced_rule_residual_v23_vs_v19.md`

## Training command

```bash
SOXL_USE_TRANSITION_V23=1 PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
    --profiles balanced_rule_residual_v23 --smoke --skip-data-refresh --seed 12345
```

## Result

- First full run completed on 2026-04-21 with seed `12345`.
- Summary artifact:
  - `artifacts/models/soxl_balanced_rule_residual_v23_summary.json`
- Pairwise review:
  - `evidence/post_train_review/balanced_rule_residual_v23_vs_v19.md`
- Decision:
  - dropped and archived

The branch improved defensive shape, but the trade-off was too expensive:

- `test_2022`: `-22.45 / 31.07 / 0.261` -> `-21.88 / 30.23 / 0.250`
- `test_2023_2024`: `43.60 / 37.34` -> `34.25 / 29.39`
- `test_proxy`: `735.10 / 0.95 / 54.83` -> `493.23 / 0.89 / 51.20`

Interpretation: the narrow transition pack did not create selective re-entry.
It mainly pushed the model toward a more defensive, lower-exposure style.
