# balanced_rule_residual_v22_continued_retention_v1

Date: 2026-05-21

## Governance

- Direction: `model_selection`
- Role: `diagnose`
- Changed layer: `selection`
- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `balanced_rule_residual_v22` final retained actor
- Target window: fixed split plus `test_proxy`; no runtime binding
- Accepted trade-off: a short continuation may prove training is not the
  bottleneck; any candidate must beat v22 without worsening `test_2022`
- External reference: checkpoint-selection discipline in
  `docs/research_knowledge/model_selection.md`
- Comparison-table row key:
  `balanced_rule_residual_v22_continued_retention_v1`

## Hypothesis

If v22 was under-selected only because intermediate actors were not retained,
a short continuation from the promoted v22 actor with replayable iter `0`,
`10`, and `20` checkpoints should find a stage that improves validation and
fixed-split evidence over the current v22 actor.

## Historical Contrast

1. Closest success: v22/post-release-cap runtime selection improved by using
   corrected walk-forward selection evidence instead of legacy training score.
2. Closest failure: phase-balanced sampling and direct SOXX actor-feature
   injection improved local or fixed-window signals but failed broader
   walk-forward / `2022` risk checks.
3. Not a duplicate: this run does not add features, reward knobs, or runtime
   overlays. It only tests whether continuing from the current v22 actor can
   produce a better replayable checkpoint now that retention is supported.

## Acceptance Criteria

- Keep only as a model candidate if a retained checkpoint beats v22 on
  `test_proxy` Sharpe or MaxDD while not worsening `test_2022` MaxDD or average
  position.
- Drop or archive as diagnostic evidence if iter `0` remains best or later
  stages widen risk.
- Do not bind to production runtime without a separate walk-forward and
  runtime-selection review.

## Run Evidence

Command:

```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v22_continued_retention_v1 --skip-data-refresh --seed 20260521
```

Selection:

| checkpoint | validation return | validation sharpe | validation maxDD | score |
|---|---:|---:|---:|---:|
| iter 0 | 55.61% | 1.69 | 25.90% | 86.24 |
| iter 10 | 51.19% | 1.47 | 28.07% | 77.14 |
| iter 20 | 46.69% | 1.33 | 32.92% | 69.26 |

Main comparison versus `balanced_rule_residual_v22`:

| window | v22 | continued retention | readout |
|---|---:|---:|---|
| test_proxy return | 695.45% | 1128.94% | higher return |
| test_proxy Sharpe | 0.94 | 1.03 | slightly higher |
| test_proxy MaxDD | 54.52% | 59.13% | worse risk |
| test_2022 return | -24.52% | -51.24% | worse stress behavior |
| test_2022 MaxDD | 32.71% | 55.26% | fails fixed-split limit |
| test_2022 avg_pos | 0.269 | 0.321 | higher exposure |

Review:

- `v22_continued_retention_review.md`
- `soxl_balanced_rule_residual_v22_continued_retention_v1_summary.json`

## Harness Finding

The first run exposed that `src/soxl_quant/app/train.py` persisted summary JSON
before `backtest_profile` corrected `test_proxy` and `test_2022` metrics.
`tests/test_train_summary_persistence.py` now covers the expected order, and
`scripts/check_harness.py` includes that test and compiles `app/train.py`.

## Decision

- Status: `dropped` / idea-only after
  `runtime/production/card/underperforming_backtest_cleanup_v1`.
- Retention support worked: iter `0`, `10`, and `20` actor files are replayable.
- The model-selection hypothesis failed because iter `0` stayed best and later
  checkpoints widened risk.
- Do not run walk-forward for this profile and do not bind it to production.
- The 2026 YTD common-window replay trails the reimplemented `soxl_backtest`
  rule policy by `15.20pp`; prune the executable `profile.yaml` and actor
  checkpoints, while retaining this note, the review, and summary JSON as the
  idea record.
