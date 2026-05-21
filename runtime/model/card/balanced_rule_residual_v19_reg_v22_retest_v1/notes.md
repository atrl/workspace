# balanced_rule_residual_v19_reg_v22_retest_v1

## Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `balanced_rule_residual_v19_reg`
- Role: `ablate`
- Target window: fixed-split first (`test_proxy`, `test_2022`,
  `test_2023_2024`, `test_2020`); only walk-forward if fixed-split clears the
  v22-relative risk/recovery trade-off.
- Accepted trade-off: may give up some `test_proxy` return if `test_2022`
  drawdown/avg-position improves and `test_2023_2024` does not collapse versus
  `balanced_rule_residual_v22`.
- Changed layer: `feature` + `network`
- External reference: VIX term-structure feature family plus dropout/weight
  decay regularization.
- Comparison-table row key:
  `balanced_rule_residual_v19_reg_v22_retest_v1`

## Hypothesis

`balanced_rule_residual_v19_reg` showed a regularization frontier, while
`balanced_rule_residual_v22` became the canonical runtime model after corrected
walk-forward selection and post-release-cap overlay approval. This retest checks
whether the regularized v19 network settings plus the v22 VIX term-structure
feature pack can improve fixed-split risk without the stale profile-only
artifact gap of `balanced_rule_residual_v19_reg_v22`.

## Acceptance Criteria

- Fixed-split summary and retained actor are written under this card.
- Promotion gate passes fixed-split checks.
- `test_2022` MaxDD is at or below the canonical v22 level, or any increase is
  offset by materially better `test_2023_2024` return.
- `test_2023_2024` return remains close enough to v22 to justify walk-forward.
- No stale `balanced_rule_residual_v19_reg_v22` artifact is reused.

## Failure Mode

- Regularization plus VIX features repeats the stale-profile pattern: fixed
  split looks acceptable but recovery weakens or drawdown widens.
- The branch peaks early and later checkpoints trade proxy return for worse
  stress risk, matching the failed v22 continuation diagnostic.

## Historical Contrast

Closest success:

- `balanced_rule_residual_v19_reg` produced a real regularization frontier and
  retained walk-forward artifacts.
- `balanced_rule_residual_v22` is the current canonical runtime model after
  corrected walk-forward model selection.

Closest failure:

- The stale `balanced_rule_residual_v19_reg_v22` card has only a profile and no
  retained actor, summary, or walk-forward outputs.
- `balanced_rule_residual_v22_continued_retention_v1` proved v22 was not simply
  undertrained.
- `balanced_rule_residual_v22_soxx` and `balanced_rule_residual_v22_release_v1`
  showed fixed-window gains can fail later validation or be absorbed by
  execution.

Why this is not a duplicate:

- It uses a new profile key and card, leaving the stale profile-only artifact as
  historical evidence.
- It requires fresh fixed-split output before any walk-forward or runtime
  interpretation.
- It tests a feature + network combination, not another v22 continuation or
  threshold-only execution tweak.

## Run Log

2026-05-21:

```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_reg_v22_retest_v1 --skip-data-refresh --seed 20260521
```

Artifacts:

- Summary:
  `runtime/model/card/balanced_rule_residual_v19_reg_v22_retest_v1/soxl_balanced_rule_residual_v19_reg_v22_retest_v1_summary.json`
- Review:
  `runtime/model/card/balanced_rule_residual_v19_reg_v22_retest_v1/v19_reg_v22_retest_review.md`
- Dashboard export:
  `dashboard_data/versioned/balanced_rule_residual_v19_reg_v22_retest_v1/SOXL.json`

Fixed-split result:

- Proxy: `1280.4%` return / `1.020` Sharpe / `59.6%` MaxDD.
- `test_2022`: `-54.1%` return / `-1.428` Sharpe / `58.3%` MaxDD /
  `0.381` avg position.
- `test_2023_2024`: `44.5%` return / `0.659` Sharpe / `40.8%` MaxDD.
- `test_2020`: `140.5%` return / `2.401` Sharpe / `23.1%` MaxDD.

Promotion checks:

- `test_proxy_sharpe`: pass, `1.020 >= 0.500`.
- `test_2022_max_dd`: fail, `58.30% > 55.00%`.
- `test_2022_avg_pos`: fail, `0.381 > 0.350`.

Decision:

- Registry status is `ablation`.
- Do not run walk-forward and do not bind runtime.
- Keep the retained actors/checkpoints as small replay evidence for this
  governed retest only.
- Interpretation: combining v19_reg regularization with the v22 VIX term pack
  increases exposure and proxy/bull-market return, but it destroys the
  bear-market risk control that made both baselines useful.
