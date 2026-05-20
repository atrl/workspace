# v22 Corrected Runtime Selection Review

Date: 2026-05-21

## Question

Is `balanced_rule_residual_v22` with
`selective_release_recovery_post_release_cap_v1` still blocked because the base
model lacks promotion evidence, or only because the runtime overlay approval
flag is intentionally false?

## Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/walkforward_backtest_validate.py --profile balanced_rule_residual_v22 --output-dir runtime/model/card/balanced_rule_residual_v22
PYTHONPATH=. qlib_venv/bin/python scripts/review_runtime_selection.py --profile balanced_rule_residual_v22 --model-summary runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json --overlay-review runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json --output-dir runtime/model/card/balanced_rule_residual_v22 --family balanced --persist-model-promotion
PYTHONPATH=. qlib_venv/bin/python scripts/review_selective_release_promotion.py --profile balanced_rule_residual_v22 --output-dir runtime/execution/card/selective_release_post_release_cap_v1 --variant selective_release_recovery_post_release_cap_v1
```

## Corrected Walk-Forward Evidence

| window | return | Sharpe | MaxDD | trades | B&H return | relative |
|---|---:|---:|---:|---:|---:|---:|
| `wf_2020` | `+136.65%` | `2.160` | `20.16%` | `22` | `+59.78%` | `+76.87pp` |
| `wf_2022` | `-48.40%` | `-1.239` | `54.67%` | `26` | `-86.59%` | `+38.18pp` |
| `wf_2023_2024` | `+50.98%` | `1.011` | `30.55%` | `29` | `-2.60%` | `+53.58pp` |

The persisted model promotion sidecar now has:

- `eligible=true`
- `walk_forward_evaluated=true`
- `wf_2023_2024_return=50.9777`
- `wf_2023_2024_sharpe=1.0114`
- `wf_2023_2024_relative=53.5811`

## Runtime Selection Readout

`runtime_selection_review.json` still has `decision=blocked`, but the model-side
checks now pass. The only blocking reason is:

- `profile overlay auto_select_allowed=false`

Current family auto-selection still returns:

- profile: `balanced_rule_residual_v19`
- actor: `soxl_balanced_rule_residual_v19_actor.pt`

`balanced_rule_residual_v22` model promotion now reports:

- `(True, 'model passed promotion gate')`

`balanced_rule_residual_v22` runtime auto-selection still reports:

- `(False, 'profile overlay selective_release_recovery_post_release_cap_v1 is not approved for runtime auto-selection')`

## Decision

The v22 model-promotion evidence gap is closed. The remaining blocker is an
explicit production governance decision on the profile overlay
`auto_select_allowed` flag. Do not run another v22 model-evidence loop for the
same question; either approve the overlay flag after production scheduler proof
or defer v22/post-release-cap and move to a distinct mechanism.

## 2026-05-21 Addendum

The production scheduler proof completed and the governance decision has now
been applied:

- `execution_overlay.auto_select_allowed=true`
- `runtime_selection_review.json`: `decision=approved`
- current auto-selected profile: `balanced_rule_residual_v22`
- runtime selection score source: `walk_forward_mean`
- runtime selection scores: v22 `64.88`, v19 `46.17`

This card remains the model-promotion evidence record; the active runtime
selection state is now captured in `runtime_selection_review.md` and
`runtime_selection_review.json`.
