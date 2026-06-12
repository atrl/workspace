# Underperforming Backtest Cleanup Ideas

Primary benchmark: `soxl_backtest_policy_reimplementation_v1` on the 2026 YTD common window.

## balanced_rule_residual_v22

- Decision: `protect_runtime_component`
- YTD return: `116.882644%`
- Delta vs primary: `-15.195334pp`
- Keep idea: Do not use as a standalone main-line replay versus the new rule policy on this window; keep as canonical runtime component and fast leg for the budget-usage composer.
- Cleanup action: `none`

## balanced_rule_residual_v22_continued_retention_v1

- Decision: `archive_idea_only`
- YTD return: `116.882644%`
- Delta vs primary: `-15.195334pp`
- Keep idea: Continuation-from-v22 idea failed: extra retained checkpoints did not improve the usable v22 path, worsened stress risk, and now trails the primary rule policy on the YTD common window.
- Cleanup action: `delete_profile_and_actor_checkpoints`

## v19reg_v22_budget_usage_bull_guard_v1

- Decision: `protect_production_runtime`
- YTD return: `3.250352%`
- Delta vs primary: `-128.827626pp`
- Keep idea: Production scheduler composer remains a protected execution/risk-control path; compare separately from research return replays because the narrow 2026 YTD result reflects its defensive runtime posture.
- Cleanup action: `none`
