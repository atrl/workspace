# Scheduler Composer Env Enabled

Date: 2026-05-21

## Goal

Move the approved `v19reg_v22_budget_usage_bull_guard_v1` runtime composer from
an available private-env path to an intentionally enabled scheduler env state.

## Change

Updated `/Users/rziiiii/.openclaw/soxl.env` without printing private env
contents. The scheduler-visible non-secret runtime fields now select:

- `SOXL_RUNTIME_COMPOSER=budget_usage`
- `SOXL_RUNTIME_COMPOSER_PROFILE=v19reg_v22_budget_usage_bull_guard_v1`
- `SOXL_RUNTIME_SLOW_PROFILE=balanced_rule_residual_v19_reg`
- `SOXL_RUNTIME_FAST_PROFILE=balanced_rule_residual_v22`
- `SOXL_RUNTIME_BULL_PARTICIPATION_GUARD=true`

Repo plist defaults were not changed. The LaunchAgent wrapper reads these fields
at runtime and appends the explicit composer CLI flags.

## Evidence

```bash
production/check_scheduler_health.sh
# runtime composer: enabled composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22 bull_guard=true
# VERCEL_TOKEN: missing in current environment and launch env file
# wf_weekly: pending until first weekly run
```

```bash
production/run_soxl_scheduler.sh \
  --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard
# using runtime composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1
# as_of=2026-05-20
# signal action=0 target=48.41% price=$173.20
# risk_gate approved=True
# dry-run: skipped order placement
# output=runtime/soxl/paper_trading/2026-05-20/dry_run
```

Artifact check:

```bash
jq '{profile:.profile, dry_run:.dry_run, executed:.executed, composition:.composition.name, composer_profile:.composition.composer_profile, slow:.composition.slow_profile, fast:.composition.fast_profile, risk_approved:.risk_gate.approved}' \
  runtime/soxl/paper_trading/2026-05-20/dry_run/summary.json
# profile=v19reg_v22_budget_usage_bull_guard_v1
# dry_run=true
# executed=false
# composition=budget_usage
# composer_profile=v19reg_v22_budget_usage_bull_guard_v1
# slow=balanced_rule_residual_v19_reg
# fast=balanced_rule_residual_v22
# risk_approved=true
```

## Decision

The next scheduled post-close run will use the approved runtime composer unless
the private env is changed back or an explicit CLI composer argument overrides
it. This is now ready for the next fresh-date non-dry paper proof.

Remaining blockers are separate:

- remote Vercel deployment still needs scheduler-visible credentials
- weekly walk-forward still needs the first Saturday scheduled artifact
- composer promotion still needs a fresh-date non-dry paper artifact
