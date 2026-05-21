# Pre-open Scheduled Composer Proof V1

Date: 2026-05-21

## Goal

Prove that the installed macOS pre-open LaunchAgent can run the production
runtime-composer path and refresh the dashboard/Vercel evidence bundle with
scheduled trigger provenance.

## Evidence

The natural `21:00 Mon-Fri Asia/Shanghai` LaunchAgent fired at
`2026-05-21T21:00:02+08:00` and wrote:

```text
[scheduler] run started at 2026-05-21T21:00:02+08:00 broker=paper dry_run=True skip_data_update=True data_provider=auto schedule_slot=preopen schedule_trigger=launchagent_preopen runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1 slow_profile=balanced_rule_residual_v19_reg fast_profile=balanced_rule_residual_v22 bull_guard=True
```

Runtime readout from the same scheduled segment:

```text
[paper] data fresh: last_date=2026-05-20 expected=2026-05-20 stale_days=0
[paper] using runtime composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22
[paper] as_of=2026-05-20
[paper] signal action=0 target=48.41% price=$173.20
[paper] risk_gate approved=True
[paper] dry-run: skipped order placement
[paper] output=/Users/rziiiii/.openclaw/workspace/runtime/soxl/paper_trading/2026-05-20/dry_run
```

Scheduler post-paper steps:

```text
[scheduler] dashboard build
[done] wrote /Users/rziiiii/.openclaw/workspace/dashboard.html
[done] wrote /Users/rziiiii/.openclaw/workspace/dashboard_data/index.json
[done] wrote /Users/rziiiii/.openclaw/workspace/dashboard_data/model_versions.json
[done] wrote /Users/rziiiii/.openclaw/workspace/dashboard_data/versioned/ (38 versions)

[scheduler] production goal gates export
production_goal_gates status=pending issues=external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof output=/Users/rziiiii/.openclaw/workspace/dashboard_data/production_goal_gates.json

[scheduler] vercel dashboard deploy
vercel_dashboard status=blocked_no_credentials output=/Users/rziiiii/.openclaw/workspace/runtime/production/card/vercel_remote_deploy_v1

[scheduler] Vercel deploy blocked by missing credentials; continuing after writing deployment evidence
[scheduler] all done
```

Health immediately after the run:

```text
preopen status: success marker found
preopen target: ok schedule_slot=preopen schedule_trigger=preopen dry_run=True skip_data_update=True runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
dashboard static: ok dashboard_static_export generated_at=2026-05-21 21:01:35 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 production_status=blocked_no_credentials goal_gates=pending paper_trading
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Dry-run artifact:

- `runtime/soxl/paper_trading/2026-05-20/dry_run/summary.json`

Key values:

- profile: `v19reg_v22_budget_usage_bull_guard_v1`
- data date: `2026-05-20`
- dry_run: `true`
- executed: `false`
- risk gate: approved
- target/current position: `0.48408568215209924` / `0.48408568215209924`
- composer mode: `crash_rebuild`
- bull guard: enabled but not ready

## Boundary

This proves scheduled pre-open inference/dashboard/Vercel-evidence refresh for
the installed runtime composer. It does not clear the production goal, because:

- Vercel remains `blocked_no_credentials` / `missing_project_link`.
- The final post-close composer target still needs a scheduled
  `schedule_slot=postclose` success segment.
- Ordinary paper proof still waits for the next new-date scheduled non-dry run;
  current same-date proof is `duplicate_skipped`.

## Model-Optimization Implication

This live scheduled proof supports the current decision to exercise the
approved `v19reg_v22_budget_usage_bull_guard_v1` composer path before reopening
training. The model-side audit still says not to run another direct SOXX actor
feature retrain or v22 continuation; both have existing negative evidence.
