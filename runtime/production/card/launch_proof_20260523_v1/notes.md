# launch_proof_20260523_v1

Date: 2026-05-23

Purpose:

- Recheck the current production-launch proof after the `06:15 Tue-Sat
  Asia/Shanghai` post-close LaunchAgent window.
- Separate local runtime readiness from external Vercel deployment setup.

Commands:

```bash
date '+%Y-%m-%d %H:%M:%S %Z %z'
PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
production/check_scheduler_health.sh
PYTHONPATH=. qlib_venv/bin/python production/check_futu_opend.py
PYTHONPATH=. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy
PYTHONPATH=. qlib_venv/bin/python scripts/export_production_goal_gates.py
PYTHONPATH=. qlib_venv/bin/python scripts/review_model_runtime_readiness.py
```

Readout:

- Local check time: `2026-05-23 23:13:06 CST +0800`.
- Scheduler service: `com.soxl.preopen.scheduler`,
  `com.soxl.daily.scheduler`, and `com.soxl.weekly.wf` are loaded, and each
  installed plist matches the repo copy.
- Post-close proof: `scheduler_postclose.log` has a current
  `2026-05-23 06:15:02+08:00` run-start marker with
  `schedule_slot=postclose`, `schedule_trigger=launchagent_postclose`,
  `runtime_composer=budget_usage`, and
  `composer_profile=v19reg_v22_budget_usage_bull_guard_v1`.
- Data freshness: `last_date=2026-05-22 expected=2026-05-22`.
- Data registry: `update_data_registry: OK symbols=6 panels=10 qlib=6`.
- Runtime model: `v19reg_v22_budget_usage_bull_guard_v1`
  (`balanced_rule_residual_v19_reg` slow component,
  `balanced_rule_residual_v22` fast component).
- Paper broker: `paper broker ready`.
- Latest non-dry paper artifact:
  `runtime/soxl/paper_trading/2026-05-22/summary.json`,
  `dry_run=false`, `profile=v19reg_v22_budget_usage_bull_guard_v1`,
  `target_position=0.7839037554314358`, `risk_gate=approved`,
  `execution_result.status=skipped`, `error=below minimum lot threshold`.
- Ordinary paper proof: `ok selected_model=v19reg_v22_budget_usage_bull_guard_v1
  execution=skipped`.
- Dashboard static export: `ok`, `versions=38`, `missing=0`,
  `model_versions=94`, `latest=paper_trading`, `paper_trading_rows=4`,
  `paper_trading_end=2026-05-22`, `model_replays=10/10@2026-05-21`.
- Futu OpenD preflight: `OK: OpenD is reachable at 127.0.0.1:11111`.
- Vercel remote deploy: `blocked_no_credentials`;
  auth is `blocked_no_credentials`, project is `missing_project_link`,
  `deployment_url=null`, and the launch env file applied no Vercel keys.
- Production goal gates: `pending external_vercel_setup`.

Decision:

- The local production loop is launch-ready for monitoring: scheduled
  post-close data refresh, inference, paper-state persistence, dashboard static
  export, and model replay refresh are all proven against the current runtime
  composer.
- This is not a public Vercel deployment proof yet. The only remaining launch
  blocker is external Vercel setup: provide `VERCEL_TOKEN` or complete
  `vercel login`, then link/configure the Vercel project with either
  `.vercel/project.json` or `VERCEL_ORG_ID` plus `VERCEL_PROJECT_ID`.
- Do not open a new training branch for this blocker; the model readiness
  decision is now `ready_for_production_monitoring`.

Artifacts:

- `dashboard_data/production_goal_gates.json`
- `dashboard_data/production_status.json`
- `runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`
- `runtime/model/card/model_runtime_readiness_v1/model_runtime_readiness_review.md`
