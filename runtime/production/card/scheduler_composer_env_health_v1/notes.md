# Scheduler Composer Env Health v1

## Goal

Make scheduler health show whether the private LaunchAgent env file has
actually enabled the approved runtime composer path.

## Issue

`production/run_soxl_scheduler.sh` can now append budget-usage composer flags
from `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}`, but
`production/check_scheduler_health.sh` only reported the file and
`VERCEL_TOKEN` status. Operators could not tell from health whether the next
scheduled run would remain on the default single-profile path, run the approved
composer, or fail because composer env fields were incomplete.

## Change

- Health now parses non-secret `SOXL_RUNTIME_*` fields from the current
  environment or launch env file without sourcing the file.
- If `SOXL_RUNTIME_COMPOSER` is unset or `none`, health reports
  `runtime composer: default single-profile path`.
- If composer mode is set but required fields are missing, health reports
  `runtime composer: incomplete ... missing=...`.
- If the required fields are present, health reports the composer/profile/slow
  profile/fast profile/bull-guard status.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_reports_launch_env_runtime_composer_configuration \
  tests/test_scheduler_health.py::test_scheduler_health_reports_incomplete_launch_env_runtime_composer_configuration -q
# red before implementation: no runtime composer health line
# green after implementation: 2 passed

production/check_scheduler_health.sh
# launch env file: present path=/Users/rziiiii/.openclaw/soxl.env
# VERCEL_TOKEN: missing in current environment and launch env file
# runtime composer: default single-profile path
# data/model/broker/dashboard ok
# selected v22 runtime proof ok via duplicate_skipped
# Vercel remains blocked_no_credentials
```

## Decision

Current installed scheduling remains on the default
`balanced_rule_residual_v22` single-profile path because the private env file
does not set `SOXL_RUNTIME_COMPOSER`. To intentionally run the approved
composer on a fresh paper date, add the required `SOXL_RUNTIME_*` fields to the
private env file and verify that health changes to `runtime composer: enabled`.
