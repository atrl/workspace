# Operator Scheduler Docs Sync V1

Date: 2026-05-21

## Goal

Keep the operator-facing scheduler instructions aligned with the current
production scheduler flags and Vercel setup path.

## Root Cause

`production/crontab.txt` and the LaunchAgent plists had been updated to pass
`--vercel-deploy --health-summary`, but the manual cron examples in
`production/FUTU_SEMI_AUTO.md` still showed the older commands without
`--health-summary`. An operator copying the documented cron lines would lose
the final `Production Goal Gates` section in scheduled logs.

The private env-file example also did not show the required Vercel keys next to
the LaunchAgent setup, which made the remaining `external_vercel_setup` gate
less actionable.

## Change

- Added a regression test that checks the operator docs include the current
  scheduler flags and Vercel env-key placeholders.
- Updated the manual cron examples to include `--health-summary`.
- Updated the Vercel scheduler smoke command to include `--health-summary`.
- Updated LaunchAgent install/uninstall commands to use
  `launchctl bootstrap` / `launchctl bootout`.
- Added commented Vercel env-key placeholders to the private env-file example
  so fake `REPLACE_ME` values are not accidentally exported.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_production_schedule.py::test_operator_doc_scheduler_examples_include_current_production_flags -q
# red before docs update: pre-open cron example lacked --health-summary
# green after docs update: 1 passed
```

## Decision

The production setup guide now points operators at the same scheduler behavior
that the repo-owned crontab and LaunchAgents use. This does not clear the live
production gates; it reduces the chance that the next manual install loses the
health-summary evidence needed to clear them.

Current live blockers remain:

```text
external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```
