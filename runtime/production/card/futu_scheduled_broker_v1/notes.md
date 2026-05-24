# Futu Scheduled Broker v1

Date: 2026-05-24

## Goal

Configure the scheduled simulated-trading path to use Futu, not IBKR and not
the repo-local paper broker.

## Root Cause

- Futu broker code already lived under `src/soxl_quant/app/futu.py`.
- IBKR had been added as an explicit optional broker.
- The active scheduled LaunchAgent and cron templates still passed
  `--broker paper`, so the scheduled simulated-trading path was not using the
  Futu simulated account.
- `production/check_scheduler_health.sh` also ran broker health without the
  configured scheduler broker, so it reported `paper broker ready` even after
  LaunchAgent broker configuration changed.

## Change

- `production/com.soxl.daily.scheduler.plist` now runs post-close with
  `--broker futu`.
- `production/com.soxl.preopen.scheduler.plist` now runs pre-open dry-run with
  `--broker futu`.
- `production/crontab.txt` and `production/FUTU_SEMI_AUTO.md` now show Futu as
  the scheduled simulated-trading broker.
- Installed LaunchAgents in `~/Library/LaunchAgents/` were synced from the repo
  plists and reloaded.
- `production/check_scheduler_health.sh` reads the installed daily LaunchAgent
  broker and passes that same broker into
  `production/soxl_paper_trading.py --healthcheck`.
- External-broker dry-runs now write under
  `runtime/soxl/paper_trading/<date>/dry_run/<broker>/` so Futu dry-run evidence
  does not overwrite top-level same-day execution proof.

## Live Evidence

OpenD:

```bash
PYTHONPATH=. qlib_venv/bin/python production/check_futu_opend.py
# OK: OpenD is reachable at 127.0.0.1:11111
```

Futu broker health:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py \
  --broker futu --healthcheck \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard --no-notify
# all_ok=true, broker.ok=true, broker.msg="futu quote ok"
```

Installed LaunchAgents:

```text
com.soxl.daily.scheduler: --broker futu --schedule-slot postclose
com.soxl.preopen.scheduler: --broker futu --dry-run --schedule-slot preopen
launchctl list: both loaded
```

Scheduler health:

```text
scheduled broker: futu
broker target: futu
data: ok last_date=2026-05-22 expected=2026-05-22
model: ok v19reg_v22_budget_usage_bull_guard_v1
broker: ok futu quote ok
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Futu dry-run:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --broker futu --dry-run --skip-data-update --skip-dashboard --no-notify \
  --schedule-slot manual \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard
# output=runtime/soxl/paper_trading/2026-05-22/dry_run/futu
# risk_gate approved=false, target step 0.775 exceeds limit 0.350
```

No Futu order was submitted in this manual validation loop because it was a
dry-run.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_production_schedule.py tests/test_scheduler_health.py \
  tests/test_paper_trading.py tests/test_futu_adapter.py \
  tests/test_daily_scheduler.py tests/test_ibkr_adapter.py -q
# 96 passed

plutil -lint production/com.soxl.daily.scheduler.plist \
  production/com.soxl.preopen.scheduler.plist \
  "$HOME/Library/LaunchAgents/com.soxl.daily.scheduler.plist" \
  "$HOME/Library/LaunchAgents/com.soxl.preopen.scheduler.plist"
# all OK
```

## Decision

- Scheduled simulated trading is now configured for Futu.
- IBKR remains available only as an explicit manual broker option; it is not the
  scheduled simulated-trading broker.
- The next natural post-close LaunchAgent run is still required for a fresh
  scheduled non-dry Futu proof.
