# SOXL Futu Simulated-Trading Workflow

This is the Futu OpenAPI integration for `SOXL`: model-driven, broker-backed, and human-approved before any order is sent to the simulated account.

## Prerequisites

1. **Install OpenD** (Futu local gateway)

   **macOS (automated)**
   ```bash
   bash scripts/install_futu_opend_macos.sh
   ```
   This downloads and extracts OpenD to `~/futu_opend/`. After installation:
   - Edit `~/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.xml`
     with your Futu credentials.
   - Start OpenD:
     ```bash
     ~/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.app/Contents/MacOS/FutuOpenD
     ```

   **macOS (manual) / Windows / Linux**
   - Download the command-line package from the official OpenAPI page.
   - Known working macOS direct URL pattern:
     `https://softwaredownload.futunn.com/Futu_OpenD_${VERSION}_Mac.tar.gz`
     (the `.zip` and `.dmg` variants currently return 404)
   - Windows: run `FutuOpenD.exe`
   - Linux: run `./FutuOpenD`

   Verify OpenD is reachable before trading:
   ```bash
   qlib_venv/bin/python production/check_futu_opend.py
   ```

2. **Authenticate**
   - OpenD will display a QR code in the terminal or popup window
   - Scan it with the 富途牛牛 mobile app
   - Complete the API questionnaire when prompted (first-time only)
   - Ensure your Futu account has US stock simulated-trading enabled

3. **Environment**
   - `futu-api` is already installed in `qlib_venv/`
   - All scripts below should be run with `qlib_venv/bin/python`

## Intended flow

1. Create or update the supervised/model-selection evidence in a runtime
   notebook, such as `runtime/model_selection/card/qlib_alpha158_supervised_cadence_v1/`.
2. Persist any promoted execution artifact under `runtime/model/card/` (must
   pass promotion gate). Legacy actor execution paths remain compatibility
   surfaces until migrated.
3. Generate a daily signal with dry-run first:

```bash
qlib_venv/bin/python production/futu_manual_session.py --profile balanced --as-of 2026-04-15
```

4. Review the payload written under `runtime/soxl/futu-live-YYYY-MM-DD/`:
   - `signal.json`
   - `risk_gate.json`
   - `account.json`
   - `reconcile.json`
5. Only after manual review, submit the order:

```bash
qlib_venv/bin/python production/futu_manual_session.py --profile balanced --as-of 2026-04-15 --submit
```

## Replay mode (historical backtest through Futu paper account)

Run a full date range against the Futu simulated account:

```bash
qlib_venv/bin/python production/soxl_futu_runner.py --profile balanced --start 2024-01-01 --end 2024-12-31
```

This will replay each day's signal and send orders to the Futu simulated account. Use with care on a clean paper account because it generates real broker state (fills, commissions, positions).

## Optional flags

- `--market-order` — submit market orders instead of limit-at-last-price
- `--unlock-password <pwd>` — auto-unlock trade password if Futu requires it for simulation
- `--host <ip>` / `--port <port>` — connect to a non-default OpenD instance

## What the risk gate blocks

- Broker (OpenD) is disconnected or unhealthy
- Existing open orders are still working
- One-shot position jump is too large
- Target position exceeds the execution cap
- Saved model summary does not include a passing `promotion` block

## Daily Paper-Trading Runner (Auto Loop)

`production/soxl_paper_trading.py` provides a fully automated daily loop:

```bash
# Repo-local paper broker
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker paper

# Futu simulated account, sized as a $10,000 notional paper account
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker futu --account-notional 10000 --unlock-password XXX

# Dry-run (signal only, no orders)
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --dry-run

# Health check
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --healthcheck

# Force a specific data provider for the online refresh
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --data-provider yahoo_api --dry-run
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --data-provider polygon --dry-run
```

The runner executes the closed loop every invocation:
1. **Data update** — incrementally fetches latest SOXL/SOXX/VIX, merges into `data/processed/panels/soxl_soxx_vix_full.csv`, refreshes `data/qlib/soxl/`. The default `--data-provider auto` tries yfinance first, then Yahoo's chart API (`yahoo_api`), then Polygon + CBOE VIX (`polygon`) on retryable provider failures. `polygon` requires `POLYGON_API_KEY`.
2. **Auto model selection** — scans `runtime/model/card/` for the best promoted model of the given family (default `balanced`). Override with `--profile <name>`.
3. **Signal generation** — runs `plan_manual_session()` for the latest available bar.
4. **Risk gate** — applies the same runtime checks as live execution.
5. **Order placement** — submits the target position through the selected broker (skipped in `--dry-run`).
6. **Notification** — pushes a summary via Bark (iOS), macOS native, or console fallback.
7. **Persistence** — writes `signal.json`, `risk_gate.json`, `account.json`, `reconcile.json`, and `summary.json` under `runtime/soxl/paper_trading/<date>/`.

### Unified Scheduler (recommended)

`production/soxl_daily_scheduler.py` runs the full pipeline in one shot:

```bash
# Repo-local paper broker
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker paper

# Futu simulated account, sized as a $10,000 notional paper account
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker futu --account-notional 10000 --unlock-password XXX

# Dry-run (signal + dashboard, no orders)
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run

# Explicit provider selection for the online data refresh
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --data-provider yahoo_api --dry-run
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --data-provider polygon --dry-run

# Cached-data smoke check (dry-run + dashboard, no orders, no notifications)
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --smoke
```

Pipeline order:
1. Market data update (incremental fetch + qlib refresh)
2. Paper-trading signal generation + execution
3. Dashboard rebuild (`dashboard.html`)
4. Completion notification via Bark / macOS

### macOS LaunchAgent (recommended over cron)

1. Store private LaunchAgent environment variables outside the repo:
   ```bash
   mkdir -p ~/.openclaw
   cat > ~/.openclaw/soxl.env <<'EOF'
   export POLYGON_API_KEY='REPLACE_ME'
   # Optional Vercel deployment; uncomment only after replacing with real values.
   # export VERCEL_TOKEN='REPLACE_ME'
   # export VERCEL_PROJECT_ID='REPLACE_ME'
   # export VERCEL_ORG_ID='REPLACE_ME'
   EOF
   chmod 600 ~/.openclaw/soxl.env
   ```
   The LaunchAgents call `production/run_soxl_scheduler.sh`, which loads
   `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` before running
   `production/soxl_daily_scheduler.py`.

   To intentionally run the approved budget-usage runtime composer through the
   same LaunchAgent wrapper, add the composer fields to that private env file:
   ```bash
   export SOXL_RUNTIME_COMPOSER='budget_usage'
   export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'
   export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'
   export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'
   export SOXL_RUNTIME_BULL_PARTICIPATION_GUARD='true'
   ```
   Leave `SOXL_RUNTIME_COMPOSER` unset or set it to `none` to keep the default
   single-profile `balanced_rule_residual_v22` scheduler path.
2. The repo LaunchAgent plists are configured for Futu simulated trading
   (`--broker futu`) with a `$10,000` notional simulation account
   (`--account-notional 10000`). Edit `production/com.soxl.daily.scheduler.plist`
   and `production/com.soxl.preopen.scheduler.plist` only if you want Bark
   notifications, a different notional size, or a temporary broker override.
3. Install:
   ```bash
   cp production/com.soxl.daily.scheduler.plist production/com.soxl.preopen.scheduler.plist ~/Library/LaunchAgents/
   launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist 2>/dev/null || true
   launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist 2>/dev/null || true
   launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
   launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist
   ```
4. Verify:
   ```bash
   launchctl list | grep 'com.soxl.*scheduler'
   ```
5. Unload when needed:
   ```bash
   launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
   launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist
   ```

### Manual cron (alternative)

```bash
# Asia/Shanghai local time:
# Mon-Fri 21:00: US pre-open cached-data dry-run + dashboard + Vercel deploy attempt
0 21 * * 1-5 SOXL_SCHEDULE_TRIGGER=cron_preopen /Users/rziiiii/dev/workspace/production/run_soxl_scheduler.sh --broker futu --account-notional 10000 --runtime-composer budget_usage --composer-profile v19reg_v22_budget_usage_bull_guard_v1 --slow-profile balanced_rule_residual_v19_reg --fast-profile balanced_rule_residual_v22 --bull-participation-guard --dry-run --no-notify --skip-data-update --vercel-deploy --health-summary --schedule-slot preopen >> /Users/rziiiii/dev/workspace/runtime/soxl/paper_trading/scheduler_preopen.log 2>&1

# Tue-Sat 06:15: US post-close data refresh + Futu simulated trading + dashboard + Vercel deploy attempt
15 6 * * 2-6 SOXL_SCHEDULE_TRIGGER=cron_postclose /Users/rziiiii/dev/workspace/production/run_soxl_scheduler.sh --broker futu --account-notional 10000 --runtime-composer budget_usage --composer-profile v19reg_v22_budget_usage_bull_guard_v1 --slow-profile balanced_rule_residual_v19_reg --fast-profile balanced_rule_residual_v22 --bull-participation-guard --no-notify --vercel-deploy --health-summary --schedule-slot postclose >> /Users/rziiiii/dev/workspace/runtime/soxl/paper_trading/scheduler_postclose.log 2>&1
```

### Notification Setup (Bark)

1. Install [Bark](https://apps.apple.com/us/app/bark-custom-notifications/id1403753865) on your iPhone.
2. Copy your device key from the app.
3. Export it in your shell profile (`~/.zshrc`):
   ```bash
   export BARK_KEY="your_device_key_here"
   ```
4. Test:
   ```bash
   PYTHONPATH=. qlib_venv/bin/python production/notify.py --title "Test" --body "Hello from SOXL"
   ```
5. Disable Bark and fall back to macOS notification:
   ```bash
   PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --no-notify
   ```

## Dashboard Auto-Update

The daily scheduler automatically rebuilds `dashboard.html` after each paper-trading run. Paper-trading results are appended to `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv` in the same format as post-train review outputs, so they appear as a new versioned series in the dashboard.

To rebuild the dashboard manually:
```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
```

Then open `dashboard.html` in a browser. For Vercel static deployment, keep the generated
`dashboard.html`, `dashboard_data/index.json`, `dashboard_data/model_versions.json`, and
`dashboard_data/versioned/*/SOXL.json` export together; `vercel.json` serves `/` from
`dashboard.html` and revalidates the dashboard data on each request.

Scheduled Vercel deployment is opt-in through the scheduler flag:
```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker futu --account-notional 10000 --dry-run --skip-data-update --vercel-deploy --health-summary
```

Without Vercel credentials, the scheduler records `blocked_no_credentials` under
`runtime/production/card/vercel_remote_deploy_v1/` and continues. To publish remotely,
make `VERCEL_TOKEN` visible to the scheduler environment or run `vercel login` for the
same user. The checkout also needs a Vercel project link: run
`vercel link --yes --team <team-id> --project <project-name-or-id>` for this
checkout, or expose `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID` to the scheduler
environment. `scripts/deploy_vercel_dashboard.py` reads
`${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` for those Vercel variables
when they are not already in the current shell, so you can run it without
`--deploy` as a non-publishing readiness probe before the next scheduled deploy
attempt. Then rerun with `--vercel-deploy`.

`production/check_scheduler_health.sh` prints a `Scheduler Private Env` section
that confirms whether `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` is
present and whether `VERCEL_TOKEN` / Vercel project env IDs are configured,
without printing their values. The `Vercel Remote Deployment` section also
reports whether the Vercel project is linked/configured and whether the evidence
probe applied Vercel variables from the env file.

## Current limitations

- This is a scaffold, not a fully proven live system
- Cancellation flow is best-effort
- Fill reconciliation assumes immediate submit semantics; async fills are not polled yet
- `futu-api` and OpenD must be installed and running separately
- Only single-symbol (US.SOXL) execution is implemented
- yfinance / Yahoo free endpoints may rate-limit on rapid successive calls; the daily runner now tries yfinance, Yahoo chart API, and Polygon + CBOE VIX in `--data-provider auto` mode, then falls back to cached dry-run if fresh data is unavailable.
