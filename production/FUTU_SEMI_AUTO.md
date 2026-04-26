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

1. Train and validate the actor with `src/soxl_quant/cli.py`.
2. Persist the selected actor under `runtime/model/card/` (must pass promotion gate).
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
# Paper broker (local simulation, default)
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py

# Futu simulated account
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker futu --unlock-password XXX

# Dry-run (signal only, no orders)
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --dry-run

# Health check
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --healthcheck
```

The runner executes the closed loop every invocation:
1. **Data update** — incrementally fetches latest SOXL/SOXX/VIX from yfinance, merges into `data/processed/panels/soxl_soxx_vix_full.csv`, refreshes `data/qlib/soxl/`.
2. **Auto model selection** — scans `runtime/model/card/` for the best promoted model of the given family (default `balanced`). Override with `--profile <name>`.
3. **Signal generation** — runs `plan_manual_session()` for the latest available bar.
4. **Risk gate** — applies the same runtime checks as live execution.
5. **Order placement** — submits the target position through the selected broker (skipped in `--dry-run`).
6. **Notification** — pushes a summary via Bark (iOS), macOS native, or console fallback.
7. **Persistence** — writes `signal.json`, `risk_gate.json`, `account.json`, `reconcile.json`, and `summary.json` under `runtime/soxl/paper_trading/<date>/`.

### Unified Scheduler (recommended)

`production/soxl_daily_scheduler.py` runs the full pipeline in one shot:

```bash
# Paper broker (default)
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py

# Futu simulated account
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker futu --unlock-password XXX

# Dry-run (signal + dashboard, no orders)
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run
```

Pipeline order:
1. Market data update (incremental fetch + qlib refresh)
2. Paper-trading signal generation + execution
3. Dashboard rebuild (`dashboard.html`)
4. Completion notification via Bark / macOS

### macOS LaunchAgent (recommended over cron)

1. Edit `production/com.soxl.daily.scheduler.plist` and fill in your `BARK_KEY`.
2. Install:
   ```bash
   cp production/com.soxl.daily.scheduler.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
   ```
3. Verify:
   ```bash
   launchctl list | grep com.soxl.daily.scheduler
   ```
4. Unload when needed:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
   ```

### Manual cron (alternative)

```bash
# Every trading day at 16:35 ET (after US market close)
35 16 * * 1-5 cd /path/to/workspace && PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py >> runtime/soxl/paper_trading/scheduler.log 2>&1
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

Then open `dashboard.html` in a browser.

## Current limitations

- This is a scaffold, not a fully proven live system
- Cancellation flow is best-effort
- Fill reconciliation assumes immediate submit semantics; async fills are not polled yet
- `futu-api` and OpenD must be installed and running separately
- Only single-symbol (US.SOXL) execution is implemented
- yfinance free tier may rate-limit on rapid successive calls; the daily runner is designed for once-per-day invocation
