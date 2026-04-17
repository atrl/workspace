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
2. Persist the selected actor under `artifacts/models/` (must pass promotion gate).
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

## Current limitations

- This is a scaffold, not a fully proven live system
- Cancellation flow is best-effort
- Fill reconciliation assumes immediate submit semantics; async fills are not polled yet
- `futu-api` and OpenD must be installed and running separately
- Only single-symbol (US.SOXL) execution is implemented
