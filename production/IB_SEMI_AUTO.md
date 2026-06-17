# SOXL IB Semi-Auto Workflow

This is the first production shape for `SOXL`: model-driven, broker-backed, and human-approved before any order is sent.

## Intended flow

1. Create or update the supervised/model-selection evidence in a runtime
   notebook, such as `runtime/model_selection/card/qlib_alpha158_supervised_cadence_v1/`.
2. Persist any promoted supervised execution artifact with a replayable daily
   signal path. Until then the unified runner records a blocked HOLD intent and
   refuses external orders.
3. Check local TWS / IB Gateway readiness:

```bash
PYTHONPATH=. qlib_venv/bin/python production/check_ibkr_gateway.py --port 7497
```

4. Generate a daily signal with:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker ibkr --ibkr-port 7497 --dry-run
```

5. Review the payload written under `runtime/soxl/live-YYYY-MM-DD/`:
   - `signal.json`
   - `risk_gate.json`
   - `account.json`
   - `reconcile.json`
6. Only after manual review, submit the order:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker ibkr --ibkr-port 7497
```

The unified daily runner can also target IBKR explicitly:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker ibkr --ibkr-port 7497 --dry-run
```

## What the risk gate blocks

- Broker is disconnected or unhealthy
- Existing open orders are still working
- One-shot position jump is too large
- Target position exceeds the execution cap

## Current limitations

- This is a scaffold, not a fully proven live system
- Cancellation flow is not implemented yet
- Fill reconciliation is minimal and assumes one-symbol execution
- `ib-insync` is installed in `qlib_venv`
- IB Gateway/TWS must already be running, logged in, and configured to allow socket clients
- Default ports are TWS live `7496`, TWS paper `7497`, IB Gateway live `4001`, IB Gateway paper `4002`
- The supervised runtime artifact must be promoted and provide a replayable
  daily signal path; otherwise the semi-auto session will refuse the order
