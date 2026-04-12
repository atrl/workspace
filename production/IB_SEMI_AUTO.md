# SOXL IB Semi-Auto Workflow

This is the first production shape for `SOXL`: model-driven, broker-backed, and human-approved before any order is sent.

## Intended flow

1. Train and validate the actor with `src/soxl_quant/cli.py`.
2. Persist the selected actor under `artifacts/models/`.
3. Generate a daily signal with:

```bash
python3 production/ib_manual_session.py --profile balanced --as-of 2026-04-11
```

4. Review the payload written under `runtime/soxl/live-YYYY-MM-DD/`:
   - `signal.json`
   - `risk_gate.json`
   - `account.json`
   - `reconcile.json`
5. Only after manual review, submit the order:

```bash
python3 production/ib_manual_session.py --profile balanced --as-of 2026-04-11 --submit
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
- `ib_insync` must be installed separately and IB Gateway/TWS must already be running
- The saved model summary must include a passing `promotion` block; otherwise the semi-auto session will refuse the order
