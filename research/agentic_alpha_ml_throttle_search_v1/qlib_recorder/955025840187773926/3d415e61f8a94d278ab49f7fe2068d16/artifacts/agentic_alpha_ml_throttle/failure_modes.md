# Failure Modes

- ML throttle keeps the base prior and controls over-100% exposure, but it can still underperform if the validated base spends the trend leg fully invested.
- Selection is validation-only; oracle test rows are diagnostic and are not promotion evidence.
- The model learns 5-day labels while the report uses daily PnL, reducing next-day churn but not eliminating sample-size risk.
- Single-asset daily ML remains fragile; production use still requires paper-sim and walk-forward refresh.
