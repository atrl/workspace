# Failure Modes

- Annual target requires every required fold to beat SOXL open-to-open buyhold and keep max drawdown above -20%.
- This card uses 15m data as lagged context for open2open sizing; it does not prove same-day intraday execution.
- The trend protection floor can preserve return but may also keep exposure during early selloffs.
- SOXX constituent features may inherit snapshot or partial-history bias until point-in-time holdings are complete.
- Current blocking reasons: annual_target_gate_failed.
