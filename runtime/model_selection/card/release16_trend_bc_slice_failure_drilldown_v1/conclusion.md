# Release16 Trend BC Slice Failure Drilldown v1

Decision: `release16_trend_bc_same_sweep_stopped_objective_handoff`.

- Same-family BC weight sweep allowed: `False`.
- Full-gate folds: `0/3`.
- Strict-DD solution folds: `1/3`.
- Relative-return solution folds: `2/3`.
- Holdout-slice solution folds: `1/3`.
- Selected min WF relative return: `12.29` pp versus target `30.00` pp.
- Selected worst WF MaxDD: `26.57%` versus target `< 20.00%`.
- Next mechanism: `risk_state_drawdown_drift_checkpoint_objective`.

Conclusion: do not start another release16 trend-target BC weight sweep. The next runnable model work should train a risk-state / drawdown-drift checkpoint objective and keep SOXLBacktest qlib walk-forward as the baseline comparison.
