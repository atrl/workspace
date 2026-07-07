# Failure Modes

- source_template_dependency: v10 does not retrain a new standalone ML model; it reuses the v2 LGB/XGB-confirmed regime template artifact.
- overfit_risk_not_zero: fixed scaling reduces outer-test fitting risk, but the source template is still a hand-designed regime prior.
- benchmark_scope: current pass is versus SOXX open-to-open, not versus SOXL buy-and-hold.
- live_execution_gap: this is recorder-backed research evidence; it still needs paper execution and slippage monitoring before production.
- selected_return=458.1614%, selected_maxDD=-8.2249%, SOXX_return=413.6565%.
