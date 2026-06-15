# dashboard_v2_latest_decision_matrix_v1

## Result

- Dashboard v2 now shows the latest SOXLBacktest baseline row through `2026-06-15`.
- Qlib actor replay rows remain cut at safe replay end `2026-06-12`, but their `latest_signal_stateless` rows for `2026-06-15` are surfaced in the first-screen decision matrix.
- The main K-line keeps historical difference markers unlabeled to avoid candle occlusion, while latest buy/sell signal markers include model labels and positions.

## Evidence

- Browser proof: `dashboard_v2_latest_decision_matrix_20260616.png`
- `scripts/check_dashboard_v2_data.py dashboard_data/v2`: `dashboard_v2_check=ok profiles=6 market_rows=4091 comparison_rows=112`
- `scripts/check_dashboard_static_export.py`: `dashboard_v2=6@2026-06-12/latest=2026-06-15`
- `scripts/check_harness.py`: `passed=336 failures=0`
