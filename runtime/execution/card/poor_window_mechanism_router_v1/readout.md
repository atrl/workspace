## Final Readout

This notebook is the primary single-run conclusion artifact.

- Decision: `diagnostic_next_step_rebound_router_no_promotion`
- Next mechanism: `state-conditioned rebound participation router with separate high-vol positive-rebound and high-vol downtrend branches`
- Repeated poor-window clusters: `2026_high_vol_rebound_underexposure, 2025_high_vol_rebound_underexposure`
- Top feature group: `volatility_path`
- Cap-like sources still fail rebound excess: `True`
- Strict MaxDD pass sources among structured selections: `0`

### Mechanism Notes

- The persistent 2025/2026 clusters have large negative excess while drawdown is often already below the 15% target, so the remaining failure is underparticipation in violent rebounds, not only account drawdown.
- Feature consensus is dominated by volatility-path and trend/rebound variables. `vol_of_vol_21`, `jump_variation_5`, `realized_vol_63`, and trend/rebound terms recur across cards.
- Tightening execution caps alone is now a duplicated failure mode: it leaves rebound underexposure while risking `wf_2022` or trend-window regression.
- The next experiment should route state, not tune another scalar cap: allow rebound participation only when high-volatility positive momentum and long-anchor recovery are both present; de-risk separately when high volatility is paired with trend breaks/downside momentum.

### Artifacts

- `runtime/execution/card/poor_window_mechanism_router_v1/poor_window_inventory.csv`
- `runtime/execution/card/poor_window_mechanism_router_v1/cluster_summary.csv`
- `runtime/execution/card/poor_window_mechanism_router_v1/feature_consensus.csv`
- `runtime/execution/card/poor_window_mechanism_router_v1/feature_group_consensus.csv`
- `runtime/execution/card/poor_window_mechanism_router_v1/rebound_cluster_comparison.csv`
