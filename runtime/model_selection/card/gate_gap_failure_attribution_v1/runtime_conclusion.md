## Runtime Conclusion

Decision: `diagnostic_no_promotion`. The latest state-conditioned reward branch is not a promotion candidate under +30pp / <=15% gates.

Key evidence:

- Latest branch dominant gate failure: `joint_tradeoff`; recommended mechanism from reusable attribution: `walk_forward_weighted_multi_objective_selection`.
- Latest branch has `1` near-miss window(s), but also `2` joint trade-off window(s) and `1` drawdown-control window(s). The near miss on `wf_2023_2024` is not enough to offset `test_proxy`, `wf_2020`, and `wf_2022` failures.
- Across recent cards, repeated severe failures are split between `joint_tradeoff` and `drawdown_control`, so another stronger global reward penalty is not the next useful experiment.
- Prior poor-window aggregate still points to underparticipation clusters around `2025-04-04, 2026-02-04, 2020-03-18`. That means the next training/selection loop must not solve drawdown by collapsing participation in rebound windows.
- Top feature-difference hypotheses to route into factor IC / single-signal / walk-forward validation, not direct promotion: `vol_of_vol_21, tracking_error_vol_21, decay_63, realized_vol_63, tracking_error_21`.

Next valid loop:

1. Build a walk-forward-weighted multi-objective selector/training objective that scores +30pp excess and <=15% MaxDD jointly, not as a single scalar validation return.
2. Add state-conditioned participation/reentry pressure only in rebound/bull-recovery states, while keeping explicit drawdown control in 2022-style stress states.
3. Before retraining, run a factor notebook for the SOXX/SOXL spread and the recurring volatility/tracking-error candidates to decide whether they are features, gates, or only diagnostics.
