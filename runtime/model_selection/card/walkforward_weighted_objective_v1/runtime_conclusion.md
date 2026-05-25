## Runtime Conclusion

Decision: `no_full_gate_candidate`. Do not promote a candidate from this screen.

Key evidence:

- Selected diagnostic candidate by lowest walk-forward weighted loss: `recovery_reentry_conditional_vol_target_v1`.
- Selected candidate still fails full target: `full_gate=False`, `max_max_dd=26.14%`, `min_relative_return=55.80pp`, `loss=237.50`.
- Strict-screen pass count: `1`; strict candidates that also pass all walk-forward windows: `0`.
- This is direct overfit evidence: strict OOS can pass while walk-forward windows still fail the <=15% drawdown target.

Next valid mechanism:

- Keep `recovery_reentry_conditional_vol_target_v1` as the closest diagnostic reference, but do not bind it to runtime.
- Build the next branch as a walk-forward-weighted multi-objective training/selection objective that optimizes relative return and drawdown jointly across `wf_2020`, `wf_2022`, and `wf_2023_2024`.
- Because top candidates are mostly drawdown-control failures, combine state-conditioned drawdown guard with rebound participation pressure instead of a single global drawdown cap or another reward penalty sweep.
