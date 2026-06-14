# release16_rulebase_reversal_drawdown_upside_state_head_actor_probe_v1 full probe conclusion

Decision: `full_probe_failed_no_actor_retention`.

The constrained full walk-forward probe is not retainable. Fixed test return is 291.0% with MaxDD 25.3%, which violates the user hard MaxDD <20% gate. The 2026 YTD daily replay returns 38.6% from 2026-01-02 to 2026-06-09, below both the SOXLBacktest qlib reference 132.1% and current-best alignment floor 213.1%. Walk-forward remains unstable: wf_2020 return 35.4% / MaxDD 19.3% / relative -298.0pp; wf_2022 return -16.5% / Sharpe -1.08; wf_2023_2024 return -7.7% / MaxDD 20.9% / relative 28.5pp.

Actor retention is blocked. The generated actor checkpoints are deleted and must not be used as dashboard runnable models. Keep only JSON/CSV evidence for diagnosis.

Next action: do not rerun this profile unchanged. Diagnose the state-head/reward mismatch that causes underparticipation versus SOXLBacktest in 2026 YTD and the negative wf_2022 / wf_2023_2024 folds before opening another retained actor branch.
