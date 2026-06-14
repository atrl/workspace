# Bull Reentry WF2020 Post-Revision Gap Diagnostic

- Decision: `wf2020_post_revision_gate_feasibility_blocker_no_promotion`.
- Promotion: blocked; no actor is promoted or retained by this diagnostic.
- Post-revision wf_2020: selected `bc_initialization` relative `-277.74pp`, MaxDD `15.49%`.
- Bull/reentry target effect: added `119` high-position target rows and improved relative return by `15.02pp`, but the remaining relative gap is `307.74pp`.
- Feasibility check: wf_2020 buy-hold is `333.33%`, so the current relative gate requires `363.33%` return with MaxDD `<20%`.
- Reference contrast: SOXLBacktest `legacy_default` return `220.78%` / MaxDD `54.28%`; `risk_cap_35_15_05` return `69.60%` / MaxDD `17.94%`.
- Next action: `calibrate_wf2020_retention_gate_or_run_reference_feasibility_screen_before_more_bull_reentry_training`.

Sidecars: `pre_post_replay_delta.csv`, `wf2020_feasibility_reference_matrix.csv`, `target_revision_marginal_effect.csv`, `target_contract_year_coverage.csv`, `selection.json`, `summary.json`.
