# soxlbacktest_reference_model_score_v1

- Decision: `no_existing_model_passes_retention_gate_use_reference_scoring_for_next_training`.
- Profiles scored: `21`; complete WF profiles: `12`.
- Retention-gate pass candidates: `none`.
- Selected next-training source by score: `state_conditioned_drawdown_rebound_reward_train_v1`.
- Best YTD profile: `smoke`; best WF drawdown profile among complete-WF candidates: `drawdown_rebound_reward_train_v1`.
- SOXLBacktest `legacy_default` remains the buy/sell reference; `wf_pass_ytd18_15_15_02` remains a low-drawdown but low-return risk floor.
- Promotion remains blocked until one candidate has all required WF windows, worst WF MaxDD <20%, YTD return >= SOXLBacktest reference, and YTD return >= current-best alignment floor.

## Sidecars
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/candidate_score_summary.csv`
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/fold_alignment_metrics.csv`
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/profile_source_audit.csv`
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/missing_evidence.csv`
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/selection.json`
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/summary.json`
- `runtime/model_selection/card/soxlbacktest_reference_model_score_v1/conclusion.md`
