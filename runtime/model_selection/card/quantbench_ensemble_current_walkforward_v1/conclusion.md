# Conclusion

Decision: `quantbench_current_walkforward_not_target_ready_no_promotion`.

Selected profile: `quantbench_ensemble_sequence_fragile_panic_budget_v1`.
Failures: `ytd_return_below_soxlbacktest_baseline;ytd_return_below_best_alignment`.
YTD return: `15.50%` versus required best `213.14%` and SOXLBacktest `132.08%`.
YTD MaxDD: `12.30%`.

Walk-forward metrics:
- wf_2020: return=5.33%, MaxDD=18.94%
- wf_2022: return=-8.56%, MaxDD=14.37%
- wf_2023_2024: return=143.75%, MaxDD=15.21%

Risk note: this card refreshes and screens a path target only; it does not train, retain, or promote an actor.
Next action: `use_failure_columns_to_tune_bull_release_or_drawdown_budget_before_actor_training`.
