# Failure Modes

- selected_model_constraint: selection uses 2025 validation selection_objective_score, not the 2026 test oracle. The strongest test diagnostic is `xgb_classifier_base_throttle_base_constrained_ref_p0-0.5_t0.85-1_s1_h0.05_hold1` and is not promoted as a selectable parameter.
- base_reference_gap: validation-selected LGB/XGB high-chop model test return 519.860% is below base_constrained_ref 568.517%; the model mainly improves high-chop gating and pacing, not unconditional superiority over the existing base prior.
- label_lag: complete 5-day forward-label backtest ends on 2026-06-25; the 2026-07-02 latest signal uses train+validation refit and an incomplete-label score frame, so it is not verified PnL.
- high_chop_scope: high_chop means near 252-day highs with low trend efficiency and realized-vol expansion; recent June/July post-high-chop exposure is low, which is risk control but can miss a V recovery.
- base_throttle_limit: selected position is not a standalone ML position forecast; base_constrained_ref drives exposure and LGBM probability only throttles that base between 0.85 and 1.0.
- close_to_close_execution_gap: daily backtest applies target_position(t) to t -> t+1 close-to-close forward_return; live promotion requires next-open, after-hours, and intraday adverse-excursion validation on minute data.
- execution_gap: this is qlib research/paper evidence without live broker execution, tax, intraday slippage, or extended-hours fill-quality proof.
