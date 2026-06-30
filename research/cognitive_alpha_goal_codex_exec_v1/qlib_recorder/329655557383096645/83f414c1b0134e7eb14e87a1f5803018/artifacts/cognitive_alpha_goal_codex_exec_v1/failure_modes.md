# Failure Modes

- position_model_promotion: not_promoted_failed_walk_forward_risk_gates
- direct_factor_role: prior_only_not_final_promotion
- direct_factor_promotion: diagnostic_direct_factor_candidate
- direct_factor_backtest 不能作为最终晋升门，只能作为 prior/diagnostic。
- 若未跑赢 constrained_weight_reference 或 walk-forward/exposure/risk gates，保持 not_promoted。
