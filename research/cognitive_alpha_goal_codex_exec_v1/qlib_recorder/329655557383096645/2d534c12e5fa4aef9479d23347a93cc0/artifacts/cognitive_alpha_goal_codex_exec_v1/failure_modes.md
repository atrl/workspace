# Failure Modes

- position_model_promotion: not_promoted_failed_position_risk_gates
- direct_factor_role: prior_only_not_final_promotion
- direct_factor_promotion: not_promoted_under_constrained_weight_reference
- direct_factor_backtest 不能作为最终晋升门，只能作为 prior/diagnostic。
- 若未跑赢 constrained_weight_reference 或 walk-forward/exposure/risk gates，保持 not_promoted。
