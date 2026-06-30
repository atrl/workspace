# Failure Modes

- goal_loop_status: complete
- terminal_reason: max_attempts_exhausted
- attempts: 2/2
- position_model_promotion: not_promoted_failed_position_risk_gates
- direct_factor_role: prior_only_not_final_promotion
- direct_factor_promotion: not_promoted_under_constrained_weight_reference
- next_action: stop_max_attempts_exhausted
- direct_factor_backtest 不能作为最终晋升门，只能作为 prior/diagnostic。
- 若未跑赢 constrained_weight_reference 或 walk-forward/exposure/risk gates 且仍有 attempt 预算，保持 needs_iteration 并继续 Codex exec 因子生成。
- 若达到 max_attempts 仍未晋升，保持 not_promoted 并停止自动循环。
