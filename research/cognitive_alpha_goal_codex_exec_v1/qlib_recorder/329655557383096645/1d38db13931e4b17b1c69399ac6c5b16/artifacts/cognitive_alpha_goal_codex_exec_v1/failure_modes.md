# Failure Modes

- goal_loop_status: needs_iteration
- terminal_reason: max_attempts_exhausted_needs_iteration
- attempts: 9/9
- position_model_promotion: not_promoted_under_base_constrained_ref
- direct_factor_role: prior_only_not_final_promotion
- direct_factor_promotion: not_promoted_under_base_constrained_ref
- next_action: continue_codex_exec_factor_generation
- direct_factor_backtest 不能作为最终晋升门，只能作为 prior/diagnostic。
- 若未跑赢 base_constrained_ref 或 walk-forward/exposure/risk gates 且仍有 attempt 预算，保持 needs_iteration 并继续 Codex exec 因子生成。
- 若达到当前 max_attempts 仍未晋升，保持 needs_iteration，并由外层 Goal refresh attempt budget 后继续。
