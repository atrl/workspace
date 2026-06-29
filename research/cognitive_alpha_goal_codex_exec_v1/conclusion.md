# 结论

- what_ran: 执行 cognitive_alpha_goal_codex_exec_v1，使用 Codex exec candidate provider 生成 CogAlpha 候选，再通过 qlib provider、Alpha158 DatasetH/DataHandlerLP、R.start recorder、position model 和 direct-factor prior 完成研究闭环。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- codex_provider: provider_id=codex_exec_candidate_provider，prompt_count=7，sandbox=read-only，openai_api_key_required=False。
- evolution: generations=2，selected_count=1，metrics=ic,icir,rank_ic,rank_icir,mutual_information。
- position_model: promotion=not_promoted_failed_walk_forward_risk_gates，selected_return=7.6562%，walk_forward_folds=2，all_gates_pass=False。
- direct_factor_backtest: promotion=diagnostic_direct_factor_candidate，return=849.9034%，role=prior_only_not_final_promotion。
- latest_signal: as_of=2026-06-24，action=HOLD，target_position=0.0000。
- promotion_decision: not_promoted_failed_walk_forward_risk_gates；production_paper_sim=blocked_not_promoted。
- next_action: 若未通过 constrained_weight_reference 或 walk-forward/exposure/risk gates，继续扩展 Codex-exec 候选和稳定性筛选，不创建 production paper/sim。
- recorder: d8c0dfd3407c46c8b2d04641b407a79e。
- risk: 本研究仅用于 SOXL qlib 研究和 paper/sim 风控，不构成买卖建议。
