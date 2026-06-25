# 结论

- what_ran: 执行 cognitive_alpha_evolution_full_v1，用 qlib.init、D.features、DatasetH/DataHandlerLP、Alpha158.get_feature_config 和 R.start 跑 soxl_full CogAlpha evolution。
- key_metrics: profile=soxl_full，generations=24，initial_pool_size=80，parent_pool_size=32，children_per_generation=96，selected_count=1，candidate_metric_rows=2289。
- artifact: qlib recorder id=f2930a051cfc46b39a12f4fe26f417a4，candidate/fitness/quality/lineage/rejected/state 全部保存在 recorder。
- promotion_decision: not_promoted_full_evolution_only。full evolution 只证明可恢复候选进化链路，不单独晋升生产。
- risks: 该 full run 是 repo-owned deterministic Code/Alpha158 生成，不依赖外部 LLM provider；下游仍需重新接 LightGBM 仓位模型和 benchmark。
- next_action: 用 full selected features 重跑 position model，并接入 signal refresh/Bark paper-sim push。
- 本研究面向交易复盘和风险控制，不构成买卖建议。
