# Failure Modes

- 这是 CogAlpha evolution -> multi-model position 的诊断卡，不是晋升卡。
- goal_manifest.json 记录 data_gate、candidate_generation、quality_checker、fitness_evaluation、evolution_loop、model_and_position、research_card、signal_push 的 goal-mode stage 证据。
- 输入 alpha 来自 soxl_full evolution pool，已跑 21-agent/80/96/24 generation full profile。
- direct_factor_backtest 是因子直连受约束仓位搜索，只能作为 benchmark/prior，不能作为最终晋升门。
- 最终晋升必须由 position_model 或 production paper/sim 通过；当前卡只保留为 research evidence。
- 本研究面向交易复盘和风险控制，不构成买卖建议。
