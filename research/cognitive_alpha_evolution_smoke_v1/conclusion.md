# 结论

- what_ran: 执行 cognitive_alpha_evolution_smoke_v1，用 qlib.init、D.features、DatasetH/DataHandlerLP、Alpha158.get_feature_config 和 R.start 跑通可恢复 CogAlpha smoke runner。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-23，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-23。
- key_metrics: profile=soxl_smoke，generations=2，selected_count=1，repair_attempt_count=1，candidate_metric_rows=26。
- regime/window: 本卡只验证候选池、质量检查、repair、五指标筛选、mutation/crossover、多代 checkpoint 和 recorder-first 证据；未做收益晋升。
- promotion_decision: not_promoted_smoke_runner_only。它是 goal-runner 验收卡，不替代现有 SOXL 仓位模型。
- risks: 默认候选是 deterministic code pool，不是外部 LLM 实时生成；SOXL 单标的 IC 近似弱于论文横截面股票池；本研究面向交易复盘和风险控制，不构成买卖建议。
- next_action: 在同一 runner 上接入 full profile / LLM provider / LightGBM 仓位层，并与 constrained_weight_reference、soxl_backtest 对比。qlib recorder id=138f6f77f10c4af7baa7a3f35f6a2999。
