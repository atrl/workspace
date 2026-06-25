# 结论

- what_ran: 执行 cognitive_alpha_external_provider_smoke_v1，用 qlib.init、D.features、DatasetH/DataHandlerLP、Alpha158.get_feature_config 和 R.start 跑通外部 command-provider 到 CogAlpha evolution 的 smoke 链路。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- key_metrics: provider=fixture_external_command_provider，prompt_count=4，response_count=4，generations=2，selected_count=2，repair_attempt_count=1。
- regime/window: 本卡只验证外部 provider 接口、prompt/response provenance、质量检查、repair、五指标筛选、mutation/crossover、多代 checkpoint 和 recorder-first 证据。
- promotion_decision: not_promoted_fixture_provider_readiness_only。它不是生产模型，不生成最新买卖点。
- risks: 当前没有真实 OpenAI API key；本次是 fixture provider，不是真实 OpenAI 推理；SOXL 单标的 IC 近似弱于论文横截面股票池；本研究面向交易复盘和风险控制，不构成买卖建议。
- next_action: 配置真实 OpenAI/最佳模型 key 后，用同一 command-provider 接口重跑 full profile，再接 LightGBM 仓位层并对比 constrained_weight_reference、soxl_backtest。qlib recorder id=02d781b184134af78f3f4a108b11da72。
