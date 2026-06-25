# 结论

- what_ran: 执行 cognitive_alpha_external_provider_smoke_v1，用 qlib.init、D.features、DatasetH/DataHandlerLP、Alpha158.get_feature_config、R.start 和 Codex CLI readiness 跑通 Codex provider 合同 smoke。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- key_metrics: codex_auth=chatgpt，provider=fixture_external_command_provider，prompt_count=4，response_count=4，generations=2，selected_count=2，repair_attempt_count=1。
- regime/window: 本卡只验证 Codex exec provider schema、prompt/response provenance、质量检查、repair、五指标筛选、mutation/crossover、多代 checkpoint 和 recorder-first 证据。
- promotion_decision: not_promoted_codex_provider_readiness_only。它不是生产模型，不生成最新买卖点。
- risks: 本次为 fixture schema smoke，不是完整 Codex 候选池生成；SOXL 单标的 IC 近似弱于论文横截面股票池；本研究面向交易复盘和风险控制，不构成买卖建议。
- next_action: 后续直接用 Codex provider 扩展候选池，不再把外部 API key 作为运行前置条件；若要推广，仍需先超过 constrained_weight_reference、soxl_backtest。qlib recorder id=2791630ea7474fe99882dfb8fcbab947。
