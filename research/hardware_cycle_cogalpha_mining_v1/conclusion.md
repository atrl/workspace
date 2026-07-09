# 结论

- what_ran: 执行 hardware_cycle_cogalpha_mining_v1，将硬件周期 qlib 日频状态特征交给 CognitiveAlpha factor mining，再把选中硬件因子输入 open-to-open probability budget position model。
- data: hardware_provider=/Users/rziiiii/dev/workspace/data/qlib/hardware_cycle_v1，feature_count=416，daily_semantics=daily_market_proxy_indices_from_trading_day_closes。
- factor_mining: selected_count=1，selected=["hardware_server_supply_chain_confirmation"]。
- key_metrics: selected_return=3.3090%，maxDD=-0.7640%。
- walk_forward: folds=4，risk_all_pass=True。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=0.0150。
- promotion_decision: not_promoted_under_base_constrained_ref。
- risks: 云 GPU 主特征是日频公开市场代理，不是直接租赁报价；月频直接价格只保留为诊断状态，不能单独晋升。
- recorder: f79b0434b22d4df5a483a79bcfaeeb6d。
