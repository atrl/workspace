# 结论

- what_ran: 执行 hardware_cycle_cogalpha_mining_v1，将硬件周期 qlib 日频状态特征交给 CognitiveAlpha factor mining，再把选中硬件因子输入 open-to-open probability budget position model。
- data: hardware_provider=/Users/rziiiii/dev/workspace/data/qlib/hardware_cycle_v1，feature_count=488，daily_semantics=daily_market_proxy_indices_from_trading_day_closes，free_gpu_snapshots=52 (2026-02-05..2026-03-28)。
- factor_mining: selected_count=1，selected=["hardware_server_supply_chain_confirmation"]。
- key_metrics: selected_return=3.3090%，maxDD=-0.7640%。
- walk_forward: folds=4，risk_all_pass=True。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=0.0150。
- promotion_decision: not_promoted_under_base_constrained_ref。
- risks: 云 GPU 长历史主特征仍是日频公开市场代理；免费日频直接报价覆盖短且缺口不前填，月频直接价格也只保留为诊断状态，不能单独晋升。
- recorder: dfd7d40fe3424856a093a7be272dd870。
