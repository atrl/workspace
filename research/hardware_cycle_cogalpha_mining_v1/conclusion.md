# 结论

- what_ran: 执行 hardware_cycle_cogalpha_mining_v1，将硬件周期 qlib 日频状态特征交给 CognitiveAlpha factor mining，再把选中硬件因子输入 open-to-open probability budget position model。
- data: hardware_provider=/Users/rziiiii/dev/workspace/data/qlib/hardware_cycle_v1，feature_count=377，daily_semantics=last_observed_state_forward_filled_to_qlib_daily_calendar。
- factor_mining: selected_count=1，selected=["hardware_cpu_compute_breadth"]。
- key_metrics: selected_return=2.9563%，maxDD=-0.6529%。
- walk_forward: folds=4，risk_all_pass=False。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=0.0150。
- promotion_decision: not_promoted_failed_walk_forward_risk_gates。
- risks: 月频云 GPU 价格只作为日频状态 carry，不是每日新增观测；短历史 cloud direct 因子必须通过 WF 才能升级。
- recorder: 0cb6256ced7c4dbdbc9e905c8f4b2c6e。
