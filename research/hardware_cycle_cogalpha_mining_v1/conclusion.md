# 结论

- what_ran: 执行 hardware_cycle_cogalpha_mining_v1，将硬件周期 qlib 日频状态特征交给 CognitiveAlpha factor mining，再把选中硬件因子输入 open-to-open probability budget position model。
- data: hardware_provider=/Users/rziiiii/dev/workspace/data/qlib/hardware_cycle_v1，feature_count=574，daily_semantics=daily_market_proxy_indices_from_trading_day_closes，free_gpu_snapshots=52 (2026-02-05..2026-03-28)，hardware_sales_ppi_obs=2644 (1992-01-01..2026-05-01)，sales_status=cache_fallback，sales_snapshot=cached_existing_raw。
- factor_mining: selected_count=3，selected=["hardware_storage_sales_price_pressure", "hardware_capex_confirmed_accelerator_momentum", "hardware_memory_bottleneck_spread"]。
- key_metrics: selected_return=3.6395%，maxDD=-1.2304%。
- walk_forward: folds=4，risk_all_pass=False。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=0.0150。
- promotion_decision: not_promoted_failed_walk_forward_risk_gates。
- risks: free_cloud_gpu_* 是云 GPU 租赁/实例小时价，不是销售价；FRED/BLS hardware_sales_* 是按披露日/as-of 生效的月频官方销售价指数状态，不是日频现货报价；硬盘和网络日频数据是公开市场代理，不能单独晋升。
- recorder: 4df2dd47bf664a48a3cde494aec003f7。
