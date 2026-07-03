# Conclusion

- what_ran: 执行 agentic_alpha_high_chop_open2open_lgb_xgb_v1，用 qlib.init + D.features 读取 SOXL/SOXX/VIX 日线，构造 high_chop_pressure、chop_breakdown_risk、chop_reentry_quality、SOXX lead、levered decay、panic reversal、volume climax 等新因子，并强制比较 LGBM/XGB classifier/regressor；训练标签为 label_open_to_open_5d，回测收益为 open_to_open_return。
- key_metrics: validation 近似打平时按 ['total_turnover:asc', 'max_drawdown_pct:desc', 'trade_count:asc'] 选择 `xgb_classifier_base_throttle_base_constrained_ref_p0-0.5_t0.55-1_s1_h0.05_hold1`；2026 已标注 open-to-open test 2026-01-02 至 2026-06-24 return 230.307%，max drawdown -20.308%，SOXL open-to-open buyhold 349.335%，goal passers 0。
- regime_window: high-chop slice return 46.168%，max drawdown -10.908%，avg position 0.591，trade count 8；post-high-chop return 201.805%。
- base_comparison: selected 230.307% vs base_constrained_ref 278.118%，gap -47.811 pct，beats_base_constrained_ref=False。
- promotion_decision: research_reject_open2open_goal_gate；open-to-open 修正了 close-to-close 执行错配，但仍未覆盖 7/2 这类开盘后继续暴跌的 minute adverse excursion，也没有 next-open 滑点/夜盘成交验证，不能进入 production review。
- latest_signal: 最新 2026-07-02 close signal target_position 0.001，对应下一可交易 open 生效；这是未完成 5 日 label 的生产式打分，不是已验证收益。
- next_action: 若要超过 base_constrained_ref，需要继续加入 intraday/semiconductor breadth/overnight gap 因子，并用分钟级 open-to-open 路径验证 7/2 类型日内回撤。
- qlib_recorder: 93fe37701bfa4f4dafbc84be53f0befc
