# 固定 5-session label 动作消融（候选，中文结论）
- 仅把目标换成 fixed 5-session label；1 候选 + 7 年度 fit（2017-2023），2016 fallback，2024-2026 carry 2023；0 调参/选符号/挑窗口。
- training_state=`fixed_parent_state_label_ablation`；continuation=original frozen net_value router；训练用冻结父状态，**不是**候选自身 on-policy 多日标签（off-policy 偏移为限制）。
- 源绑定：diag recorder `5adde25b815042a78396492ea71169d1` labels_10 sha `6f1582fd490e`；control recorder `e4d888a42c914bf69660ab78a35ba6e7`。
- candidate rows：{'parent_rows': 2690, 'candidate_rows_with_finite_5d_label': 2646, 'excluded_unknown_or_crossyear': 44, 'null_label_filled_with_zero': 0, 'unknown_rows_retained': 44, 'missing_filled_count': 0, 'window_status_counts': {'available': 2646, 'unavailable_cross_year_upstream_identity': 36, 'unavailable_segment_or_gap': 4, 'unavailable_terminal_short': 4}, 'illegal_hold_counterfactual_rows': 374, 'note': 'source identity features are not filtered by routing_eligible; illegal-HOLD rows are counted as a limitation'}。
- 决定：`not_promoted`；reasons=['nine_gates_failed', 'dev_paired_ci_vs_control_not_positive', 'dev_paired_ci_vs_soxx_not_positive']；stop_route=end_fixed_label_route_no_parameter_search。
### 原九项门（仅 development 2017-2022，outer 不入 annual 门）
- ref no_hold_reference：status=ok all_pass=False {'development_return_above_baseline': True, 'stress_25bps_return_above_baseline': True, 'max_drawdown_not_worse': True, 'worst_day_not_worse': False, 'cvar5_not_worse': False, 'worst_5d_not_worse': True, 'open_trade_days_reduced_8pct': True, 'turnover_not_increased': True, 'annual_wealth_floor_pass': False, 'all_pass': False}
- ref active_hold_baseline：status=ok all_pass=False {'development_return_above_baseline': True, 'stress_25bps_return_above_baseline': True, 'max_drawdown_not_worse': True, 'worst_day_not_worse': False, 'cvar5_not_worse': False, 'worst_5d_not_worse': True, 'open_trade_days_reduced_8pct': True, 'turnover_not_increased': True, 'annual_wealth_floor_pass': False, 'all_pass': False}
### SOXX 同费用比较（relative wealth = candidate 相对 SOXX 的比值；非相对 control）
- development_walk_forward @10bp：cand 243.30% vs SOXX 320.00%；relative_wealth(vs SOXX) -18.26%；return_delta -76.70pp；annual win rate 0.2857142857142857。
- development_walk_forward @25bp：cand 153.88% vs SOXX 319.37%；relative_wealth(vs SOXX) -39.46%；return_delta -165.50pp；annual win rate 0.2857142857142857。
- outer_reused_diagnostic @10bp：cand 188.10% vs SOXX 398.09%；relative_wealth(vs SOXX) -42.16%；return_delta -209.99pp；annual win rate 0.25。
- outer_reused_diagnostic @25bp：cand 82.72% vs SOXX 397.34%；relative_wealth(vs SOXX) -63.26%；return_delta -314.62pp；annual win rate 0.0。
### paired 20-session block uncertainty（cost×segment 独立）
- development_walk_forward @10bp vs control_all_state：mean log rel 0.7591610230077076 CI [-3.4751773049469783,5.118994371916654] blocks=1737 status=ok。
- development_walk_forward @10bp vs soxx_buy_hold：mean log rel -1.1483054657248597 CI [-6.091697341927533,3.8786254136290736] blocks=1737 status=ok。
- development_walk_forward @25bp vs control_all_state：mean log rel -0.15374526384879753 CI [-4.015693198838716,3.882141612769863] blocks=1737 status=ok。
- development_walk_forward @25bp vs soxx_buy_hold：mean log rel -2.8582629596646165 CI [-8.18806676844673,2.5565201845051155] blocks=1737 status=ok。
- outer_reused_diagnostic @10bp vs control_all_state：mean log rel -5.077915812120868 CI [-11.717828213665308,2.5299297396204965] blocks=915 status=ok。
- outer_reused_diagnostic @10bp vs soxx_buy_hold：mean log rel -5.8616244209897115 CI [-11.106207762047015,2.336803691659608] blocks=915 status=ok。
- outer_reused_diagnostic @25bp vs control_all_state：mean log rel -9.031009497883245 CI [-16.22957612078406,-1.0327603758259134] blocks=915 status=ok。
- outer_reused_diagnostic @25bp vs soxx_buy_hold：mean log rel -10.72054859424792 CI [-16.540205563703005,-1.74474833266595] blocks=915 status=ok。
- native：passed。
- 边界：outer=reused_diagnostic（非 fresh）；历史看似通过仍需独立前向，不替换现役；诊断 CI 不明确只是探索，不预称 predictive gate 通过。
- 子 Recorder：8a5ecfdade5c4f20a9786617b3350254。
