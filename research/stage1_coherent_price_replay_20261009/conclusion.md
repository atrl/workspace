# Stage1 coherent 价格重放（增量 1，中文结论）
- 做了什么：冻结 base forecast/selection 与 control 动作模型（身份不变），仅在新 coherent 价格 panel 上重放（价格效应）；再在同一 panel 上训练费用一致候选（训练效应）；两者分开报告，不合并。
- coherent provider：`data/qlib/stage1_coherent_daily_marks_20261009`；intraday provider：`data/qlib/stage1_repaired_intraday_20261009`；execution mark provider：`data/qlib/stage1_execution_marks_20261009`。
- 首小时输入修复：全部 EARLY_FIELDS（ret/range/from-low/volume/bar_count × SOXL/SOXX）统一从 repaired 源重建；{'source': '/Users/rziiiii/dev/workspace/data/qlib/stage1_repaired_intraday_20261009', 'repaired_sessions': 1944, 'repaired_from': '2018-12-31', 'prior_vintage_prefix_sessions': 752, 'fields': ['early_ret', 'early_range', 'early_from_low', 'early_volume', 'early_bar_count'], 'symbols': ['SOXL', 'SOXX'], 'note': 'all first-hour fields are rebuilt together from one repaired source in the covered region; the unmatched prefix carries no recovery'}。
- price control 来源：`analysis/training_results.pkl` key=2022 的年度 models，直接读回不 refit；control parity=`passed`。
- 决定：`coherent_price_replay_reported`。
### 价格效应（同冻结模型、仅换价格；旧值为父 Recorder 精确对象指标）
- development_walk_forward @10bp：coherent 200.4859% vs old daily 200.4894%（Δ -0.0035pp），MaxDD -29.0278% vs -29.0277%（Δ -0.0001pp）
- development_walk_forward @25bp：coherent 160.8489% vs old daily 160.8518%（Δ -0.0029pp），MaxDD -30.1793% vs -30.1792%（Δ -0.0000pp）
- outer_reused_diagnostic @10bp：coherent 362.7927% vs old daily 362.8372%（Δ -0.0445pp），MaxDD -26.4977% vs -26.4822%（Δ -0.0155pp）
- outer_reused_diagnostic @25bp：coherent 324.6125% vs old daily 324.6494%（Δ -0.0369pp），MaxDD -26.8667% vs -26.8513%（Δ -0.0154pp）
（以上仅为这些已测路径上的影响幅度，不是普适结论。）
- 注意：本卡价格效应只测 economic control 路径；legacy/rules-only 的 state feedback 路径可能对 source vintage 明显更敏感，需单独核验。
### 训练效应（同 coherent panel 新候选项）
- development_walk_forward @10bp：114.86% / MaxDD -30.87%
- development_walk_forward @25bp：167.66% / MaxDD -30.01%
- outer_reused_diagnostic @10bp：173.87% / MaxDD -31.27%
- outer_reused_diagnostic @25bp：142.36% / MaxDD -28.00%
- native 对账：status=`passed`，boundary=`2018-12-31`；2016 prefix open-only、2018-12-31+ intraday 承接 prefix 末现金/股数；prefix 选择被承接但不重新验证。
- 门：候选须超过同价冻结 control；[{'candidate': 'training_effect_candidate', 'segment': 'development_walk_forward', 'cost_bps': 10.0, 'reference': 'price_effect_control', 'return_above_control': False, 'max_drawdown_not_worse': False, 'worst_day_not_worse': False, 'cvar5_not_worse': False, 'worst_5d_not_worse': False, 'all_pass': False}, {'candidate': 'training_effect_candidate', 'segment': 'development_walk_forward', 'cost_bps': 25.0, 'reference': 'price_effect_control', 'return_above_control': True, 'max_drawdown_not_worse': True, 'worst_day_not_worse': True, 'cvar5_not_worse': False, 'worst_5d_not_worse': True, 'all_pass': False}, {'candidate': 'training_effect_candidate', 'segment': 'outer_reused_diagnostic', 'cost_bps': 10.0, 'reference': 'price_effect_control', 'return_above_control': False, 'max_drawdown_not_worse': False, 'worst_day_not_worse': False, 'cvar5_not_worse': True, 'worst_5d_not_worse': True, 'all_pass': False}, {'candidate': 'training_effect_candidate', 'segment': 'outer_reused_diagnostic', 'cost_bps': 25.0, 'reference': 'price_effect_control', 'return_above_control': False, 'max_drawdown_not_worse': False, 'worst_day_not_worse': True, 'cvar5_not_worse': True, 'worst_5d_not_worse': True, 'all_pass': False}]
- 仅历史 opening/10:30 close 代理，不证明真实 09:31/10:31 成交；候选 not_promoted 直到独立 forward。
- 子 Recorder：9775b262dfab496e947a433a18ab6c88；保存真实 reports/positions/reconciliation/replays。
