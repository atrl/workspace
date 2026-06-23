# Failure Modes

- factor_library_reversal_reentry: 反转再入场可以产生很高的局部收益，但高杠杆版本承受深度 SOXL 路径回撤；回撤 gate 内收益远低于生产基准。
- factor_library_trend_sell_gate: 趋势持有叠加 sell_score 能把 MaxDD 降到 20% 附近，但需要把平均暴露压低到约十几个百分点，复利不足。
- factor_library_vol_managed_score: volatility-managed score 降低了路径风险，但在本窗口内主要通过降暴露达成，return_gate 失败。
- data_boundary: 因子库 IC/信号回测是检索证据，不是 promotion 证据；promotion 只看本卡用 qlib provider 和 production baseline 重新计算的 return/DD。
- next_action: 若继续寻找，需要更换信息源或模型家族，例如更长历史期权风险、资金流、订单流或非阈值化模型，而不是重复相同 reversal 阈值。
