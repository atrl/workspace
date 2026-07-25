# 结论

- 已按同构口径重做：两个组合使用完全相同的过去式通用特征、网络结构、目标、约束和训练截止日，但分别只用自己的历史训练。没有把 SOXL 仓位套到 TQQQ。
- SOXL/SOXX/cash outer 10bp：收益 302.02%，SOXX 364.60%，财富比 0.8653，MaxDD -52.57%，平均现金 21.07%。
  同平均仓位静态组合收益 366.50%，动态/静态财富比 0.8618。
- TQQQ/QQQ/cash outer 10bp：收益 165.26%，QQQ 163.60%，财富比 1.0063，MaxDD -26.58%，平均现金 14.21%。
  同平均仓位静态组合收益 179.40%，动态/静态财富比 0.9494。
- 这不是只在 outer 才出现的过拟合：开发期 walk-forward 中，SOXL 通用模型/同仓位静态财富比为 0.9550，TQQQ 为 0.8856；动态择时在开发期已经没有增益。
- 同构泛化判断：`not_validated_on_tqqq_homologous_pair`；失败门：SOXL_SOXX_CASH:outer_return_10bps, SOXL_SOXX_CASH:outer_return_25bps, SOXL_SOXX_CASH:dynamic_wealth_vs_matched_static, SOXL_SOXX_CASH:dynamic_calmar_vs_matched_static, SOXL_SOXX_CASH:maxdd, SOXL_SOXX_CASH:worst_rolling_5d, SOXL_SOXX_CASH:annual_relative_stability, SOXL_SOXX_CASH:annual_risk_stability, TQQQ_QQQ_CASH:outer_return_25bps, TQQQ_QQQ_CASH:dynamic_wealth_vs_matched_static, TQQQ_QQQ_CASH:dynamic_calmar_vs_matched_static, TQQQ_QQQ_CASH:average_cash, TQQQ_QQQ_CASH:annual_relative_stability。
- 这张卡验证的是日频直接仓位核心的可迁移性；SOXL 生产模型额外使用半导体专属特征和盘中恢复模块，二者不能混为同一个模型。
- Outer 2023-2026 全程只报告，未参与模型或参数选择。
- Qlib recorder：`f2f5e90be7c94c31bf61a014c866a97e`。
