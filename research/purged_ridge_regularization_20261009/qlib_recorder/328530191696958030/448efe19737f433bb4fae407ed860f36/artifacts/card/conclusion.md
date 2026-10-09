# 严格过去验证选择Ridge正则化

本轮保留25价格特征、H5标签与5日执行，只改变收益头正则选择。每年用更早训练期拟合10/100/1000及历史均值预测，以前一年严格成熟验证误差选择，再用当前年开始前成熟数据重拟合。

25bp比较如下；完整开发年2016–2022，其后仍为复用历史：

- development/soxx：收益325.11%，相对SOXX终值1.0000，MaxDD -47.37%，CVaR5 -4.42%，换手1.00。
- reused_diagnostic/soxx：收益398.62%，相对SOXX终值1.0000，MaxDD -42.79%，CVaR5 -5.30%，换手0.00。
- development/fixed_ridge10：收益256.50%，相对SOXX终值0.8386，MaxDD -41.39%，CVaR5 -4.34%，换手20.50。
- reused_diagnostic/fixed_ridge10：收益387.84%，相对SOXX终值0.9784，MaxDD -42.79%，CVaR5 -5.30%，换手2.00。
- development/validation_selected：收益250.56%，相对SOXX终值0.8246，MaxDD -41.39%，CVaR5 -4.36%，换手11.62。
- reused_diagnostic/validation_selected：收益398.62%，相对SOXX终值1.0000，MaxDD -42.79%，CVaR5 -5.30%，换手0.00。

历史门通过：False；年度选择：{'2016': 'prior', '2017': 'ridge_1000', '2018': 'prior', '2019': 'ridge_1000', '2020': 'prior', '2021': 'ridge_1000', '2022': 'ridge_10', '2023': 'prior', '2024': 'prior', '2025': 'prior', '2026': 'ridge_1000'}。整体目标仍在进行，不晋升。

历史均值预测与全仓SOXX组合是两个不同对照；若选择prior，仍须看真实优化权重、费用与收益，不能称其学到了alpha。内层两资产误差按各自内层训练方差归一，避免SOXL主导，绝不按外层收益选择。

年度/窗口差异与所有44个内层候选分数保留于selection_scores、selection_audit、annual_metrics。预测MSE/IC及组合费用后稳定性分开评价，没有根据后验结果翻号、改alpha或改期限。

全部历史已复用；先前年度验证仍可能不适合下一年。H5重叠、价格口径/次来源观察、历史开盘、连续份额与现金零收益限制保留。

若选择修正仍未通过，下一步依据选择结果、预测增量与真实持仓差异继续定位信息或配置目标的问题；本批归档不关闭稳定超额目标。

Qlib Recorder：448efe19737f433bb4fae407ed860f36；as_of=2026-09-25。
