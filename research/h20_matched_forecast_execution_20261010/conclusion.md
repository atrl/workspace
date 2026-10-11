# H20 匹配期限实验

实际执行 55 次拟合和 8 个原生回放；现金、数量、费用和三个资产权重对账通过。

- development/10bp：收益 325.75%，相对 SOXX 终值 1.0000，MaxDD -47.37%，CVaR5 -4.42%。
- reused_diagnostic/10bp：收益 398.12%，相对 SOXX 终值 1.0000，MaxDD -42.79%，CVaR5 -5.30%。
- development/25bp：收益 325.11%，相对 SOXX 终值 1.0000，MaxDD -47.37%，CVaR5 -4.42%。
- reused_diagnostic/25bp：收益 397.38%，相对 SOXX 终值 1.0000，MaxDD -42.79%，CVaR5 -5.30%。

历史门：False；诊断门：False。不晋升，路线决定：end_h20_route_no_further_search。
这仍是已复用历史；H5 经济参考未核验，不能把差异归因于单独期限。固定 HAC5 不覆盖所有 H20 重叠标签依赖；历史开盘是报价成交代理，并非真实 09:31 成交。
若历史及诊断门通过，需新的冻结身份及至少 60 个独立合格前向日；否则停止 H20，不重搜期限、alpha 或阈值。
Recorder `72455b5609ab4cc5b37808d7e10b2cb0`，FINISHED。

补充预测诊断：两个资产、两个窗口的 MSE 均高于各自历史均值 prior；点 rank IC 为负，但本轮未对 IC 计算置信区间，不能据此证明所有信息均无效。见 forecast_information.csv 与分析 Recorder 3556aa4c0f424e4f9d05070869b2e839。
