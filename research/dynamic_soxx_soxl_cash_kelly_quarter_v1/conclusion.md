# 中文结论

- 做了什么：在现有 v2/v10 source regime template signal 上叠加 1/4 Kelly sizing；Kelly 的 edge/variance 只用 `template_strategy_return.shift(1)` 的过去滚动窗口估计。
- 参数：kelly_fraction=0.25，lookback=63，min_obs=20，max_kelly_scale=0.45，SOXX floor=0.1。
- 核心指标：SOXX return=413.6565%，maxDD=-43.0919%；quarter-Kelly selected return=349.0079%，maxDD=-8.6907%，excess=-64.6486%。
- 对比 v10 固定缩放：v10 fixed return=458.1614%，maxDD=-8.2249%；Kelly return gap=-109.1534pp，drawdown gap=-0.4657pp。
- 泛化判断：Kelly 叠加解决的是仓位大小随近期 edge/variance 变化，不解决 source template 自身是否过拟合；所有参数在本卡固定，不用 test segment 选择。
- 是否晋升：not_promoted_quarter_kelly_failed_current_goal；研究证据可用于 sizing 对比，不能直接实盘。
- qlib recorder：1af807e6b0654e59891fdba4d212bf5b
- 不构成买卖建议。
