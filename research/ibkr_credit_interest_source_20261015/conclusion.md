# IBKR现金利息：发现账务缺项，但历史来源尚不足

现有SOXL/SOXX/CASH研究沿用现金零利息，可能低估现金收益；本轮未向任何账户添加利息，也未更改已报告收益。完成官方来源获取、解析、Qlib入库读回及NAS证据归档，0拟合、0新增回测。

核查官方LLC月度基准下载端点，共请求2022-12至2026-10的47个月。38个月HTTP成功但仅返回表头，不能当0利率。2026-02至2026-10有182条USD观测，最早2026-02-02、最晚2026-10-13；缺少完整2023–2025和2026年1月数据。本结论只描述本次端点响应，不断言所有官方渠道均无更早历史。月度逐项URL、哈希和行数见monthly_coverage.csv；原始空响应及非空响应全部在Recorder保留。

官方规则入口：
- [现金利率与分层](https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php)：美元首10,000现金免息，NAV门槛及Pro利差影响付息。
- [计息计算](https://www.interactivebrokers.com/en/pricing/pricing-calculations-int.php)：使用已结算现金；NAV低于100,000时按比例缩减；一般USD采用360天基数；应计与最终月度入账分开。
- [基准定义](https://www.interactivebrokers.com/en/pricing/reference-benchmark-rates-int.php)：IBKR基准由多种市场参考综合确定，不能直接用Fed Funds或当前利率代替历史。
- [历史月度入口](https://www.interactivebrokers.com/en/accounts/fees/monthlyInterestRates.php)：星号表示沿用前日数据，本轮单独保留标记，不假定已获当日新定值。

这些是当前页面规则快照，不是2023以来每期合同和点时版本认证。源数据as_of为2026-10-14，与旧价格回测截至2026-10-09分开；之后的观测不会进入旧回测。基准百分数字段转换为小数（3.880%→0.0388），N/A保留未知，空月份不填充，负值括号和前日标记均测试；请求月份错位及超过as_of会拒绝。仅真实观测进入Qlib，不做跨缺口前向填充。

现有账簿ending_cash是交易后即时现金，不等于官方计息的已结算现金。完整复现还需要交易结算日历/结算延迟、日终NAV、应计余额与月度支付处理，并同时重算SOXX基准；利息改变净值和风险路径，不能只加到策略期末财富。应计余额亦不能简单当作立即复利的已结算现金。现阶段不声称已完成利息引擎或精确IBKR账单复现。

决策needs_data_historical_benchmarks_and_settlement_ledger：历史基准覆盖、历史利差合同及结算账本均未合格。已取得来源可复用，但不是模型新因子，也不证明现金利息能填平超额缺口。后续先核查其他可验证官方历史材料或既有账单证据，再决定完整结算实现；不得用近月利率回填历史来获得漂亮回测。没有请求账户凭证或发送外部消息，生产未改，稳定净SOXX超额目标未完成。

1项针对性解析合同测试通过；执行notebook及原生Qlib读回通过；NAS冷恢复逐对象哈希通过。源HTML和全币种下载为Recorder证据，不放大卡根目录。
