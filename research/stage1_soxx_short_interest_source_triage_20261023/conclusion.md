# Nasdaq 上市 SOXX 空头余额：历史来源资格初筛

这项假说关注尚未平仓的空头**余额**，与已失败的 FINRA 当日卖空**成交量占比**不是同一变量。它可能反映仓位压力，但目前没有预定预测方向或可交易规则。

[Nasdaq Trader](https://nasdaqtrader.com/trader.aspx?id=ShortInterest)说明公开逐证券查询覆盖滚动12个月，按半月结算日统计，并在公布日美东16:00后发布；不能把结算日当决策时点。[Nasdaq Data Link](https://data.nasdaq.com/databases/NSHRT)的官方搜索索引简介写明该产品从2022-09-30结算日开始，单靠它不能覆盖Stage1的2011–2022训练期；本工具未渲染该JavaScript产品页，尚不能把索引简介当作原始文件覆盖认证。[SOXX当前页面](https://www.nasdaq.com/market-activity/etf/soxx/short-interest)本次显示数据不可用。[Nasdaq产品页](https://www.nasdaq.com/products/data/equities/short-interest)描述订阅/SFTP，但没有在已查材料中证明提供2011–2022完整历史及修订版本。FINRA的[2021监管说明](https://www.finra.org/sites/default/files/2021-06/Regulatory-Notice-21-19.pdf)区分上市证券空头余额与OTC公开数据；不能拿旧FINRA卖空成交量代理。

决定：`needs_2011_2022_Nasdaq_listed_SOXX_short_interest_vintages_no_fit`。本卡仅审查官方网页渲染内容与Data Link搜索索引简介，未接收原始数据文件字节，因此没有可归档的市场快照或历史首次发布时间认证。下一步先取得可授权的2011/2018/2022三期样本，核对SOXX证券身份、结算/公布时刻、拆股基准、修订及完整范围，再决定是否接入原Stage1 39项对照。未购买、未读SOXX收益、未拟合、未运行账户；Goal仍未完成。
