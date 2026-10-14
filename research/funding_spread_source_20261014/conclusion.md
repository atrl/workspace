# 融资压力原始来源：已采集，尚不具备预测时点资格

本轮0拟合、0回测、0收益相关筛选。针对尚未在现有结论中找到的美元融资压力机制，取得NYFed官方SOFR及FRED所载美联储准备金付息率，独立保存载荷、取回时间、SHA256和原生Qlib存储；没有改变现役策略。

SOFR观察期2018-04-02至2026-10-09，共2130行。逐有效日期匹配准备金付息率，无缺口、不前向填充；2021-07-28以前用IOER，2021-07-29起用IORB，保留明确制度标记。两种制度的定义变化由美联储实施说明确认，不能将其静默当同一原始系列。利差统一为100*(SOFR百分数-准备金利率百分数)，单位bp，样本范围-21至315bp；没有按这个范围选择阈值。

2019-05-31和2021-08-05的SOFR分位数在官方载荷中为NA，但中央利率和成交量存在。保留这两日，不补分位数0、不删除利率。其余已知分位数的顺序检查通过。API当前全部revisionIndicator为空，只表示当前载荷标记为空，不等于历史首发版本已认证。

最重要的边界：SOFR有效日期是底层交易日期，通常下一业务日约08:00纽约时间公布，可能于同日约14:30修订。Treasury业务日与股票交易日不一致。当前provider明确calendar_role=underlying_rate_effective_dates_not_publication_dates、forecast_input_qualified=false，不得直接按有效日期与股票特征拼接，不得用今日下载的最终值冒充当日开盘前已知值。

决定：raw_funding_source_collected_needs_historical_availability_alignment。新增的是独立于股票OHLC的原始信息来源，还不是合格预测特征。融资压力到半导体收益或杠杆暴露的传导仅为待证机制；该利差不是SOXL实际融资费用，也不能直接转成现金规则。

下一步核对历史公布/修订日历与首版本或足够保守的可用时点；完成时点资格前不做收益检验。旧恢复分数和新低收敛的失败路线保持关闭。稳定净超额未实现。

主要来源：[NYFed公布与修订说明](https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates)、[Markets API](https://markets.newyorkfed.org/static/docs/markets-api.html)、[美联储2021年实施说明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20210728a1.htm)、[FRED IORB](https://fred.stlouisfed.org/series/IORB)、[FRED IOER](https://fred.stlouisfed.org/series/IOER)。

Recorder `86169eecce9a4e1eb1150dc77af64560`。[年度来源范围](annual_source_range.csv)。
