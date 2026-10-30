# 报告分钟成交额作为活动强度：几乎重复价×量

**SOXX：历史Goal未通过。** 只读2018–2020两ETF原件及同钟Qlib原价收盘/成交量，零收益标签、模型或账户。在每个交易日去均值后，`log(turnover)`与已用`log(raw_close×raw_volume)`的秩相关，各年SOXL/SOXX均**>0.999999**，高于事前允许的0.995重复门；年度精数见[`audit.json`](audit.json)、[`annual_activity_novelty.csv`](annual_activity_novelty.csv)。这是活动**强弱排序**近重复，不等于两列逐字相等：报告成交额/量相对收盘仍有中位约1–5bp残差。但将该残差解释为真实bar VWAP的独立价格压力，已被同bar高低价语义门挡住。

越界主要在小量bar：SOXL ≤100股仅18.16%落在放宽1bp的高低范围，>1000股约99.31%；SOXX对应35.68%/99.60%，仍不满足原99.9%全来源门。按已见数据改成“只选大成交量”会是阈值救援，本卡不做。固定报告成交额强度路线决定`reported_turnover_activity_near_duplicate_and_vwap_semantics_failed`，不称ETF申赎流、不拟合。旧日报告换手/相对成交额代理已有各自封存作用域；当前无合格新信息或SOXX账户。
