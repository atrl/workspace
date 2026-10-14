# EPS一致预期口径核查：不能直接指定GAAP或公司非GAAP为actual

固定读取已有105条Refinitiv简报预期中的全部35次NVDA事件，逐日期绑定48季度原始SEC申报文本；未查看股价或事件后收益，0拟合、0回测。来源并非只剩早期12条，已扩展的105条记录及PDF都在父Recorder，未重新下载或修数。

LSEG官方说明：I/B/E/S actuals按多数分析师采用的经营/可比口径调整；comparable actual、reported GAAP actual、restated actual和go-forward actual是不同概念。公司自行定义的非GAAP数字不能自动代替供应商可比actual。依据：https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/actuals 和 https://lipperalpha.refinitiv.com/2023/03/earnings-insight-how-are-companies-actually-reporting-earnings/ 。

本轮对简报“去年EPS”做事后数值核对，半美分仅为预登记的描述性接近尺度，不是口径认证：35例中5例接近拆股换算后的原报告GAAP，30例接近原报告公司非GAAP；比较当期报告里的去年栏时，GAAP5例、公司非GAAP28例，4例两者都不接近。各计数允许重叠，不加总成35，也不按最近距离选actual。所有比较和差异完整保留。

|简报日期|简报去年EPS|同股份口径原GAAP|同股份口径原公司非GAAP|当期比较非GAAP|
|---|---:|---:|---:|---:|
|2019-02-14|1.57|1.78|1.72|1.72|
|2022-05-25|0.92|0.7575|0.915|0.91|
|2026-05-20|0.96|0.76|0.81|0.78|
|2026-08-26|1.05|1.08|1.05|1.01|

这些差异不能全归为数据错误。2022例涉及拆股和显示精度；2026-05简报0.96与公司FY2026Q1原报告单列的“剔除H20费用及相关税项”0.96相同，而普通非GAAP为0.81（原始附件：https://www.sec.gov/Archives/edgar/data/1045810/000104581025000115/q1fy26pr.htm）。这能反证统一取公司非GAAP的简单规则，不能仅凭数字相同证明供应商的完整调整政策。2019例尚未查清，不推导补数或挑可比值。

更重要的是，本轮简报的“去年EPS”只是在当前简报日期看到的后来历史值，可能反映重述或后来口径；禁止把它回填为去年公告时已知的供应商actual。当前预期与当前可比actual、历史预期版本、提前预警/修订及发布时间需要分别绑定。既有原公司报告值保持不变。

决定needs_vendor_comparable_actuals_and_vintage，不生成surprise、不重新训练旧供应商EPS路线。下一具体信息需求是同期可比actual及其口径/生效版本，不能用GAAP、公司非GAAP或次年简报去年值替代。105份预期PDF的历史公开时间认证也仍缺失；数字可以核对不等于首发或生产资格。此来源仍可做进一步限定核查，不能把“尚无成功模型”直接当成全部研究已无路可走。

1项解析合同测试通过（新增非GAAP列读取及非供应商口径标记），48份非GAAP结果在增加表头检查后逐值不变，NAS冷恢复逐对象哈希通过。生产及既有收益不变，稳定净SOXX超额未达成。
