# SEC原始申报EPS核查原型：两季度文本已留存，连续训练序列仍缺

本轮未训练、未回测。公司事实API、submissions刷新及2016Q2财务报表ZIP直接下载均返回403；不能把本轮请求失败说成来源不存在。官方说明companyfacts按申报聚合，frames偏向最后申报匹配期间，后者不能直接作为当时版本。财务报表批次在2024年重新处理过，当前下载批次也不能自动当成2016年的归档字节。

改为使用已有申报编号，读取两个SEC附件：FY2017Q1当前GAAP diluted EPS0.33、上年比较0.24；FY2018Q2当前0.92、上年比较0.41。标题、GAAP与Non-GAAP区分、正文和表格均经检查，与已有公司报告语料数值一致。0.41是后来报告中的上年比较值，不替换FY2017Q2最初的0.40。FY2017Q2、FY2018Q1两个索引此次读取报内部错误，未据命名猜附件内容。

原始来源链接：
- https://www.sec.gov/Archives/edgar/data/1045810/000104581016000243/q117pr.htm
- https://www.sec.gov/Archives/edgar/data/1045810/000104581017000117/q2fy18pr.htm

已保存web工具实际返回的完整文本响应及SHA256，含来源URL与行定位，原始内容存Qlib Recorder/NAS；不是只记忆一个链接或手写数值。表示层明确为渲染文本，不是原始HTML，不宣称首次披露字节、精确首次发布时间或完整申报正文均已认证。工具读取成功也不等于本地HTTP下载成功。已有submissions元数据仍保持此前版本身份，未将失败刷新替换原数据。

当前只有2个不连续季度的原始申报文本核对，不能计算连续历史要求的SUE，更不能把单一发行人代理叫全SOXX盈利预期。源报表、后来可比值、文件申报时间和首发公告时点继续分开；不猜UTC/美东转换，不用延迟假设代替版本核对。

结论needs_continuous_original_filing_series_and_clock，training_input_qualified=false。缺连续原始申报数值和可用时间证据；不是已完成特征训练，也不是超额改进。下一步为补齐剩余季度的原始附件和时间绑定，不重跑已关闭的一致预期或指引路线。生产模型及原账户收益不变，稳定超额目标未完成。

notebook执行核对来源哈希及与父Recorder数值一致，NAS冷恢复逐对象哈希通过。本轮无新增业务逻辑，不增加测试套件。官方接口语义：https://www.sec.gov/search-filings/edgar-application-programming-interfaces；季度批次说明：https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets。
