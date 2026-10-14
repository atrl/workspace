# DTCC年度结算预告原件：2026已取得，其余年份仍缺

本轮取得DTCC通知23036-25的单页PDF原件，文件签名为PDF，文内签发日2025-10-15，主题为2026年预期假日日程。原始网址为https://www.dtcc.com/-/media/Files/pdf/2025/10/15/23036-25.pdf ，通过2026-04-01 18:21:51 UTC的Internet Archive捕获取得；Memento时间一致，下载内容SHA1/Base32与CDX摘要ALQ5VTCACE7XVZL6A57X7AV55PLULNRL一致，另保存SHA256。捕获时间不是首次发布时间认证。

按PDF技能渲染并视觉检查了全部表格和脚注，转录12行预告日期，未把HTML错误页当作PDF。正文明确每个假日前还会另发具体服务通知，因此年度预告本身不提供所有服务的最终实际状态。

关键区分：Closed为关闭；Open1为有限结算服务、需核查具体服务；Open2为开放但不提供结算。2026年Good Friday（4月3日）和独立日观察日（7月3日）标Open1，不能直接视为普通股票可结算。Columbus Day（10月12日）和Veterans Day（11月11日）标Open2，也不能把设施开放当作可结算。其他列出的8个日期为Closed。完整转录见anticipated_holidays.csv；超过2026-10-14的日期只作为已公布计划保留，不是假称未来实际观测。

本轮对2023–2026四个通知的Globals路径和media PDF路径分别查询存档，共8个路径族；7个无匹配捕获，2026 media路径获得上述原件。未再次请求已返回404的实时原件端点。空索引结果不证明其他路径或其他服务无历史资料；2023–2025本次路径没有被认证。

决策needs_data_remaining_years_and_specific_service_notices。没有把年度表扩展成完整自然日/交易日日历，没有认证券商实际本金及费用value date，也没有把原件中的有限服务日直接写成true。现有Qlib provider仅作Recorder运行上下文，本轮未创建或宣称新的合格结算日历provider。

0拟合、0新增回测、未给原收益计息、生产未改。下一项仍需要其余年份原件及服务级例外通知，尤其要区分普通股票CNS与有限交割服务。来源原件、提取文本、搜索索引和转录表已进入Qlib Recorder与NAS，冷恢复逐对象哈希通过。数据资格补齐不等于稳定净SOXX超额，目标未完成。
