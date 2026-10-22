# H.15 收益率斜率：ALFRED 历史版本导出预检

官方 [ALFRED T10Y2Y 下载页](https://alfred.stlouisfed.org/series/downloaddata?seid=T10Y2Y)允许填写历史版本日期。按冻结计划只尝试一次 ZIP CSV 导出，指定 2011-03-08 与 2020-03-23 两个 as-of 日期及原始百分比单位。网页的修订日期选项共 3,147 个，首项为 2014-01-27；这不足以证明 2011 年 as-of 查询没有数据。

点击下载后浏览器进入 HTTP 响应失败页，未返回 ZIP 路径或原始数据字节；浏览器工具未暴露具体状态码。没有重复导出、猜测 URL、调用 FRED 当前历史文件或取得 API key。因未收到源字节，不声称 2011/2020 历史版本存在或不存在，也没有可归档的原始 ZIP。精确选择与停止状态见 `transport_result.json`。

决定 `stop_fixed_ALFRED_form_export_probe_needs_transport_or_authorized_API_key`。收益率曲线斜率仍是与 Stage1 39 项不同的候选信息，但本来源门未通过；0 SOXX 收益读取、Qlib 调用、拟合及账户。若以后有授权的 FRED API key 或新的官方原始版本入口，须另冻范围和预算，再验证两期原值及下一 Stage1 决策前可用时刻；不能用当前回算历史补上本次缺口。
