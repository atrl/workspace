# ALFRED DFII10 初次发布值：固定表单尝试失败

按冻结计划仅提交一次公开的“Initial Release Only／Zipped CSV”表单，范围 2018-10-01 至 2026-10-07。HTTPS 请求在等待响应 **60 秒后 ReadTimeout**，没有收到 HTTP 状态、响应头或数据体；没有重试或改表单参数。请求时钟、字段、异常和空响应清单已写入 `source_receipt.json`，并在 Yanxuan NAS 做远端哈希及定向回读。

因此本固定传输版本的结论是 `needs_ALFRED_download_or_NAS`，具体依赖是公开表单 POST 在当前传输路径无响应。官方[ALFRED 下载页](https://alfred.stlouisfed.org/series/downloaddata?seid=DFII10)说明可以按历史日期检索当时版本，但网页说明不等于已取得原始值，更不证明具体发布时间。原财政部合并 CSV 仍只有回溯来源资格，不能拟合、运行完整 Stage1 0bp 账户或声称超过 SOXX。

这次 1 次 POST、0 重定向 GET、0 SOXX 结果读取、0 拟合、0 账户。若继续寻找历史初值，必须另冻不同的官方只读传输路径；不得在本固定版本上反复提交或更改日期与输出格式。
