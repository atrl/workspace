# Chicago Fed NFCI 历史版本来源：需要访问

经济假说是：芝加哥联储的每周金融条件指标可能含有 Stage1 原 39 项价格/VIX/广度输入以外的系统性信用与杠杆压力信息，帮助区分应该防守的广泛紧缩与可参与的半导体修复。官方页面说明该指标通常于周三美东 08:30 发布上周五的数据，历史值会随新输入和权重修订；[Chicago Fed NFCI](https://www.chicagofed.org/research/data/nfci/current-data)、[ALFRED NFCI](https://alfred.stlouisfed.org/series?seid=NFCI)。这只构成来源线索，不构成可交易信息。

为取得可审计原件，按冻结预算依次尝试了同一官方 ALFRED 下载表单：Python `requests` 40 秒读超时；直接 curl 的 HTTP/2 连接返回流内部错误；限定的 HTTP/1.1 请求 120 秒后零字节超时。三次实际请求均未收到 HTML 字节，且各次收据/失败 notebook 保留。**命令行没有原始 HTML、数值 ZIP、NAS 来源副本或 Qlib 因子。** 随后按单独冻结的只读浏览器计划看到官方表单的 801 个可选版本日期（2011-05-25 至 2026-10-21）；这是渲染 DOM 元数据，不是原始 HTTP 字节或版本数值。再按单独冻结计划仅提交一次 2011-05-25 Zipped CSV 样本，60 秒内未观察到下载事件或文件。当前决定为 `needs_source_delivery_after_three_bounded_zero_byte_transport_failures_and_one_unobserved_browser_download`；不继续重复请求、购买数据或用今日修订序列回填 2011 年训练日。

本卡 0 市场结果、0 模型拟合、0 资金账户、0 合格前向日；SOXX 稳定超额未实现。后续只有在官方原件传输恢复或获得已授权的等价首发版本来源时，才能另冻来源资格核查。

浏览器限定收据见 `browser_fallback_result.json`、`single_vintage_download_result.json`；早于 2011-05-25 的 Stage1 训练信号缺历史版本，不能以后来的版本填充。

官方 FRED API 的 `series/observations` 可按版本日期返回历史观测，但要求已注册 API key；当前工作环境未发现配置的 FRED key。官方 FRED MCP 免 API key，但需要独立账户登录和条款接受，目前未连接。此来源仍 `needs_source_delivery_or_auth`；详细替代路径及边界见 `api_route_assessment.json`。

免 key 的 FRED 图表 CSV 来源探针已拒绝：URL 明确请求 `vintage_date=2011-05-25`，实际返回 1971-01-08 至 2026-10-16 的 2911 条 NFCI 行，说明不能把它当作 2011 年版本。该响应原字节仅作为拒绝证据保存到 NAS 并定点恢复（SHA‑256 `220de8b46bd636a0e9fce8818bdda0b53dbcaab5ae8015d5b4393b83ac5390c5`），绝不用于模型。见 `graph_vintage_probe_result.json`。
