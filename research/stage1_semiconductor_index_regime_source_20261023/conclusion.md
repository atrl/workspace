# SOXX 与 SOXL 的 2021 年底层指数变更

冻结三份官方 SEC 原件各请求一次，HTTP 均为 200；原文、响应头、URL、UTC 请求时刻与 SHA-256 已保存在本机并归档 Yanxuan NAS，远端哈希和限定逐文件恢复通过。响应头含临时 Cookie，原字节不提交 Git。0 市场收益读取、0 拟合、0 账户。

[SOXX 的 SEC 补充文件](https://www.sec.gov/Archives/edgar/data/1100663/000119312521126827/d82642d497.htm)当时预告在 2021-06-21 **左右**从 PHLX Semiconductor Sector Index 改为 ICE Semiconductor Index。[SOXL 的 SEC 补充文件](https://www.sec.gov/Archives/edgar/data/1424958/000119312521195149/d177232d497.htm)预告在 2021-08-25 **左右**作相同更换；[8 月 27 日交易所申报](https://www.sec.gov/Archives/edgar/vprr/2100/21009136.pdf)已将 ICE 列为 SOXL 底层指数。由此可将 2021 年夏季作为外生训练域变化的来源假设，但这三份原文没有认证 SOXX 的确切生效秒点。后续只读归因应把 2021-06-14 至 08-27 保留为过渡未知，2021-06-11 及以前作旧域，2021-08-30 起作共同新域，避免用“左右”硬切交易日。

这个来源只证明两个基金发生过结构变化及保守时段；它不能说明 Stage1 失败由此造成，更不能把变更日期变成择时阈值。决定：`conditional_official_2021_index_transition_boundary_source_passed_no_outcomes`。下一步须在匹配 Qlib 原训练与复用评价价格面上，比较旧域、过渡、新域的 SOXL−SOXX 跟踪残差及标签分布；再判断是否值得冻结一个真正的 Stage1 0bp 新机制。SOXX 目标尚未实现。
