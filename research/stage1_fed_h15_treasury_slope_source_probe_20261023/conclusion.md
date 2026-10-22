# 美联储 H.15 国债 10 年减 2 年期限差：固定原版网址探针停止

按冻结四次 GET 上限实际请求两次。[2011-03-07 原始 H.15](https://www.federalreserve.gov/releases/h15/20110307/h15.htm) 返回 HTTP 200，原页标明 2:30 p.m. Eastern Time，包含页面上的 2 年与 10 年 Treasury constant maturity 行；但本次 HTTP 响应的 `Last-Modified` 为 2011-03-14，比名义发布日期晚一周，不能认证 3 月 7 日当时的原字节。预定 `https://www.federalreserve.gov/releases/h15/20200320/` 返回 HTTP 404，按计划停止，未请求 2026 锚点或当前修订历史。**这是该精确归档网址/版式的传输失败，不是 2020 年国债收益率不存在，也不是期限差信息检验失败。** 美联储现行 H.15 页说明每日发布时刻为 4:15 p.m. Eastern Time；不同年代时钟需要按各原件保留，任何后续 Stage1 信号只能用原版已发布后的实际 Qlib 会话。

两份响应的原字节、状态和安全响应头已由 Yanxuan NAS 归档并远端限定恢复，SHA-256 `e7307909f97a0c73abd7505ce95fc45712b8a62cc72da40fe9f60f685c9187ed`；本卡 0 SOXX 收益读取、0 拟合、0 账户、0 Qlib 特征映射。决定 `source_transport_v1_stopped_2020_dated_H15_URL_HTTP404_no_factor`，保留 2011 日期页面与 2020 失败，不改 URL 在本版本重试。下一版本先查明 2016 年后官方日版历史接口及修订语义，再决定能否冻结全 Stage1 日期源。
