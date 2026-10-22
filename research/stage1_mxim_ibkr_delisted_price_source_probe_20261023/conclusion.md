# 旧 Maxim（MXIM）IBKR 历史价格来源：接口未返回明细

只读 `search_contracts("MXIM")` 返回 `MAXIM INTEGRATED PRODUCTS` 股票旧合约 ID `272207`；这个发现调用发生在冻结历史请求计划前，原始搜索响应未留存，故它只定位本次单合约探针，不构成已认证的旧证券映射。随后按冻结计划向该 ID 请求一次最多 1000 根日线、包含公司行动的历史图表；连接器仅返回 `Details currently unavailable. Please try again later and contact client services if the issue persists.`，没有任何价格 bar。没有重试、账户读取、订单、拟合或账户。

完整历史请求返回 JSON 的原字节和 SHA-256 已保存在本机并归档 Yanxuan NAS，远端哈希与限定恢复通过。决定：`needs_market_data_IBKR_old_MXIM_history_details_unavailable`。这是连接器的访问/服务失败，**不能证明旧 MXIM 的 2017 年价格不存在**，也不能修复 2017 原 SOXX 篮子至少 14.2745% 的现有 Qlib 成分价格身份缺口。后续需要另行冻结并验证其他历史证券数据来源；不按本次错误提示立即重试。SOXX v2 目标仍未达成。
