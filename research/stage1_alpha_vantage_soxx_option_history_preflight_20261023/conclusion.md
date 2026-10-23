# SOXX 历史期权链：已有 Alpha Vantage 凭据权限预检

[Alpha Vantage 官方 API 文档](https://www.alphavantage.co/documentation/#historical-options)提供 `HISTORICAL_OPTIONS`，描述为按日期返回期权链及 IV/Greeks，并明确属于 premium。按冻结计划，只以当前环境已有凭据读取 SOXX 2011-10-03 一次，不购买、不接触供应商。HTTP 为 200，但原始 JSON 只有 `Information`，内容要求订阅 premium；**零合约、零 IV/Delta**。因此没有请求预定的第二个 2024 日期。HTTP 200 不是数据授权通过。

响应原字节 SHA-256 `6c7d1922…`、请求收据和冻结计划均已归档 Yanxuan NAS，归档 SHA-256 `8aaa7727…` 完成远端哈希与限定恢复；回执和 Git 工件都不含 API key。零 SOXX 未来收益读取、零模型拟合、零账户、零购买。

**决定：关闭当前 Alpha Vantage 凭据的历史期权权限路径。** 这不证明 SOXX 2011–2026 期权历史不存在，也不证明购买 premium 后一定有 SOXX 全期覆盖、双边报价时刻或可用的25Δ恒定期限样本。历史 SOXX/SMH 期权链仍需先取得有权限的具体日期样本及首发/修订/缺失审计；现有 ORATS 只询样草稿尚待用户授权发送，不能据此运行偏斜信息门或 Stage1 0bp账户。
