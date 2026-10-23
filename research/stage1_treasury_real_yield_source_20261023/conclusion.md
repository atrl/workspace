# 美国财政部 10 年期实际收益率：Stage1 来源预检

## 中文结论

按冻结计划一次读取美国财政部四份官方 CSV，覆盖 2003-01-02 至 2026-10-22，共 5,956 个唯一来源日期。原始响应、头部、接收时间和 SHA-256 已存入 Yanxuan NAS；远端校验及定向冷恢复通过。财政部说明该利率来自收盘指示性国债报价，因此同日 09:46 开盘决策不能使用当日数值。[官方历史档案](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives)与[口径说明](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions)是来源依据。

只用冻结 Stage1 配套的原生 Qlib 交易日历，在决策日之前至少隔两个 Qlib 会话取最近的来源日期。修正后的来源映射在 2018–2022 原训练期 **1060/1061** 日有数，2023–2026 复用评价期 **944/944** 日有数。唯一未知训练日是 **2018-10-15**：配套 provider 最早只到 2018-10-12，不能在同一 Qlib 时钟下证明已有两个前序会话。首次来源映射因裁掉了日历前缀误报 1059/1061；首次修复又错误要求 1061/1061，在新 Recorder 写入前停下。两次过程均保留，第二次机械修复才得到上述保守覆盖。

这条来源可能补充 Stage1 原有 39 项价格、VIX 和广度输入，但**历史首次公开和修订版本尚未核证**；两会话延迟只是保守研究假设，不能把今天的回溯 CSV 自动变成逐日当时版本。尚未定义收益率冲击坐标或做与旧输入的冗余检验，更未读取 SOXX 结果、拟合模型或运行 0bp 账户。下一步应先核对原始版本和决策时钟，再冻结单一坐标、接收层与信息门；只有合格候选才从完整 Stage1 f696+b472 链进入 0bp 加 Tiered 手续费的完整资金账户及 SOXX 硬门。

来源映射 Qlib Recorder `3139aaaaad954c37a938b9f23677e41a` 与 NAS 归档 SHA-256 `cb3fbbd6ee5f04273d6164efb0f8970a9a6687bfd8a64827740fe78e0bb0bf94` 已定向恢复。原始首次 Recorder `4d4fefd49efa4a67a3da0db8d8d378b6` 和错误映射仍保留。SOXX 超额、2/5bp 和独立前瞻证据均未通过；Goal 继续。
