# 原始2011年SOXX持仓表来源预检

按冻结计划只读取一份ASX托管的BlackRock/iShares原始PDF。HTTP 200、application/pdf，精确PDF为426页、3,412,380字节，SHA-256为`90b7f7e171e026c12bd64f72e0c3c72122fec2e4bb3af469ff4474482b4bbf62`；第361–362页明确标为2011-10-31的SOXX投资明细，含证券名称、股数和市值。SanDisk、AMD、NVIDIA和TSM ADR是可见正控。原PDF及响应头已归档NAS，远端SHA-256核对一致（`eeaedaf893c002d641acd3099e836b73f46c3e20474fe17f7d9806a44ae3397d`）。

PDF创建元数据为2012-01-03，ASX URL日期亦为2012-01-03；页面印刷流程时间在2011-12-22，现有HTTP Last-Modified为2023-02-21。三者均不能替代最早公开/接收时钟。现只证明**一份2011年原始报告存在**，尚未取得2010–2022全期报告、2011年首个信号以前的已公开篮子、所有成员的历史证券身份和价格。不得把10月底报告倒填到当年更早训练日。

决定：`2011_SOXX_original_schedule_confirmed_full_Stage1_history_unqualified`。0因子值、0市场收益读取、0拟合、0账户；SOXX稳定超额目标未完成。下一步先核对2010–2022原始报告及提交时钟和证券价格覆盖，再决定是否能冻结Stage1的匹配信息实验。
