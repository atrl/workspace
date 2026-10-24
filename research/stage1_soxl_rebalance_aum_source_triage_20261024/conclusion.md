# SOXL实际再平衡需求的发行人来源仍缺日级历史与公布时钟

按冻结计划只请求一次[Direxion官方SOXL/SOXS产品页](https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs)，收到HTTP 200、**1,204,411字节**HTML；UTC请求/响应时刻、响应头和SHA-256记录于`result.json`。原HTML压缩包SHA-256 `0ffa12b3f888d6df690b64d3ccdc1430ff182c3ee3f095dac55fb6ce0756ade4` 已上传Yanxuan NAS、核对远端哈希并限定恢复。未向供应商联系、购买或使用第三方估计；0市场结果、拟合、账户。

该页面当前呈现NAV/市场价与“Daily Fund Holdings (csv)”字样，但静态HTML中的该链接指回**同一产品页**；固定计划只允许跟随一个明确的同域直接CSV，故未执行第二次请求。页面文本未给出“shares outstanding”或“AUM”字段，也没有2011–2022逐日资产/份额表及各日原始首次公布版本。**这只是对本次官方产品页入口的限定结论**，不证明发行人其他动态接口或档案不存在。

按来源门决定 `source_only_needs_historical_daily_SOXL_AUM_shares_and_first_public_vintages`。没有日级基金规模与申赎份额，就不能把前卡的尾盘SOXL−3×SOXX价格残差冒充实际杠杆ETF再平衡需求，也不能重开已失败的固定局部信息门。若后续发现官方历史接口，需新来源计划核对覆盖、单位/拆分、修订及在下一Stage1决策前的可用时间，再冻结不同的因子与经济接收器。

首次执行在HTTP调用前因状态变量类型错误停止，`attempt1_preflight_failure.json`记录0 GET；机械修复后仅完成1次官方请求，没有追加预算或备用站点。SOXX稳定净超额、2/5bp和独立前向均未达到。

**已归档页面内嵌数据补审（零新请求）：** `__NEXT_DATA__`有140条当前基金记录，其中仅一条SOXL，交易日期2026-10-22；该SOXL记录没有AUM、份额数或净流入字段。四个“Daily Fund Holdings”锚点的原始`href`均为`#`，产品字段也没有直接历史CSV链接。详见[embedded_page_audit_result.json](embedded_page_audit_result.json)。这不排除另有官方动态接口，但该接口尚未获得或认证。
