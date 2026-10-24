# Direxion产品页脚本只暴露持仓弹窗，未给出历史AUM接口

从先前已归档的官方SOXL产品页中固定选取其唯一 `pages/product/[slug]` JavaScript资源，仅向Direxion同域请求一次。得到HTTP 200、**64,769字节** `application/javascript`，精确响应与UTC时间、SHA-256在`result.json`；原字节压缩包SHA-256 `cfb0fbd9e74d1f2e84064785c4f2d3122e1d4a2e7e51120a35bc2730403e3ae8` 已归档Yanxuan NAS、远端哈希核对并定向恢复。未执行脚本、跟随其引用或请求其他chunk/API；0市场结果、拟合、账户和供应商联系。

此产品页脚本的“Daily Fund Holdings”点击处理只是调用 **`show-daily-holdings` 模态窗**。文本含持仓/下载代码，但没有 AUM、份额字段、直接历史CSV URL或可直接识别的 `/api/` 历史路径；唯一出现的“history”是浏览器地址历史处理，不是基金时间序列。完整限定关键字上下文见`result.json`。

决定 `source_only_product_chunk_no_daily_historical_AUM_shares_endpoint_in_scope`。这只排除**本次所选产品页chunk**直接给出2011–2022日级AUM/份额与首发版本；共享chunk或发行人其他官方动态接口未读，不能说全站不存在。实际再平衡需求因子仍为 `needs_data`，不会把此前已失败的尾盘价格残差改名为基金流量重测。若继续追共享模块，必须另冻URL、次数、响应保存及首次公布门，再决定是否值得进入Stage1信息/账户检验。
