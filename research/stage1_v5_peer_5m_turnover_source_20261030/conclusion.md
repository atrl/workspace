# QQQ/SMH原始5分钟成交额：同bar成交均价语义门失败

**SOXX判定：历史Goal未通过。** 本卡只检查现有NAS已备份的Futu `NONE` QQQ/SMH K_5M原件，不读取未来标签、零新拟合、零资金账户。两parquet本地和NAS逐件SHA一致，见[`nas_original_readback.json`](nas_original_readback.json)；Qlib同行5分钟provider读取与原件时间戳逐行对齐，2018–2022各**1056个来源日**、合计**164088行**，原价成交量最大差**0**，OHLC浮点最大差约1.52×10⁻⁵。正成交量且价格有效的bar中报告成交额均为正。官方[富途历史K线文档](https://openapi.futunn.com/futu-api-doc/quote/request-history-kline.html)把`turnover`列为“成交额”，但没有证明它与bar高低价使用同一成交过滤口径。

冻结必要语义门要求每只ETF的`turnover/volume`在原价OHLC高低范围放宽1bp后**至少99.9%**落入；实际QQQ仅**93.2457%**、SMH**99.7416%**，两者均失败。发现期两者相对OHLC4仍有覆盖全部来源日的bp级残差；数值差异不足以认证为本bar真实成交均价。2018–2022逐年覆盖与残差精数见[`audit.json`](audit.json)、[`annual_source_summary.csv`](annual_source_summary.csv)。低量与高量的事后筛法不能作为这版来源门救援，SMH接近99.9%也不能只为其放宽；无需、也未发起新的Futu请求或供应商联络。

决定`peer_reported_5m_turnover_semantics_failed_before_fit`。该字段在现有原件中的**同bar VWAP解释**未获资格，不能直接并入后续固定股数五会话信息/账户；不等于所有原始成交额信息在理论上无价值，也不证明Futu交易额“错误”，具体成交过滤/时点差异仍`unknown`。旧QQQ/SMH价量路径、AMD/NVDA/SMH广度和已失败条件抗跌版本保持原作用域；当前没有新的SOXX超额或2023+候选账户。下一步只按当前Qlib/NAS已登记来源与关闭路线做限定队列准入审计，若仍无实质不同、点时可用的信息和匹配接收任务，记录`needs_distinct_decision_time_mechanism`，不继续轮换价量字段。
