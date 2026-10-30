# 原始逐分钟成交额：Qlib对齐通过，隐含VWAP语义门失败

**SOXX：历史Goal未通过。** 只读已NAS快照备份的Futu `NONE` SOXL/SOXX一分钟原件，源manifest SHA `18f1fb3f17907932ddd13881aea9199369181a92c9b6c77395d0b255100c68c3`；本地两parquet及NAS快照逐件SHA一致，见[`nas_original_readback.json`](nas_original_readback.json)。Qlib双ETF2018–2022共**820440行**与原件时间戳一对一、原价成交量最大差**0**，原价OHLC浮点最大差约3.05×10⁻⁵；已知量bar报告成交额正值率100%。官方[富途历史K线说明](https://openapi.futunn.com/futu-api-doc/quote/request-history-kline.html)把`turnover`列为“成交额”、美股`time_key`为美东时间，但没有给出该成交额与高低价使用同一成交流的保证。

若将`turnover/volume`当**同一分钟成交均价**，在放宽1bp后只有**90.77%**位于该bar原价高低范围，未达事前**99.9%**语义门。2018–2020每个来源日都存在相对OHLC4的bp级残差，说明字段不是简单列复制；数值新颖性不能弥补口径不一致。来源门决定`reported_minute_turnover_same_bar_vwap_semantics_failed`：不把它加入Qlib作为已认证真实VWAP特征，不读取收益、不拟合或跑账户。历史真实首收仍`needs_data`。旧`close×volume`15分钟代理和`open×volume`时序路线另有封存失败作用域；本卡没有重开它们。
