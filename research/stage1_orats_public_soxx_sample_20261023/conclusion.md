# ORATS 官方公开样本：SOXX 25Δ 行权价链存在

[ORATS 官方历史数据页](https://orats.com/university/historical-data)公开的 2024-01-03 近收盘 ZIP，经冻结1次 HEAD、1次 GET 获取；压缩原件66,808,802字节、SHA-256 `e04a3731…`，716,822条全市场行权价记录中 **1,124条为 SOXX**。Yanxuan NAS 归档 SHA-256 `a91c6d1b…` 已远端哈希与限定恢复；原件可按收据恢复，不依赖未归档本地缓存。零购买、零 SOXX 未来收益读取、零拟合、零账户。

SOXX 样本中1,050条看涨和957条看跌行同时具备正值双边报价、正值mid IV与有效 Delta。[ORATS 字段说明](https://orats.com/near-eod-data)规定该单列 `delta` 是看涨 Delta，看跌 Delta 为其减1。距离30日最近的两个月期限分别是 **16日和44日**；两期限上，+0.25看涨和−0.25看跌均有有效报价与IV的上下行权价括号。具体行权价、bid/ask、IV、Delta与合约标识见 `source_audit.json`。这证明**单日 SOXX 链的特征可构造性**，未计算或检验收益预测。

**来源仍未通过历史门。** 当前样本只有2024-01-03，不能推断2011–2022 SOXX逐日覆盖、缺失与拆分后合约身份；CSV只有交易日期，没有每行首次报价/文件送达时间。ORATS产品说明称近收盘数据夜间交付，但未认证每个历史日期在下一交易日 Stage1 09:46 前已按相同数值可得。已更新[只询样邮件草稿](../soxx_option_vendor_sample_preflight_20261023/orats_inquiry_draft.md)以集中请求2011/2012原始 SOXX 样本和发布/修订时钟，尚未发送或购买。之后只有完整来源通过，才可冻结25Δ约30日偏斜的匹配信息门与完整Stage1 0bp Tiered账户。
