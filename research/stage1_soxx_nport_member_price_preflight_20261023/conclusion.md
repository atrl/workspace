# SOXX 原始 N-PORT 成分股价格来源门

13 份原始 N-PORT 的 **390 条**普通股市值与既有 ticker 桥逐 accession、逐原值闭合：357 条原始 CUSIP 与市值均匹配，33 条原件 CUSIP 为 `000000000`／`N/A`，只具备同报告内名称及市值匹配，未据占位符推断跨报告身份。这里的 ticker 桥仅用于诊断现有价格来源，不改变原始证券标识。

按当时已申报的篮子逐日检查 2019–2022 年 780 个 N-PORT 生效交易日，要求每只成员在信号日及之前 20 个真实 Qlib 交易日有正且有限的调整收盘价。现有 `soxx_pit_leveraged_training_v1` Qlib 价格库在篮子所涉 39 个 ticker 中缺 **BRKS、CCMP、CREE、IPHI、MXIM、WOLF、XLNX** 七个；**完整 30 股价格窗口为 0/780 日**，严格 CUSIP 身份加完整价格也为 0/780。每天缺价成员中位数 4，缺失的原始普通股权重中位数 5.70%，最大 7.69%；没有丢弃成员或重归一缺口。工作区另登记 WOLF/CREE 与 BRKS 的补充来源，因此七个是**本 Qlib provider 的缺口**，不能扩大为所有备份均不存在；其余缺口及价格的历史首次接收版本仍需另证。

本卡使用原生 Qlib `D.calendar`／一次 `D.features` 输入价格读取、DatasetH 和 Recorder `073fd00eeb594f82be3741ad091091c6`。Recorder 的 Yanxuan NAS SHA-256 为 `acc9cac4842d3cbd84af121d50f38d73b22c6daef409a1d64cf3c80c196db51d`，远端哈希和限定恢复通过。新增 0 HTTP 请求、0 SOXX 收益／标签读取、0 拟合、0 原生账户。

决定：`NPORT_member_price_source_coverage_diagnostic_no_Stage1_feature_or_account`。按当前价格库，预定的完整篮子“新低扩散／下跌集中度”机制**不能进入信息拟合**；对原始披露市值直接计算的前五股占比／HHI 不依赖成分股价格，待全 Stage1 来源合格后可另立一个有界、可证伪机制。此前失败的四项广度特征不重开。
