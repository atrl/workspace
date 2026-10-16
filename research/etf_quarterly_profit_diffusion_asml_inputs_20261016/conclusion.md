# 季度利润扩散：ASML来源补齐版本结论

本版只接入已核对的ASML US GAAP EUR直接季度净利润渠道，其余同口径同比公式、基金权重、95%覆盖、180日时效、最新季度未知标记和全部月份规则不变。独立的原19合格月输入版本保留，未重新命名年度估值、价格广度或单一TSMC月营收路线。

完整保留93个月末，79个月通过源覆盖门，从2019-11至2026-08；2490个发行人/月中2335个同比配对有效。2019有2个合格月，2020–2023各12个，2024为11个、2025为10个、2026截至9月为8个。未披露篮子前10个月保留；另有2024-12、2025-01、2025-02、2026-09未通过，覆盖约89.90%、89.90%、85.76%、92.53%，没有用旧季度、零或重归一补齐。

四个主字段为profit_improve_weight、profit_deteriorate_weight、profit_bounded_growth、quarterly_loss_weight。四个共享源控制字段为known_profit_age_mean_days、holdings_age_days、pair_coverage_weight、current_profit_coverage_weight。不合格月份主字段为未知，当前利润亏损权重与同比配对覆盖分别记录；无汇率/股份/价格分母，也无跨币种净金额汇总。

时间为月末16:15纽约信号、财务来源下一交易日09:30研究生效；CIK/会计标签/范围/币种严格分开，ASML实际财政季度及其已知日期锚保留。权重仍是迟报篮子代理；MRVL旧报告主体映射不冒充当前法律组合。源修正不提供新的独立市场观察。

已执行notebook并完成Qlib月末字段读回、原生Recorder/NAS和独立源评估，没有在本卡读取目标收益、拟合或运行账户。79个月是来源资格，不是稳定超额或晋升。下游信息与配置实验分别依据自己的冻结规则和独立评估作结论。

缓存交付：新Recorder的inputs/data_cache_files保存78份外国季度、ASML、原季度扩散与本版本的实际provider/snapshot缓存文件；data_cache_manifest.json给出逐文件hash，provider/snapshot紧凑manifest保留卡根。上游封存Recorder未改动，精确8条ignore路径覆盖可重建副本。所有金融重建输入与当前规范帧均在NAS，无需为恢复缓存重新研究或拟合。
