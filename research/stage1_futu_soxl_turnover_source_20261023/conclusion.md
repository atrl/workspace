# SOXL 换手率字段纠正（优先结论）

2010–2011 的444个 `turnover_rate=0` 行均有正成交量和成交额，其中2011年252日属于 Stage1 训练期。零值不能当真实换手；有效训练覆盖为 **2766/3020**，低于原计划3000日门。此同字段相对换手率版本在拟合前关闭；此前有限/非负覆盖通过的结论由 [`field_semantics_correction.json`](field_semantics_correction.json) 覆盖。原始响应、NAS备份和 Qlib Recorder 保留。

# Stage1 Futu SOXL 日换手率：同口径来源与覆盖门

Futu `US.SOXL` 完成日 K 线、AuType.NONE 只读取得换手率。原始表和请求回执均按 SHA-256 备份 NAS 并逐件恢复；原生 Qlib 来源与原 Stage1 信号日历核对。没有新闻、未来收益、拟合或资金账户。

2011–2022 训练 3018/3020 日、2023–2026 复用评价 946/946 日有值；缺日 ['2012-01-13', '2013-10-28']。来源门 5/5，决定 `conditional_source_coverage_pass_first_public_unknown`。

历史首次公开版本、修订与每日收盘后可用时间仍未认证。此卡仅为后续 SOXL 相对 SOXX 换手假设补全同字段来源；已关闭的 SOXX 单边换手异常不能改名重开。下一步先按原39项和 SOXX 换手做无结果去重，再另冻接收任务与信息门；通过后才可能考虑原 f696 父本0bp Tiered完整账户。

原始 NAS SHA `a53917c27afbd40a704a02a84818233d88586d899dfd8a328001828fdac09307`；Qlib Recorder `cc528c4f412a4336bd644f09f0a57caa` NAS SHA `67c881ae0d22aa279b569ffffc9eb3d56f5c5a649d47d11514f33373339d07dd`。SOXX 超额、2/5bp及独立前向仍未实现。
