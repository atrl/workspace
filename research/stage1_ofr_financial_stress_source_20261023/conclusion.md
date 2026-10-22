# OFR 金融压力指数：官方全历史来源与修订边界

官方 CSV 收到 6,784 个无重复日期（2000-01-03 至 2026-10-20），`Funding` 等九列均有数值；官方修订工作簿与 CSV 原字节已归档至 NAS，远端哈希及每个原始成员限定恢复通过。0 SOXX 结果读取、0 拟合、0 账户。

来源风险是实质性的：2018-05-18 的 158 个改值日期、2018-12-04 的 2 个日期，以及 2023-06-27 的 382 个日期均有官方旧/新值；2023 年替换资金类别变量后，Funding 绝对变动中位数 0.307、最大 0.764 个指数点。工作簿“新值”与当前 CSV 在对应日期逐项一致，不能把它当作当年旧值。官网另提示 2026 年数据订正，旧版本不在 2023 工作簿内。

[OFR 于 2017-10-25 发布的说明](https://www.financialresearch.gov/from-the-director/2017/10/25/the-ofr-introduces-financial-stability-index-and-financial-vulnerabilities-monitor/)把现行每日 FSI 列为新工具；[2014 年 OFR 报告](https://www.financialresearch.gov/annual-reports/files/office-of-financial-research-annual-report-2014.pdf)描述较早的 FSI，但不能证明现行 33 指标序列的 2000–2017 回填与当时版本完全相同。因此**不能**把今天 CSV 的 2011–2017 数值直接并入冻结 Stage1 全训练期当点时信号。

[当前 OFR 页面](https://www.financialresearch.gov/financial-stress-index/)称公布值较观测日滞后两个工作日；这不是各历史日的首次接收证明。下一步只可对 2018–2022 的 10:32 恢复接收器做单独的条件性来源映射，保守滞后、复原官方已列修订旧值、保留未知；通过后还须检验旧 21 输入增量和完整 0bp 账户。

来源 NAS SHA-256 `074e2c56f6f95a4f56382a32e25c65ad560e5a53ee6e4bb7fd58e50bc247b369`。本卡决定：`official_full_numeric_history_received_pre2017_PIT_and_revision_boundaries_unqualified`，不授予交易时点、SOXX 超额或晋升资格。
