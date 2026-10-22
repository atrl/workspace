# SOXX Series ID 检索：来源入口已核对

按冻结 URL 读取 SEC EDGAR 的 `S000004354` series 申报页，两页均 HTTP 200；页面标题区同时标明 iShares Trust CIK `0001100663`、Series `S000004354` 和现名 iShares Semiconductor ETF。N-Q 页列出 **27 条**（2006–2019），其中 2011–2018 有 **15 条**；NPORT-P 搜索页列出 **28 条 NPORT-P 与 1 条 NPORT-P/A**（2019–2026）。两页原始字节和响应头已归档 NAS，SHA-256 `14789c4727e1a518511e56dbc6112a975798a2dc0efc2def0e234de54f8bdc90`，远端哈希与每个原件限定恢复一致。

零网络解析中的第一版把 NPORT-P/A 误当成异常而停止；保留失败回执后按固定补充规则只允许该修订形式，未增加 GET 或改变任何 accession。2011–2018 的 15 条 N-Q accession 全在原信托目录 188 条未修订候选中。已核实的 2011-04-30 SOXX 正文 accession 在 series 页；两条已核实不含 SOXX 的 2011-01-31 信托 N-Q 则不在。详见 `series_index_parse.json`。

第一阶段取得较精确的 **SOXX 系列申报索引**。当前基金名称不代表历史每期正文已核对。页面只有申报日与链接，不能代替 SEC 接受时钟、报告期、原始证券数值或历史可交易信息。2 个索引 GET、0 正文、0 市场收益、0 因子、0 拟合、0 账户；不能用今天的 ETF 成分回填旧日。

## N-CSR/N-CSRS 系列索引与时间绑定

另冻两个索引 URL。第一份 `type=N-CSR` 搜索页 HTTP 200，SEC 实际把 N-CSR 22、N-CSRS 21、N-CSR/A 3、N-CSRS/A 2 一起返回（共 48 条）；初版只接受 N-CSR 而停止，失败回执保留。新零网络解析方案从**同一原页**按四类分开读取，因此没有再发原计划的第二个重复 GET。该页面原件及收据在 NAS 归档 SHA-256 `c22146476851d3c9853d9c1da489fe54402f113693f72e56e176d98111f5da71`，远端哈希与原字节恢复通过。

合并 N-Q 与 N-CSR/N-CSRS 的 series 索引，并以已封存信托目录逐 accession 补入 `reportDate` 和 `acceptanceDateTime`：**2011–2018 年有 32 条未修订的不同报告期候选，另有 5 条系列级修订申报**。其中 2011-01-31 报告期对应 N-CSRS `0001193125-11-093502`，SEC 接受时刻为 2011-04-11 20:08:56 UTC，而不是之前被排除的两份 1 月 N-Q。完整索引和来源哈希见 `series_report_calendar_inventory.json`。主来源卡已另行确认 2011 四个报告期的原始 SOXX 表身份；本卡的 32 条目录不能替代其逐正文核验。

最终决定：`SEC_series_specific_2011_2018_report_metadata_32_base_5_amendments_bodies_pending`。本卡合计 **3 个索引 GET、0 申报正文、0 收益、0 因子、0 拟合、0 账户**。2019–2022 NPORT 的 series accession 已列出，接受时钟和原始持仓值仍待核对；历史成分股每日价格也是独立缺口。只有这些来源和完整 Stage1 日历通过，才有资格冻结新低扩散／下跌集中度信息检验。
